import copy
import time
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.progress import Progress, track

from pytorch_tabular import TabularModel, models
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.utils import (
    OOMException,
    OutOfMemoryHandler,
    available_models,
    get_logger,
    int_to_human_readable,
    suppress_lightning_logs,
)

from pytorch_lightning.tuner.tuning import Tuner
from pathlib import Path
import os

logger = get_logger("pytorch_tabular")

MODEL_SWEEP_PRESETS = {
    "lite": (
        ("CategoryEmbeddingModelConfig", {"layers": "256-128-64"}),
        ("GANDALFConfig", {"gflu_stages": 6}),
        (
            "TabNetModelConfig",
            {
                "n_d": 32,
                "n_a": 32,
                "n_steps": 3,
                "gamma": 1.5,
                "n_independent": 1,
                "n_shared": 2,
            },
        ),
    ),
    "standard": (
        ("CategoryEmbeddingModelConfig", {"layers": "256-128-64"}),
        ("CategoryEmbeddingModelConfig", {"layers": "512-128-64"}),
        ("GANDALFConfig", {"gflu_stages": 6}),
        ("GANDALFConfig", {"gflu_stages": 15}),
        (
            "TabNetModelConfig",
            {
                "n_d": 32,
                "n_a": 32,
                "n_steps": 3,
                "gamma": 1.5,
                "n_independent": 1,
                "n_shared": 2,
            },
        ),
        (
            "TabNetModelConfig",
            {
                "n_d": 32,
                "n_a": 32,
                "n_steps": 5,
                "gamma": 1.5,
                "n_independent": 2,
                "n_shared": 3,
            },
        ),
        ("FTTransformerConfig", {"num_heads": 4, "num_attn_blocks": 4}),
    ),
    "full": (m for m in available_models() if m not in ["MDNConfig", "NodeConfig"]),
    "high_memory": (m for m in available_models() if m not in ["MDNConfig"]),
}


def _validate_args(
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: Optional[List[Union[str, Callable]]] = None,
    metrics_params: Optional[List[dict]] = None,
    metrics_prob_input: Optional[List[bool]] = None,
    common_model_args: Optional[dict] = {},
    rank_metric: Optional[str] = "loss",
):
    assert task in [
        "classification",
        "regression",
    ], f"task must be one of ['classification', 'regression'], but got {task}"
    assert isinstance(train, pd.DataFrame), f"train must be a pandas DataFrame, but got {type(train)}"
    assert isinstance(test, pd.DataFrame), f"test must be a pandas DataFrame, but got {type(test)}"
    if metrics is not None:
        assert isinstance(metrics, list), f"metrics must be a list of strings or callables, but got {type(metrics)}"
        assert all(
            isinstance(m, (str, Callable)) for m in metrics
        ), f"metrics must be a list of strings or callables, but got {metrics}"
        assert metrics_params is not None, "metric_params cannot be None when metrics is not None"
        assert metrics_prob_input is not None, "metrics_prob_inputs cannot be None when metrics is not None"
        assert isinstance(
            metrics_params, list
        ), f"metric_params must be a list of dicts, but got {type(metrics_params)}"
        assert isinstance(metrics_prob_input, list), (
            "metrics_prob_inputs must be a list of bools, but got" f" {type(metrics_prob_input)}"
        )
        assert len(metrics) == len(metrics_params), (
            "metrics and metric_params must be of the same length, but got" f" {len(metrics)} and {len(metrics_params)}"
        )
        assert len(metrics) == len(metrics_prob_input), (
            "metrics and metrics_prob_inputs must be of the same length, but got"
            f" {len(metrics)} and {len(metrics_prob_input)}"
        )
        assert all(
            isinstance(m, dict) for m in metrics_params
        ), f"metric_params must be a list of dicts, but got {metrics_params}"
    if common_model_args is not None:
        # all args should be members of ModelConfig
        assert all(k in ModelConfig.__dataclass_fields__.keys() for k in common_model_args.keys()), (
            "common_model_args must be a subset of ModelConfig, but got" f" {common_model_args.keys()}"
        )
    if rank_metric[0] not in ["loss", "accuracy", "mean_squared_error"]:
        assert rank_metric[0] in metrics, f"rank_metric must be one of {metrics}, but got {rank_metric}"
    assert rank_metric[1] in [
        "lower_is_better",
        "higher_is_better",
    ], (
        "rank_metric[1] must be one of ['lower_is_better', 'higher_is_better'], but" f" got {rank_metric[1]}"
    )


def _validate_arg_model_list(model_list, task):
    assert model_list is not None, "models cannot be None"
    assert isinstance(
        model_list, (str, list)
    ), f"models must be a string or list of strings, but got {type(model_list)}"
    if isinstance(model_list, str):
        assert (
                model_list in MODEL_SWEEP_PRESETS.keys()
        ), f"models must be one of {MODEL_SWEEP_PRESETS.keys()}, but got {model_list}"
    else:  # isinstance(models, list):
        assert all(
            isinstance(m, (str, ModelConfig)) for m in model_list
        ), f"models must be a list of strings or ModelConfigs, but got {model_list}"
        assert all(task == m.task for m in model_list if isinstance(m, ModelConfig)), (
            f"task must be the same as the task in ModelConfig, but got {task} and"
            f" {[m.task for m in model_list if isinstance(m, ModelConfig)]}"
        )


def model_sweep_custom(
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: Union[DataConfig, str],
    optimizer_config: Union[OptimizerConfig, str],
    trainer_config: Union[TrainerConfig, str],
    model_list: Union[str, List[Union[ModelConfig, str]]] = "lite",
    metrics: Optional[List[Union[str, Callable]]] = None,
    metrics_params: Optional[List[dict]] = None,
    metrics_prob_input: Optional[List[bool]] = None,
    validation: Optional[pd.DataFrame] = None,
    experiment_config: Optional[Union[ExperimentConfig, str]] = None,
    common_model_args: Optional[dict] = {},
    rank_metric: Optional[Tuple[str, str]] = ("loss", "lower_is_better"),
    return_best_model: bool = True,
    seed: int = 42,
    ignore_oom: bool = True,
    progress_bar: bool = True,
    verbose: bool = True,
    suppress_lightning_logger: bool = True,
    min_lr: float = 1e-8,
    max_lr: float = 1,
    num_training: int = 100,
    mode: str = "exponential",
    early_stop_threshold: Optional[float] = 4.0,
    **kwargs,
):
    """Compare multiple models on the same dataset.

    Args:
        task (str): The type of prediction task. Either 'classification' or 'regression'

        train (pd.DataFrame): The training data

        test (pd.DataFrame): The test data on which performance is evaluated

        data_config (Union[DataConfig, str]): DataConfig object or path to the yaml file.

        optimizer_config (Union[OptimizerConfig, str]): OptimizerConfig object or path to the yaml file.

        trainer_config (Union[TrainerConfig, str]): TrainerConfig object or path to the yaml file.

        model_list (Union[str, List[Union[ModelConfig, str]]], optional): The list of models to compare.
                This can be one of the presets defined in ``pytorch_tabular.tabular_model_sweep.MODEL_SWEEP_PRESETS``
                or a list of ``ModelConfig`` objects. Defaults to "lite".

        metrics (Optional[List[str]]): the list of metrics you need to track during training. The metrics
                should be one of the functional metrics implemented in ``torchmetrics``. By default, it is
                accuracy if classification and mean_squared_error for regression

        metrics_prob_input (Optional[bool]): Is a mandatory parameter for classification metrics defined in
                the config. This defines whether the input to the metric function is the probability or the class.
                Length should be same as the number of metrics. Defaults to None.

        metrics_params (Optional[List]): The parameters to be passed to the metrics function. `task` is forced to
                be `multiclass` because the multiclass version can handle binary as well and for simplicity we are
                only using `multiclass`.

        validation (Optional[DataFrame], optional):
                If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation.
                Defaults to None.

        experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to
                the yaml file.

        common_model_args (Optional[dict], optional): The model argument which are common to all models. The list
                of params can be found in ``ModelConfig``. If not provided, will use defaults. Defaults to {}.

        rank_metric (Optional[Tuple[str, str]], optional): The metric to use for ranking the models. The first element
                of the tuple is the metric name and the second element is the direction.
                Defaults to ('loss', "lower_is_better").

        return_best_model (bool, optional): If True, will return the best model. Defaults to True.

        seed (int, optional): The seed for reproducibility. Defaults to 42.

        ignore_oom (bool, optional): If True, will ignore the Out of Memory error and continue with the next model.

        progress_bar (bool, optional): If True, will show a progress bar. Defaults to True.

        verbose (bool, optional): If True, will print the progress. Defaults to True.

        suppress_lightning_logger (bool, optional): If True, will suppress the lightning logger. Defaults to True.

        min_lr (Optional[float], optional): minimum learning rate to investigate

        max_lr (Optional[float], optional): maximum learning rate to investigate

        num_training (Optional[int], optional): number of learning rates to test

        mode (Optional[str], optional): search strategy, either 'linear' or 'exponential'. If set to
            'linear' the learning rate will be searched by linearly increasing
            after each batch. If set to 'exponential', will increase learning
            rate exponentially.

        early_stop_threshold (Optional[float], optional): threshold for stopping the search. If the
            loss at any point is larger than early_stop_threshold*best_loss
            then the search is stopped. To disable, set to None.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

        Returns:
            results: Training results.

            best_model: If return_best_model is True, return best_model otherwise return None.
    """
    _validate_args(
        task=task,
        train=train,
        test=test,
        metrics=metrics,
        metrics_params=metrics_params,
        metrics_prob_input=metrics_prob_input,
        common_model_args=common_model_args,
        rank_metric=rank_metric,
    )
    _validate_arg_model_list(model_list, task)

    if suppress_lightning_logger:
        suppress_lightning_logs()

    if progress_bar:
        if trainer_config.progress_bar != "none":
            # Turning off thie internal progress bar to avoid conflict with sweep progress bar
            warnings.warn(
                "Training Progress bar is not `none`. Set `progress_bar=none` in"
                " `trainer_config` to remove this warning"
            )
            trainer_config.progress_bar = "none"

    if model_list in ["full", "high_memory"]:
        warnings.warn(
            "The full model list is quite large and uses a lot of memory. "
            "Consider using `lite` or define configs yourselves for a faster run"
        )
    _model_args = ["metrics", "metrics_params", "metrics_prob_input"]
    # Replacing the common model args with the ones passed in the function
    for arg in _model_args:
        if locals()[arg] is not None:
            common_model_args[arg] = locals()[arg]
    if isinstance(model_list, str):
        model_list = copy.deepcopy(MODEL_SWEEP_PRESETS[model_list])
        model_list = [
            (
                getattr(models, model_config[0])(task=task, **model_config[1], **common_model_args)
                if isinstance(model_config, Tuple)
                else (
                    getattr(models, model_config)(task=task, **common_model_args)
                    if isinstance(model_config, str)
                    else model_config
                )
            )
            for model_config in model_list
        ]

    def _init_tabular_model(m):
        return TabularModel(
            data_config=data_config,
            model_config=m,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            experiment_config=experiment_config,
            verbose=False,
        )

    init_tabular_model = _init_tabular_model(model_list[0])

    prep_dl_kwargs, prep_model_kwargs, train_kwargs = init_tabular_model._split_kwargs(kwargs)

    datamodule = init_tabular_model.prepare_dataloader(train=train, validation=validation, seed=seed, **prep_dl_kwargs)
    results = []
    best_model = None
    is_lower_better = rank_metric[1] == "lower_is_better"
    best_score = 1e9 if is_lower_better else -1e9
    it = track(model_list, description="Sweeping Models") if progress_bar else model_list
    ctx = Progress() if progress_bar else nullcontext()
    with ctx as progress:
        if progress_bar:
            task_p = progress.add_task("Sweeping Models", total=len(model_list))
        for model_config in model_list:
            if isinstance(model_config, str):
                model_config = getattr(models, model_config)(task=task, **common_model_args)
            else:
                for key, val in common_model_args.items():
                    if hasattr(model_config, key):
                        setattr(model_config, key, val)
                    else:
                        raise ValueError(
                            f"ModelConfig {model_config.name} does not have an" f" attribute {key} in common_model_args"
                        )
            params = model_config.__dict__
            start_time = time.time()
            tabular_model = _init_tabular_model(model_config)
            name = tabular_model.name
            if verbose:
                logger.info(f"Training {name}")
            model = tabular_model.prepare_model(datamodule, **prep_model_kwargs)
            if progress_bar:
                progress.update(task_p, description=f"Training {name}", advance=1)
            with OutOfMemoryHandler(handle_oom=True) as handler:

                # Copy from train() method with additional lr_find parameters
                handle_oom = False
                tabular_model._prepare_for_training(model, datamodule, **train_kwargs)
                train_loader, val_loader = (
                    tabular_model.datamodule.train_dataloader(),
                    tabular_model.datamodule.val_dataloader(),
                )
                tabular_model.model.train()
                if tabular_model.config.auto_lr_find and (not tabular_model.config.fast_dev_run):
                    if tabular_model.verbose:
                        logger.info("Auto LR Find Started")
                    with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
                        lr_finder = Tuner(tabular_model.trainer).lr_find(
                            tabular_model.model,
                            train_dataloaders=train_loader,
                            val_dataloaders=val_loader,
                            min_lr=min_lr,
                            max_lr=max_lr,
                            num_training=num_training,
                            mode=mode,
                            early_stop_threshold=early_stop_threshold,
                        )
                    if oom_handler.oom_triggered:
                        raise OOMException(
                            "OOM detected during LR Find. Try reducing your batch_size or the"
                            " model parameters." + "/n" + "Original Error: " + oom_handler.oom_msg
                        )
                    if tabular_model.verbose:
                        logger.info(
                            f"Suggested LR: {lr_finder.suggestion()}. For plot and detailed"
                            " analysis, use `find_learning_rate` method."
                        )
                    tabular_model.model.reset_weights()
                    # Parameters in models needs to be initialized again after LR find
                    tabular_model.model.data_aware_initialization(tabular_model.datamodule)
                tabular_model.model.train()
                if tabular_model.verbose:
                    logger.info("Training Started")
                with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
                    tabular_model.trainer.fit(tabular_model.model, train_loader, val_loader)
                if oom_handler.oom_triggered:
                    raise OOMException(
                        "OOM detected during Training. Try reducing your batch_size or the"
                        " model parameters."
                        "/n" + "Original Error: " + oom_handler.oom_msg
                    )
                tabular_model._is_fitted = True
                if tabular_model.verbose:
                    logger.info("Training the model completed")
                if tabular_model.config.load_best:
                    tabular_model.load_best_model()

            res_dict = {
                "model": name,
                'learning_rate': lr_finder.suggestion(),
                "# Params": int_to_human_readable(tabular_model.num_params),
            }
            if handler.oom_triggered:
                if not ignore_oom:
                    raise OOMException(
                        "Out of memory error occurred during cross validation. "
                        "Set ignore_oom=True to ignore this error."
                    )
                else:
                    res_dict.update(
                        {
                            f"test_{rank_metric[0]}": (np.inf if is_lower_better else -np.inf),
                            "epochs": "OOM",
                            "time_taken": "OOM",
                            "time_taken_per_epoch": "OOM",
                        }
                    )
                    res_dict["model"] = name + " (OOM)"
            else:
                if (
                    tabular_model.trainer.early_stopping_callback is not None
                    and tabular_model.trainer.early_stopping_callback.stopped_epoch != 0
                ):
                    res_dict["epochs"] = tabular_model.trainer.early_stopping_callback.stopped_epoch
                else:
                    res_dict["epochs"] = tabular_model.trainer.max_epochs

                # Update results with train metrics
                train_metrics = tabular_model.evaluate(test=train, verbose=False)[0]
                metrics_names = list(train_metrics.keys())
                for m_name in metrics_names:
                    train_metrics[m_name.replace('test', 'train')] = train_metrics.pop(m_name)
                res_dict.update(train_metrics)

                # Update results with validation metrics
                validation_metrics = tabular_model.evaluate(test=validation, verbose=False)[0]
                metrics_names = list(validation_metrics.keys())
                print(validation_metrics)
                for m_name in metrics_names:
                    validation_metrics[m_name.replace('test', 'validation')] = validation_metrics.pop(m_name)
                res_dict.update(validation_metrics)

                # Update results with test metrics
                res_dict.update(tabular_model.evaluate(test=test, verbose=False)[0])

                res_dict["time_taken"] = time.time() - start_time
                res_dict["time_taken_per_epoch"] = res_dict["time_taken"] / res_dict["epochs"]

                if verbose:
                    logger.info(f"Finished Training {name}")
                    logger.info("Results:" f" {', '.join([f'{k}: {v}' for k,v in res_dict.items()])}")
                res_dict["params"] = params

                if tabular_model.trainer.checkpoint_callback:
                    res_dict["checkpoint"] = tabular_model.trainer.checkpoint_callback.best_model_path
                    save_dir = str(Path(res_dict["checkpoint"]).parent).replace('\\', '/') + '/' + Path(res_dict["checkpoint"]).stem
                    tabular_model.save_model(save_dir)
                    os.remove(res_dict["checkpoint"])

                results.append(res_dict)

                if tabular_model.config['checkpoints_path']:
                    try:
                        pd.DataFrame(results).style.background_gradient(
                            subset=[
                                "train_loss",
                                "validation_loss",
                                "test_loss",
                                "time_taken",
                                "time_taken_per_epoch"
                            ], cmap="RdYlGn_r"
                        ).to_excel(f"{tabular_model.config['checkpoints_path']}/progress.xlsx")
                    except PermissionError:
                        pass

                if return_best_model:
                    tabular_model.datamodule = None
                    if best_model is None:
                        best_model = copy.deepcopy(tabular_model)
                        best_score = res_dict[f"test_{rank_metric[0]}"]
                    else:
                        if is_lower_better:
                            if res_dict[f"test_{rank_metric[0]}"] < best_score:
                                best_model = copy.deepcopy(tabular_model)
                                best_score = res_dict[f"test_{rank_metric[0]}"]
                        else:
                            if res_dict[f"test_{rank_metric[0]}"] > best_score:
                                best_model = copy.deepcopy(tabular_model)
                                best_score = res_dict[f"test_{rank_metric[0]}"]
    if verbose:
        logger.info("Model Sweep Finished")
        logger.info(f"Best Model: {best_model.name}")
    results = pd.DataFrame(results).sort_values(by=f"test_{rank_metric[0]}", ascending=is_lower_better)
    if return_best_model and best_model is not None:
        best_model.datamodule = datamodule
        return results, best_model
    else:
        return results, None
