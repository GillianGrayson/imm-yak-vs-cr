import copy
import time
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.progress import Progress, track

from pytorch_tabular import TabularModel, models
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    GANDALFConfig,
    TabNetModelConfig,
    FTTransformerConfig,
    DANetConfig
)
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.models.common.heads import LinearHeadConfig
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
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

import optuna

from src.utils.configs import read_parse_config

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from pathlib import Path
import os

logger = get_logger("pytorch_tabular")


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def get_model_config_trial(
        trial: optuna.Trial,
        model_config_default
):
    model_config = copy.deepcopy(model_config_default)
    model_config['head_config']['dropout'] = trial.suggest_float('head_dropout', 0.0, 0.3)
    if model_config_default._model_name == 'GANDALFModel':
        model_config['gflu_stages'] = trial.suggest_int('gflu_stages', 1, 20)
        model_config['gflu_dropout'] = trial.suggest_float('gflu_dropout', 0.0, 0.25)
        model_config['gflu_feature_init_sparsity'] = trial.suggest_float('gflu_feature_init_sparsity', 0.05, 0.55)
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1.00, log=True)
    elif model_config_default._model_name == 'DANetModel':
        model_config['n_layers'] = trial.suggest_int('n_layers', 16, 32)
        model_config['abstlay_dim_1'] = trial.suggest_categorical('abstlay_dim_1', [8, 16, 32])
        model_config['k'] = trial.suggest_int('k', 2, 3)
        model_config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.05, 0.25)
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.0005, 0.5, log=True)
    elif model_config_default._model_name == 'CategoryEmbeddingModel':
        model_config['layers'] = trial.suggest_categorical('layers', ["256-128-64", "512-256-128", "256-128-64", "32-16", "64-32-16", "32-16-8", "128-64", "128-128", "16-16"])
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 1.0, log=True)
    elif model_config_default._model_name == 'TabNetModel':
        model_config['n_d'] = trial.suggest_int('n_d', 4, 64, step=4)
        model_config['n_a'] = trial.suggest_int('n_a', 4, 64, step=4)
        model_config['n_steps'] = trial.suggest_int('n_steps', 3, 7)
        model_config['gamma'] = trial.suggest_float('gamma', 1.3, 1.8)
        model_config['n_independent'] = trial.suggest_int('n_independent', 1, 4)
        model_config['n_shared'] = trial.suggest_int('n_shared', 1, 4)
        model_config['mask_type'] = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 10.0, log=True)
    elif model_config_default._model_name == 'FTTransformerModel':
        model_config['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
        model_config['num_attn_blocks'] = trial.suggest_int('num_attn_blocks', 2, 16, step=2)
        model_config['attn_dropout'] = trial.suggest_float('attn_dropout', 0.0, 0.25)
        model_config['add_norm_dropout'] = trial.suggest_float('add_norm_dropout', 0.0, 0.25)
        model_config['ff_dropout'] = trial.suggest_float('ff_dropout', 0.0, 0.25)
        # model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.000001, 0.5, log=True) # For Immunomarkers
        model_config['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 5.0, log=True) # For EpImAge
    else:
        raise ValueError(f"Model {model_config_default._model_name} not supported for Optuna trials")

    return model_config


def get_optimizer_config_trial(
        trial: optuna.Trial,
        optimizer_config_default
):
    optimizer_config = copy.deepcopy(optimizer_config_default)

    if optimizer_config_default.optimizer == 'Adam':
        optimizer_config['optimizer_params']['weight_decay'] = trial.suggest_float('optimizer_params_weight_decay', 1e-8, 1e-4, log=True)
    else:
        raise ValueError(f"Optimizer {optimizer_config_default.optimizer} not supported for Optuna trials")

    if optimizer_config_default.lr_scheduler == 'ReduceLROnPlateau':
        optimizer_config['lr_scheduler_params']['factor'] = trial.suggest_float('lr_scheduler_params_factor', 0.01, 0.99, log=False)
    elif optimizer_config_default.lr_scheduler == 'StepLR':
        pass
    else:
        raise ValueError(f"Learning Rate Scheduler {optimizer_config_default.lr_scheduler} not supported for Optuna trials")

    return optimizer_config


def get_data_config_trial(
        trial: optuna.Trial,
        data_config_default
):
    data_config = copy.deepcopy(data_config_default)

    data_config['continuous_feature_transform'] = trial.suggest_categorical(
        'continuous_feature_transform',
        [None, "yeo-johnson", "box-cox", "quantile_normal", "quantile_uniform"]
    )

    return data_config


def train_hyper_opt(
        trial: optuna.Trial,
        trials_results: List[dict],
        opt_metrics: List[Tuple[str, str]],
        opt_parts: List[str],
        model_config_default: Union[ModelConfig, str],
        data_config_default: Union[DataConfig, str],
        optimizer_config_default: Union[OptimizerConfig, str],
        trainer_config_default: Union[TrainerConfig, str],
        experiment_config_default: Optional[Union[ExperimentConfig, str]],
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        datamodule: TabularDatamodule,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        handle_oom: bool = True,
        ignore_oom: bool = True,
        verbose: bool = False,
        suppress_lightning_logger: bool = True,
        **kwargs,
):
    """Trains the model with hyperparameter selection from Optuna trials.

    Args:

        trial (optuna.Trial):
            Optuna trial object, which varies hyperparameters.

        trials_results (List[dict]):
            List with results of optuna trials.

        opt_metrics (List[Tuple[str, str]]):
            List of pairs ('metric name', 'direction') for optimization.

        opt_parts (List[str]):
            List of optimization parts: 'train', 'validation', 'test'.

        model_config_default (Union[ModelConfig, str]):
            A subclass of ModelConfig or path to the yaml file with default model configuration.
            Determines which model to run from the type of config.

        data_config_default (Union[DataConfig, str]):
            DataConfig object or path to the yaml file. Defaults to None.

        optimizer_config_default (Union[OptimizerConfig, str]): The OptimizerConfig for the TabularModel.
            If str is passed, will initialize the OptimizerConfig using the yaml file in that path.

        trainer_config_default (Union[TrainerConfig, str]): The TrainerConfig for the TabularModel.
            If str is passed, will initialize the TrainerConfig using the yaml file in that path.

        experiment_config_default (Union[ExperimentConfig, str]): ExperimentConfig object or path to the yaml file.

        train (pd.DataFrame): The training data.

        validation (pd.DataFrame): The validation data while training.
            Used in Early Stopping and Logging.

        test (pd.DataFrame): The test data on which performance is evaluated.

        datamodule (TabularDatamodule): The datamodule.

        min_lr (Optional[float], optional): minimum learning rate to investigate

        max_lr (Optional[float], optional): maximum learning rate to investigate

        num_training (Optional[int], optional): number of learning rates to test

        mode (Optional[str], optional): search strategy, either 'linear' or 'exponential'.
            If set to 'linear' the learning rate will be searched by linearly increasing after each batch.
            If set to 'exponential', will increase learning rate exponentially.

        early_stop_threshold (Optional[float], optional): threshold for stopping the search.
            If the loss at any point is larger than early_stop_threshold*best_loss then the search is stopped.
            To disable, set to None.

        handle_oom (bool): If True, will try to handle OOM errors elegantly.

        ignore_oom (bool, optional): If True, will ignore the Out of Memory error and continue with the next model.

        verbose (bool, optional): If True, will print the progress.

        suppress_lightning_logger (bool, optional): If True, will suppress the lightning logger.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

    Returns:
        pl.Trainer: The PyTorch Lightning Trainer instance
    """
    if suppress_lightning_logger:
        suppress_lightning_logs()

    data_config_trial = get_data_config_trial(trial, read_parse_config(data_config_default, DataConfig))
    model_config_trial = get_model_config_trial(trial, read_parse_config(model_config_default, ModelConfig))
    optimizer_config_trial = get_optimizer_config_trial(trial, read_parse_config(optimizer_config_default, OptimizerConfig))

    tabular_model = TabularModel(
        data_config=data_config_trial,
        model_config=model_config_trial,
        optimizer_config=optimizer_config_trial,
        trainer_config=trainer_config_default,
        experiment_config=experiment_config_default,
        verbose=verbose,
        suppress_lightning_logger=suppress_lightning_logger
    )

    prep_dl_kwargs, prep_model_kwargs, train_kwargs = tabular_model._split_kwargs(kwargs)

    start_time = time.time()

    model = tabular_model.prepare_model(datamodule, **prep_model_kwargs)

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
            with suppress_stdout_stderr():
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
        "model": tabular_model.name,
        'learning_rate': tabular_model.model.hparams.learning_rate,
        "# Params": int_to_human_readable(tabular_model.num_params),
    }
    if oom_handler.oom_triggered:
        if not ignore_oom:
            raise OOMException(
                "Out of memory error occurred during cross validation. "
                "Set ignore_oom=True to ignore this error."
            )
        else:
            res_dict.update(
                {
                    f"test_loss": np.inf,
                    f"validation_loss": np.inf,
                    "epochs": "OOM",
                    "time_taken": "OOM",
                    "time_taken_per_epoch": "OOM",
                }
            )
            for part in opt_parts:
                for metric_pair in opt_metrics:
                    res_dict[f"{part}_{metric_pair[0]}"] = np.inf if metric_pair[1] == 'minimize' else -np.inf
            res_dict["model"] = tabular_model.name + " (OOM)"
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
        for m_name in metrics_names:
            validation_metrics[m_name.replace('test', 'validation')] = validation_metrics.pop(m_name)
        res_dict.update(validation_metrics)

        # Update results with test metrics
        res_dict.update(tabular_model.evaluate(test=test, verbose=False)[0])

        res_dict["time_taken"] = time.time() - start_time
        res_dict["time_taken_per_epoch"] = res_dict["time_taken"] / res_dict["epochs"]

        if verbose:
            logger.info(f"Finished Training {tabular_model.name}")
            logger.info("Results:" f" {', '.join([f'{k}: {v}' for k, v in res_dict.items()])}")

        res_dict["model_params"] = model_config_trial
        res_dict["data_params"] = data_config_trial
        res_dict["optimizer_params"] = optimizer_config_trial

        if tabular_model.trainer.checkpoint_callback:
            res_dict["checkpoint"] = tabular_model.trainer.checkpoint_callback.best_model_path
            save_dir = str(Path(res_dict["checkpoint"]).parent).replace('\\', '/') + '/' + Path(res_dict["checkpoint"]).stem
            tabular_model.save_model(save_dir)
            os.remove(res_dict["checkpoint"])

        trials_results.append(res_dict)

        if tabular_model.config['checkpoints_path']:
            try:
                pd.DataFrame(trials_results).style.background_gradient(
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

    result = []
    for part in opt_parts:
        for metric_pair in opt_metrics:
            result.append(res_dict[f"{part}_{metric_pair[0]}"])

    return result
