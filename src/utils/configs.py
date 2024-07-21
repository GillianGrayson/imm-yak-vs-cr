import os
from omegaconf import OmegaConf
from pytorch_tabular.config import ModelConfig
from pytorch_tabular.utils import getattr_nested


def read_parse_config(config, cls):
    if isinstance(config, str):
        if os.path.exists(config):
            _config = OmegaConf.load(config)
            if cls == ModelConfig:
                cls = getattr_nested(_config._module_src, _config._config_name)
            config = cls(
                **{
                    k: v
                    for k, v in _config.items()
                    if (k in cls.__dataclass_fields__.keys()) and (cls.__dataclass_fields__[k].init)
                }
            )
        else:
            raise ValueError(f"{config} is not a valid path")
    config = OmegaConf.structured(config)
    return config
