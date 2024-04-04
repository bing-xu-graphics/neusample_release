"""Configuration manager with defaults"""
import dataclasses

from typing import Dict, Optional, TypeVar
from omegaconf import OmegaConf

@dataclasses.dataclass
class BaseNeuMatConfig:
    """Default Config"""
    name: str = 'unnamed'
    training_dataset: str = "data/training_data/demo.h5"

    num_epochs: int = 100

    training_interface: str = "NeuMIPv1Module"
    # Optional list of keyword parameters to pass to the model interface class
    # constructor, see 'interface.py'
    interface_params: Optional[Dict] = dataclasses.field(default=None)

    data_interface: str = "NeuMIPv1DataModule"
    # Optional list of keyword parameters to pass to the data interface class
    # constructor, see 'interface.py'
    data_interface_params: Optional[Dict] = dataclasses.field(default=None)

    model: str = "NeuMIPv1SingleRes"
    # Optional list of keyword parameters to pass to the model class constructor,
    # see 'models.py'
    model_params: Optional[Dict] = dataclasses.field(default=None)

    @staticmethod
    def default_config():
        """Instantiate a default configuration."""
        return OmegaConf.structured(BaseNeuMatConfig)

T = TypeVar("T")

def merge(conf1: T, conf2: T) -> T:
    """Merge two configurations."""
    return OmegaConf.merge(conf1, conf2)

def get_config(config_filename):
    conf = BaseNeuMatConfig.default_config()

    if config_filename is not None:
        file_conf = OmegaConf.load(config_filename)
        conf = OmegaConf.merge(conf, file_conf)

    return conf