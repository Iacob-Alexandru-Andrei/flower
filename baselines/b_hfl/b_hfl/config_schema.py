"""Pydnatic Config schemas for datasets."""
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel


class ClientTrainConfig(BaseModel):
    """Pydantic schema for client training configuration."""

    num_rounds: int
    fit_fraction: float
    train_children: bool
    train_chain: bool
    train_proxy: bool


class ClientTestConfig(BaseModel):
    """Pydantic schema for client testing configuration."""

    eval_fraction: float
    test_children: bool
    test_chain: bool
    test_proxy: bool


class DataloaderConfig(BaseModel):
    """Pydantic schema for dataloader configuration."""

    batch_size: int
    num_workers: int
    shuffle: bool
    test: bool


class NetConfig(BaseModel):
    """Pydantic schema for neural network configuration."""

    pass


class ParameterConfig(BaseModel):
    """Pydantic schema for parameter configuration."""

    pass


class NodeOptConfig(BaseModel):
    """Pydantic schema for parameter configuration."""

    pass


class RunConfig(BaseModel):
    """Pydantic schema for run configuration."""

    epochs: int
    client_learning_rate: float
    weight_decay: float


class RecClientConf(BaseModel):
    """Pydantic schema for recursive client configuration."""

    client_config: Union[ClientTrainConfig, ClientTestConfig]
    dataloader_config: DataloaderConfig
    parameter_config: Optional[ParameterConfig]
    net_config: Optional[NetConfig]
    run_config: RunConfig


class RecClientTrainConf(RecClientConf):
    """Pydantic schema for recursive client train confs."""

    client_config: ClientTrainConfig


class RecClientTestConf(RecClientConf):
    """Pydantic schema for recursive client test confs."""

    client_config: ClientTestConfig


def get_recursive_client_train_configs() -> Callable:
    return lambda **dict: RecClientTrainConf(**dict)


def get_recursive_client_test_configs() -> Callable:
    return lambda **dict: RecClientTestConf(**dict)
