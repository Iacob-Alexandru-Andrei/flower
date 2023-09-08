"""Pydnatic Config schemas for datasets."""
import abc
from typing import Callable, Dict, Optional, Tuple, Union

import flwr as fl
from flwr.common import NDArrays
from pydantic import BaseModel


class ClientConfig(BaseModel):
    """Pydantic schema for client configuration."""

    parent_round: Optional[int]


class ClientTrainConfig(ClientConfig):
    """Pydantic schema for client training configuration."""

    num_rounds: int
    fit_fraction: float
    train_children: bool
    train_chain: bool
    train_proxy: bool


class ClientTestConfig(ClientConfig):
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


class ParameterConfig(BaseModel):
    """Pydantic schema for parameter configuration."""


class NodeOptConfig(BaseModel):
    """Pydantic schema for parameter configuration."""


class GetParametersConfig(BaseModel):
    """Pydantic schema for requiesting parameters."""


class DatasetGeneratorConfig(BaseModel):
    """Pydantic schema for dataset generator configuration."""


class RunConfig(BaseModel):
    """Pydantic schema for run configuration."""

    epochs: int
    client_learning_rate: float
    weight_decay: float


class RecClientRuntimeConf(BaseModel):
    """Pydantic schema for recursive client runtime configuration."""

    client_config: Union[ClientTrainConfig, ClientTestConfig]
    server_round: int
    global_seed: int
    client_seed: int
    dataloader_config: Dict
    parameter_config: Dict
    net_config: Dict
    run_config: Dict
    node_optimizer_config: Dict
    get_parameters_config: Dict
    dataset_generator_config: Dict


class RecClientRuntimeTrainConf(RecClientRuntimeConf):
    """Pydantic schema for recursive client train configuration."""

    client_config: ClientTrainConfig


class RecClientRuntimeTestConf(RecClientRuntimeConf):
    """Pydantic schema for recursive client test configuration."""

    client_config: ClientTestConfig


class RecClientConf(BaseModel):
    """Pydantic schema for recursive client configuration."""

    client_config: Union[ClientTrainConfig, ClientTestConfig]
    dataloader_config: DataloaderConfig
    run_config: RunConfig
    server_round: Optional[int]
    global_seed: Optional[int]
    client_seed: Optional[int]
    parameter_config: Optional[ParameterConfig]
    net_config: Optional[NetConfig]
    node_optimizer_config: Optional[NodeOptConfig]
    get_parameters_config: Optional[GetParametersConfig]
    dataset_generator_config: Optional[DatasetGeneratorConfig]


class RecClientTrainConf(RecClientConf):
    """Pydantic schema for recursive client train confs."""

    client_config: ClientTrainConfig


class RecClientTestConf(RecClientConf):
    """Pydantic schema for recursive client test confs."""

    client_config: ClientTestConfig


def get_recursive_client_train_configs() -> Callable:
    """Return a callable that returns a recursive client train config.

    For both validation and instantiation.
    """
    return lambda: RecClientTrainConf


def get_recursive_client_test_configs() -> Callable:
    """Return a callable that returns a recursive client test config.

    For both validation and instantiation.
    """
    return lambda: RecClientTestConf


class ConfigurableRecClient(fl.client.NumPyClient, abc.ABC):
    """Abstract class for recursive clients with configurable schemas."""

    @abc.abstractmethod
    def fit(
        self, parameters: NDArrays, config: Union[Dict, RecClientRuntimeTrainConf]
    ) -> Tuple[NDArrays, int, Dict]:
        """Execute the node subtree and fit the node model."""

    @abc.abstractmethod
    def evaluate(
        self, parameters: NDArrays, config: Union[Dict, RecClientRuntimeTestConf]
    ) -> Tuple[float, int, Dict]:
        """Execute the node subtree and evaluate the node model."""
