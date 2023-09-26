"""Pydnatic Config schemas for datasets."""
import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import NDArrays
from pydantic import BaseModel


class ClientTrainConfig(BaseModel):
    """Pydantic schema for client training configuration."""

    num_rounds: int
    fit_fraction: float
    train_children: bool
    train_chain: bool
    train_proxy: bool
    root_to_leaf_residuals: List[Any]
    leaf_to_root_residuals: List[Any]


class ClientTrainRuntimeConfig(ClientTrainConfig):
    """Pydantic schema for client runtime training configuration."""

    parent_round: Optional[int]
    parent_num_examples: Optional[int]
    parent_metrics: Optional[Dict]


class ClientTestConfig(BaseModel):
    """Pydantic schema for client testing configuration."""

    eval_fraction: float
    test_children: bool
    test_chain: bool
    test_proxy: bool


class ClientTestRuntimeConfig(ClientTestConfig):
    """Pydantic schema for client runtime testing configuration."""

    parent_round: Optional[int]


class RecClientConf(BaseModel):
    """Pydantic schema for recursive client configuration."""

    dataloader_config: Dict
    run_config: Dict
    parameter_config: Dict
    net_config: Dict
    state_loader_config: Dict
    state_generator_config: Dict
    anc_node_optimizer_config: Dict
    desc_node_optimizer_config: Dict
    get_parameters_config: Dict
    dataset_generator_config: Dict


class RecClientTrainConf(RecClientConf):
    """Pydantic schema for recursive client train confs."""

    client_config: ClientTrainConfig


class RecClientTestConf(RecClientConf):
    """Pydantic schema for recursive client test confs."""

    client_config: ClientTestConfig


class RecClientRuntimeConf(RecClientConf):
    """Pydantic schema for recursive client runtime configuration."""

    server_round: int
    global_seed: int
    client_seed: int


class RecClientRuntimeTrainConf(RecClientRuntimeConf):
    """Pydantic schema for recursive client train configuration."""

    client_config: ClientTrainRuntimeConfig


class RecClientRuntimeTestConf(RecClientRuntimeConf):
    """Pydantic schema for recursive client test configuration."""

    client_config: ClientTestRuntimeConfig


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
