"""Optimize node parameters for hierarchical clients."""
from typing import Callable, Dict, Iterable, List, Tuple, Union

from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate
from pydantic import BaseModel

from b_hfl.typing.common_types import FitRes, DescNodeOpt, State


class WeightedAvgConfig(BaseModel):
    """Pydantic schema for weighted average configuration."""

    alpha: float


def get_fedavg_weighted_node_opt(
    node_aggregate: Callable[[List[Tuple[NDArrays, int]]], NDArrays] = aggregate,
) -> DescNodeOpt:
    """Get the federated averaging strategy for node optimizer."""

    def fedavg_weighted_node_opt(
        state: State,
        parameters: NDArrays,
        child_parameters: Iterable[FitRes],
        residuals: Iterable[FitRes],
        config: Union[Dict, WeightedAvgConfig],
    ) -> Tuple[NDArrays, State]:
        """Federated averaging strategy for node optimizer.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        child_parameters : Iterable[NDArrays]
            The gradients received from the clients.

        Returns
        -------
        parameters : NDArrays
            The updated (global) model parameters.
        """
        if isinstance(config, Dict):
            config = WeightedAvgConfig(**config)

        child_results = [
            (params, num_examples)
            for (params, num_examples, _) in child_parameters
            if num_examples > 0
        ]
        children_parameters = (
            node_aggregate(child_results) if child_results else parameters
        )
        # return (
        #     [
        #         parent_layer * config.alpha + child_layer * (1 - config.alpha)
        #         for parent_layer, child_layer in zip(parameters, children_parameters)
        #     ],
        #     state,
        # )
        return children_parameters, state

    def initialise_state(_config: Dict) -> State:
        return (0, {}, {})

    return initialise_state, fedavg_weighted_node_opt
