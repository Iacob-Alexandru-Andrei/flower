"""Optimize node parameters for hierarchical clients."""
from typing import Callable, Dict, Iterable, List, Tuple

from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate


def get_fedavg_weighted_node_opt(
    alpha: float,
    node_aggregate: Callable[[List[Tuple[NDArrays, int]]], NDArrays] = aggregate,
) -> Callable[[NDArrays, Iterable[Tuple[NDArrays, int, Dict]], Dict], NDArrays]:
    """Get the federated averaging strategy for node optimizer."""

    def fedavg_weighted_node_opt(
        parameters: NDArrays,
        child_parameters: Iterable[Tuple[NDArrays, int, Dict]],
        _config: Dict,
    ) -> NDArrays:
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
        child_results = [
            (params, num_examples)
            for (params, num_examples, _) in child_parameters
            if num_examples > 0
        ]
        children_parameters = (
            node_aggregate(child_results) if child_results else parameters
        )
        return [
            parent_layer * alpha + child_layer * (1 - alpha)
            for parent_layer, child_layer in zip(parameters, children_parameters)
        ]

    return fedavg_weighted_node_opt
