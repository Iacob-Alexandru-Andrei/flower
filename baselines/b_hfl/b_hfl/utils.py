"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import flwr as fl
from typing import List, Iterable
from flwr.common.typing import NDArrays, Scalar


def process_children(
    result_metrics: List,
    parameters: NDArrays,
    children: Iterable[fl.client.NumPyClient],
):
    for child, conf in children:
        child_params, *everything_else = child.fit(parameters, conf)
        result_metrics.append(everything_else)
        yield child_params, *everything_else
