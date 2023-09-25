"""Handles asynchronous collection of FitRes and EvalRes."""

import concurrent.futures
from typing import Generator, List

from flwr.common import NDArrays

from b_hfl.schema.client_schema import (
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)
from b_hfl.typing.common_types import (
    ClientEvaluateFutureList,
    ClientFitFutureList,
    EvalRes,
    FitRes,
)


def process_fit_results_and_accumulate_metrics(
    result_metrics: List,
    parameters: NDArrays,
    parent_conf: RecClientRuntimeTrainConf,
    fit_res_generators: ClientFitFutureList,
) -> Generator[FitRes, None, None]:
    """Fit the children and accumulate the metrics."""
    futures = [
        child_gen(parameters, conf, parent_conf)
        for child_gen, conf in fit_res_generators
    ]
    while futures:
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            futures.remove(future)
            future_result = future.result()
            result_metrics.append(future_result[1:])
            yield future_result


def process_evaluate_results(
    parameters: NDArrays,
    parent_conf: RecClientRuntimeTestConf,
    eval_res_generator: ClientEvaluateFutureList,
) -> Generator[EvalRes, None, None]:
    """Evaluate the children."""
    futures = [
        child_gen(parameters, conf, parent_conf)
        for child_gen, conf in eval_res_generator
    ]
    while futures:
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            futures.remove(future)
            future_result = future.result()
            yield future_result
