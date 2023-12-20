"""Handles asynchronous collection of FitRes and EvalRes."""

import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

import ray
from flwr.common import (
    NDArrays,
)
from torch.utils.data import Dataset

import b_hfl.utils.utils as utils

from b_hfl.schema.client_schema import (
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)
from b_hfl.typing.common_types import (
    ClientEvaluateFutureList,
    ClientFitFutureList,
    DataloaderGenerator,
    EvalRes,
    FitRes,
    NetGenerator,
    TestFunc,
    TrainFunc,
)
from b_hfl.utils.utils import (
    get_parameters_copy,
    set_parameters,
)


# pylint: disable=too-many-locals,too-many-arguments
@ray.remote
def remote_train(
    net_generator: NetGenerator,
    train_func: TrainFunc,
    train_dataset_generator: Callable[[Dict], Dataset],
    create_dataloader: DataloaderGenerator,
    parameters: NDArrays,
    config: RecClientRuntimeTrainConf,
    mode: str,
    relative_id: Path,
    sep,
) -> FitRes:
    """Train the model remotely on ray."""
    net = net_generator(config.net_config)

    device = utils.get_device()

    set_parameters(net, parameters, to_copy=True)
    net.to(device)
    net.train()
    num_examples, metrics = train_func(
        net,
        create_dataloader(
            train_dataset_generator(config.dataset_generator_config),
            config.dataloader_config | {"test": False},
        ),
        config.run_config | {"device": device},
    )
    trained_parameters = get_parameters_copy(net)

    return (
        trained_parameters,
        num_examples,
        {f"{relative_id}{sep}{mode}{sep}{key}": val for key, val in metrics.items()},
    )


# pylint: disable=too-many-locals,too-many-arguments
@ray.remote
def remote_test(
    net_generator: NetGenerator,
    test_func: TestFunc,
    test_dataset_generator: Callable[[Dict], Dataset],
    create_dataloader: DataloaderGenerator,
    parameters: NDArrays,
    config: RecClientRuntimeTestConf,
    mode: Optional[str],
    relative_id,
    sep,
) -> EvalRes:
    """Test the model remotely on ray."""
    if test_dataset_generator is not None and mode is not None:
        net = net_generator(config.net_config)
        device = utils.get_device()

        set_parameters(net, parameters, to_copy=True)
        net.to(device)
        net.eval()
        loss, num_examples, metrics = test_func(
            net,
            create_dataloader(
                test_dataset_generator(config.dataset_generator_config),
                config.dataloader_config | {"test": False},
            ),
            config.run_config | {"device": device},
        )

        return (
            loss,
            num_examples,
            {
                f"{relative_id}{sep}{key}{sep}{mode}": val
                for key, val in metrics.items()
            },
        )

    return 0.0, 0, {}


class DropoutException(Exception):
    """Exception raised when a client is dropped out of the tree."""


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
            failure = future.exception()
            if failure is not None and not isinstance(failure, DropoutException):
                raise failure

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
            failure = future.exception()
            if failure is not None and not isinstance(failure, DropoutException):
                raise failure

            future_result = future.result()
            yield future_result
