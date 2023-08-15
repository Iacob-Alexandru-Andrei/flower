"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common_types import (
    ClientFN,
    LoadConfig,
    RecursiveBuilder,
    TransformType,
    ParametersLoader,
    DatasetLoader,
)
from dataset_preparation import FileHierarchy
from flwr.common.typing import NDArrays
from flwr.server.history import History
from torch.utils.data import Dataset, TensorDataset
from flwr.common import ndarrays_to_parameters, Parameters


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.
    """
    return lambda: x


def decorate_client_fn_with_recursive_builder(
    get_client_recursive_builder: Callable[[FileHierarchy], RecursiveBuilder],
    path_dict: FileHierarchy,
) -> Callable[[ClientFN], Callable[[str], fl.client.Client]]:
    """Decorate a client function with a recursive builder."""
    i: int = 0

    def client_fn_wrapper(
        client_fn: ClientFN,
    ) -> Callable[[str], fl.client.Client]:
        def wrap_client_fn(cid: str) -> fl.client.Client:
            nonlocal i
            nonlocal path_dict
            recursive_builder: RecursiveBuilder = get_client_recursive_builder(
                path_dict["children"][i]
            )
            i += 1
            return client_fn(
                cid, path_dict["children"][i]["path"], None, recursive_builder
            )  # type: ignore

        return wrap_client_fn

    return client_fn_wrapper


def decorate_dataset_with_transforms(
    transform: TransformType,
    target_transform: TransformType,
) -> Callable[
    [
        Callable[
            [Path, TransformType, TransformType],
            Dataset,
        ]
    ],
    Callable[[Path], Dataset],
]:
    """Decorate a dataset function with transforms."""

    def decorator_transform(
        func: Callable[
            [Path, TransformType, TransformType],
            Dataset,
        ]
    ) -> DatasetLoader:
        def wrapper_transform(path: Path) -> Dataset:
            return func(path, transform, target_transform)

        return wrapper_transform

    return decorator_transform


def load_pandas_file(path: Path) -> Dataset:
    """Load a pandas dataset from a csv file."""
    df = pd.read_csv(path)
    x_tensor = torch.tensor(df["x"].values)
    y_tensor = torch.tensor(df["y"].values)
    return TensorDataset(x_tensor, y_tensor)


@lazy_wrapper
def load_tensor_dataset(path: Path) -> Dataset:
    """Load a torch dataset from a pt file."""
    return TensorDataset(*torch.load(path))


@lazy_wrapper
def save_parameters_to_file(path: Path, parameters: NDArrays) -> None:
    """Save parameters to a file.

    The file type is inferred from the file extension. Add new file types here.
    """
    if path.suffix == ".npy" or path.suffix == ".npz" or path.suffix == ".np":
        np.savez_compressed(path.with_suffix(""), parameters)
    else:
        raise ValueError(f"Unknown parameter format: {path.suffix}")


@lazy_wrapper
def load_parameters_file(path: Path) -> NDArrays:
    """Load parameters from a file."""
    return [
        arr for val in np.load(file=path, allow_pickle=True).values() for arr in val
    ]


def get_device() -> torch.device:
    """Get the device (CPU or GPU) for torch."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
    output_directory: Path,
) -> None:
    """Plot classification accuracy from history.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.savefig(output_directory / "graph.png")
    plt.close()


def get_config(config_name: str) -> Callable[[int, Path], Dict]:
    """Get the config for a given round."""

    def file_on_fit_config_fn(server_round: int, true_id: Path) -> Dict:
        """Return the config for the given round."""
        with open(true_id / f"{config_name}_config.json") as f:
            configs = json.load(f)
            if len(configs["rounds"]) < server_round:
                return configs["rounds"][-1]
            else:
                return configs["rounds"][server_round]

    return file_on_fit_config_fn


def extract_file_from_files(files: List[Path], file_type) -> Optional[Path]:
    """Extract a file of a given type from a list of files."""
    to_extract: Optional[Path] = None
    any((to_extract := file) for file in files if file_type in file.name)

    return to_extract


@lazy_wrapper
def get_initial_parameters(
    path_dict: FileHierarchy,
    load_parameters_file: ParametersLoader,
    net_generator: Callable,
    on_fit_config_fn: LoadConfig,
) -> Parameters:
    """Get the initial parameters for the server."""
    parameters_file = extract_file_from_files(path_dict["files"], "parameters")
    if parameters_file is not None:
        return ndarrays_to_parameters(load_parameters_file(parameters_file))
    else:
        return ndarrays_to_parameters(
            [
                val.cpu().numpy()
                for _, val in net_generator(
                    on_fit_config_fn(0, path_dict["path"])["net_config"]
                )
                .state_dict()
                .items()
            ]
        )


def get_fed_eval_fn(
    root_path: Path,
    client_fn: ClientFN,
    recursive_builder: RecursiveBuilder,
    on_evaluate_config_function: LoadConfig,
) -> Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]:
    """Get the federated evaluation function."""
    client = client_fn(str(root_path), root_path, None, recursive_builder)

    def fed_eval_fn(
        server_round: int, parameters: NDArrays, config: Dict
    ) -> Optional[Tuple[float, Dict]]:
        real_config = on_evaluate_config_function(server_round, root_path)
        results = client.evaluate(parameters, real_config)
        loss, _, metrics = results
        return loss, metrics

    return fed_eval_fn


# TODO implement once requirements are clearer
def get_on_fit_metrics_agg_fn() -> None:
    return None
