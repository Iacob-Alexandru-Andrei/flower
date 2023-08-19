"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common_types import (
    ClientFN,
    DatasetLoader,
    LoadConfig,
    ParametersLoader,
    RecursiveBuilder,
    TransformType,
)
from dataset_preparation import FolderHierarchy
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.common.typing import NDArrays
from server import History
from torch.utils.data import Dataset, TensorDataset
import ray


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.
    """
    return lambda: x


def decorate_client_fn_with_recursive_builder(
    get_client_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    path_dict: FolderHierarchy,
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
        np.savez(str(path.with_suffix("")), *parameters)
    elif path.suffix == ".pt":
        torch.save(parameters, path)
    else:
        raise ValueError(f"Unknown parameter format: {path.suffix}")


@lazy_wrapper
def load_parameters_file(path: Path) -> NDArrays:
    """Load parameters from a file."""
    if path.suffix == ".npy" or path.suffix == ".npz" or path.suffix == ".np":
        return list(np.load(file=str(path), allow_pickle=True).values())
    elif path.suffix == ".pt":
        return torch.load(path)
    else:
        raise ValueError(f"Unknown parameter format: {path.suffix}")


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
    output_directory: Path,
    name: str,
) -> None:
    """Plot classification accuracy from history.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.losses_centralized)
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100)  # Accuracy 0-100%
    plt.title("Centralized Validation")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(output_directory / f"{name}.png")
    plt.close()


def get_config(config_name: str) -> Callable[[int, Path], Dict]:
    """Get the config for a given round."""

    def file_on_fit_config_fn(server_round: int, true_id: Path) -> Dict:
        """Return the config for the given round."""
        with open(true_id / f"{config_name}_config.json") as f:
            configs = json.load(f)
            if len(configs["rounds"]) <= server_round:
                return configs["rounds"][-1]
            else:
                return configs["rounds"][server_round]

    return file_on_fit_config_fn


def extract_file_from_files(files: Path, file_type) -> Optional[Path]:
    """Extract a file of a given type from a list of files."""
    to_extract: Optional[Path] = None
    any(
        (to_extract := file)
        for file in files.iterdir()
        if file.is_file() and file_type in file.name
    )

    return to_extract


@lazy_wrapper
def get_initial_parameters(
    path_dict: FolderHierarchy,
    load_parameters_file: ParametersLoader,
    net_generator: Callable,
    on_fit_config_fn: LoadConfig,
) -> Parameters:
    """Get the initial parameters for the server."""
    parameters_file = extract_file_from_files(path_dict["path"], "parameters")
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


# TODO implement once requirements are clearer
def get_on_fit_metrics_agg_fn() -> None:
    """Aggregate fit metrics fof hierarchical clients."""
    return None


def cleanup(path_dict: FolderHierarchy, to_clean: List[str]) -> None:
    """Cleanup the files in the path_dict."""
    for file in path_dict["path"].iterdir():
        if file.is_file():
            for clean_token in to_clean:
                if clean_token in file.name:
                    if file.exists():
                        file.unlink()
                        break

    for child in path_dict["children"]:
        cleanup(child, to_clean)


def save_files(
    path_dict: FolderHierarchy, output_dir: Path, ending: str, to_save: List[str]
) -> None:
    """Save the files in the path_dict."""
    output_dir = output_dir / path_dict["path"].name

    for file in path_dict["path"].iterdir():
        if file.is_file():
            for save_token in to_save:
                if save_token in file.name:
                    if file.exists():
                        destination_file = (
                            output_dir / file.with_stem(f"{file.stem}{ending}").name
                        )
                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, destination_file)

    for child in path_dict["children"]:
        save_files(child, output_dir, ending, to_save)


def get_save_files_every_round(
    path_dict: FolderHierarchy,
    output_dir: Path,
    to_save: List[str],
    save_frequency: int,
) -> Callable[[int], None]:
    def save_files_round(round: int) -> None:
        if round % save_frequency == 0:
            save_files(path_dict, output_dir, f"_{round}", to_save)

    return save_files_round


class FileSystemManager:
    """A context manager for cleaning up files."""

    def __init__(
        self,
        path_dict: FolderHierarchy,
        output_dir,
        to_clean: List[str],
        to_save_once: List[str],
    ) -> None:
        self.to_clean = to_clean
        self.path_dict = path_dict
        self.output_dir = output_dir
        self.to_save_once = to_save_once
        pass

    def __enter__(self):
        """Initialize the context manager and cleanup."""
        print(f"Pre-cleaning {self.to_clean}")
        cleanup(self.path_dict, self.to_clean)
        os.sync()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        print(f"Saving {self.to_save_once}")
        save_files(self.path_dict, self.output_dir, "", self.to_save_once)
        os.sync()
        print(f"Post-cleaning {self.to_clean}")
        cleanup(self.path_dict, self.to_clean)
        if ray.is_initialized():
            temp_dir = Path(ray.worker._global_node.get_session_dir_path())  # type: ignore
            ray.shutdown()
            shutil.rmtree(temp_dir)
            print(f"Cleaned up ray temp session: {temp_dir}")


def save_histories(
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    type: str,
) -> None:
    for folder, _, history in histories:
        with open(folder / f"history{type}.json", "w", encoding="utf-8") as f:
            json.dump(history.__dict__, f, ensure_ascii=False)
        with open(
            output_directory / f"history{type}_root_{folder.stem}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(history.__dict__, f, ensure_ascii=False)


def plot_histories(
    plotting_fn: Callable[[History, Path, str], None],
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    type: str,
) -> None:
    for folder, _, history in histories:
        plotting_fn(history, folder, f"history{type}")
        plotting_fn(history, output_directory, f"history{type}_root_{folder.stem}")


def process_histories(
    plotting_fn: Callable[[History, Path, str], None],
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    type: str,
) -> None:
    save_histories(
        histories=histories,
        output_directory=output_directory,
        type="_optimal",
    )
    plot_histories(
        plotting_fn=plotting_fn,
        histories=histories,
        output_directory=output_directory,
        type=type,
    )
