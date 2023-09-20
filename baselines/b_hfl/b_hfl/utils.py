"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import json
import os
import random
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from flwr.common import Metrics, NDArrays, Parameters, ndarrays_to_parameters
from torch import nn
from torch.utils.data import Dataset
from traitlets import Bool

import wandb
from b_hfl.common_types import (
    ClientFN,
    DatasetLoader,
    LoadConfig,
    ParametersLoader,
    RecursiveBuilder,
    TransformType,
)
from b_hfl.dataset_preparation import FolderHierarchy
from b_hfl.modified_flower.server import History


def get_parameters(net: nn.Module) -> NDArrays:
    """Get the parameters of a network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: NDArrays, to_copy=False) -> None:
    """Set the parameters of a network."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.Tensor(v if not to_copy else v.copy()) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


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
    root: Path = path_dict.path

    recursive_builder: RecursiveBuilder = get_client_recursive_builder(path_dict)

    def client_fn_wrapper(
        client_fn: ClientFN,
    ) -> Callable[[str], fl.client.Client]:
        def wrap_client_fn(cid: str) -> fl.client.Client:
            nonlocal path_dict

            return client_fn(
                cid, path_dict.path, root, None, recursive_builder
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
    if path.suffix == ".pt":
        return torch.load(path)

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
    plt.savefig(output_directory / f"{name}.png")
    plt.close()


def get_config(config_name: str, seed: int) -> Callable[[int, Path], Dict]:
    """Get the config for a given round."""
    configs_dict: Dict[Path, List[Dict]] = {}
    seed_generator = random.Random(seed)

    def file_on_fit_config_fn(server_round: int, true_id: Path) -> Dict:
        """Return the config for the given round."""
        if true_id not in configs_dict:
            client_seed: int = seed_generator.randint(0, 2**32 - 1)
            with open(true_id / f"{config_name}_config.json") as f:
                configs: List[Dict] = json.load(f)
                for i, config in enumerate(configs):
                    config["server_round"] = i
                    config["global_seed"] = seed
                    config["client_seed"] = client_seed
                with open(true_id / f"{config_name}_config_runtime.json", "w") as f:
                    json.dump(configs, f)

        else:
            configs = configs_dict[true_id]

        if len(configs) <= server_round:
            return configs[-1]

        return configs[server_round]

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
    load_params_file: ParametersLoader,
    net_generator: Callable,
    on_fit_config_fn: LoadConfig,
) -> Parameters:
    """Get the initial parameters for the server."""
    parameters_file = extract_file_from_files(path_dict.path, "parameters")
    if parameters_file is not None:
        return ndarrays_to_parameters(load_params_file(parameters_file))

    return ndarrays_to_parameters(
        [
            val.cpu().numpy()
            for _, val in net_generator(
                on_fit_config_fn(0, path_dict.path)["net_config"]
            )
            .state_dict()
            .items()
        ]
    )


@lazy_wrapper
def get_metrics_agg_fn(metrics_list: List[Tuple[int, Metrics]], sep="::") -> Metrics:
    """Aggregate fit metrics fof hierarchical clients."""
    result: Metrics = {}
    metrics_dict: Dict = defaultdict(list)
    root = os.path.commonprefix(
        [key.split(sep)[0] for _, metric in metrics_list for key in metric.keys()]
    )[:-1]

    num_examples_array: Dict = defaultdict(list)
    for num_examples, metrics in metrics_list:
        for key, value in metrics.items():
            relative_id, mode, metric = key.split(sep)

            if root == os.path.dirname(relative_id) and "#" not in metric:
                metrics_dict[f"{mode}{sep}{metric}"].append(value)
                num_examples_array[f"{mode}{sep}{metric}"].append(num_examples)

        result.update(metrics)

    for metric in metrics_dict.keys():
        result[f"{root}{sep}{metric}#M"] = float(np.mean(metrics_dict[metric]))
        result[f"{root}{sep}{metric}#WM"] = float(
            np.average(metrics_dict[metric], weights=num_examples_array[metric])
        )
        result[f"{root}{sep}{metric}#S"] = float(np.std(metrics_dict[metric]))

    return result


def cleanup(path_dict: FolderHierarchy, to_clean: List[str]) -> None:
    """Cleanup the files in the path_dict."""
    for file in path_dict.path.iterdir():
        if file.is_file():
            for clean_token in to_clean:
                if clean_token in file.name:
                    if file.exists():
                        file.unlink()
                        break

    for child in path_dict.children:
        cleanup(child, to_clean)
    os.sync()


def save_files(
    path_dict: FolderHierarchy, output_dir: Path, ending: str, to_save: List[str]
) -> None:
    """Save the files in the path_dict."""
    output_dir = output_dir / path_dict.path.name

    for file in path_dict.path.iterdir():
        if file.is_file():
            for save_token in to_save:
                if save_token in file.name:
                    if file.exists():
                        destination_file = (
                            output_dir / file.with_stem(f"{file.stem}{ending}").name
                        )
                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, destination_file)

    for child in path_dict.children:
        save_files(child, output_dir, ending, to_save)
    os.sync()


def get_save_files_every_round(
    path_dict: FolderHierarchy,
    output_dir: Path,
    to_save: List[str],
    save_frequency: int,
) -> Callable[[int], None]:
    """Get a function that saves files every save_frequency rounds."""

    def save_files_round(cur_round: int) -> None:
        if cur_round % save_frequency == 0:
            save_files(path_dict, output_dir, f"_{cur_round}", to_save)

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

    def __enter__(self):
        """Initialize the context manager and cleanup."""
        print(f"Pre-cleaning {self.to_clean}")
        cleanup(self.path_dict, self.to_clean)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        print(f"Saving {self.to_save_once}")
        save_files(self.path_dict, self.output_dir, "", self.to_save_once)
        print(f"Post-cleaning {self.to_clean}")
        cleanup(self.path_dict, self.to_clean)
        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path()  # type: ignore
            )
            ray.shutdown()
            shutil.rmtree(temp_dir)
            print(f"Cleaned up ray temp session: {temp_dir}")


def save_histories(
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    history_type: str,
) -> None:
    """Save the histories.

    Saves them both in the client folder and in a flat fashion.
    """
    for folder, _, history in histories:
        with open(folder / f"history{history_type}.json", "w", encoding="utf-8") as f:
            json.dump(history.__dict__, f, ensure_ascii=False)

        with open(
            output_directory / f"history{history_type}_from_{folder.stem}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(history.__dict__, f, ensure_ascii=False)


def plot_histories(
    plotting_fn: Callable[[History, Path, str], None],
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    history_type: str,
) -> None:
    """Plot the histories.

    Both in the client folders and in a flat hierarchy.
    """
    for folder, _, history in histories:
        plotting_fn(history, folder, f"plot_history_{history_type}")
        plotting_fn(
            history, output_directory, f"plot_history_{history_type}_root_{folder.stem}"
        )


def process_histories(
    plotting_fn: Callable[[History, Path, str], None],
    histories: List[Tuple[Path, FolderHierarchy, History]],
    output_directory: Path,
    history_type: str,
) -> None:
    """Process the histories.

    Save and plot them in the client folders and in a flat fashion.
    """
    save_histories(
        histories=histories,
        output_directory=output_directory,
        history_type=history_type,
    )
    plot_histories(
        plotting_fn=plotting_fn,
        histories=histories,
        output_directory=output_directory,
        history_type=history_type,
    )


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""


def wandb_init(wandb_enabled: Bool, *args, **kwargs):
    """Initialize wandb if enabled."""
    if wandb_enabled:
        return wandb.init(*args, **kwargs)

    return NoOpContextManager()


def hash_combine(*args) -> int:
    """Combine hashes of multiple objects."""
    combined_hash = 0
    for arg in args:
        if arg is not None:
            combined_hash ^= hash(arg)
    return combined_hash


def get_seeded_rng(
    global_seed: int, client_seed: int, server_round: int, parent_round: Optional[int]
) -> random.Random:
    """Get a seeded random number generator."""
    return random.Random(
        hash_combine(global_seed, client_seed, server_round, parent_round)
    )
