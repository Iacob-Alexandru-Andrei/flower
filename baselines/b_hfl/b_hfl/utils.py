"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import torch
from flwr.common.typing import NDArrays
from torch.utils.data import Dataset, TensorDataset

PathLike = Union[str, Path]


def lazy_wrapper(x: Any) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.
    """
    return lambda: x


@lazy_wrapper
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
def save_general_parameters(path: Path, parameters: NDArrays) -> None:
    """Save parameters to a file.

    The file type is inferred from the file extension. Add new file types here.
    """
    if path.suffix == ".pt":
        torch.save(parameters, path)
    elif path.suffix == ".npy" or path.suffix == ".npz" or path.suffix == ".np":
        np.savez(path, parameters)
    else:
        raise ValueError(f"Unknown parameter format: {path.suffix}")


def get_device() -> torch.device:
    """Get the device (CPU or GPU) for torch."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
