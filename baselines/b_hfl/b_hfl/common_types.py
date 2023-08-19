"""Common types used throughout the project."""
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import flwr as fl
import torch
from flwr.common import NDArrays
from mypy_extensions import NamedArg
from state_management import DatasetManager, ParameterManager
from torch import nn
from torch.utils.data import DataLoader, Dataset

PathLike = Union[str, Path]

TransformType = Callable[[Any], torch.Tensor]

DatasetLoader = Callable[[Path], Dataset]

DatasetLoaderNoTransforms = Callable[[Path, TransformType, TransformType], Dataset]

ParametersLoader = Callable[[Path], NDArrays]

ClientGeneratorList = Sequence[Tuple[Callable[[], fl.client.NumPyClient], Dict]]


# Any stands in for recursive types
# because mypy doesn't support them properly.
RecursiveStructure = Tuple[
    Callable[[Dict], Optional[NDArrays]],
    Callable[[Dict], Optional[Dataset]],
    Callable[[Dict], Optional[Dataset]],
    ClientGeneratorList,
    Callable[[Tuple[NDArrays, Dict], NamedArg(bool, "final")], None],
]

ClientFN = Callable[
    [str, PathLike, Optional[fl.client.NumPyClient], Any], fl.client.NumPyClient
]

RecursiveBuilder = Callable[
    [
        fl.client.NumPyClient,
        ClientFN,
        NamedArg(bool, "test"),
    ],
    RecursiveStructure,
]


NetGenerator = Callable[[Dict], nn.Module]


NodeOpt = Callable[[NDArrays, Iterable[Tuple[NDArrays, int, Dict]], Dict], NDArrays]


OptimizerGenerator = Callable[
    [Iterator[nn.parameter.Parameter], Dict], torch.optim.Optimizer
]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


TestFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[float, int, Dict]]

DataloaderGenerator = Callable[[Optional[Dataset], Dict], Optional[DataLoader]]

LoadConfig = Callable[[int, Path], Dict]


class FolderHierarchy(TypedDict):
    """Dictionary representation of the file system."""

    root: Path
    parent: Optional[Any]
    parent_path: Optional[Path]
    path: Path
    children: List[Any]


class ClientFolderHierarchy(TypedDict):
    """A dictionary, filesyste-based representation of the client hieararchy."""

    name: str
    children: List[Any]


class ConfigFolderHierarchy(TypedDict):
    """A dictionary, filesyste-based representation of the client hieararchy."""

    on_fit_config: Dict
    on_evaluate_config: Dict
    children: List[Any]


RecursiveBuilderWrapper = Callable[
    [
        Path,
        FolderHierarchy,
        DatasetLoader,
        DatasetManager,
        ParametersLoader,
        ParameterManager,
        LoadConfig,
        LoadConfig,
        str,
        str,
    ],
    RecursiveBuilder,
]
