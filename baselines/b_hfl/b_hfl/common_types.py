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
from flwr.common import NDArrays
from mypy_extensions import NamedArg
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from state_management import DatasetManager, ParameterManager


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


NodeOpt = Callable[[NDArrays, Iterable[Tuple[NDArrays, int, Dict]]], NDArrays]


OptimizerGenerator = Callable[
    [Iterator[nn.parameter.Parameter], Dict], torch.optim.Optimizer
]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


TestFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[float, int, Dict]]

DataloaderGenerator = Callable[[Optional[Dataset], Dict], Optional[DataLoader]]

LoadConfig = Callable[[int, Path], Dict]


class FileHierarchy(TypedDict):
    """Dictionary representation of the file system."""

    parent: Optional[Any]
    parent_path: Optional[Path]
    path: Path
    files: List[Path]
    children: List[Any]


class ClientFileHierarchy(TypedDict):
    """A dictionary, filesyste-based representation of the client hieararchy."""

    name: str
    children: List[Any]


class ConfigFileHierarchy(TypedDict):
    """A dictionary, filesyste-based representation of the client hieararchy."""

    on_fit_config: Dict
    on_evaluate_config: Dict
    children: List[Any]


RecursiveBuilderWrapper = Callable[
    [
        FileHierarchy,
        DatasetLoader,
        DatasetManager,
        ParametersLoader,
        ParameterManager,
        LoadConfig,
        LoadConfig,
    ],
    RecursiveBuilder,
]
