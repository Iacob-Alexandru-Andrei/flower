"""Common types used throughout the project."""
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import flwr as fl
import torch
from file_system_schema import FolderHierarchy
from flwr.common import NDArrays
from mypy_extensions import NamedArg
from state_management import DatasetManager, ParameterManager
from torch import nn
from torch.utils.data import DataLoader, Dataset

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

# The re-definition of recursive builder is necessary
# because mypy 0.961 doesn't support mutually-recursive types
# the usage of any is necessary because of recursive typing.
ClientFN = Callable[
    [
        str,
        Path,
        Path,
        Optional[fl.client.NumPyClient],
        Callable[
            [
                fl.client.NumPyClient,
                Any,
                NamedArg(bool, "test"),
            ],
            RecursiveStructure,
        ],
    ],
    fl.client.NumPyClient,
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
