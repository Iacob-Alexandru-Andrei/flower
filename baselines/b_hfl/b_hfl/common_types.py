"""Common types used throughout the project."""
from concurrent.futures import Executor, Future
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
    Type,
    Union,
)

import torch
from flwr.common import NDArrays
from mypy_extensions import NamedArg
from pydantic import BaseModel
from b_hfl.schemas.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)
from schemas.file_system_schema import FolderHierarchy
from state_management import DatasetManager, ParameterManager
from torch import nn
from torch.utils.data import DataLoader, Dataset

ConfigSchemaGenerator = Callable[[], Type[BaseModel]]
TransformType = Callable[[Any], torch.Tensor]

DatasetLoader = Callable[[Path], Dataset]

DatasetLoaderNoTransforms = Callable[[Path, TransformType, TransformType], Dataset]

ParametersLoader = Callable[[Path], NDArrays]

FitRes = Tuple[NDArrays, int, Dict]
EvalRes = Tuple[float, int, Dict]

ClientFitFuture = Tuple[
    Callable[[NDArrays, Dict, RecClientRuntimeTrainConf], Future[FitRes]], Dict
]

ClientEvaluateFuture = Tuple[
    Callable[[NDArrays, Dict, RecClientRuntimeTestConf], Future[EvalRes]], Dict
]

ClientFitFutureList = Sequence[ClientFitFuture]
ClientEvaluateFutureList = Sequence[ClientEvaluateFuture]

ClientResGeneratorList = Union[ClientFitFutureList, ClientEvaluateFutureList]


# Any stands in for recursive types
# because mypy doesn't support them properly.
RecursiveStructure = Tuple[
    Optional[Callable[[Dict], NDArrays]],
    Optional[Callable[[Dict], Dataset]],
    Optional[Callable[[Dict], Dataset]],
    ClientResGeneratorList,
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
        Optional[ConfigurableRecClient],
        ConfigSchemaGenerator,
        ConfigSchemaGenerator,
        Callable[
            [
                ConfigurableRecClient,
                Any,
                NamedArg(bool, "test"),
            ],
            RecursiveStructure,
        ],
    ],
    ConfigurableRecClient,
]

RecursiveBuilder = Callable[
    [
        ConfigurableRecClient,
        ClientFN,
        NamedArg(bool, "test"),
    ],
    RecursiveStructure,
]


NetGenerator = Callable[[Dict], nn.Module]


NodeOpt = Callable[[NDArrays, Iterable[FitRes], Dict], NDArrays]


OptimizerGenerator = Callable[
    [Iterator[nn.parameter.Parameter], Dict], torch.optim.Optimizer
]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


TestFunc = Callable[[nn.Module, DataLoader, Dict], EvalRes]

DataloaderGenerator = Callable[[Dataset, Dict], DataLoader]

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
        Executor,
        ConfigSchemaGenerator,
        ConfigSchemaGenerator,
    ],
    RecursiveBuilder,
]


EvaluateFunc = Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]
