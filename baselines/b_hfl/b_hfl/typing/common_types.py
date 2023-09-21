"""Common types used throughout the project."""
from concurrent.futures import Future
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from b_hfl.schemas.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)

# Everything in FitRes except for the NDArrays
State = Tuple[int, Dict]

ConfigSchemaGenerator = Callable[[], Type[BaseModel]]
TransformType = Callable[[Any], torch.Tensor]

DatasetLoader = Callable[[Path], Dataset]

DatasetLoaderNoTransforms = Callable[[Path, TransformType, TransformType], Dataset]

ParametersLoader = Callable[[Path], NDArrays]

StateLoader = Callable[[Path], State]

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


GetResiduals = Callable[[NamedArg(bool, "leaf_to_root")], Iterable[FitRes]]
SendResiduals = Callable[[Any, FitRes, NamedArg(bool, "leaf_to_root")], None]


FitRecursiveStructure = Tuple[
    Optional[Callable[[Dict], NDArrays]],
    Optional[Callable[[Dict], State]],
    Optional[Callable[[Dict], Dataset]],
    Optional[Callable[[Dict], Dataset]],
    ClientResGeneratorList,
    GetResiduals,
    SendResiduals,
    Callable[[Tuple[NDArrays, State], NamedArg(bool, "final")], None],
]

EvalRecursiveStructure = Tuple[
    Optional[Callable[[Dict], NDArrays]],
    Optional[Callable[[Dict], Dataset]],
    Optional[Callable[[Dict], Dataset]],
    ClientResGeneratorList,
]

RecursiveStructure = Union[FitRecursiveStructure, EvalRecursiveStructure]


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


NodeOpt = Callable[[FitRes, Iterable[FitRes], Iterable[FitRes], Dict], FitRes]


OptimizerGenerator = Callable[
    [Iterator[nn.parameter.Parameter], Dict], torch.optim.Optimizer
]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


TestFunc = Callable[[nn.Module, DataLoader, Dict], EvalRes]

DataloaderGenerator = Callable[[Dataset, Dict], DataLoader]

LoadConfig = Callable[[int, Path], Dict]


EvaluateFunc = Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]
