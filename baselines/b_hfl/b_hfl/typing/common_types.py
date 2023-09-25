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
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, Dataset

from b_hfl.schema.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)

State = Tuple[int, Dict, Any]

# Everything in FitRes except for the NDArrays

ConfigSchemaGenerator = Callable[[], Type[BaseModel]]
TransformType = Callable[[Any], torch.Tensor]

DatasetLoader = Callable[[Path], Dataset]

DatasetLoaderNoTransforms = Callable[[Path, TransformType, TransformType], Dataset]

ParametersLoader = Callable[[Path], NDArrays]

StateLoader = Callable[[Path], State]

FitRes = Tuple[NDArrays, int, Dict]
# Can contain the params, optional num_examples, state and metrics dict
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


GetResiduals = Callable[[bool], Iterable[FitRes]]
SendResiduals = Callable[[Any, FitRes, bool], None]
RecursiveStep = Callable[[Tuple[NDArrays, State, State], bool], None]

FitRecursiveStructure = Tuple[
    Optional[Callable[[Dict], NDArrays]],
    Optional[Callable[[Dict], State]],
    Optional[Callable[[Dict], State]],
    Optional[Callable[[Dict], Dataset]],
    Optional[Callable[[Dict], Dataset]],
    ClientResGeneratorList,
    GetResiduals,
    SendResiduals,
    RecursiveStep,
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
                bool,
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
        bool,
    ],
    RecursiveStructure,
]


NetGenerator = Callable[[Dict], nn.Module]


StateGenerator = Callable[[Dict], State]
NodeOptFunction = Callable[
    [State, NDArrays, Iterable[FitRes], Iterable[FitRes], Dict],
    Tuple[NDArrays, State],
]

NodeOpt = Tuple[
    StateGenerator,
    NodeOptFunction,
]


OptimizerGenerator = Callable[
    [Iterator[nn.parameter.Parameter], Dict], torch.optim.Optimizer
]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


TestFunc = Callable[[nn.Module, DataLoader, Dict], EvalRes]

DataloaderGenerator = Callable[[Dataset, Dict], DataLoader]

LoadConfig = Callable[[int, Path], Dict]


EvaluateFunc = Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]
