"""Train/eval functions for a given dataset/task."""

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import torch
from flwr.common.typing import NDArrays
from PIL.Image import Image as ImageType
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from b_hfl.typing.common_types import OptimizerGenerator, TrainFunc
from b_hfl.utils.utils import lazy_wrapper


# Load with appropriate transforms
@lazy_wrapper
def to_tensor_transform(x: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor.

    Args:
        p (Any): object to transform

    Returns
    -------
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(x)


def get_image_to_tensor_transform() -> Callable[[ImageType], torch.Tensor]:
    """Get the transform to convert an image to a PyTorch Tensor."""
    return transforms.ToTensor()


def optimizer_generator_decorator(
    optimizer_generator: OptimizerGenerator,
) -> Callable[
    [Callable[[nn.Module, DataLoader, Dict, OptimizerGenerator], Tuple[int, Dict]]],
    TrainFunc,
]:
    """Transfer the optimizer generator to the train function.

    Args:
        optimizer_generator (OptimizerGenerator): optimizer generator function.

    Returns
    -------
        Callable[[Callable], TrainFunc]: wrapped train function.
    """

    def wrapper(
        train_func: Callable[
            [nn.Module, DataLoader, Dict, OptimizerGenerator], Tuple[int, Dict]
        ]
    ) -> TrainFunc:
        def wrapped(
            net: nn.Module, train_loader: DataLoader, cfg: Dict
        ) -> Tuple[int, Dict]:
            return train_func(net, train_loader, cfg, optimizer_generator)

        return wrapped

    return wrapper


def set_model_parameters(net: nn.Module, parameters: NDArrays) -> nn.Module:
    """Put a set of parameters into the model object.

    Args:
        net (nn.Module): model object.
        parameters (NDArrays): set of parameters to put into the model.

    Returns
    -------
        nn.Module: updated model object.
    """
    weights = parameters
    params_dict = zip(net.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_model_parameters(net: nn.Module) -> NDArrays:
    """Get the current model parameters as NDArrays.

    Args:
        net (nn.Module): current model object.

    Returns
    -------
        NDArrays: set of parameters from the current model.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
