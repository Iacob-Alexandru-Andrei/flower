"""Train/eval functions for a given dataset/task."""

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import torch
from common_types import OptimizerGenerator, TrainFunc
from flwr.common.typing import NDArrays
from PIL.Image import Image as ImageType
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import lazy_wrapper


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


@lazy_wrapper
def optimizer_generator_femnist(
    parameters: Iterator[nn.parameter.Parameter], cfg: Dict
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer for the femnist dataset."""
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg["client_learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )


@lazy_wrapper
def train_femnist(
    net: nn.Module,
    train_loader: DataLoader,
    cfg: Dict,
    optimizer_generator: OptimizerGenerator,
) -> Tuple[int, Dict]:
    """Trains the network on the training set.

    Args:
        net (nn.Module): generic nn.Module object describing the network to train.
        train_loader (DataLoader): dataloader to iterate during the training.
        epochs (int): number of epochs of training.
        device (str): device name onto which perform the computation.
        optimizer (torch.optim.Optimizer): optimizer object.
        criterion (nn.Module): generic nn.Module describing the loss function.

    Returns
    -------
        float: the final epoch mean train loss.
    """
    net.train()
    running_loss, total = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optimizer_generator(net.parameters(), cfg)
    for _ in range(cfg["epochs"]):
        running_loss = 0.0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(cfg["device"]), labels.to(cfg["device"])

            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            running_loss += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
    return total, {"avg_loss_train": running_loss / total}


@lazy_wrapper
def test_femnist(
    net: nn.Module,
    test_loader: DataLoader,
    cfg: Dict,
) -> Tuple[float, int, Dict]:
    """Validate the network on a test set.

    Args:
        net (nn.Module): generic nn.Module object describing the network to test.
        test_loader (DataLoader): dataloader to iterate during the testing.
        device (str):  device name onto which perform the computation.
        criterion (nn.Module): generic nn.Module describing the loss function.

    Returns
    -------
        Tuple[float, float]: average test loss and average accuracy on test set.
    """
    correct, total, loss = 0, 0, 0.0
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # for data, labels in tqdm(test_loader):
        for data, labels in test_loader:
            data, labels = data.to(cfg["device"]), labels.to(cfg["device"])
            outputs = net(data)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return (
        loss / total,
        total,
        {"accuracy_test": accuracy, "avg_loss_test": loss / total},
    )


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


# def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
#     """Generic function to combine results from multiple clients
#     following training or evaluation.

#     Args:
#         metrics (List[Tuple[int, dict]]): collected clients metrics

#     Returns:
#         dict: result dictionary containing the aggregate of the metrics passed.
#     """
#     average_dict: dict = defaultdict(list)
#     total_examples: int = 0
#     for num_examples, metrics_dict in metrics:
#         for key, val in metrics_dict.items():
#             if isinstance(val, numbers.Number):
#                 average_dict[key].append((num_examples, val))  # type:ignore
#         total_examples += num_examples
#     return {
#         key: {
#             "avg": float(
#                 sum([num_examples * metr for num_examples, metr in val])
#                 / float(total_examples)
#             ),
#             "all": val,
#         }
#         for key, val in average_dict.items()
#     }


# def get_federated_evaluation_function(
#     data_dir: Path,
#     centralized_mapping: Path,
#     device: str,
#     batch_size: int,
#     num_workers: int,
#     model_generator: Callable[[], nn.Module],
#     criterion: nn.Module,
# ) -> Callable[[int, NDArrays, Dict[str, Any]], Tuple[float, Dict[str, Scalar]]]:
#     """Wrapper function for the external federated evaluation function.
#     It provides the external federated evaluation function with some
#     parameters for the dataloader, the model generator function, and
#     the criterion used in the evaluation.

#     Args:
#         data_dir (Path): path to the dataset folder.
#         centralized_mapping (Path): path to the mapping .csv file chosen.
#         device (str):  device name onto which perform the computation.
#         batch_size (int): batch size of the test set to use.
#         num_workers (int): correspond to `num_workers` param in the Dataloader object.
#         model_generator (Callable[[], nn.Module]):  model generator function.
#         criterion (nn.Module): evluation criterion.

#     Returns:
#         Callable: fed-eval.
#     """

#     full_file: Path = centralized_mapping
#     dataset: Dataset = load_femnist_dataset(data_dir, full_file, "val")

#     def federated_evaluation_function(
#         server_round: int,
#         parameters: NDArrays,
#         fed_eval_config: Dict[
#             str, Any
#         ],  # mandatory argument, even if it's not being used
#     ) -> Tuple[float, Dict[str, Scalar]]:
#         """Evaluation function external to the federation.
#         It uses the centralized val set for sake of simplicity.

#         Args:
#             server_round (int): current federated round.
#             parameters (NDArrays): current model parameters.
#             fed_eval_config (Dict[str, Any]): mandatory argument in Flower

#         Returns:
#             Tuple[float, Dict[str, Scalar]]: evaluation results
#         """
#         net: nn.Module = set_model_parameters(model_generator(), parameters)
#         net.to(device)

#         valid_loader = DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             drop_last=False,
#         )

#         loss, acc = test_femnist(
#             net=net,
#             test_loader=valid_loader,
#             device=device,
#             criterion=criterion,
#         )
#         return loss, {"accuracy": acc}

#     return federated_evaluation_function


# def get_default_train_config() -> Dict[str, Any]:
#     return {
#         "epochs": 8,
#         "batch_size": 32,
#         "client_learning_rate": 0.01,
#         "weight_decay": 0.001,
#         "num_workers": 2,
#     }


# def get_default_test_config() -> Dict[str, Any]:
#     return {
#         "batch_size": 32,
#         "num_workers": 2,
#     }
