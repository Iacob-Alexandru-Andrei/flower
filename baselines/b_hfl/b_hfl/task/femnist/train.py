"""Training and testing functions for the FEMNIST dataset."""
from typing import Any, Dict, Iterator, Optional, Tuple, Union, cast

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from b_hfl.typing.common_types import OptimizerGenerator
from b_hfl.utils.utils import lazy_wrapper


class RunConfig(BaseModel):
    """Pydantic schema for run configuration."""

    device: Union[str, torch.device]
    epochs: int
    client_learning_rate: float
    weight_decay: float

    class Config:
        arbitrary_types_allowed = True


@lazy_wrapper
def optimizer_generator_femnist(
    parameters: Iterator[nn.parameter.Parameter], cfg: Union[Dict, RunConfig]
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer for the femnist dataset."""
    if isinstance(cfg, Dict):
        cfg = RunConfig(**cfg)

    return torch.optim.AdamW(
        parameters,
        lr=float(cfg.client_learning_rate),
        weight_decay=float(cfg.weight_decay),
    )


@lazy_wrapper
def train_femnist(
    net: nn.Module,
    train_loader: DataLoader,
    cfg: Union[Dict, RunConfig],
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
    if isinstance(cfg, Dict):
        cfg = RunConfig(**cfg)

    net.train()
    running_loss, total = 0.0, 0
    start_loss: Optional[float] = None
    end_loss: Optional[float] = None
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optimizer_generator(net.parameters(), cast(Dict, cfg))
    for _ in range(cfg.epochs):
        for data, labels in train_loader:
            data, labels = data.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            running_loss += loss.item()
            start_loss = loss.item() if start_loss is None else start_loss
            end_loss = loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

    return total, {
        "loss_avg": running_loss / total,
        "loss_start": start_loss,
        "loss_end:": end_loss,
    }


class TestConfig(BaseModel):
    """Pydantic schema for run configuration."""

    device: Union[str, torch.device]

    class Config:
        arbitrary_types_allowed = True


@lazy_wrapper
def test_femnist(
    net: nn.Module,
    test_loader: DataLoader,
    cfg: Union[Dict, TestConfig],
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
    if isinstance(cfg, Dict):
        cfg = TestConfig(**cfg)

    correct, total, loss = 0, 0, 0.0
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # for data, labels in tqdm(test_loader):
        for data, labels in test_loader:
            data, labels = data.to(cfg.device), labels.to(cfg.device)
            outputs = net(data)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return (
        loss / total,
        total,
        {"acc": accuracy, "loss_avg": loss / total},
    )
