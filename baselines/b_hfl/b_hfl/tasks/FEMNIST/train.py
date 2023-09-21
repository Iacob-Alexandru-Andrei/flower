from typing import Dict, Iterator, Tuple
from torch import nn
import torch
from b_hfl.typing.common_types import OptimizerGenerator
from b_hfl.utils.utils import lazy_wrapper
from torch.utils.data import DataLoader


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
