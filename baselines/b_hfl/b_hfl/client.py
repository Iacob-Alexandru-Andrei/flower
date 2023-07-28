"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
from typing import Iterable, Tuple, Dict, Callable, List, OrderedDict, Union
import utils
from itertools import chain
import flwr as fl
from flwr.common import NDArrays, Scalar
import numpy as np
import torch


class RecursiveClient(fl.client.NumPyClient):
    pass


class RecursiveClient(fl.client.NumPyClient):
    def __init__(
        self,
        parent: fl.client.NumPyClient,
        net_generator: Callable[[], torch.nn.Module],
        num_training_epochs: int,
        num_rounds: int,
        learning_rate: float,
        learning_rate_decay: float,
        client_parameters: NDArrays,
        children: Iterable[Tuple[fl.client.NumPyClient, Dict]],
        node_opt: Callable[[NDArrays, Iterable[NDArrays]], NDArrays],
        train: Callable[[NDArrays, Dict], Tuple[int, Dict]],
    ):  # pylint: disable=too-many-arguments
        """Recursive client, which can be used to create a tree of clients. Client execution is replaced with subtree execution + training."""

        self.parent = parent
        self.net = None
        self.net_generator = net_generator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_training_epochs
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.client_parameters = client_parameters
        self.children = children
        self.train = train

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[NDArrays, int, Dict]:
        self.client_parameters = self.node_opt(self.client_parameters, [parameters])
        results = {
            "children_results": [],
            "train_results": [],
        }
        total_examples = 0
        for i in range(self.num_rounds):
            children_results = []
            self.client_parameters = self.node_opt(
                self.client_parameters,
                utils.process_children(
                    children_results, self.client_parameters, self.children
                ),
            )

            self.set_parameters(self.client_parameters)
            train_examples, train_metrics = self.train(self.client_parameters, config)
            self.client_parameters = self.get_parameters(config)

            total_examples += (
                sum((num_examples for num_examples, _ in children_results))
                + train_examples
            )

            results["children_results"].append(children_results)
            results["train_results"].append((train_examples, train_metrics))

        return (
            self.client_parameters,
            float(total_examples) / self.num_rounds,
            results,
        )

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def get_parameters(self, config: Dict) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        if self.net is None:
            self.net = self.net_generator()
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    stragglers: float,
    model: DictConfig,
) -> Tuple[
    Callable[[str], FlowerClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    num_rounds: int
        The number of rounds in the experiment. This is used to construct
        the scheduling for stragglers
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    stragglers : float
        Proportion of stragglers in the clients, between 0 and 1.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # Defines a staggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of staggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            stragglers_mat[int(cid)],
        )

    return client_fn
