"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Tuple,
)

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import NDArrays
from hydra.utils import instantiate
from mypy_extensions import NamedArg
from omegaconf import DictConfig
from torch.utils.data import DataLoader

ClientGenerator = Generator[Tuple[fl.client.NumPyClient, Dict], None, None]


RecursiveStructure = Tuple[Any, DataLoader, ClientGenerator]


class RecursiveClient(fl.client.NumPyClient):
    """Implement recursive client, which can be used to create a tree of clients.

    This class is used to create a tree of clients where data can flow bidirectionally,
    client execution is replaced with subtree execution.
    """

    def __init__(
        self,
        parent: fl.client.NumPyClient,
        net_generator: Callable[[], nn.Module],
        num_training_epochs: int,
        num_rounds: int,
        learning_rate: float,
        learning_rate_decay: float,
        client_parameters: NDArrays,
        node_opt: Callable[[NDArrays, Iterable[NDArrays]], NDArrays],
        train: Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]],
        test: Callable[[nn.Module, DataLoader, Dict], Tuple[float, int, Dict]],
        state: Any,
        level: int,
        recursive_builder: Callable[
            [Any, int, NamedArg(bool, "test")], RecursiveStructure
        ],
    ):  # pylint: disable=too-many-arguments
        """Initialise the recursive client.

        Use the given local parameters, child IDs and node_opt,train and test functions.
        #TODO: Add detailed explanation
        """
        self.parent = parent
        self.net: Optional[nn.Module] = None
        self.net_generator = net_generator
        self.device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.num_epochs = num_training_epochs
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.client_parameters = client_parameters

        # Procedures necessary for combining models, training models and testing models
        self.node_opt = node_opt
        self.train_func = train
        self.test_func = test

        # Handles building the hierarchical structure of the clients
        self.state = state
        self.level = level
        self.recursive_builder = recursive_builder

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[NDArrays, int, Dict]:
        """Execute the node subtree and fit the node model.

        #TODO: Add detailed explanation
        """
        results: Dict[str, List] = {
            "children_results": [],
            "train_results": [],
        }

        self.client_parameters = self.node_opt(self.client_parameters, [parameters])
        total_examples: int = 0
        for _i in range(self.num_rounds):
            children_results: List[Tuple[int, Dict]] = []

            recursive_structure: RecursiveStructure = self.recursive_builder(
                self.state, self.level, test=False
            )

            state, train_loader, children_generator = recursive_structure

            self.state = state

            self.client_parameters = self.node_opt(
                self.client_parameters,
                self.__train_process_children(
                    children_results, self.client_parameters, children_generator
                ),
            )

            train_examples, train_metrics = self.__train(
                train_loader, self.client_parameters, config
            )
            self.client_parameters = self.get_parameters(config)

            total_examples += (
                sum((num_examples for num_examples, _ in children_results))
                + train_examples
            )

            results["children_results"].append(children_results)
            results["train_results"].append((train_examples, train_metrics))

        return (
            self.client_parameters,
            int(float(total_examples) / self.num_rounds),
            results,
        )

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the node model and lazily evaluate the children models + parent.

        #TODO: Add detailed explanation
        """
        recursive_structure: RecursiveStructure = self.recursive_builder(
            self.state, self.level, test=True
        )

        state, test_loader, children_generator = recursive_structure
        self.state = state

        children_results = (
            child.evaluate(parameters, conf) for child, conf in children_generator
        )

        loss, num_examples, test_metrics = self.__test(
            test_loader, self.client_parameters, config
        )

        parent_results = (
            self.__test(test_loader, parameters, config) for _ in range(1)
        )

        return (
            loss,
            num_examples,
            {
                "test_metrics": test_metrics,
                "parent_results": parent_results,
                "children_results": children_results,
            },
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the parameters of the current net."""
        self.net = self.net if self.net is not None else self.net_generator()

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        self.net = self.net if self.net is not None else self.net_generator()

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def __train_process_children(
        self,
        result_metrics: List,
        parameters: NDArrays,
        children_generator: ClientGenerator,
    ):
        for child, conf in children_generator:
            child_params, *everything_else = child.fit(parameters, conf)
            result_metrics.append(everything_else)
            yield child_params, *everything_else

    def __train(self, train_loader: DataLoader, parameters: NDArrays, config: Dict):
        self.net = self.net if self.net is not None else self.net_generator()

        self.set_parameters(parameters)
        self.net.to(self.device)
        self.net.train()
        return self.train_func(self.net, train_loader, config)

    def __test(self, test_loader: DataLoader, parameters: NDArrays, config: Dict):
        self.net = self.net if self.net is not None else self.net_generator()

        self.set_parameters(parameters)
        self.net.to(self.device)
        self.net.eval()
        return self.test_func(self.net, test_loader, config)


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
    Callable[[str], RecursiveClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients.

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

    def client_fn(cid: str) -> RecursiveClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return RecursiveClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            stragglers_mat[int(cid)],
        )

    return client_fn
