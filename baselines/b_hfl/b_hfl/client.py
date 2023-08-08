"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import random
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
)

import flwr as fl
import torch
import torch.nn as nn
import utils
from dataset_preparation import FileHierarchy
from flwr.common import NDArrays
from mypy_extensions import NamedArg
from state_management import (
    DatasetManager,
    ParameterManager,
    extract_file_from_files,
    load_parameters,
    process_chain_file,
    process_file,
)
from torch.utils.data import DataLoader, Dataset
from utils import PathLike

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

ClientGeneratorFunction = Callable[
    [PathLike, fl.client.NumPyClient, Any], fl.client.NumPyClient
]

RecursiveBuilder = Callable[
    [
        fl.client.NumPyClient,
        ClientGeneratorFunction,
        NamedArg(bool, "test"),
    ],
    RecursiveStructure,
]


class RecursiveClient(fl.client.NumPyClient):
    """Implement recursive client, which can be used to create a tree of clients.

    This class is used to create a tree of clients where data can flow bidirectionally,
    client execution is replaced with subtree execution.
    """

    def __init__(
        self,
        cid: PathLike,
        parent: Any,
        net_generator: Callable[[Dict], nn.Module],
        node_opt: Callable[[NDArrays, Iterable[NDArrays]], NDArrays],
        train: Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]],
        test: Callable[[nn.Module, DataLoader, Dict], Tuple[float, int, Dict]],
        create_dataloader: Callable[[Optional[Dataset], Dict], Optional[DataLoader]],
        recursive_builder: RecursiveBuilder,
        client_fn: ClientGeneratorFunction,
    ) -> None:  # pylint: disable=too-many-arguments,too-many-instance-attributes
        """Initialise the recursive client.

        Use the given local parameters, child IDs and node_opt,train and test functions.
        #TODO: Add detailed explanation
        """
        self.cid = cid
        self.parent = parent
        self.net: Optional[nn.Module] = None
        self.net_generator = net_generator

        # Procedures necessary for combining models, training models and testing models
        self.node_opt = node_opt
        self.train_func = train
        self.test_func = test
        self.create_dataloader = create_dataloader

        # Handles building the hierarchical structure of the clients
        self.recursive_builder = recursive_builder
        self.client_fn = client_fn

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
            "train_chain_results": [],
        }
        recursive_structure: RecursiveStructure = self.recursive_builder(
            self,
            self.client_fn,
            test=False,  # type: ignore
        )

        (
            parameter_generator,
            train_proxy_dataset_generator,
            train_chain_dataset_generator,
            children_generator_list,
            recursive_step,
        ) = recursive_structure

        client_parameters = parameter_generator(config)
        if client_parameters is None:
            client_parameters = parameters
        client_parameters = self.node_opt(client_parameters, [parameters])
        total_examples: int = 0
        for _i in range(config["num_rounds"]):
            children_results: List[Tuple[int, Dict]] = []

            selected_children: List[Tuple[Callable[[], Any], Dict]] = random.sample(
                children_generator_list, config["fit_fraction"]
            )

            client_parameters = self.node_opt(
                client_parameters,
                process_children_and_accumulate_metrics(
                    children_results,
                    client_parameters,
                    selected_children,
                ),
            )
            if config["train"]:
                train_examples, train_metrics = self.__train(
                    self.create_dataloader(
                        train_chain_dataset_generator(config), config
                    ),
                    client_parameters,
                    config,
                )
                total_examples += (
                    sum((num_examples for num_examples, _ in children_results))
                    + train_examples
                )
                results["train_results"].append((train_examples, train_metrics))

                client_parameters = self.get_parameters(config)

            results["children_results"].append(children_results)

            recursive_step(config, final=False)  # type: ignore

        recursive_step(config, final=True)  # type: ignore
        return (
            client_parameters,
            int(float(total_examples) / config["num_rounds"]),
            results,
        )

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the node model and lazily evaluate the children models + parent.

        #TODO: Add detailed explanation
        """
        recursive_structure: RecursiveStructure = self.recursive_builder(
            self, self.client_fn, test=True  # type: ignore
        )

        (
            parameter_generator,
            test_proxy_dataset_generator,
            test_chain_dataset_generator,
            children_generator_list,
            recursive_step,
        ) = recursive_structure
        client_parameters = parameter_generator(config)

        if client_parameters is None:
            client_parameters = parameters

        self.selected_children = random.sample(
            children_generator_list, config["eval_fraction"]
        )

        children_results = (
            child_generator().evaluate(parameters, conf)
            if config["test_children"]
            else {}
            for child_generator, conf in children_generator_list
        )

        test_self_results = (
            self.__test(
                self.create_dataloader(test_proxy_dataset_generator(config), config),
                client_parameters,
                config,
            )
            if config["test_self"]
            else {}
            for i in range(1)
        )

        loss, num_examples, test_metrics = (
            self.__test(
                self.create_dataloader(test_chain_dataset_generator(config), config),
                client_parameters,
                config,
            )
            if config["test_chain"]
            else (0.0, 0, {})
        )

        recursive_step([client_parameters, config], final=True)  # type: ignore

        return (
            loss,
            num_examples,
            {
                "test_chain_metrics": test_metrics,
                "test_self_results_generator": test_self_results,
                "children_results_generator": children_results,
            },
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the parameters of the current net."""
        self.net = self.net if self.net is not None else self.net_generator(config)

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays, config: Dict) -> None:
        """Change the parameters of the model using the given ones."""
        self.net = self.net if self.net is not None else self.net_generator(config)

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def __train(
        self, train_loader: Optional[DataLoader], parameters: NDArrays, config: Dict
    ) -> Tuple[int, Dict]:
        if train_loader is not None:
            self.net = self.net if self.net is not None else self.net_generator(config)
            device = utils.get_device()

            self.set_parameters(parameters, config)
            self.net.to(device)
            self.net.train()
            return self.train_func(self.net, train_loader, config | {"device": device})

        return 0, {}

    def __test(
        self, test_loader: Optional[DataLoader], parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict]:
        if test_loader is not None:
            self.net = self.net if self.net is not None else self.net_generator(config)
            device = utils.get_device()

            self.set_parameters(parameters, config)
            self.net.to(device)
            self.net.eval()
            return self.test_func(self.net, test_loader, config | {"device": device})
        else:
            return 0.0, 0, {}


def process_children_and_accumulate_metrics(
    result_metrics: List,
    parameters: NDArrays,
    children_generators: ClientGeneratorList,
) -> Generator[NDArrays, None, None]:
    """Process the children and accumulate the metrics."""
    for child_gen, conf in children_generators:
        parameters, *everything_else = child_gen().fit(parameters, conf)
        result_metrics.append(everything_else)
        yield parameters


def generate_recursive_builder(
    path_dict: FileHierarchy,
    load_dataset_file: Callable[[Path], Dataset],
    dataset_manager: DatasetManager,
    load_parameters_file: Callable[[Path], NDArrays],
    parameter_manager: ParameterManager,
) -> RecursiveBuilder:
    """Generate a recursive builder for a given file hierarchy.

    This is recursively caled within the inner function for each child of a given node.
    """

    def recursive_builder(
        current_client: fl.client.NumPyClient,
        client_fn: ClientGeneratorFunction,
        test: bool = False,
    ) -> RecursiveStructure:
        """Generate a recursive structure for a given client."""
        file_type: str = "test" if test else "train"

        dataset_file: Optional[Path] = extract_file_from_files(
            path_dict["files"], file_type
        )

        def dataset_generator(_config) -> Optional[Dataset]:
            return (
                process_file(dataset_file, load_dataset_file, dataset_manager)
                if dataset_file is not None
                else None
            )

        chain_dataset_file: Optional[Path] = extract_file_from_files(
            path_dict["files"], file_type
        )

        def chain_dataset_generator(_config: Dict) -> Optional[Dataset]:
            return (
                process_chain_file(
                    chain_dataset_file, load_dataset_file, dataset_manager
                )
                if chain_dataset_file is not None
                else None
            )

        parameter_file: Optional[Path] = extract_file_from_files(
            path_dict["files"], "parameters"
        )

        def parameter_generator(_config: Dict) -> Optional[NDArrays]:
            return (
                load_parameters(parameter_file, load_parameters_file, parameter_manager)
                if parameter_file is not None
                else None
            )

        def get_child_generator(
            child_path_dict: FileHierarchy,
        ) -> Callable[[], fl.client.NumPyClient]:
            return lambda: client_fn(
                child_path_dict["path"],
                current_client,
                generate_recursive_builder(
                    child_path_dict,
                    load_dataset_file,
                    dataset_manager,
                    load_parameters_file,
                    parameter_manager,
                ),
            )

        child_generator: ClientGeneratorList = [
            (
                get_child_generator(child_path_dict),
                {},
            )
            for child_path_dict in path_dict["children"]
        ]

        def recursive_step(state: Tuple[NDArrays, Dict], final: bool) -> None:
            parameter_manager.set_parameters(
                path_dict["path"] / "parameters.pt", state[0]
            )

            children_dataset_files: Generator[Optional[Path], None, None] = (
                file
                for file in (
                    extract_file_from_files(child_dict["files"], file_type)
                    for child_dict in path_dict["children"]
                )
                if file is not None
            )
            if chain_dataset_file is not None:
                dataset_manager.unload_chain_dataset(chain_dataset_file)

            if final:
                dataset_manager.unload_children_datasets(children_dataset_files)
                parameter_manager.unload_children_parameters(children_dataset_files)

        return (
            parameter_generator,
            dataset_generator,
            chain_dataset_generator,
            child_generator,
            recursive_step,
        )

    return recursive_builder


def gen_client_fn(
    net_generator: Callable[[Dict], nn.Module],
    node_opt: Callable[[NDArrays, Iterable[NDArrays]], NDArrays],
    train: Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]],
    test: Callable[[nn.Module, DataLoader, Dict], Tuple[float, int, Dict]],
    create_dataloader: Callable[[Optional[Dataset], Dict], Optional[DataLoader]],
) -> Callable[
    [PathLike, fl.client.NumPyClient, RecursiveBuilder], fl.client.NumPyClient
]:  # pylint: disable=too-many-arguments
    """Generate a client function with the given methods for the client."""

    def client_fn(
        cid: PathLike,
        parent: fl.client.NumPyClient,
        recursive_builder: RecursiveBuilder,
    ) -> RecursiveClient:
        """Create a Flower client with a hierarchical structure.

        Since each client needs to create their own children, the function is passed in
        as an argument to each client.
        """
        return RecursiveClient(
            cid=cid,
            parent=parent,
            net_generator=net_generator,
            node_opt=node_opt,
            train=train,
            test=test,
            create_dataloader=create_dataloader,
            recursive_builder=recursive_builder,
            client_fn=client_fn,
        )

    return client_fn
