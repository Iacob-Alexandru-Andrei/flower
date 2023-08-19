"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import os
import random
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, OrderedDict, Tuple

import flwr as fl
import torch
import torch.nn as nn
import utils
from common_types import (
    ClientFN,
    ClientGeneratorList,
    DataloaderGenerator,
    DatasetLoader,
    LoadConfig,
    NetGenerator,
    NodeOpt,
    ParametersLoader,
    PathLike,
    RecursiveBuilder,
    RecursiveBuilderWrapper,
    RecursiveStructure,
    TestFunc,
    TrainFunc,
)
from dataset_preparation import FolderHierarchy
from flwr.common import Code, GetPropertiesIns, GetPropertiesRes, NDArrays, Status
from state_management import (
    DatasetManager,
    ParameterManager,
    load_parameters,
    process_chain_file,
    process_file,
)
from torch.utils.data import DataLoader, Dataset
from utils import extract_file_from_files


class RecursiveClient(fl.client.NumPyClient):
    """Implement recursive client, which can be used to create a tree of clients.

    This class is used to create a tree of clients where data can flow bidirectionally,
    client execution is replaced with subtree execution.
    """

    def __init__(
        self,
        cid: str,
        true_id: PathLike,
        parent: Optional[fl.client.NumPyClient],
        net_generator: NetGenerator,
        node_opt: NodeOpt,
        train: TrainFunc,
        test: TestFunc,
        create_dataloader: DataloaderGenerator,
        recursive_builder: RecursiveBuilder,
        client_fn: ClientFN,
    ) -> None:  # pylint: disable=too-many-arguments,too-many-instance-attributes
        """Initialise the recursive client.

        Use the given local parameters, child IDs and node_opt,train and test functions.
        #TODO: Add detailed explanation
        """
        self.cid = cid
        self.true_id = true_id
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

    def get_properties(self, config: GetPropertiesIns) -> GetPropertiesRes:
        """Return the properties of the client."""
        return GetPropertiesRes(
            Status(Code.OK, ""),
            {"true_id": self.true_id, "cid": self.cid},  # type: ignore
        )

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

        client_parameters = parameter_generator(config["parameter_config"])
        if client_parameters is None:
            client_parameters = parameters
        client_parameters = self.node_opt(
            client_parameters, [(parameters, 1, {})], config
        )
        total_examples: int = 0
        for _i in range(config["client_config"]["num_rounds"]):
            children_results: List[Tuple[int, Dict]] = []
            if config["client_config"]["train_children"]:
                selected_children: List[Tuple[Callable[[], Any], Dict]] = random.sample(
                    children_generator_list,
                    int(
                        config["client_config"]["fit_fraction"]
                        * len(children_generator_list)
                    ),
                )

                client_parameters = self.node_opt(
                    client_parameters,
                    process_children_and_accumulate_metrics(
                        children_results,
                        client_parameters,
                        selected_children,
                    ),
                    config,
                )
                total_examples += sum(
                    (num_examples for num_examples, _ in children_results)
                )

            if config["client_config"]["train_chain"]:
                train_examples, train_metrics = self._train(
                    self.create_dataloader(
                        train_chain_dataset_generator(config),
                        config["dataloader_config"] | {"test": False},
                    ),
                    client_parameters,
                    config,
                )
                total_examples += train_examples

                results["train_chain_results"].append((train_examples, train_metrics))

            if config["client_config"]["train_proxy"]:
                train_examples, train_metrics = self._train(
                    self.create_dataloader(
                        train_proxy_dataset_generator(config),
                        config["dataloader_config"] | {"test": False},
                    ),
                    client_parameters,
                    config,
                )
                total_examples += train_examples

                results["train_proxy_results"].append((train_examples, train_metrics))

            client_parameters = self.get_parameters(config)

            results["children_results"].append(children_results)

            recursive_step((client_parameters, config), final=False)  # type: ignore

        recursive_step((client_parameters, config), final=True)  # type: ignore
        return (
            client_parameters,
            int(float(total_examples) / config["client_config"]["num_rounds"]),
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
            _recursive_step,
        ) = recursive_structure
        client_parameters = parameter_generator(config["parameter_config"])

        if client_parameters is None:
            client_parameters = parameters

        self.selected_children = random.sample(
            children_generator_list,
            int(
                config["client_config"]["eval_fraction"] * len(children_generator_list)
            ),
        )

        children_results = [
            child_generator().evaluate(parameters, conf)
            if config["client_config"]["test_children"]
            else {}
            for child_generator, conf in children_generator_list
        ]

        loss, num_examples, test_metrics = (
            self._test(
                self.create_dataloader(
                    test_chain_dataset_generator(config),
                    config["dataloader_config"] | {"test": False},
                ),
                client_parameters,
                config,
            )
            if config["client_config"]["test_chain"]
            else (0.0, 0, {})
        )

        test_proxy_results = [
            self._test(
                self.create_dataloader(
                    test_proxy_dataset_generator(config),
                    config["dataloader_config"] | {"test": False},
                ),
                client_parameters,
                config,
            )
            if config["client_config"]["test_proxy"]
            else {}
            for i in range(1)
        ]

        return (
            loss,
            num_examples,
            {
                "test_chain_metrics": test_metrics,
                "test_proxy_results": test_proxy_results,
                "children_results": children_results,
            },
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the parameters of the current net."""
        self.net = (
            self.net
            if self.net is not None
            else self.net_generator(config["net_config"])
        )

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays, config: Dict) -> None:
        """Change the parameters of the model using the given ones."""
        self.net = (
            self.net
            if self.net is not None
            else self.net_generator(config["net_config"])
        )

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def _train(
        self, train_loader: Optional[DataLoader], parameters: NDArrays, config: Dict
    ) -> Tuple[int, Dict]:
        if train_loader is not None:
            self.net = (
                self.net
                if self.net is not None
                else self.net_generator(config["net_config"])
            )
            device = utils.get_device()

            self.set_parameters(parameters, config)
            self.net.to(device)
            self.net.train()
            return self.train_func(
                self.net, train_loader, config["run_config"] | {"device": device}
            )

        return 0, {}

    def _test(
        self, test_loader: Optional[DataLoader], parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict]:
        if test_loader is not None:
            self.net = (
                self.net
                if self.net is not None
                else self.net_generator(config["net_config"])
            )
            device = utils.get_device()

            self.set_parameters(parameters, config)
            self.net.to(device)
            self.net.eval()
            return self.test_func(
                self.net, test_loader, config["run_config"] | {"device": device}
            )
        else:
            return 0.0, 0, {}


def process_children_and_accumulate_metrics(
    result_metrics: List,
    parameters: NDArrays,
    children_generators: ClientGeneratorList,
) -> Generator[Tuple[NDArrays, int, Dict], None, None]:
    """Process the children and accumulate the metrics."""
    for child_gen, conf in children_generators:
        parameters, num_examples, metrics = child_gen().fit(parameters, conf)
        result_metrics.append((num_examples, metrics))
        yield parameters, num_examples, metrics


def get_recursive_builder(
    root: Path,
    path_dict: FolderHierarchy,
    load_dataset_file: DatasetLoader,
    dataset_manager: DatasetManager,
    load_parameters_file: ParametersLoader,
    parameter_manager: ParameterManager,
    on_fit_config_fn: LoadConfig,
    on_evaluate_config_fn: LoadConfig,
    parameters_file_name: str,
    parameters_ext: str = ".pt",
) -> RecursiveBuilder:
    """Generate a recursive builder for a given file hierarchy.

    This is recursively caled within the inner function for each child of a given node.
    """
    round: int = 0

    def recursive_builder(
        current_client: fl.client.NumPyClient,
        client_fn: ClientFN,
        test: bool = False,
    ) -> RecursiveStructure:
        """Generate a recursive structure for a given client."""
        file_type: str = "test" if test else "train"

        dataset_file: Optional[Path] = extract_file_from_files(
            path_dict["path"], file_type
        )

        def dataset_generator(_config) -> Optional[Dataset]:
            return (
                process_file(dataset_file, load_dataset_file, dataset_manager)
                if dataset_file is not None
                else None
            )

        chain_dataset_file: Optional[Path] = extract_file_from_files(
            path_dict["path"], f"{file_type}_chain"
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
            path_dict["path"], parameters_file_name
        )

        def parameter_generator(_config: Dict) -> Optional[NDArrays]:
            return (
                load_parameters(parameter_file, load_parameters_file, parameter_manager)
                if parameter_file is not None
                else None
            )

        def get_child_generator(
            child_path_dict: FolderHierarchy,
        ) -> Callable[[], fl.client.NumPyClient]:
            return lambda: client_fn(
                str(child_path_dict["path"]),
                child_path_dict["path"],
                current_client,
                get_recursive_builder(
                    root=root,
                    path_dict=child_path_dict,
                    load_dataset_file=load_dataset_file,
                    dataset_manager=dataset_manager,
                    load_parameters_file=load_parameters_file,
                    parameter_manager=parameter_manager,
                    on_fit_config_fn=on_fit_config_fn,
                    on_evaluate_config_fn=on_evaluate_config_fn,
                    parameters_file_name=parameters_file_name,
                    parameters_ext=parameters_ext,
                ),
            )

        config_fn = on_fit_config_fn if not test else on_evaluate_config_fn
        child_generator: ClientGeneratorList = [
            (
                get_child_generator(child_path_dict),
                config_fn(round, child_path_dict["path"]),
            )
            for child_path_dict in path_dict["children"]
        ]

        def recursive_step(state: Tuple[NDArrays, Dict], final: bool) -> None:
            nonlocal round
            round += 1

            if final:
                parameters_save_name = (
                    path_dict["path"] / f"{parameters_file_name}{parameters_ext}"
                    if parameter_file is None
                    else parameter_file
                )

                parameter_manager.set_parameters(parameters_save_name, state[0])
                # The root client will die
                if root == path_dict["path"]:
                    parameter_manager.cleanup()
                else:
                    if chain_dataset_file is not None:
                        dataset_manager.unload_chain_dataset(chain_dataset_file)

                    children_dataset_files: Generator[Optional[Path], None, None] = (
                        file
                        for file in (
                            extract_file_from_files(child_dict["path"], file_type)
                            for child_dict in path_dict["children"]
                        )
                        if file is not None
                    )

                    children_chain_dataset_files: Generator[
                        Optional[Path], None, None
                    ] = (
                        file
                        for file in (
                            extract_file_from_files(
                                child_dict["path"], f"{file_type}_chain"
                            )
                            for child_dict in path_dict["children"]
                        )
                        if file is not None
                    )

                    dataset_manager.unload_children_datasets(
                        chain(children_chain_dataset_files, children_dataset_files)
                    )

                    round = 0

        return (
            parameter_generator,
            dataset_generator,
            chain_dataset_generator,
            child_generator,
            recursive_step,
        )

    return recursive_builder


def get_client_fn(
    net_generator: NetGenerator,
    node_opt: NodeOpt,
    train: TrainFunc,
    test: TestFunc,
    create_dataloader: DataloaderGenerator,
) -> ClientFN:  # pylint: disable=too-many-arguments
    """Generate a client function with the given methods for the client."""

    def client_fn(
        cid: str,
        true_id: PathLike,
        parent: Optional[fl.client.NumPyClient],
        recursive_builder: RecursiveBuilder,
    ) -> RecursiveClient:
        """Create a Flower client with a hierarchical structure.

        Since each client needs to create their own children, the function is passed in
        as an argument to each client.
        """
        return RecursiveClient(
            cid=cid,
            true_id=true_id,
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


def get_recursive_builder_wrapper() -> RecursiveBuilderWrapper:
    """Get a recursive builder wrapper."""
    return get_recursive_builder
