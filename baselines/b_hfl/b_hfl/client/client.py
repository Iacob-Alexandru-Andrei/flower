"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import ray
import utils
from flwr.common import (
    Code,
    GetPropertiesIns,
    GetPropertiesRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Status,
)
from torch.utils.data import Dataset
from utils import get_parameters, get_seeded_rng, set_parameters

from b_hfl.common_types import (
    ClientEvaluateFutureList,
    ClientFitFutureList,
    ClientFN,
    ClientResGeneratorList,
    ConfigSchemaGenerator,
    DataloaderGenerator,
    EvalRecursiveStructure,
    EvalRes,
    FitRecursiveStructure,
    FitRes,
    NetGenerator,
    NodeOpt,
    RecursiveBuilder,
    RecursiveStructure,
    TestFunc,
    TrainFunc,
)
from b_hfl.schemas.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)


class RecursiveClient(ConfigurableRecClient):
    """Implement recursive client, which can be used to create a tree of clients.

    This class is used to create a tree of clients where data can flow bidirectionally,
    client execution is replaced with subtree execution.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        cid: str,
        true_id: Path,
        root: Path,
        parent: Optional[ConfigurableRecClient],
        net_generator: NetGenerator,
        anc_node_opt: NodeOpt,
        desc_node_opt: NodeOpt,
        train: TrainFunc,
        test: TestFunc,
        create_dataloader: DataloaderGenerator,
        recursive_builder: RecursiveBuilder,
        client_fn: ClientFN,
        fit_metrics_aggregation_fn: MetricsAggregationFn,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn,
        train_config_schema: ConfigSchemaGenerator,
        test_config_schema: ConfigSchemaGenerator,
        resources: Dict,
        timeout: Optional[float],
    ) -> None:
        """Initialise the recursive client.

        Use the given local parameters, child IDs and node_opt,train and test functions.
        #TODO: Add detailed explanation
        """
        self.cid = cid
        self.true_id = true_id
        self.parent = parent
        self.client_parameters: Optional[NDArrays] = None
        self.net_generator = net_generator

        # Procedures necessary for combining models, training models and testing models
        self.anc_node_opt = anc_node_opt
        self.desc_node_opt = desc_node_opt
        self.train_func = train
        self.test_func = test
        self.create_dataloader = create_dataloader

        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.train_schema = train_config_schema()
        self.test_schema = test_config_schema()

        # Handles building the hierarchical structure of the clients
        self.recursive_builder = recursive_builder
        self.client_fn = client_fn
        self.sep = "::"
        self.root = root

        # Ray execution resources
        self.resources = resources
        self.timeout = timeout

    def get_properties(self, config: GetPropertiesIns) -> GetPropertiesRes:
        """Return the properties of the client."""
        return GetPropertiesRes(
            Status(Code.OK, ""),
            {"true_id": self.true_id, "cid": self.cid},  # type: ignore
        )

    # pylint: disable=too-many-locals
    def fit(
        self,
        parameters: NDArrays,
        in_config: Union[Dict, RecClientRuntimeTrainConf],
    ) -> FitRes:
        """Execute the node subtree and fit the node model.

        #TODO: Add detailed explanation
        """
        if isinstance(in_config, Dict):
            self.test_schema(**in_config)
            config = RecClientRuntimeTrainConf(**in_config)
        else:
            config = in_config

        rng = get_seeded_rng(
            config.global_seed,
            config.client_seed,
            config.server_round,
            config.client_config.parent_round,
        )

        recursive_structure: RecursiveStructure = self.recursive_builder(
            self,
            self.client_fn,
            test=False,  # type: ignore
        )
        recursive_structure = cast(FitRecursiveStructure, recursive_structure)

        (
            parameter_generator,
            state_generator,
            train_proxy_dataset_generator,
            train_chain_dataset_generator,
            children_res_generator_list,
            get_residuals,
            send_residual,
            recursive_step,
        ) = recursive_structure

        children_res_generator_list = cast(
            ClientFitFutureList, children_res_generator_list
        )
        prev_round_examples, state_dict = (
            state_generator(config.state_generator_config)
            if state_generator is not None
            else config.client_config.initial_state
        )
        self.client_parameters = parameters

        if parameter_generator is not None:
            self.client_parameters = parameter_generator(config.parameter_config)

        self.client_parameters, merged_example_cnt, state_dict = self.anc_node_opt(
            (self.client_parameters, prev_round_examples, state_dict),
            [(parameters, config.client_config.num_examples, {})],
            get_residuals(False),
            config.node_optimizer_config,
        )
        total_examples: Union[float, int] = 0
        prev_round_examples: Optional[Union[float, int]] = None
        train_chain_examples: int = 0
        train_proxy_examples: int = 0
        children_results: List[Tuple[int, Dict]] = []
        train_proxy_metrics: Dict = {}
        train_chain_metrics: Dict = {}
        for i in range(config.client_config.num_rounds):
            # Synchronise clients
            # used to set seeds for reproducibility
            config.client_config.parent_round = i
            round_examples: Union[float, int] = 0

            children_results = []

            if config.client_config.train_children:
                for client_id in config.client_config.root_to_leaf_residuals:
                    send_residual(
                        client_id,
                        (
                            self.client_parameters,
                            int(prev_round_examples)
                            if prev_round_examples is not None
                            else merged_example_cnt,
                            state_dict,
                        ),
                        False,
                    )
                selected_children: ClientResGeneratorList = rng.sample(
                    children_res_generator_list,
                    int(
                        config.client_config.fit_fraction
                        * len(children_res_generator_list)
                    ),
                )

                (
                    self.client_parameters,
                    _,
                    state_dict,
                ) = self.desc_node_opt(
                    (
                        self.get_parameters(config.get_parameters_config),
                        1,
                        state_dict,
                    ),
                    process_fit_results_and_accumulate_metrics(
                        children_results,
                        self.get_parameters(config.get_parameters_config),
                        config,
                        selected_children,
                    ),
                    get_residuals(True),
                    config.node_optimizer_config,
                )
                round_examples += sum(
                    (num_examples for num_examples, _ in children_results)
                ) / len(children_results)

            (
                self.client_parameters,
                train_chain_examples,
                train_chain_metrics,
            ) = self._train(
                train_chain_dataset_generator,
                self.get_parameters(config.get_parameters_config),
                config,
                mode="" if config.client_config.train_chain else None,
            )

            round_examples += train_chain_examples

            (
                self.client_parameters,
                train_proxy_examples,
                train_proxy_metrics,
            ) = self._train(
                train_proxy_dataset_generator,
                self.get_parameters(config.get_parameters_config),
                config,
                mode="proxy" if config.client_config.train_proxy else None,
            )
            round_examples += train_proxy_examples
            total_examples += round_examples
            prev_round_examples = round_examples

            self.client_parameters = self.get_parameters(config.get_parameters_config)
            recursive_step(
                (self.client_parameters, config),
                final=False,  # type: ignore
            )

        for client_id in config.client_config.leaf_to_root_residuals:
            send_residual(
                client_id,
                (self.client_parameters, int(total_examples), state_dict),
                True,
            )

        results: Metrics = {}

        children_results_final_round = self.fit_metrics_aggregation_fn(children_results)
        results.update(children_results_final_round)

        train_self_metrics_final_round = self.fit_metrics_aggregation_fn(
            [
                (train_chain_examples, train_chain_metrics),
                (train_proxy_examples, train_proxy_metrics),
            ],
        )
        results.update(train_self_metrics_final_round)
        results.update(train_chain_metrics)
        results.update(train_proxy_metrics)

        recursive_step((self.client_parameters, config), final=True)  # type: ignore
        return (
            self.get_parameters(config.get_parameters_config),
            int(float(total_examples) / config.client_config.num_rounds),
            results,
        )

    # pylint: disable=too-many-locals
    def evaluate(
        self, parameters: NDArrays, in_config: Union[Dict, RecClientRuntimeTestConf]
    ) -> EvalRes:
        """Evaluate the node model and lazily evaluate the children models + parent.

        #TODO: Add detailed explanation
        """
        if isinstance(in_config, Dict):
            self.test_schema(**in_config)
            config = RecClientRuntimeTestConf(**in_config)
        else:
            config = in_config

        rng = get_seeded_rng(
            config.global_seed,
            config.client_seed,
            config.server_round,
            config.client_config.parent_round,
        )

        recursive_structure: RecursiveStructure = self.recursive_builder(
            self, self.client_fn, test=True  # type: ignore
        )

        recursive_structure = cast(EvalRecursiveStructure, recursive_structure)

        (
            parameter_generator,
            test_proxy_dataset_generator,
            test_chain_dataset_generator,
            children_res_generator_list,
        ) = recursive_structure

        children_res_generator_list = cast(
            ClientEvaluateFutureList, children_res_generator_list
        )

        self.client_parameters = parameters

        if parameter_generator is not None:
            self.client_parameters = parameter_generator(config.parameter_config)

        selected_children = rng.sample(
            children_res_generator_list,
            int(config.client_config.eval_fraction * len(children_res_generator_list)),
        )

        children_results = (
            process_evaluate_results(parameters, config, selected_children)
            if config.client_config.test_children
            else []
        )

        loss, num_examples, test_metrics = self._test(
            test_chain_dataset_generator,
            self.get_parameters(config.get_parameters_config),
            config,
            mode="chain" if config.client_config.test_chain else None,
        )

        _, proxy_num_examples, proxy_metrics = self._test(
            test_proxy_dataset_generator,
            self.get_parameters(config.get_parameters_config),
            config,
            mode="proxy" if config.client_config.test_proxy else None,
        )
        results: Metrics = {}
        children_results_metrics = self.evaluate_metrics_aggregation_fn(
            [(num_examples, metrics) for _, num_examples, metrics in children_results]
        )
        results.update(children_results_metrics)

        test_self_metrics = self.evaluate_metrics_aggregation_fn(
            [(num_examples, test_metrics), (proxy_num_examples, proxy_metrics)]
        )
        results.update(test_self_metrics)

        return (
            loss,
            num_examples,
            results,
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the parameters of the current net."""
        if self.client_parameters is not None:
            return self.client_parameters

        return get_parameters(self.net_generator(config))

    # pylint: disable=unused-argument
    def set_parameters(self, parameters: NDArrays, config: Dict) -> None:
        """Change the parameters of the model using the given ones."""
        self.client_parameters = parameters

    def _train(
        self,
        train_dataset_generator: Optional[Callable[[Dict], Dataset]],
        parameters: NDArrays,
        config: RecClientRuntimeTrainConf,
        mode: Optional[str],
    ) -> FitRes:
        if train_dataset_generator is not None and mode is not None:
            remote_fit_res = remote_train.options(  # type: ignore
                **self.resources
            ).remote(
                self.net_generator,
                self.train_func,
                train_dataset_generator,
                self.create_dataloader,
                parameters,
                config,
                mode,
                self.true_id,
                self.root,
                self.sep,
            )
            return cast(
                FitRes,
                ray.get(remote_fit_res, timeout=self.timeout),
            )

        return parameters, 0, {}

    def _test(
        self,
        test_dataset_generator: Optional[Callable[[Dict], Dataset]],
        parameters: NDArrays,
        config: RecClientRuntimeTestConf,
        mode: Optional[str],
    ) -> EvalRes:
        if test_dataset_generator is not None and mode is not None:
            remote_fit_res = remote_test.options(  # type: ignore
                **self.resources
            ).remote(
                self.net_generator,
                self.test_func,
                test_dataset_generator,
                self.create_dataloader,
                parameters,
                config,
                mode,
                self.true_id,
                self.root,
                self.sep,
            )
            return cast(
                EvalRes,
                ray.get(remote_fit_res, timeout=self.timeout),
            )

        return 0.0, 0, {}


# pylint: disable=too-many-locals,too-many-arguments
@ray.remote
def remote_train(
    net_generator: NetGenerator,
    train_func: TrainFunc,
    train_dataset_generator: Callable[[Dict], Dataset],
    create_dataloader: DataloaderGenerator,
    parameters: NDArrays,
    config: RecClientRuntimeTrainConf,
    mode: str,
    true_id: Path,
    root: Path,
    sep,
) -> FitRes:
    """Train the model remotely on ray."""
    net = net_generator(config.net_config)

    device = utils.get_device()

    set_parameters(net, parameters, to_copy=True)
    net.to(device)
    net.train()
    num_examples, metrics = train_func(
        net,
        create_dataloader(
            train_dataset_generator(config.dataset_generator_config),
            config.dataloader_config | {"test": False},
        ),
        config.run_config | {"device": device},
    )
    relative_id = true_id.relative_to(root.parent)

    return (
        get_parameters(net),
        num_examples,
        {f"{relative_id}{sep}{mode}{sep}{key}": val for key, val in metrics.items()},
    )


# pylint: disable=too-many-locals,too-many-arguments
@ray.remote
def remote_test(
    net_generator: NetGenerator,
    test_func: TestFunc,
    test_dataset_generator: Callable[[Dict], Dataset],
    create_dataloader: DataloaderGenerator,
    parameters: NDArrays,
    config: RecClientRuntimeTestConf,
    mode: Optional[str],
    true_id: Path,
    root: Path,
    sep,
) -> EvalRes:
    """Test the model remotely on ray."""
    if test_dataset_generator is not None and mode is not None:
        net = net_generator(config.net_config)
        device = utils.get_device()

        set_parameters(net, parameters, to_copy=True)
        net.to(device)
        net.eval()
        loss, num_examples, metrics = test_func(
            net,
            create_dataloader(
                test_dataset_generator(config.dataset_generator_config),
                config.dataloader_config | {"test": False},
            ),
            config.run_config | {"device": device},
        )
        relative_id = true_id.relative_to(root.parent)

        return (
            loss,
            num_examples,
            {
                f"{relative_id}{sep}{key}{sep}{mode}": val
                for key, val in metrics.items()
            },
        )

    return 0.0, 0, {}


def process_fit_results_and_accumulate_metrics(
    result_metrics: List,
    parameters: NDArrays,
    parent_conf: RecClientRuntimeTrainConf,
    fit_res_generators: ClientFitFutureList,
) -> Generator[FitRes, None, None]:
    """Fit the children and accumulate the metrics."""
    futures = [
        child_gen(parameters, conf, parent_conf)
        for child_gen, conf in fit_res_generators
    ]
    while futures:
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            futures.remove(future)
            future_result = future.result()
            result_metrics.append(future_result[1:])
            yield future_result


def process_evaluate_results(
    parameters: NDArrays,
    parent_conf: RecClientRuntimeTestConf,
    eval_res_generator: ClientEvaluateFutureList,
) -> Generator[EvalRes, None, None]:
    """Evaluate the children."""
    futures = [
        child_gen(parameters, conf, parent_conf)
        for child_gen, conf in eval_res_generator
    ]
    while futures:
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            futures.remove(future)
            future_result = future.result()
            yield future_result


# pylint: disable=too-many-arguments
def get_client_fn(
    net_generator: NetGenerator,
    anc_node_opt: NodeOpt,
    desc_node_opt: NodeOpt,
    train: TrainFunc,
    test: TestFunc,
    create_dataloader: DataloaderGenerator,
    fit_metrics_aggregation_fn: MetricsAggregationFn,
    evaluate_metrics_aggregation_fn: MetricsAggregationFn,
    resources: Dict,
    timeout: Optional[float],
) -> ClientFN:
    """Generate a client function with the given methods for the client."""

    def client_fn(
        cid: str,
        true_id: Path,
        root: Path,
        parent: Optional[ConfigurableRecClient],
        train_config_schema: ConfigSchemaGenerator,
        test_config_schema: ConfigSchemaGenerator,
        recursive_builder: RecursiveBuilder,
    ) -> RecursiveClient:
        """Create a Flower client with a hierarchical structure.

        Since each client needs to create their own children, the function is passed in
        as an argument to each client.
        """
        return RecursiveClient(
            cid=cid,
            true_id=true_id,
            root=root,
            parent=parent,
            net_generator=net_generator,
            anc_node_opt=anc_node_opt,
            desc_node_opt=desc_node_opt,
            train=train,
            test=test,
            create_dataloader=create_dataloader,
            recursive_builder=recursive_builder,
            client_fn=client_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            train_config_schema=train_config_schema,
            test_config_schema=test_config_schema,
            resources=resources,
            timeout=timeout,
        )

    return client_fn
