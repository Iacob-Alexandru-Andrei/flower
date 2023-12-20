"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import ray
from flwr.common import (
    Code,
    GetPropertiesIns,
    GetPropertiesRes,
    Metrics,
    NDArrays,
    Properties,
    Status,
)
from torch.utils.data import Dataset

from b_hfl.client.async_train import (
    process_evaluate_results,
    process_fit_results_and_accumulate_metrics,
    remote_test,
    remote_train,
)
from b_hfl.schema.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)
from b_hfl.typing.common_types import (
    ClientEvaluateFutureList,
    ClientFitFutureList,
    ClientFN,
    DataloaderGenerator,
    EvalRecursiveStructure,
    EvalRes,
    FitRecursiveStructure,
    FitRes,
    GetResiduals,
    MetricsAggregationFn,
    NetGenerator,
    DescNodeOpt,
    RecursiveBuilder,
    RecursiveStructure,
    SendResiduals,
    State,
    TestFunc,
    TrainFunc,
)
from b_hfl.utils.utils import (
    get_norm_of_parameter_difference,
    get_parameters_copy,
    get_seeded_rng,
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
        anc_node_opt: DescNodeOpt,
        desc_node_opt: DescNodeOpt,
        train: TrainFunc,
        test: TestFunc,
        create_dataloader: DataloaderGenerator,
        recursive_builder: RecursiveBuilder,
        client_fn: ClientFN,
        fit_metrics_aggregation_fn: MetricsAggregationFn,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn,
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
        self.anc_state_generator, self.anc_node_opt = anc_node_opt
        self.desc_state_generator, self.desc_node_opt = desc_node_opt
        self.train_func = train
        self.test_func = test
        self.create_dataloader = create_dataloader

        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        # Handles building the hierarchical structure of the clients
        self.recursive_builder = recursive_builder
        self.client_fn = client_fn
        self.sep = "::"
        self.root = root
        self.relative_id = self.true_id.relative_to(self.root.parent)

        # Ray execution resources
        self.resources = resources
        self.timeout = timeout

    # type: ignore[override]
    def get_properties(
        self, config: GetPropertiesIns  # type: ignore[override]
    ) -> GetPropertiesRes:  # type: ignore[override]
        """Return the properties of the client."""
        return GetPropertiesRes(
            Status(Code.OK, ""),
            cast(Properties, {"true_id": self.true_id, "cid": self.cid}),
        )

    # pylint: disable=too-many-locals,arguments-differ
    def fit(
        self,
        parameters: NDArrays,
        in_config: Union[Dict, RecClientRuntimeTrainConf],
    ) -> FitRes:
        """Execute the node subtree and fit the node model.

        #TODO: Add detailed explanation
        """
        if isinstance(in_config, Dict):
            config = RecClientRuntimeTrainConf(**in_config)
        else:
            config = in_config

        recursive_structure: RecursiveStructure = self.recursive_builder(
            self, self.client_fn, False
        )
        recursive_structure = cast(FitRecursiveStructure, recursive_structure)

        (
            parameter_generator,
            anc_state_loader,
            desc_state_loader,
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

        anc_state = (
            anc_state_loader(config.state_loader_config)
            if anc_state_loader is not None
            else self.anc_state_generator(config.state_generator_config)
        )

        desc_state = (
            desc_state_loader(config.state_loader_config)
            if desc_state_loader is not None
            else self.desc_state_generator(config.state_generator_config)
        )

        self.set_parameters(parameters, config.get_parameters_config)

        if parameter_generator is not None:
            self.set_parameters(
                parameter_generator(config.parameter_config),
                config.get_parameters_config,
            )

        saved_parameters = None
        if config.client_config.track_parameter_changes:
            saved_parameters = deepcopy(
                self.get_parameters(config.get_parameters_config)
            )

        merged_parameters, anc_state = self.anc_node_opt(
            anc_state,
            self.get_parameters(config.get_parameters_config),
            [
                (
                    parameters,
                    (1),
                    ({}),
                )
            ],
            get_residuals(False),
            config.anc_node_optimizer_config,
        )
        self.set_parameters(merged_parameters, config.get_parameters_config)

        results: Metrics = {}

        saved_parameters = self._add_norm(saved_parameters, results, "anc", config)

        total_examples: Union[float, int] = 0
        prev_round_examples: int = 0

        train_chain_examples: int = 0
        train_proxy_examples: int = 0
        children_results: List[Tuple[int, Dict]] = []
        train_proxy_metrics: Dict = {}
        train_chain_metrics: Dict = {}

        for i in range(config.client_config.num_rounds):
            # Synchronise clients
            # used to set seeds for reproducibility
            desc_state = (prev_round_examples, desc_state[-1])

            config.client_config.parent_round = i

            (
                parameters_updated,
                round_examples,
                children_results,
            ), desc_state = self._train_children(
                desc_state=desc_state,
                children_res_generator_list=children_res_generator_list,
                send_residual=send_residual,
                get_residuals=get_residuals,
                prev_round_examples=prev_round_examples,
                config=config,
            )

            self.set_parameters(parameters_updated, config.get_parameters_config)

            (
                trained_parameters,
                train_chain_examples,
                train_chain_metrics,
            ) = self._train(
                train_chain_dataset_generator,
                self.get_parameters(config.get_parameters_config),
                config,
                mode="chain" if config.client_config.train_chain else None,
            )

            self.set_parameters(trained_parameters, config.get_parameters_config)

            round_examples += train_chain_examples

            (
                trained_parameters,
                train_proxy_examples,
                train_proxy_metrics,
            ) = self._train(
                train_proxy_dataset_generator,
                self.get_parameters(config.get_parameters_config),
                config,
                mode="proxy" if config.client_config.train_proxy else None,
            )
            self.set_parameters(trained_parameters, config.get_parameters_config)

            round_examples += train_proxy_examples
            total_examples += round_examples
            prev_round_examples = round_examples

            recursive_step(
                (
                    self.get_parameters(config.get_parameters_config),
                    anc_state,
                    desc_state,
                ),
                False,
            )

        children_results_final_round = self.fit_metrics_aggregation_fn(
            children_results,
            self.relative_id,  # type: ignore[arg-type]
        )

        train_chain_metrics_final_round = self.fit_metrics_aggregation_fn(
            [
                (train_chain_examples, train_chain_metrics),
            ],
            self.relative_id,  # type: ignore[arg-type]
        )

        train_proxy_metrics_final_round = self.fit_metrics_aggregation_fn(
            [
                (train_proxy_examples, train_proxy_metrics),
            ],
            self.relative_id,  # type: ignore[arg-type]
        )

        results.update(children_results_final_round)
        results.update(train_chain_metrics_final_round)
        results.update(train_proxy_metrics_final_round)

        saved_parameters = self._add_norm(saved_parameters, results, "desc", config)

        recursive_step(
            (self.get_parameters(config.get_parameters_config), anc_state, desc_state),
            True,
        )
        return (
            self.get_parameters(config.get_parameters_config),
            int(float(total_examples) / config.client_config.num_rounds),
            results,
        )

    # pylint: disable=too-many-locals,arguments-differ
    def evaluate(
        self, parameters: NDArrays, in_config: Union[Dict, RecClientRuntimeTestConf]
    ) -> EvalRes:
        """Evaluate the node model and lazily evaluate the children models + parent.

        #TODO: Add detailed explanation
        """
        if isinstance(in_config, Dict):
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
            self, self.client_fn, True
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

        self.set_parameters(parameters, config.get_parameters_config)

        if parameter_generator is not None:
            self.set_parameters(
                parameter_generator(config.parameter_config),
                config.get_parameters_config,
            )

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
            [(num_examples, metrics) for _, num_examples, metrics in children_results],
            self.relative_id,
        )

        test_chain_metrics = self.evaluate_metrics_aggregation_fn(
            [(num_examples, test_metrics)],
            self.relative_id,
        )
        test_proxy_metrics = self.evaluate_metrics_aggregation_fn(
            [(proxy_num_examples, proxy_metrics)],
            self.relative_id,
        )

        results.update(children_results_metrics)
        results.update(test_chain_metrics)
        results.update(test_proxy_metrics)

        return (
            loss,
            num_examples,
            results,
        )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the parameters of the current net."""
        if self.client_parameters is not None:
            return self.client_parameters

        return get_parameters_copy(self.net_generator(config))

    # pylint: disable=unused-argument
    def set_parameters(self, parameters: NDArrays, config: Dict) -> None:
        """Change the parameters of the model using the given ones."""
        self.client_parameters = parameters

    def _train_children(
        self,
        desc_state: State,
        children_res_generator_list: ClientFitFutureList,
        send_residual: SendResiduals,
        get_residuals: GetResiduals,
        prev_round_examples: int,
        config: RecClientRuntimeTrainConf,
    ) -> Tuple[Tuple[NDArrays, int, List[Tuple[int, Dict]]], State]:
        if config.client_config.train_children:
            rng = get_seeded_rng(
                global_seed=config.global_seed,
                client_seed=config.client_seed,
                server_round=config.server_round,
                parent_round=config.client_config.parent_round,
            )
            for client_id in config.client_config.root_to_leaf_residuals:
                send_residual(
                    client_id,
                    (
                        self.get_parameters(config.get_parameters_config),
                        int(prev_round_examples),
                        {},
                    ),
                    False,
                )

            selected_children: ClientFitFutureList = rng.sample(
                children_res_generator_list,
                int(
                    config.client_config.fit_fraction * len(children_res_generator_list)
                ),
            )

            children_results: List[Tuple[int, Dict]] = []
            parameters_updated, desc_state = self.desc_node_opt(
                desc_state,
                self.get_parameters(config.get_parameters_config),
                process_fit_results_and_accumulate_metrics(
                    children_results,
                    self.get_parameters(config.get_parameters_config),
                    config,
                    selected_children,
                ),
                get_residuals(True),
                config.desc_node_optimizer_config,
            )

            round_examples = int(
                sum((num_examples for num_examples, _ in children_results))
                / len(children_results)
            )

            return (parameters_updated, round_examples, children_results), desc_state

        return (self.get_parameters(config.get_parameters_config), 0, []), desc_state

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
                self.relative_id,
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
                self.relative_id,
                self.sep,
            )
            return cast(
                EvalRes,
                ray.get(remote_fit_res, timeout=self.timeout),
            )

        return 0.0, 0, {}

    def _add_norm(
        self,
        saved_parameters: Optional[NDArrays],
        results: Dict,
        mode: str,
        config: RecClientRuntimeTrainConf,
    ) -> Optional[NDArrays]:
        if (
            config.client_config.track_parameter_changes
            and saved_parameters is not None
        ):
            (
                results[f"{self.relative_id}{self.sep}norm_pre{self.sep}anc"],
                results[f"{self.relative_id}{self.sep}norm_post{self.sep}anc"],
                results[f"{self.relative_id}{self.sep}norm_diff{self.sep}anc"],
            ) = get_norm_of_parameter_difference(
                saved_parameters, self.get_parameters(config.get_parameters_config)
            )

            return deepcopy(self.get_parameters(config.get_parameters_config))
        return None


# pylint: disable=too-many-arguments
def get_client_fn(
    net_generator: NetGenerator,
    anc_node_opt: DescNodeOpt,
    desc_node_opt: DescNodeOpt,
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
            resources=resources,
            timeout=timeout,
        )

    return client_fn
