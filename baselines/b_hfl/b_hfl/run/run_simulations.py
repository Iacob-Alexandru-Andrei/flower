"""Run either federated or centralised simulations.

Experiments execute hierarchically based on the folder structure of the data folder.
"""
import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import flwr as fl
from flwr.common import NDArrays, Parameters
from flwr.server import ServerConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig

from b_hfl.client.client import get_client_fn
from b_hfl.client.recursive_builder import (
    get_recursive_builder as recursive_bulder_generator,
)
from b_hfl.modified_flower.app import start_simulation
from b_hfl.modified_flower.client_manager import DeterministicClientManager
from b_hfl.modified_flower.server import History, Server
from b_hfl.schema.client_schema import ConfigurableRecClient
from b_hfl.schema.file_system_schema import FolderHierarchy
from b_hfl.strategy.logging_fed_avg import LoggingFedAvg
from b_hfl.typing.common_types import (
    ClientFN,
    DataloaderGenerator,
    DatasetLoader,
    DatasetLoaderNoTransforms,
    EvaluateFunc,
    LoadConfig,
    NetGenerator,
    NodeOpt,
    ParametersLoader,
    RecursiveBuilder,
    StateLoader,
    TestFunc,
    TrainFunc,
    TransformType,
)
from b_hfl.utils.dataset_preparation import (
    ConfigFolderHierarchy,
    config_map_to_file_hierarchy,
)
from b_hfl.utils.task_utils import optimizer_generator_decorator
from b_hfl.utils.utils import (
    decorate_client_fn_with_recursive_builder,
    decorate_dataset_with_transforms,
    get_metrics_agg_fn,
    get_save_files_every_round,
    seed_everything,
)


# pylint: disable=too-many-arguments
def get_fed_eval_fn(
    root_path: Path,
    client_fn: ClientFN,
    recursive_builder: RecursiveBuilder,
    on_evaluate_config_function: LoadConfig,
) -> EvaluateFunc:
    """Get the federated evaluation function."""
    client = client_fn(
        str(root_path),
        root_path,
        root_path.parent,
        None,
        recursive_builder,
    )

    # pylint: disable=unused-argument
    def fed_eval_fn(
        server_round: int, parameters: NDArrays, config: Dict
    ) -> Optional[Tuple[float, Dict]]:
        real_config = on_evaluate_config_function(server_round, root_path)
        results = client.evaluate(parameters, real_config)
        loss, _, metrics = results
        return loss, metrics

    return fed_eval_fn


def unroll_hierarchy(
    x: FolderHierarchy, recursively_train_all: bool
) -> List[FolderHierarchy]:
    """Unroll the hierarchy of the file system.

    For executing clients sequentially.
    """
    ret = [x]
    if recursively_train_all:
        for child in x.children:
            ret.extend(
                unroll_hierarchy(x=child, recursively_train_all=recursively_train_all)
            )
    return ret


# pylint: disable=too-many-locals
def build_hydra_client_fn_and_recursive_builder_generator(
    cfg: DictConfig,
    path_dict: FolderHierarchy,
    executor: concurrent.futures.Executor,
) -> Tuple[
    ClientFN,
    Callable[[str, str], Callable[[FolderHierarchy], RecursiveBuilder]],
    LoadConfig,
    LoadConfig,
    ParametersLoader,
    NetGenerator,
]:
    """Build the client function and recursive builder generator.

    They encapsulate all the client logic and state logic this allows evaluation and
    training outside of a simulation without logic duplication.
    """
    config_mapping: ConfigFolderHierarchy = call(
        config=cfg.task.client.get_config_mapping,
        path_dict=path_dict,
        cfg=cfg,
        _recursive_=False,
    )

    # Turn the logical mapping into a file system mapping
    config_map_to_file_hierarchy(
        logical_mapping=config_mapping,
    )

    net_generator: NetGenerator = call(cfg.task.client.get_net_generator)

    train: TrainFunc = optimizer_generator_decorator(
        optimizer_generator=call(cfg.task.client.get_optimizer_generator)
    )(call(config=cfg.task.client.get_train))
    test: TestFunc = call(config=cfg.task.client.get_test)

    anc_node_opt: NodeOpt = call(config=cfg.task.client.get_anc_node_opt)
    desc_node_opt: NodeOpt = call(config=cfg.task.client.get_desc_node_opt)

    create_dataloader: DataloaderGenerator = call(cfg.task.data.get_create_dataloader)

    client_resources = {
        "num_cpus": cfg.fed.cpus_per_client,
        "num_gpus": cfg.fed.gpus_per_client,
    }

    client_fn: ClientFN = get_client_fn(
        net_generator=net_generator,
        anc_node_opt=anc_node_opt,
        desc_node_opt=desc_node_opt,
        train=train,
        test=test,
        create_dataloader=create_dataloader,
        fit_metrics_aggregation_fn=get_metrics_agg_fn(),
        evaluate_metrics_aggregation_fn=get_metrics_agg_fn(),
        resources=client_resources,
        timeout=cfg.fed.timeout,
    )

    load_dataset_file_no_transforms: DatasetLoaderNoTransforms = call(
        cfg.task.data.get_load_dataset_file
    )
    transform: TransformType = call(cfg.task.data.get_transform)
    target_transform: TransformType = call(cfg.task.data.get_target_transform)

    load_dataset_file: DatasetLoader = decorate_dataset_with_transforms(
        transform, target_transform
    )(load_dataset_file_no_transforms)

    load_parameters_file: ParametersLoader = call(cfg.state.get_load_parameters_file)

    load_state_file: StateLoader = call(cfg.state.get_load_state_file)

    on_fit_config_function: LoadConfig = call(cfg.fed.get_on_fit_config_fn)

    on_evaluate_config_function: LoadConfig = call(cfg.fed.get_on_evaluate_config_fn)

    def get_client_recursive_builder_for_parameter_type(
        parameters_file_name: str,
        experiment_type: str,
    ) -> Callable[[FolderHierarchy], RecursiveBuilder]:
        """Get the recursive builder for a given parameter type."""

        def get_client_recursive_builder(x: FolderHierarchy) -> RecursiveBuilder:
            return recursive_bulder_generator(
                root=x.path,
                path_dict=x,
                load_dataset_file=load_dataset_file,
                dataset_manager=call(cfg.state.get_dataset_manager),
                load_params_file=load_parameters_file,
                parameter_manager=call(
                    cfg.state.get_parameter_manager,
                    save_parameters_to_file=call(cfg.state.get_save_parameters_to_file),
                ),
                load_state_file=load_state_file,
                state_manager=call(
                    cfg.state.get_state_manager,
                    save_state_to_file=call(cfg.state.get_save_state_to_file),
                ),
                residuals_manager={},
                on_fit_config_fn=on_fit_config_function,
                on_evaluate_config_fn=on_evaluate_config_function,
                experiment_type=experiment_type,
                parameters_file_name=parameters_file_name,
                parameters_ext=cfg.state.parameters_extension,
                executor=executor,
            )

        return get_client_recursive_builder

    return (
        client_fn,
        get_client_recursive_builder_for_parameter_type,
        on_fit_config_function,
        on_evaluate_config_function,
        load_parameters_file,
        net_generator,
    )


# pylint: disable=too-many-arguments
def get_run_fed_simulation(
    cfg: DictConfig,
    client_fn: ClientFN,
    get_client_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_evaluate_config_function: LoadConfig,
    on_fit_config_function: LoadConfig,
    client_output_directory: Path,
    initial_parameters: Parameters,
    executor: concurrent.futures.Executor,
) -> Callable[[FolderHierarchy], History]:
    """Get the function to run a federated simulation."""

    def run_fed_simulation(path_dict: FolderHierarchy) -> History:
        seed_everything(cfg.fed.seed)

        root_recursive_builder: RecursiveBuilder = get_client_recursive_builder(
            path_dict,
        )

        wrapped_client_fn: Callable[
            [str], ConfigurableRecClient
        ] = decorate_client_fn_with_recursive_builder(
            get_client_recursive_builder=get_client_recursive_builder,
            path_dict=path_dict,
        )(
            client_fn
        )

        evaluate_fn: EvaluateFunc = get_fed_eval_fn(
            root_path=path_dict.path,
            client_fn=client_fn,
            recursive_builder=root_recursive_builder,
            on_evaluate_config_function=on_evaluate_config_function,
        )

        parameters_path: Path = path_dict.path / f"parameters{cfg.fed_type}.npz"

        strategy: LoggingFedAvg = instantiate(
            cfg.strategy.init,
            parameters_path=parameters_path,
            save_parameters_to_file=call(cfg.state.get_save_parameters_to_file),
            save_files=get_save_files_every_round(
                path_dict=path_dict,
                output_dir=client_output_directory,
                to_save=cfg.state.save_per_round,
                save_frequency=cfg.state.save_frequency,
            ),
            fraction_fit=(
                float(cfg.fed.num_clients_per_round) / cfg.fed.num_total_clients
            ),
            fraction_evaluate=(
                float(cfg.fed.num_evaluate_clients_per_round)
                / cfg.fed.num_total_clients
            ),
            min_fit_clients=cfg.fed.num_clients_per_round,
            min_evaluate_clients=cfg.fed.num_evaluate_clients_per_round,
            min_available_clients=cfg.fed.num_total_clients,
            on_fit_config_fn=on_fit_config_function,
            on_evaluate_config_fn=on_evaluate_config_function,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            accept_failures=False,
            fit_metrics_aggregation_fn=call(cfg.fed.get_on_fit_metrics_agg_fn),
            evaluate_metrics_aggregation_fn=call(
                cfg.fed.get_on_evaluate_metrics_agg_fn
            ),
        )
        # Solve from here down
        server: Server = Server(
            strategy=strategy,
            client_manager=DeterministicClientManager(cfg.fed.seed),
            use_wandb=cfg.use_wandb,
            eval_only=cfg.eval_only,
            executor=executor,
        )

        return start_simulation(
            client_fn=cast(Callable[[str], fl.client.Client], wrapped_client_fn),
            num_clients=cfg.fed.num_total_clients,
            client_resources={
                "num_cpus": cfg.fed.cpus_per_client,
                "num_gpus": cfg.fed.gpus_per_client,
            },
            server=server,
            config=ServerConfig(num_rounds=cfg.fed.num_rounds),
            strategy=strategy,
            ray_init_args={"include_dashboard": False},
        )

    return run_fed_simulation


# pylint: disable=too-many-arguments
def run_fed_simulations_recursive(
    path_dict: FolderHierarchy,
    cfg: DictConfig,
    client_fn: ClientFN,
    get_client_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_evaluate_config_function: LoadConfig,
    on_fit_config_function: LoadConfig,
    client_output_directory: Path,
    initial_parameters: Parameters,
    recursively_fed_all: bool,
    executor: concurrent.futures.Executor,
) -> List[Tuple[Path, FolderHierarchy, History]]:
    """Run federated simulations recursively.

    Starting with each client in-turn as the root of the simulation.
    """
    run_fed_simulation_fn: Callable[
        [FolderHierarchy], History
    ] = get_run_fed_simulation(
        cfg=cfg,
        client_fn=client_fn,
        get_client_recursive_builder=get_client_recursive_builder,
        on_evaluate_config_function=on_evaluate_config_function,
        on_fit_config_function=on_fit_config_function,
        client_output_directory=client_output_directory,
        initial_parameters=initial_parameters,
        executor=executor,
    )

    histories: List[Tuple[Path, FolderHierarchy, History]] = []

    for client_dict in unroll_hierarchy(path_dict, recursively_fed_all):
        histories.append(
            (client_dict.path, client_dict, run_fed_simulation_fn(client_dict))
        )

    return histories


# pylint: disable=too-many-arguments
def get_train_and_eval_optimal(
    client_fn: ClientFN,
    root: Path,
    get_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_fit_config_function: LoadConfig,
    on_evaluate_config_function: LoadConfig,
    initial_parameters: NDArrays,
    seed: int,
) -> Callable[[FolderHierarchy], History]:
    """Get function to train a client on the chain dataset."""

    def train_client(path_dict: FolderHierarchy) -> NDArrays:
        client = client_fn(
            str(path_dict.path),
            path_dict.path,
            root,
            None,
            get_recursive_builder(path_dict),
        )
        config = on_fit_config_function(0, path_dict.path)
        config["client_config"]["train_chain"] = True
        config["client_config"]["train_proxy"] = False
        config["client_config"]["train_children"] = False
        model, *_ = client.fit(initial_parameters, config)
        return model

    def train_and_eval_client(path_dict: FolderHierarchy):
        hist: History = History(use_wandb=False)
        seed_everything(seed)
        trained_parameters = train_client(path_dict)
        seed_everything(seed)
        eval_fn: EvaluateFunc = get_fed_eval_fn(
            root_path=path_dict.path,
            client_fn=client_fn,
            recursive_builder=get_recursive_builder(path_dict),
            on_evaluate_config_function=on_evaluate_config_function,
        )
        res = eval_fn(0, trained_parameters, {})
        if res is None:
            raise ValueError("Evaluation failed")

        loss, metrics = res
        hist.add_loss_centralized(0, loss)
        hist.add_metrics_centralized(0, metrics)
        return hist

    return train_and_eval_client


# pylint: disable=too-many-arguments
def train_and_evaluate_optimal_models_from_hierarchy(
    path_dict: FolderHierarchy,
    client_fn: ClientFN,
    get_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_fit_config_function: LoadConfig,
    on_evaluate_config_function: LoadConfig,
    initial_parameters: NDArrays,
    seed: int,
    recursively_train_all: bool,
) -> List[Tuple[Path, FolderHierarchy, History]]:
    """Run centralised training with each node in the tree."""
    train_client_fn = get_train_and_eval_optimal(
        client_fn,
        path_dict.path,
        get_recursive_builder,
        on_fit_config_function,
        on_evaluate_config_function=on_evaluate_config_function,
        initial_parameters=initial_parameters,
        seed=seed,
    )
    histories: List[Tuple[Path, FolderHierarchy, History]] = []
    for client_dict in unroll_hierarchy(path_dict, recursively_train_all):
        print("Training optimal model for client: ", client_dict.path)
        histories.append((client_dict.path, client_dict, train_client_fn(client_dict)))

    return histories
