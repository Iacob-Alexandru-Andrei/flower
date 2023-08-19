from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from omegaconf import DictConfig
import flwr as fl
from flwr.common import NDArrays
from strategy import LoggingFedAvg
from common_types import (
    ClientFN,
    ConfigFolderHierarchy,
    DataloaderGenerator,
    DatasetLoader,
    DatasetLoaderNoTransforms,
    FolderHierarchy,
    NetGenerator,
    ParametersLoader,
    RecursiveBuilder,
    TrainFunc,
    TestFunc,
    NodeOpt,
    TransformType,
    ParametersLoader,
    LoadConfig,
    RecursiveBuilderWrapper,
)
from hydra.utils import call, instantiate
from task_utils import optimizer_generator_decorator
from utils import (
    decorate_client_fn_with_recursive_builder,
    decorate_dataset_with_transforms,
    get_save_files_every_round,
    seed_everything,
)
from server import Server, History
from client_manager import DeterministicClientManager
from flwr.server import ServerConfig
from flwr.common import Parameters


def get_fed_eval_fn(
    root_path: Path,
    client_fn: ClientFN,
    recursive_builder: RecursiveBuilder,
    on_evaluate_config_function: LoadConfig,
) -> Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]]:
    """Get the federated evaluation function."""
    client = client_fn(str(root_path), root_path, None, recursive_builder)

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
    ret = [x]
    if recursively_train_all:
        return ret
    for child in x["children"]:
        ret.extend(unroll_hierarchy(child, recursively_train_all))
    return ret


def build_hydra_client_fn_and_recursive_builder_generator(
    cfg: DictConfig,
    path_dict: FolderHierarchy,
) -> Tuple[
    ClientFN,
    Callable[[str], Callable[[FolderHierarchy], RecursiveBuilder]],
    LoadConfig,
    LoadConfig,
    ParametersLoader,
    NetGenerator,
]:
    get_config_mapping: Callable[[FolderHierarchy], ConfigFolderHierarchy] = call(
        cfg.data.get_config_mapping
    )

    config_mapping: ConfigFolderHierarchy = get_config_mapping(path_dict)

    call(
        cfg.data.create_configs,
        logical_mapping=config_mapping,
        train_schema=call(cfg.data.get_train_config_schema),  # type: ignore
        test_schema=call(cfg.data.get_test_config_schema),
    )

    net_generator: NetGenerator = call(cfg.client.get_net_generator)

    train: TrainFunc = optimizer_generator_decorator(
        call(cfg.client.get_optimizer_generator)
    )(call(cfg.client.get_train))
    test: TestFunc = call(cfg.client.get_test)

    node_opt: NodeOpt = call(cfg.client.get_node_opt)

    create_dataloader: DataloaderGenerator = call(cfg.data.get_create_dataloader)

    client_fn: ClientFN = call(
        cfg.client.get_client_fn,
        net_generator=net_generator,
        node_opt=node_opt,
        train=train,
        test=test,
        create_dataloader=create_dataloader,
    )

    load_dataset_file_no_transforms: DatasetLoaderNoTransforms = call(
        cfg.data.get_load_dataset_file
    )
    transform: TransformType = call(cfg.data.get_transform)
    target_transform: TransformType = call(cfg.data.get_target_transform)

    load_dataset_file: DatasetLoader = decorate_dataset_with_transforms(
        transform, target_transform
    )(load_dataset_file_no_transforms)

    load_parameters_file: ParametersLoader = call(cfg.state.get_load_parameters_file)

    on_fit_config_function: LoadConfig = call(cfg.fed.get_on_fit_config_fn)

    on_evaluate_config_function: LoadConfig = call(cfg.fed.get_on_evaluate_config_fn)
    recursive_builder_wrapper: RecursiveBuilderWrapper = call(
        cfg.state.get_recursive_builder
    )

    def get_client_recursive_builder_for_parameter_type(
        parameters_file_name: str,
    ) -> Callable[[FolderHierarchy], RecursiveBuilder]:
        def get_client_recursive_builder(x: FolderHierarchy) -> RecursiveBuilder:
            return recursive_builder_wrapper(
                root=x["path"],  # type: ignore
                path_dict=x,
                load_dataset_file=load_dataset_file,
                dataset_manager=call(cfg.state.get_dataset_manager),
                load_parameters_file=load_parameters_file,
                parameter_manager=call(
                    cfg.state.get_parameter_manager,
                    save_parameters_to_file=call(cfg.state.get_save_parameters_to_file),
                ),
                on_fit_config_fn=on_fit_config_function,
                on_evaluate_config_fn=on_evaluate_config_function,
                parameters_file_name=parameters_file_name,
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


def get_run_fed_simulation(
    cfg: DictConfig,
    client_fn: ClientFN,
    get_client_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_evaluate_config_function: LoadConfig,
    on_fit_config_function: LoadConfig,
    client_output_directory: Path,
    initial_parameters: Parameters,
) -> Callable[[FolderHierarchy], History]:
    def run_fed_simulation(path_dict: FolderHierarchy) -> History:
        seed_everything(cfg.fed.seed)

        root_recursive_builder: RecursiveBuilder = get_client_recursive_builder(
            path_dict
        )

        wrapped_client_fn: Callable[
            [str], fl.client.Client
        ] = decorate_client_fn_with_recursive_builder(
            get_client_recursive_builder, path_dict
        )(
            client_fn
        )

        evaluate_fn: Callable[
            [int, NDArrays, Dict], Optional[Tuple[float, Dict]]
        ] = call(
            cfg.fed.get_fed_eval_fn,
            root_path=path_dict["path"],
            client_fn=client_fn,
            recursive_builder=root_recursive_builder,
            on_evaluate_config_function=on_evaluate_config_function,
        )

        strategy: LoggingFedAvg = instantiate(
            cfg.strategy.init,
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
        )

        return fl.simulation.start_simulation(
            client_fn=wrapped_client_fn,
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
) -> List[Tuple[Path, FolderHierarchy, History]]:
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
    )

    histories: List[Tuple[Path, FolderHierarchy, History]] = []

    for client_dict in unroll_hierarchy(path_dict, recursively_fed_all):
        histories.append(
            (client_dict["path"], client_dict, run_fed_simulation_fn(client_dict))
        )

    return histories


def get_train_and_eval_optimal(
    client_fn: ClientFN,
    get_recursive_builder: Callable[[FolderHierarchy], RecursiveBuilder],
    on_fit_config_function: LoadConfig,
    on_evaluate_config_function: LoadConfig,
    initial_parameters: NDArrays,
    seed: int,
) -> Callable[[FolderHierarchy], History]:
    """Get function to train a client on the chain dataset."""

    def train_client(path_dict: FolderHierarchy) -> NDArrays:
        client = client_fn(
            str(path_dict["path"]),
            path_dict["path"],
            None,
            get_recursive_builder(path_dict),
        )
        config = on_fit_config_function(0, path_dict["path"])
        config["client_config"]["train_chain"] = True
        config["client_config"]["train_proxy"] = False
        config["client_config"]["train_children"] = False
        model, *_ = client.fit(initial_parameters, config)
        return model

    def train_and_eval_client(path_dict: FolderHierarchy):
        hist: History = History()
        seed_everything(seed)
        trained_parameters = train_client(path_dict)
        seed_everything(seed)
        eval_fn = get_fed_eval_fn(
            root_path=path_dict["path"],
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
    train_client_fn = get_train_and_eval_optimal(
        client_fn,
        get_recursive_builder,
        on_fit_config_function,
        on_evaluate_config_function=on_evaluate_config_function,
        initial_parameters=initial_parameters,
        seed=seed,
    )
    histories: List[Tuple[Path, FolderHierarchy, History]] = []

    for client_dict in unroll_hierarchy(path_dict, recursively_train_all):
        histories.append(
            (client_dict["path"], client_dict, train_client_fn(client_dict))
        )

    return histories
