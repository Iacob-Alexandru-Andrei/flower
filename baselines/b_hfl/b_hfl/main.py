"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import hydra
import wandb
import yaml
import os

from common_types import (
    ClientFN,
    DataloaderGenerator,
    LoadConfig,
    NetGenerator,
    NodeOpt,
    RecursiveBuilder,
    TestFunc,
    TrainFunc,
    DatasetLoader,
    DatasetLoaderNoTransforms,
    ParametersLoader,
    TransformType,
    FileHierarchy,
    RecursiveBuilderWrapper,
)
from client_manager import DeterministicClientManager
from flwr.common import NDArrays
from flwr.server.strategy import Strategy
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from server import History, Server
from task_utils import optimizer_generator_decorator
from utils import (
    decorate_client_fn_with_recursive_builder,
    decorate_dataset_with_transforms,
    seed_everything,
)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ):
        seed_everything(cfg.fed.seed)

        output_directory = Path(
            hydra.utils.to_absolute_path(HydraConfig.get().runtime.output_dir)
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

        load_parameters_file: ParametersLoader = call(
            cfg.state.get_load_parameters_file
        )

        path_dict: FileHierarchy = call(cfg.data.get_file_hierarchy)

        on_fit_config_function: LoadConfig = call(cfg.fed.get_on_fit_config_fn)

        on_evaluate_config_function: LoadConfig = call(
            cfg.fed.get_on_evaluate_config_fn
        )
        recursive_builder_wrapper: RecursiveBuilderWrapper = call(
            cfg.state.get_recursive_builder
        )

        def get_client_recursive_builder(x: FileHierarchy) -> RecursiveBuilder:
            return recursive_builder_wrapper(
                path_dict=x,  # type: ignore
                load_dataset_file=load_dataset_file,
                dataset_manager=call(cfg.state.get_dataset_manager),
                load_parameters_file=load_parameters_file,
                parameter_manager=call(
                    cfg.state.get_parameter_manager,
                    save_parameters_to_file=call(cfg.state.get_save_parameters_to_file),
                ),
                on_fit_config_fn=on_fit_config_function,
                on_evaluate_config_fn=on_evaluate_config_function,
            )

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
            [ClientFN, FileHierarchy, RecursiveBuilder],
            Callable[[int, NDArrays, Dict], Optional[Tuple[float, Dict]]],
        ] = call(
            cfg.fed.get_fed_eval_fn,
            root_path=path_dict["path"],
            client_fn=client_fn,
            recursive_builder=root_recursive_builder,
            on_evaluate_config_function=on_evaluate_config_function,
        )

        strategy: Strategy = instantiate(
            cfg.strategy.init,
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
            initial_parameters=call(cfg.fed.get_initial_parameters)(
                path_dict=path_dict,
                load_parameters_file=load_parameters_file,
                net_generator=net_generator,
                on_fit_config_fn=on_fit_config_function,
            ),
            accept_failures=False,
            fit_metrics_aggregation_fn=call(cfg.fed.get_on_fit_metrics_agg_fn),
            evaluate_metrics_aggregation_fn=call(
                cfg.fed.get_on_evaluate_metrics_agg_fn
            ),
        )
        # Solve from here down
        server: Server = Server(
            strategy=strategy, client_manager=DeterministicClientManager(cfg.fed.seed)
        )

        hist: History = fl.simulation.start_simulation(
            client_fn=wrapped_client_fn,
            num_clients=cfg.fed.num_total_clients,
            client_resources={
                "num_cpus": cfg.fed.cpus_per_client,
                "num_gpus": cfg.fed.gpus_per_client,
            },
            server=server,
            strategy=strategy,
            ray_init_args={"include_dashboard": False},
        )

        call(cfg.plot_results, hist=hist, output_directory=output_directory)
        print(hist.__dict__)
        with open(output_directory / "hist.json", "w", encoding="utf-8") as f:
            json.dump(hist.__dict__, f, ensure_ascii=False, indent=4)
        wandb.save(str((output_directory / "hist.json").resolve()))

        # 2. Prepare your dataset
        # here you should call a function in datasets.py that returns
        # whatever is needed to:
        # (1) ensure the server can access the dataset used to evaluate your model after
        # aggregation
        # (2) tell each client what dataset partitions they should use
        # (e.g. a this could be a location in the file system,
        # a list of dataloader, a list of ids to extract
        # from a dataset, it's up to you)

        # 3. Define your clients
        # Define a function that returns another function
        # that will be used during simulation to
        # instantiate each individual client
        # client_fn = client.<my_function_that_returns_a_function>()

        # 4. Define your strategy
        # pass all relevant argument
        # (including the global dataset used after aggregation,
        # if needed by your method.)
        # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

        # 5. Start Simulation
        # history = fl.simulation.start_simulation(<arguments for simulation>)

        # 6. Save your results
        # Here you can save the `history` returned by the simulation and include
        # also other buffers, statistics, info needed to be saved in order to later
        # on generate the plots you provide in the README.md. You can for instance
        # access elements that belong to the strategy for example:
        # data = strategy.get_my_custom_data() -- assuming you have such method defined.
        # Hydra will generate for you a directory each time you run the code. You
        # can retrieve the path to that directory with this:
        # save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
