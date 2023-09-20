"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import concurrent.futures
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import hydra
import yaml
from flwr.common import Parameters, parameters_to_ndarrays
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

import wandb
from b_hfl.common_types import RecursiveBuilder
from b_hfl.modified_flower.server import History
from b_hfl.run_simulations import (
    build_hydra_client_fn_and_recursive_builder_generator,
    run_fed_simulations_recursive,
    train_and_evaluate_optimal_models_from_hierarchy,
)
from b_hfl.schemas.file_system_schema import FolderHierarchy
from b_hfl.utils import FileSystemManager, process_histories, wandb_init


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    datetime.now()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["OC_CAUSE"] = "1"

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as run:
        path_dict: FolderHierarchy = call(cfg.task.data.get_folder_hierarchy)

        output_directory = Path(
            hydra.utils.to_absolute_path(HydraConfig.get().runtime.output_dir)
        )

        if cfg.reuse_output_dir is not None:
            output_directory = Path(cfg.reuse_output_dir)

        with open(
            output_directory / f"config_root_{cfg.task.data.client_folder}.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(wandb_config, f, allow_unicode=True)

        client_output_directory = output_directory / "client_data"
        client_output_directory.mkdir(parents=True, exist_ok=True)

        plots_output_directory = output_directory / "plots"
        plots_output_directory.mkdir(parents=True, exist_ok=True)

        histories_output_directory = output_directory / "histories"
        histories_output_directory.mkdir(parents=True, exist_ok=True)

        with FileSystemManager(
            path_dict=path_dict,
            output_dir=client_output_directory,
            to_clean=cfg.state.to_clean,
            to_save_once=cfg.state.to_save_once,
        ) as _, concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.fed.max_workers
        ) as executor:
            # Contains the root node used to run the experiments
            # Its FolderHierarchy and History results
            optimal_model_histories: List[Tuple[Path, FolderHierarchy, History]] = []
            federated_models_histories: List[Tuple[Path, FolderHierarchy, History]] = []

            (
                client_fn,
                get_client_recursive_builder_for_parameter_type,
                on_fit_config_function,
                on_evaluate_config_function,
                train_config_schema,
                test_config_schema,
                load_parameters_file,
                net_generator,
            ) = build_hydra_client_fn_and_recursive_builder_generator(
                cfg, path_dict, executor
            )

            initial_parameters: Parameters = call(cfg.fed.get_initial_parameters)(
                path_dict=path_dict,
                load_params_file=load_parameters_file,
                net_generator=net_generator,
                on_fit_config_fn=on_fit_config_function,
            )

            def plot_history(x, output_dir, name):
                return call(
                    cfg.fed.plot_results,
                    hist=x,
                    output_directory=output_dir,
                    name=name,
                )

            # Train optimal models at every level
            if cfg.train_optimal:
                get_centralised_client_recursive_builder: Callable[
                    [FolderHierarchy], RecursiveBuilder
                ] = get_client_recursive_builder_for_parameter_type(
                    f"parameters{cfg.optimal_type}"
                )
                optimal_model_histories.extend(
                    train_and_evaluate_optimal_models_from_hierarchy(
                        path_dict=path_dict,
                        client_fn=client_fn,
                        get_recursive_builder=get_centralised_client_recursive_builder,
                        on_fit_config_function=on_fit_config_function,
                        on_evaluate_config_function=on_evaluate_config_function,
                        train_config_schema=train_config_schema,
                        test_config_schema=test_config_schema,
                        initial_parameters=parameters_to_ndarrays(initial_parameters),
                        seed=cfg.fed.seed,
                        recursively_train_all=cfg.recursively_train_all_optimal_models,
                    )
                )

                process_histories(
                    plotting_fn=plot_history,
                    histories=optimal_model_histories,
                    output_directory=histories_output_directory,
                    history_type=cfg.optimal_type,
                )
            # Train federated models
            if cfg.train_fed:
                get_client_recursive_builder: Callable[
                    [FolderHierarchy], RecursiveBuilder
                ] = get_client_recursive_builder_for_parameter_type(
                    f"parameters{cfg.fed_type}"
                )

                federated_models_histories.extend(
                    run_fed_simulations_recursive(
                        path_dict=path_dict,
                        cfg=cfg,
                        client_fn=client_fn,
                        get_client_recursive_builder=get_client_recursive_builder,
                        on_fit_config_function=on_fit_config_function,
                        on_evaluate_config_function=on_evaluate_config_function,
                        client_output_directory=client_output_directory,
                        initial_parameters=initial_parameters,
                        recursively_fed_all=cfg.recursively_fed_simulate_all,
                        executor=executor,
                    )
                )

            process_histories(
                plotting_fn=plot_history,
                histories=federated_models_histories,
                output_directory=histories_output_directory,
                history_type=cfg.fed_type,
            )
        if run is not None:
            run.save(
                str((output_directory / "*").resolve()),
                str((output_directory).resolve()),
                "now",
            )

        print(
            subprocess.run(
                ["wandb", "sync", "--clean-old-hours", "24"],
                capture_output=True,
                text=True,
                check=True,
            )
        )

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
