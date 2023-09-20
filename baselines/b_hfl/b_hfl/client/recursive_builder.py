"""Client recursive builder for a given file hierarchy."""

import concurrent.futures
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, Optional

from flwr.common import NDArrays
from torch.utils.data import Dataset

from b_hfl.common_types import (
    ClientFN,
    ClientResGeneratorList,
    ConfigSchemaGenerator,
    DatasetLoader,
    EvalRes,
    FitRes,
    LoadConfig,
    ParametersLoader,
    RecursiveBuilder,
    RecursiveStructure,
    State,
    StateLoader,
)
from b_hfl.schemas.client_schema import (
    ConfigurableRecClient,
    RecClientRuntimeTestConf,
    RecClientRuntimeTrainConf,
)
from b_hfl.schemas.file_system_schema import FolderHierarchy
from b_hfl.state_management import (
    DatasetManager,
    ParameterManager,
    StateManager,
    load_parameters,
    load_state,
    process_file,
)
from b_hfl.utils import extract_file_from_files


# pylint: disable=too-many-arguments
def get_recursive_builder(
    root: Path,
    path_dict: FolderHierarchy,
    load_dataset_file: DatasetLoader,
    dataset_manager: DatasetManager,
    load_params_file: ParametersLoader,
    parameter_manager: ParameterManager,
    load_state_file: StateLoader,
    state_manager: StateManager,
    residuals_manager: Dict[Any, Dict[Any, FitRes]],
    on_fit_config_fn: LoadConfig,
    on_evaluate_config_fn: LoadConfig,
    parameters_file_name: str,
    parameters_ext: str,
    executor: concurrent.futures.Executor,
    train_config_schema: ConfigSchemaGenerator,
    test_config_schema: ConfigSchemaGenerator,
) -> RecursiveBuilder:
    """Generate a recursive builder for a given file hierarchy.

    This is recursively caled within the inner function for each child of a given node.
    """
    cur_round: int = 0

    def recursive_builder(
        current_client: ConfigurableRecClient,
        client_fn: ClientFN,
        test: bool = False,
    ) -> RecursiveStructure:
        """Generate a recursive structure for a given client."""
        file_type: str = "test" if test else "train"

        dataset_file: Optional[Path] = extract_file_from_files(
            path_dict.path, file_type
        )

        chain_dataset_file: Optional[Path] = extract_file_from_files(
            path_dict.path, f"{file_type}_chain"
        )

        parameter_file: Optional[Path] = extract_file_from_files(
            path_dict.path, parameters_file_name
        )

        state_file: Optional[Path] = extract_file_from_files(
            path_dict.path, f"{parameters_file_name}_state"
        )

        def get_fit_child_generator(
            child_path_dict: FolderHierarchy,
            cid: str,
        ) -> Callable[
            [NDArrays, Dict, RecClientRuntimeTrainConf],
            concurrent.futures.Future[FitRes],
        ]:
            def fit_client(
                params: NDArrays, in_conf: Dict, parent_conf: RecClientRuntimeTrainConf
            ) -> FitRes:
                train_config_schema(**in_conf)
                config = RecClientRuntimeTrainConf(**in_conf)
                # Syncrhonise client
                config.client_config.parent_round = (
                    parent_conf.client_config.parent_round
                )

                return client_fn(
                    cid,
                    child_path_dict.path,
                    root,
                    current_client,
                    train_config_schema,
                    test_config_schema,
                    get_recursive_builder(
                        root=root,
                        path_dict=child_path_dict,
                        load_dataset_file=load_dataset_file,
                        dataset_manager=dataset_manager,
                        load_params_file=load_params_file,
                        parameter_manager=parameter_manager,
                        load_state_file=load_state_file,
                        state_manager=state_manager,
                        residuals_manager=residuals_manager,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        parameters_file_name=parameters_file_name,
                        parameters_ext=parameters_ext,
                        executor=executor,
                        train_config_schema=train_config_schema,
                        test_config_schema=test_config_schema,
                    ),
                ).fit(params, config)

            return lambda params, conf, parent_conf: executor.submit(
                fit_client, params, conf, parent_conf
            )

        def get_evaluate_child_generator(
            child_path_dict: FolderHierarchy,
            cid: str,
        ) -> Callable[
            [NDArrays, Dict, RecClientRuntimeTestConf],
            concurrent.futures.Future[EvalRes],
        ]:
            def evaluate_client(
                params: NDArrays, in_conf: Dict, parent_conf: RecClientRuntimeTestConf
            ) -> EvalRes:
                test_config_schema(**in_conf)
                config = RecClientRuntimeTestConf(**in_conf)
                config.client_config.parent_round = (
                    parent_conf.client_config.parent_round
                )

                return client_fn(
                    cid,
                    child_path_dict.path,
                    root,
                    current_client,
                    train_config_schema,
                    test_config_schema,
                    get_recursive_builder(
                        root=root,
                        path_dict=child_path_dict,
                        load_dataset_file=load_dataset_file,
                        dataset_manager=dataset_manager,
                        load_params_file=load_params_file,
                        parameter_manager=parameter_manager,
                        load_state_file=load_state_file,
                        state_manager=state_manager,
                        residuals_manager=residuals_manager,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        parameters_file_name=parameters_file_name,
                        parameters_ext=parameters_ext,
                        executor=executor,
                        train_config_schema=train_config_schema,
                        test_config_schema=test_config_schema,
                    ),
                ).evaluate(params, config)

            return lambda params, conf, parent_conf: executor.submit(
                evaluate_client, params, conf, parent_conf
            )

        config_fn = on_fit_config_fn if not test else on_evaluate_config_fn
        child_generator: ClientResGeneratorList = []
        if not test:
            child_generator = [
                (
                    get_fit_child_generator(child_path_dict, str(i)),
                    config_fn(cur_round, child_path_dict.path),
                )
                for i, child_path_dict in enumerate(path_dict.children)
            ]
        else:
            child_generator = [
                (
                    get_evaluate_child_generator(child_path_dict, str(i)),
                    config_fn(cur_round, child_path_dict.path),
                )
                for i, child_path_dict in enumerate(path_dict.children)
            ]

        # TODO Finish state management implementation here
        def recursive_step(fit_rest: FitRes, final: bool) -> None:
            nonlocal cur_round
            cur_round += 1

            if final:
                parameters_save_name = (
                    path_dict.path / f"{parameters_file_name}{parameters_ext}"
                    if parameter_file is None
                    else parameter_file
                )
                parameter_manager.set_parameters(parameters_save_name, fit_rest[0])

                if chain_dataset_file is not None:
                    dataset_manager.unload_chain_dataset(chain_dataset_file)

                children_dataset_files: Generator[Optional[Path], None, None] = (
                    file
                    for file in (
                        extract_file_from_files(child_dict.path, file_type)
                        for child_dict in path_dict.children
                    )
                    if file is not None
                )

                children_chain_dataset_files: Generator[Path, None, None] = (
                    file
                    for file in (
                        extract_file_from_files(child_dict.path, f"{file_type}_chain")
                        for child_dict in path_dict.children
                    )
                    if file is not None
                )

                dataset_manager.unload_children_datasets(
                    chain(children_chain_dataset_files, children_dataset_files)
                )

                cur_round = 0

                # The root client will die
                if root == path_dict.path:
                    parameter_manager.cleanup()

        def get_dataset_generator(
            dataset_file: Optional[Path],
        ) -> Optional[Callable[[Dict], Dataset]]:
            if dataset_file is not None:

                def dataset_generator(_config) -> Dataset:
                    return process_file(
                        dataset_file, load_dataset_file, dataset_manager
                    )

                return dataset_generator

            return None

        def get_parameter_generator(
            parameter_file: Optional[Path],
        ) -> Optional[Callable[[Dict], NDArrays]]:
            if parameter_file is not None:

                def parameter_generator(_config: Dict) -> NDArrays:
                    return load_parameters(
                        parameter_file, load_params_file, parameter_manager
                    )

                return parameter_generator

            return None

        def get_state_generator(
            state_file: Optional[Path],
        ) -> Optional[Callable[[Dict], State]]:
            if state_file is not None:

                def state_generator(_config: Dict) -> State:
                    return load_state(state_file, load_state_file, state_manager)

                return state_generator

            return None

        def get_residuals(leaf_to_root: bool) -> Iterable[FitRes]:
            id = path_dict.path / "leaf_to_root" / f"{leaf_to_root}"
            if id not in residuals_manager:
                residuals_manager[id] = {}
            return residuals_manager[id].values()

        def send_residuals(send_to: Path, residual: FitRes, leaf_to_root: bool) -> None:
            send_to_id = send_to / "leaf_to_root" / f"{leaf_to_root}"
            send_from_id = path_dict.path
            if send_to_id not in residuals_manager:
                residuals_manager[send_to_id] = {}
            residuals_manager[send_to_id][send_from_id] = residual

        if not test:
            return (
                get_parameter_generator(parameter_file),
                get_state_generator(state_file),
                get_dataset_generator(dataset_file),
                get_dataset_generator(chain_dataset_file),
                child_generator,
                get_residuals,
                send_residuals,
                recursive_step,
            )
        else:
            return (
                get_parameter_generator(parameter_file),
                get_dataset_generator(dataset_file),
                get_dataset_generator(chain_dataset_file),
                child_generator,
            )

    return recursive_builder
