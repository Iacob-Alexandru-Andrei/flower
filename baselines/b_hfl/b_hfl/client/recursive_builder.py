"""Client recursive builder for a given file hierarchy."""

import concurrent.futures
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple

from flwr.common import NDArrays

from b_hfl.typing.common_types import (
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
from b_hfl.state_management.dataset import (
    DatasetManager,
    get_dataset_generator,
)
from b_hfl.state_management.parameters import (
    ParameterManager,
    get_parameter_generator,
)
from b_hfl.state_management.node_state import (
    StateManager,
    get_state_generator,
)
from b_hfl.state_management.residuals import (
    get_residuals,
    send_residuals,
)

from b_hfl.utils.utils import extract_file_from_files


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
    state_file_name: str = "_state",
) -> RecursiveBuilder:
    """Generate a recursive builder for a given file hierarchy.

    This is recursively caled within the inner function for each child of a given node.
    """
    cur_round: int = 0
    state_file_name = f"{parameters_file_name}{state_file_name}"

    def recursive_builder(
        current_client: ConfigurableRecClient,
        client_fn: ClientFN,
        test: bool = False,
    ) -> RecursiveStructure:
        """Generate a recursive structure for a given client."""
        file_type: str = "test" if test else "train"

        dataset_file: Optional[Path] = extract_file_from_files(
            files=path_dict.path, file_type=file_type
        )

        chain_dataset_file: Optional[Path] = extract_file_from_files(
            files=path_dict.path, file_type=f"{file_type}_chain"
        )

        parameter_file: Optional[Path] = extract_file_from_files(
            files=path_dict.path, file_type=parameters_file_name
        )

        state_file: Optional[Path] = extract_file_from_files(
            files=path_dict.path, file_type=state_file_name
        )

        def recursive_step(
            params_and_state: Tuple[NDArrays, State], final: bool
        ) -> None:
            nonlocal cur_round
            cur_round += 1

            if final:
                parameters, state = params_and_state
                parameters_save_name = (
                    path_dict.path / f"{parameters_file_name}{parameters_ext}"
                    if parameter_file is None
                    else parameter_file
                )

                state_save_name = (
                    path_dict.path / state_file_name
                    if state_file is None
                    else state_file
                )
                parameter_manager.set_parameters(
                    path=parameters_save_name, parameters=parameters
                )

                state_manager.set_state(path=state_save_name, state=state)

                if chain_dataset_file is not None:
                    dataset_manager.unload_chain_dataset(chain_dataset_file)

                children_dataset_files: Generator[Optional[Path], None, None] = (
                    file
                    for file in (
                        extract_file_from_files(
                            files=child_dict.path, file_type=file_type
                        )
                        for child_dict in path_dict.children
                    )
                    if file is not None
                )

                children_chain_dataset_files: Generator[Path, None, None] = (
                    file
                    for file in (
                        extract_file_from_files(
                            files=child_dict.path, file_type=f"{file_type}_chain"
                        )
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
                    state_manager.cleanup()

        if not test:
            return (
                get_parameter_generator(
                    parameter_file=parameter_file,
                    load_params_file=load_params_file,
                    parameter_manager=parameter_manager,
                ),
                get_state_generator(
                    state_file=state_file,
                    load_state_file=load_state_file,
                    state_manager=state_manager,
                ),
                get_dataset_generator(
                    dataset_file=dataset_file,
                    load_dataset_file=load_dataset_file,
                    dataset_manager=dataset_manager,
                ),
                get_dataset_generator(
                    dataset_file=chain_dataset_file,
                    load_dataset_file=load_dataset_file,
                    dataset_manager=dataset_manager,
                ),
                _get_child_generators(
                    current_client=current_client, client_fn=client_fn, test=test
                ),
                lambda leaf_to_root: get_residuals(
                    residuals_manager=residuals_manager,
                    path_dict=path_dict,
                    leaf_to_root=leaf_to_root,
                ),
                lambda send_to, residual, leaf_to_root: send_residuals(
                    residuals_manager=residuals_manager,
                    path_dict=path_dict,
                    send_to=send_to,
                    residual=residual,
                    leaf_to_root=leaf_to_root,
                ),
                recursive_step,
            )
        else:
            return (
                get_parameter_generator(
                    parameter_file=parameter_file,
                    load_params_file=load_params_file,
                    parameter_manager=parameter_manager,
                ),
                get_dataset_generator(
                    dataset_file=dataset_file,
                    load_dataset_file=load_dataset_file,
                    dataset_manager=dataset_manager,
                ),
                get_dataset_generator(
                    dataset_file=chain_dataset_file,
                    load_dataset_file=load_dataset_file,
                    dataset_manager=dataset_manager,
                ),
                _get_child_generators(
                    current_client=current_client, client_fn=client_fn, test=test
                ),
            )

    def _get_child_generators(
        current_client: ConfigurableRecClient, client_fn: ClientFN, test: bool
    ) -> ClientResGeneratorList:
        if test:
            return [
                (
                    _get_evaluate_child_generator(
                        current_client=current_client,
                        client_fn=client_fn,
                        child_path_dict=child_path_dict,
                        cid=str(i),
                    ),
                    on_evaluate_config_fn(cur_round, child_path_dict.path),
                )
                for i, child_path_dict in enumerate(iterable=path_dict.children)
            ]

        return [
            (
                _get_fit_child_generator(
                    current_client=current_client,
                    client_fn=client_fn,
                    child_path_dict=child_path_dict,
                    cid=str(i),
                ),
                on_fit_config_fn(cur_round, child_path_dict.path),
            )
            for i, child_path_dict in enumerate(iterable=path_dict.children)
        ]

    def _get_fit_child_generator(
        current_client: ConfigurableRecClient,
        client_fn: ClientFN,
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
            config.client_config.parent_round = parent_conf.client_config.parent_round

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
            ).fit(parameters=params, config=config)

        return lambda params, conf, parent_conf: executor.submit(
            fit_client, params, conf, parent_conf
        )

    def _get_evaluate_child_generator(
        current_client: ConfigurableRecClient,
        client_fn: ClientFN,
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
            config.client_config.parent_round = parent_conf.client_config.parent_round

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
            ).evaluate(parameters=params, config=config)

        return lambda params, conf, parent_conf: executor.submit(
            evaluate_client, params, conf, parent_conf
        )

    return recursive_builder
