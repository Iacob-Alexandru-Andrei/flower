"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import csv
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from b_hfl.schema.client_schema import RecClientTrainConf
from b_hfl.schema.file_system_schema import (
    ClientFolderHierarchy,
    ConfigFolderHierarchy,
    FolderHierarchy,
)
from b_hfl.typing.common_types import ConfigSchemaGenerator


def get_parameter_convertor(
    convertors: Iterable[Tuple[Any, Callable]]
) -> Callable[[Callable], Callable]:
    """Get a decorator that converts parameters to the right type."""

    def convert(param: Any) -> bool:
        for param_type, convertor in convertors:
            if isinstance(param, param_type):
                return convertor(param)
        return param

    def convert_params(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(convert(arg))
            for kwarg_name, kwarg_value in kwargs.items():
                new_kwargs[kwarg_name] = convert(kwarg_value)
            return func(*new_args, **new_kwargs)

        return wrapper

    return convert_params


pathify_params = get_parameter_convertor([(str, Path)])


@pathify_params
def get_folder_hierarchy(
    path: Path,
    parent_path: Optional[Path] = None,
    parent: Optional[FolderHierarchy] = None,
) -> FolderHierarchy:
    """Build a dictionary representation of the file system."""
    # Schema of the path_dict

    def rec_get_folder_hierarchy(
        root: Path,
        path: Path,
        parent_path: Optional[Path] = None,
        parent: Optional[FolderHierarchy] = None,
    ) -> FolderHierarchy:
        path_dict: FolderHierarchy = FolderHierarchy(
            **{
                "root": root,
                "parent": parent,
                "parent_path": parent_path,
                "path": path,
                "children": [],
            }
        )
        # Build the tree

        for _, child_path in enumerate(path.iterdir()):
            if child_path.is_dir():
                path_dict.children.append(
                    rec_get_folder_hierarchy(
                        root,
                        child_path,
                        parent_path,
                        path_dict,
                    )
                )
        return path_dict

    return rec_get_folder_hierarchy(path, path, parent_path, parent)


def extract_child_mapping(root: Path) -> Dict[str, Path]:
    """Extract a mapping from a one-layer child hierarchy.

    Parameters
    ----------
    root : Path
        The root directory where the file hierarchy will be created.

    Returns
    -------
    Dict[str, Path]
        A mapping from child names to paths.
    """
    path_dict = get_folder_hierarchy(root)
    mapping: Dict[str, Path] = {}
    for child in path_dict.children:
        mapping[f"r_{child['path'].name}"] = child.path
    return mapping


def child_map_to_file_hierarchy(
    logical_mapping: ClientFolderHierarchy, in_root: Path, out_root: Path
) -> None:
    """Create a file hierarchy based on a logical mapping.

    Parameters
    ----------
    mapping : Dict[str, Path]
        A mapping from logical names to paths.
    root : Path
        The root directory where the file hierarchy will be created.
    """
    child_mapping = extract_child_mapping(in_root)

    # pylint: disable=too-many-locals
    def rec_child_map_to_file_hierarchy(
        client_file_hierarchy: ClientFolderHierarchy,
        parent_path: Path,
    ) -> Dict[str, Path]:
        nonlocal child_mapping

        name = client_file_hierarchy.name

        is_base_case_client = name.startswith("r_")
        children = client_file_hierarchy.children

        chain_files: Dict[str, List[Path]] = defaultdict(list)

        cur_path = parent_path / name
        cur_path.mkdir(parents=True, exist_ok=True)

        if is_base_case_client:
            client_directory = child_mapping[name]
            for path in client_directory.iterdir():
                if path.is_file():
                    if not path.name.startswith("_") and not path.name.startswith("."):
                        chain_files[f"{path.stem}"].append(path)

        for child in children:
            child_chain_files = rec_child_map_to_file_hierarchy(
                child,
                cur_path,
            )
            for key, path in child_chain_files.items():
                chain_files[key].append(path)

        return_chain_files: Dict[str, Path] = {}
        for file_type, files in chain_files.items():
            new_path = cur_path / f"{file_type}_chain.csv"
            with open(new_path, "w") as f:
                writer = csv.writer(f)
                for path in files:
                    writer.writerow([str(path)])

            return_chain_files[file_type] = new_path

        return return_chain_files

    rec_child_map_to_file_hierarchy(logical_mapping, out_root)


@get_parameter_convertor([(str, Path), (DictConfig, OmegaConf.to_container)])
def config_map_to_file_hierarchy(
    logical_mapping: ConfigFolderHierarchy,
) -> None:
    """Create a file hierarchy based on a logical mapping.

    Parameters
    ----------
    mapping : Dict[str, Path]
        A mapping from logical names to paths.
    root : Path
        The root directory where the file hierarchy will be created.
    """

    def rec_config_map_to_file_hierarchy(
        config_folder_hierarchy: ConfigFolderHierarchy,
    ) -> None:
        on_fit_configs = [
            on_fit_config.dict()
            for on_fit_config in config_folder_hierarchy.on_fit_configs
        ]

        on_evaluate_configs = [
            on_evaluate_config.dict()
            for on_evaluate_config in config_folder_hierarchy.on_evaluate_configs
        ]
        with open(config_folder_hierarchy.path / "on_fit_config.json", "w") as f:
            json.dump(on_fit_configs, f)

        with open(config_folder_hierarchy.path / "on_evaluate_config.json", "w") as f:
            json.dump(on_evaluate_configs, f)

        for child_config in config_folder_hierarchy.children:
            rec_config_map_to_file_hierarchy(child_config)

    rec_config_map_to_file_hierarchy(logical_mapping)
    os.sync()


def get_uniform_configs_wrapped(
    train_config_schema: ConfigSchemaGenerator,
    test_config_schema: ConfigSchemaGenerator,
    on_fit_configs: List[Dict],
    on_evaluate_configs: List[Dict],
) -> Callable[[FolderHierarchy], ConfigFolderHierarchy]:
    """Get a uniform config hierarchy."""
    train_schema = train_config_schema()
    test_schema = test_config_schema()
    new_on_fit_configs: List[BaseModel] = [
        train_schema(**on_fit_config) for on_fit_config in on_fit_configs
    ]
    new_on_evaluate_configs: List[BaseModel] = [
        test_schema(**on_evaluate_config) for on_evaluate_config in on_evaluate_configs
    ]

    def wrapped_get_uniform_config(
        path_dict: FolderHierarchy,
    ) -> ConfigFolderHierarchy:
        return get_uniform_configs(
            path_dict, new_on_fit_configs, new_on_evaluate_configs
        )

    return wrapped_get_uniform_config


def get_uniform_configs(
    path_dict: FolderHierarchy,
    on_fit_configs: List[BaseModel],
    on_evaluate_configs: List[BaseModel],
) -> ConfigFolderHierarchy:
    """Get a uniform config hierarchy."""
    new_fit_configs = deepcopy(on_fit_configs)
    new_evaluate_configs = deepcopy(on_evaluate_configs)

    if path_dict.path.name.startswith("r_"):
        for new_fit_config in new_fit_configs:
            if isinstance(new_fit_config, RecClientTrainConf):
                new_fit_config.client_config.train_chain = True
    children = [
        get_uniform_configs(child, on_fit_configs, on_evaluate_configs)
        for child in path_dict.children
    ]
    return ConfigFolderHierarchy(
        **{
            "path": path_dict.path,
            "on_fit_configs": new_fit_configs,
            "on_evaluate_configs": new_evaluate_configs,
            "children": children,
        }
    )


# def flat_hierarchy_to_edge_assingment(src_folder: Path, assign: Callable[[str, Path], int]) -> ClientFolderHierarchy:
#     """ Assign leaf clients based on a predicate."""
#     edge_servers:
#     child_mapping = extract_child_mapping(src_folder)
#     for child, path in child_mapping.items():
