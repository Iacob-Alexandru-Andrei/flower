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

from b_hfl.schema.client_schema import RecClientTestConf, RecClientTrainConf
from b_hfl.schema.file_system_schema import (
    ClientFolderHierarchy,
    ConfigFolderHierarchy,
    FolderHierarchy,
)


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
        level: int,
        parent_path: Optional[Path] = None,
        parent: Optional[FolderHierarchy] = None,
    ) -> FolderHierarchy:
        path_dict: FolderHierarchy = FolderHierarchy(
            root=root,
            parent=parent,
            parent_path=parent_path,
            level=level,
            max_level=level,
            path=path,
            children=[],
        )

        # Build the tree
        true_max_level = level
        for _, child_path in enumerate(path.iterdir()):
            if child_path.is_dir():
                child_path_dict = rec_get_folder_hierarchy(
                    root=root,
                    path=child_path,
                    level=level + 1,
                    parent_path=parent_path,
                    parent=path_dict,
                )
                path_dict.children.append(child_path_dict)
                true_max_level = max(true_max_level, child_path_dict.max_level)

        path_dict.max_level = true_max_level

        return path_dict

    return rec_get_folder_hierarchy(
        root=path, path=path, level=0, parent_path=parent_path, parent=parent
    )


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
        mapping[f"r_{child.path.name}"] = child.path
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
            json.loads(on_fit_config.json())
            for on_fit_config in config_folder_hierarchy.on_fit_configs
        ]

        on_evaluate_configs = [
            json.loads(on_evaluate_config.json())
            for on_evaluate_config in config_folder_hierarchy.on_evaluate_configs
        ]
        with open(config_folder_hierarchy.path / "on_fit_config.json", "w") as f:
            json.dump(on_fit_configs, f)

        with open(config_folder_hierarchy.path / "on_evaluate_config.json", "w") as f:
            json.dump(on_evaluate_configs, f)

        for child_config in config_folder_hierarchy.children:
            rec_config_map_to_file_hierarchy(child_config)

    rec_config_map_to_file_hierarchy(logical_mapping)


def get_uniform_configs_wrapped(
    on_fit_configs: List[Dict],
    on_evaluate_configs: List[Dict],
) -> Callable[[FolderHierarchy], ConfigFolderHierarchy]:
    """Get a uniform config hierarchy."""
    new_on_fit_configs: List[RecClientTrainConf] = [
        RecClientTrainConf(**on_fit_config) for on_fit_config in on_fit_configs
    ]
    new_on_evaluate_configs: List[RecClientTestConf] = [
        RecClientTestConf(**on_evaluate_config)
        for on_evaluate_config in on_evaluate_configs
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
    on_fit_configs: List[RecClientTrainConf],
    on_evaluate_configs: List[RecClientTestConf],
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


def hierarchy_to_edge_assingment(
    src_folder: Path,
    assign_flat: Callable[[str, Path], Optional[str]],
    assign_rec: List[Callable[[ClientFolderHierarchy], str]],
) -> ClientFolderHierarchy:
    """Assign leaf clients based on a predicate."""
    hierarchy_dict: Dict[str, ClientFolderHierarchy] = {}
    child_mapping = extract_child_mapping(src_folder)
    for child, path in child_mapping.items():
        edge_server_index = assign_flat(child, path)
        if edge_server_index is None:
            continue

        if edge_server_index not in hierarchy_dict:
            hierarchy_dict[edge_server_index] = ClientFolderHierarchy(
                **{"name": edge_server_index, "children": []}
            )

        hierarchy_dict[edge_server_index].children.append(
            ClientFolderHierarchy(**{"name": child, "children": []})
        )
    full_hierarchy = hierarchy_to_edge_assingment_rec(hierarchy_dict, assign_rec)
    if len(full_hierarchy) != 1:
        raise ValueError("The hierarchy should have only one root")
    return next(iter(full_hierarchy.values()))


def hierarchy_to_edge_assingment_rec(
    hierarchy_dict: Dict[str, ClientFolderHierarchy],
    assign_rec: List[Callable[[ClientFolderHierarchy], str]],
) -> Dict[str, ClientFolderHierarchy]:
    """Assign internal nodes based on a predicate."""
    if len(assign_rec) == 0:
        return hierarchy_dict

    level_assign = assign_rec.pop(0)
    level_hierarchy_dict: Dict[str, ClientFolderHierarchy] = {}
    for _name, hierarchy in hierarchy_dict.items():
        server_index = level_assign(hierarchy)

        if server_index not in level_hierarchy_dict:
            level_hierarchy_dict[server_index] = ClientFolderHierarchy(
                **{"name": server_index, "children": []}
            )
        level_hierarchy_dict[server_index].children.append(hierarchy)
    return hierarchy_to_edge_assingment_rec(level_hierarchy_dict, assign_rec)


def get_modulo_assign_flat(
    num_edge_servers: int,
    max_edge_clients: int,
) -> Callable[[str, Path], Optional[str]]:
    """Get a modulo assignment function."""
    client_cnt = 0

    def assign_flat(
        child: str,
        path: Path,
    ) -> Optional[str]:
        nonlocal client_cnt

        if client_cnt >= max_edge_clients:
            return None

        to_ret = f"{client_cnt % num_edge_servers}"
        client_cnt += 1
        return to_ret

    return assign_flat


def get_modulo_assign_rec(
    num_servers_per_level: List[int],
) -> List[Callable[[ClientFolderHierarchy], str]]:
    """Get a modulo assignment function per-level."""

    def get_assign_modulo(
        num_servers: int,
    ) -> Callable[[ClientFolderHierarchy], str]:
        client_cnt = 0

        def assign_modulo(folder_hierarchy: ClientFolderHierarchy) -> str:
            nonlocal client_cnt
            to_ret = f"{client_cnt % num_servers}"
            client_cnt += 1
            return to_ret

        return assign_modulo

    return [
        get_assign_modulo(num_edge_servers)
        for num_edge_servers in num_servers_per_level
    ]


def get_configs_per_level(
    path_dict: FolderHierarchy,
    level_config_generator: Callable[
        [FolderHierarchy], Tuple[List[RecClientTrainConf], List[RecClientTestConf]]
    ],
) -> ConfigFolderHierarchy:
    """Get a config hierarchy per level."""
    on_fit_configs, on_evaluate_configs = level_config_generator(path_dict)

    children = [
        get_configs_per_level(child, level_config_generator)
        for child in path_dict.children
    ]
    return ConfigFolderHierarchy(
        **{
            "path": path_dict.path,
            "on_fit_configs": on_fit_configs,
            "on_evaluate_configs": on_evaluate_configs,
            "children": children,
        }
    )
