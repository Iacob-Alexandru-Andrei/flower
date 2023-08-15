"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from cerberus.validator import BareValidator, Validator, schema_registry
from common_types import FileHierarchy, ClientFileHierarchy, ConfigFileHierarchy
from config_schema import get_recursive_client_schema

import gdown
from regex import B


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)


def download_FEMNIST(dataset_dir: Path = Path("data/femnist")) -> None:
    """Download and extract the FEMNIST dataset."""
    #  Download compressed dataset
    data_file = dataset_dir / "femnist.tar.gz"
    if not data_file.exists():
        id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            str(dataset_dir / "femnist.tar.gz"),
        )

    decompressed_dataset_dir = dataset_dir / "femnist"
    # Decompress dataset
    if not decompressed_dataset_dir.exists():
        with tarfile.open(data_file, "r:gz") as tar:
            tar.extractall(decompressed_dataset_dir)

    print(f"Dataset extracted in {dataset_dir}")


# Any is used to represent "FilehHierarchy" because it is a recursive type
# and MyPy does not have proper support for recursive types.


def get_wrapped_file_hierarchy(path: str) -> FileHierarchy:
    return get_file_hierarchy(Path(path))


def get_file_hierarchy(
    path: Path,
    parent_path: Optional[Path] = None,
    parent: Optional[FileHierarchy] = None,
) -> FileHierarchy:
    """Build a dictionary representation of the file system."""
    # Schema of the path_dict
    path_dict: FileHierarchy = {
        "parent": parent,
        "parent_path": parent_path,
        "path": path,
        "files": [],
        "children": [],
    }
    # Build the tree

    for _, child_path in enumerate(path.iterdir()):
        if child_path.is_file():
            if not child_path.name.startswith("_") and not child_path.name.startswith(
                "."
            ):
                path_dict["files"].append(child_path)
        else:
            path_dict["children"].append(
                get_file_hierarchy(
                    child_path,
                    parent_path,
                    path_dict,
                )
            )
    return path_dict


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
    path_dict = get_file_hierarchy(root)
    mapping: Dict[str, Path] = {}
    for child in path_dict["children"]:
        mapping[f"r_{child['path'].name}"] = child["path"]
    return mapping


schema_registry.add(
    "client_file_hierarchy_schema",
    {
        "name": {"type": "string", "required": True},
        "children": {
            "type": "list",
            "schema": {"type": "dict", "schema": "client_file_hierarchy_schema"},
        },
    },
)


def child_map_to_file_hierarchy(
    logical_mapping: ClientFileHierarchy, in_root: Path, out_root: Path
) -> None:
    """Create a file hierarchy based on a logical mapping.

    Parameters
    ----------
    mapping : Dict[str, Path]
        A mapping from logical names to paths.
    root : Path
        The root directory where the file hierarchy will be created.
    """
    client_file_hierarchy_schema = schema_registry.get("client_file_hierarchy_schema")

    validator: BareValidator = Validator(client_file_hierarchy_schema)  # type: ignore

    if not validator.validate(logical_mapping):
        raise ValueError(f"Invalid client hierarchy: {validator.errors}")

    child_mapping = extract_child_mapping(in_root)

    def rec_child_map_to_file_hierarchy(
        client_file_hierarchy: ClientFileHierarchy,
        parent_path: Path,
    ) -> Dict[str, Path]:
        nonlocal child_mapping

        name = client_file_hierarchy["name"]
        children = client_file_hierarchy["children"]

        chain_files: Dict[str, List[Path]] = defaultdict(list)

        cur_path = parent_path / name
        cur_path.mkdir(parents=True, exist_ok=True)

        if name.startswith("r_"):
            client_directory = child_mapping[name]
            for path in client_directory.iterdir():
                if path.is_file():
                    if not path.name.startswith("_") and not path.name.startswith("."):
                        chain_files[f"{path.stem}_base"].append(path)

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


schema_registry.add(
    "config_file_hierarchy_schema",
    {
        "on_fit_config": {"type": "dict", "required": True},
        "on_evaluate_config": {"type": "dict", "required": True},
        "children": {
            "type": "list",
            "schema": {"type": "dict", "schema": "config_file_hierarchy_schema"},
        },
    },
)


def config_map_to_file_hierarchy(
    logical_mapping: ConfigFileHierarchy,
    out_root: Path,
    train_config_schema: Dict[str, Any],
    test_config_schema: Dict[str, Any],
) -> None:
    """Create a file hierarchy based on a logical mapping.

    Parameters
    ----------
    mapping : Dict[str, Path]
        A mapping from logical names to paths.
    root : Path
        The root directory where the file hierarchy will be created.
    """
    config_file_hierarchy_schema = schema_registry.get("config_file_hierarchy_schema")
    validator: BareValidator = Validator(config_file_hierarchy_schema)  # type: ignore

    if not validator.validate(logical_mapping):
        raise ValueError(
            f"Invalid config hierarchy: {validator.errors}, {logical_mapping}"
        )

    train_config_validator: BareValidaVator = Validator(train_config_schema)  # type: ignore
    test_config_validator: BareValidaVator = Validator(test_config_schema)  # type: ignore

    def rec_config_map_to_file_hierarchy(
        config_file_hierarchy: ConfigFileHierarchy,
        cur_path: Path,
    ) -> None:
        on_fit_config = config_file_hierarchy["on_fit_config"]

        if not train_config_validator.validate(on_fit_config):
            raise ValueError(
                f"Invalid config: {train_config_validator.errors}, {on_fit_config}"
            )

        on_evaluate_config = config_file_hierarchy["on_evaluate_config"]

        if not test_config_validator.validate(on_evaluate_config):
            raise ValueError(
                f"Invalid config: {train_config_validator.errors}, {on_evaluate_config}"
            )

        with open(cur_path / "on_fit_config.json", "w") as f:
            json.dump(on_fit_config, f)

        with open(cur_path / "on_evaluate_config.json", "w") as f:
            json.dump(on_evaluate_config, f)

        folders = (folder for folder in cur_path.iterdir() if folder.is_dir())

        for child_config, folder in zip(config_file_hierarchy["children"], folders):
            rec_config_map_to_file_hierarchy(child_config, folder)

    rec_config_map_to_file_hierarchy(logical_mapping, out_root)


def get_uniform_configs(
    path_dict: FileHierarchy, on_fit_config: Dict, on_evaluate_config: Dict
) -> ConfigFileHierarchy:
    """Get a uniform config hierarchy."""
    return {
        "on_fit_config": on_fit_config,
        "on_evaluate_config": on_evaluate_config,
        "children": [
            get_uniform_configs(child, on_fit_config, on_evaluate_config)
            for child in path_dict["children"]
        ],
    }


default_FEMNIST_recursive_fit_config = {
    "rounds": [
        {
            "client_config": {
                "num_rounds": 1,
                "fit_fraction": 1.0,
                "train_children": True,
                "train_chain": False,
                "train_proxy": False,
            },
            "dataloader_config": {
                "batch_size": 8,
                "num_workers": 2,
                "shuffle": False,
                "test": False,
            },
            "parameter_config": {},
            "net_config": {},
            "run_config": {
                "epochs": 1,
                "client_learning_rate": 0.01,
                "weight_decay": 0.001,
            },
        }
    ]
}

default_FEMNIST_recursive_evaluate_config = {
    "rounds": [
        {
            "client_config": {
                "eval_fraction": 1.0,
                "test_children": True,
                "test_chain": True,
                "test_proxy": False,
            },
            "dataloader_config": {
                "batch_size": 8,
                "num_workers": 2,
                "shuffle": False,
                "test": False,
            },
            "parameter_config": {},
            "net_config": {},
            "run_config": {
                "epochs": 1,
                "client_learning_rate": 0.01,
                "weight_decay": 0.001,
            },
        }
    ]
}


if __name__ == "__main__":
    src_folder = Path(
        "/home/aai30/nfs-share/b_hfl/femnist_local/femnist/femnist/client_data_mappings/fed_natural"
    )
    out_folder = Path(
        "/home/aai30/nfs-share/b_hfl/femnist_local/femnist/femnist/client_data_mappings/hierarchical_test"
    )

    child_map_to_file_hierarchy(
        {
            "name": "0",
            "children": [
                {
                    "name": "1",
                    "children": [
                        {
                            "name": "3",
                            "children": [
                                {"name": "r_0", "children": []},
                                {"name": "r_1", "children": []},
                            ],
                        },
                        {
                            "name": "4",
                            "children": [
                                {"name": "r_0", "children": []},
                                {"name": "r_0", "children": []},
                            ],
                        },
                    ],
                },
                {
                    "name": "2",
                    "children": [
                        {
                            "name": "5",
                            "children": [
                                {"name": "r_0", "children": []},
                                {"name": "r_1", "children": []},
                            ],
                        },
                        {
                            "name": "6",
                            "children": [
                                {"name": "r_2", "children": []},
                                {"name": "r_0", "children": []},
                            ],
                        },
                    ],
                },
            ],
        },
        src_folder,
        out_folder,
    )

    path_dict = get_file_hierarchy(out_folder)
    uniform_config_map = get_uniform_configs(
        path_dict,
        default_FEMNIST_recursive_fit_config,
        default_FEMNIST_recursive_evaluate_config,
    )

    train_config_schema = get_recursive_client_schema(False)
    test_config_schema = get_recursive_client_schema(True)

    config_map_to_file_hierarchy(
        uniform_config_map, out_folder, train_config_schema, test_config_schema
    )

    # download_and_preprocess()
