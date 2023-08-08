"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


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

# if __name__ == "__main__":

import csv

#     download_and_preprocess()
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Callable

import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset


# Any is used to represent "FilehHierarchy" because it is a recursive type
# and MyPy does not have proper support for recursive types.
class FileHierarchy(TypedDict):
    """Dictionary representation of the file system."""

    parent: Optional[Any]
    parent_path: Optional[Path]
    path: Path
    files: List[Path]
    children: List[Any]


def build_path_dict(
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
    if path.is_file():
        path_dict["files"].append(path)
    else:
        for _, child_path in enumerate(path.iterdir()):
            path_dict["children"].append(
                build_path_dict(
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
    path_dict = build_path_dict(root)
    mapping: Dict[str, Path] = {}
    for child in path_dict["children"]:
        mapping[f"r_{child['path'].name}"] = child["path"]
    return mapping


class ClientFileHierarchy(TypedDict):
    name: str
    children: List[Any]


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
                    chain_files[path.stem].append(path)

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
                    writer.writerow(str(path))
            print(files)
            return_chain_files[file_type] = new_path
        print(cur_path)

        return return_chain_files

    rec_child_map_to_file_hierarchy(logical_mapping, out_root)


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
    Path("data/test/in"),
    Path("data/test/out"),
)
import tarfile

import gdown


def download_FEMNIST(dataset_dir: Path = Path("/data/femnist")) -> None:
    """Download and extract the FEMNIST dataset."""
    #  Download compressed dataset
    if not (dataset_dir / "femnist.tar.gz").exists():
        id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            str(dataset_dir / "femnist.tar.gz"),
        )

    # Decompress dataset
    if not dataset_dir.exists():
        with tarfile.open(dataset_dir / "femnist.tar.gz", "r:gz") as tar:
            tar.extractall(dataset_dir)

    print(f"Dataset extracted in {dataset_dir}")


class FEMNIST(Dataset):
    """Create a PyTorch dataset from the FEMNIST dataset."""

    def __init__(
        self,
        mapping: Path,
        data_dir: Path,
        name: str = "train",
        transform: Optional[Callable[[ImageType], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
    ):
        """Initialize the FEMNIST dataset.

        Args:
            mapping (Path): path to the mapping folder containing the .csv files.
            data_dir (Path): path to the dataset folder. Defaults to data_dir.
            name (str): name of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional): transform function to be applied to the ImageType object.
            target_transform (Optional[Callable[[int], Any]], optional): transform function to be applied to the label.
        """
        self.data_dir = data_dir
        self.mapping = mapping
        self.name = name

        self.data: Sequence[Tuple[str, int]] = self._load_dataset()
        self.transform: Optional[Callable[[ImageType], Any]] = transform
        self.target_transform: Optional[Callable[[int], Any]] = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Function used by PyTorch to get a sample.

        Args:
            index (_type_): index of the sample.

        Returns
        -------
            Tuple[Any, Any]: couple (sample, label).
        """
        sample_path, label = self.data[index]

        # Convert to the full path
        full_sample_path: Path = self.data_dir / self.name / sample_path

        img: ImageType = Image.open(full_sample_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """Function used by PyTorch to get the length of the dataset as number of
        samples.

        Returns
        -------
            int: the length of the dataset.
        """
        return len(self.data)

    def _load_dataset(self) -> Sequence[Tuple[str, int]]:
        """Load the paths and labels of the partition Preprocess the dataset for faster
        future loading If opened for the first time.

        Raises
        ------
            ValueError: raised if the mapping file doesn't exists

        Returns
        -------
            Sequence[Tuple[str, int]]: partition asked as a sequence of couples (path_to_file, label)
        """
        preprocessed_path: Path = (self.mapping / self.name).with_suffix(".pt")
        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            csv_path = (self.mapping / self.name).with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")
            else:
                with open(csv_path, mode="r") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    # Ignore header
                    next(csv_reader)

                    # Extract the samples and the labels
                    partition: Sequence[Tuple[str, int]] = [
                        (sample_path, int(label_id))
                        for _, sample_path, _, label_id in csv_reader
                    ]

                    # Save for future loading
                    torch.save(partition, preprocessed_path)
                    return partition
