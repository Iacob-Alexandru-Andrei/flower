import abc
import os
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from flwr.common import NDArrays
from torch.utils.data import ChainDataset, ConcatDataset, Dataset

from b_hfl.typing.common_types import (
    DatasetLoader,
    FitRes,
    ParametersLoader,
    State,
    StateLoader,
)
from b_hfl.schemas.file_system_schema import FolderHierarchy


class DatasetManager(ABC):
    """Abstract base class for dataset managers."""

    @abc.abstractmethod
    def register_dataset(self, path: Path, dataset: Dataset) -> Any:
        """Register a dataset with the dataset manager."""

    @abc.abstractmethod
    def register_chain_dataset(
        self,
        path: Path,
        chain_files: Iterable[Path],
        chain_dataset: Union[ChainDataset, ConcatDataset],
    ) -> Any:
        """Register a chain dataset with the dataset manager."""

    @abc.abstractmethod
    def get_dataset(self, path: Path) -> Any:
        """Get a dataset from the dataset manager."""

    @abc.abstractmethod
    def unload_children_datasets(self, paths: Iterable[Path]) -> None:
        """Unload the children datasets of a given path."""

    @abc.abstractmethod
    def unload_chain_dataset(self, path: Path) -> None:
        """Unload a chain dataset."""

    @abc.abstractmethod
    def __contains__(self, path: Path) -> bool:
        """Check if a dataset is in the dataset manager."""


class EvictionDatasetManager(DatasetManager):
    """Dataset manager that evicts datasets when it reaches a certain size."""

    def __init__(
        self,
        dataset_limit: int,
        eviction_proportion: float = 0.1,
    ) -> None:
        """Initialize the eviction dataset manager."""
        self.dataset_dict: Dict[Path, Dataset] = {}
        self.chained_paths: Dict[Path, Iterable[Path]] = {}
        self.dataset_limit: int = dataset_limit
        self.eviction_proportion: float = eviction_proportion

    def register_dataset(self, path: Path, dataset: Dataset) -> None:
        """Register a dataset with the dataset manager."""
        if len(self.dataset_dict) >= self.dataset_limit:
            self._evict()
        self.dataset_dict[path] = dataset

    def register_chain_dataset(
        self, path: Path, chain_files: Iterable[Path], chain_dataset: ChainDataset
    ) -> None:
        """Register a chain dataset with the dataset manager."""
        self.chained_paths[path] = chain_files
        self.dataset_dict[path] = chain_dataset

    def get_dataset(self, path: Path) -> Dataset:
        """Get a dataset from the dataset manager."""
        return self.dataset_dict[path]

    def unload_children_datasets(self, paths: Iterable[Path]) -> None:
        """Unload the children datasets of a given path."""
        for path in paths:
            self.dataset_dict.pop(path, None)

    def unload_chain_dataset(self, path: Path) -> None:
        """Unload a chain dataset."""
        if "base" not in path.name:
            self.chained_paths.pop(path, None)
            self.dataset_dict.pop(path, None)

    def __contains__(self, path: Path) -> bool:
        """Check if a dataset is in the dataset manager."""
        return path in self.dataset_dict

    def _evict(self) -> None:
        """Evict a fraction of datasets from the dataset manager."""
        eviction_iterator: Iterator[Path] = iter(self.dataset_dict.keys())
        to_evict: List[Path] = []
        while len(to_evict) < int(self.eviction_proportion * self.dataset_limit):
            try:
                item = next(eviction_iterator)
                if (
                    item not in self.chained_paths.values()
                    and item not in self.chained_paths.keys()
                ):
                    to_evict.append(item)
            except StopIteration:
                break
        for path in to_evict:
            del self.dataset_dict[path]


class NoOpDatasetManager(DatasetManager):
    """Dataset manager that evicts datasets when it reaches a certain size."""

    def __init__(
        self,
    ) -> None:
        """Initialize the eviction dataset manager."""

    def register_dataset(self, path: Path, dataset: Dataset) -> None:
        """Register a dataset with the dataset manager."""

    def register_chain_dataset(
        self, path: Path, chain_files: Iterable[Path], chain_dataset: ChainDataset
    ) -> None:
        """Register a chain dataset with the dataset manager."""

    def get_dataset(self, path: Path) -> Dataset:
        """Get a dataset from the dataset manager."""
        return None  # type: ignore

    def unload_children_datasets(self, paths: Iterable[Path]) -> None:
        """Unload the children datasets of a given path."""

    def unload_chain_dataset(self, path: Path) -> None:
        """Unload a chain dataset."""

    def __contains__(self, path: Path) -> bool:
        """Check if a dataset is in the dataset manager."""
        return False


def get_eviction_dataset_manager(
    dataset_limit: int, eviction_proportion: float
) -> EvictionDatasetManager:
    """Get an eviction dataset manager."""
    return EvictionDatasetManager(dataset_limit, eviction_proportion)


def get_noop_dataset_manager() -> NoOpDatasetManager:
    """Get an eviction dataset manager."""
    return NoOpDatasetManager()


def get_chain_paths(path: Path) -> Iterable[Path]:
    """Get the paths of the files in a chain file."""
    if path.suffix == ".csv":
        with open(path, "r") as file:
            return [Path(line.strip()) for line in file.readlines()]
    else:
        raise ValueError(f"File {path} is not a valid chain file.")


def load_dataset(
    path: Path,
    load_dataset_file: Callable[[Path], Dataset],
    dataset_manager: DatasetManager,
) -> Dataset:
    """Load a dataset from a file/chain if it not in manager.

    This function is used to load a dataset from a file or chain file. If the dataset is
    already in the dataset manager, it is returned from the dataset manager. Other ise,
    the dataset is loaded from the file and registered with the dataset manager.
    """
    if path in dataset_manager:
        return dataset_manager.get_dataset(path)

    dataset: Dataset = load_dataset_file(path)
    dataset_manager.register_dataset(path, dataset)
    return dataset


def process_chain_file(
    path: Path,
    load_file: Callable[[Path], Dataset],
    dataset_manager: DatasetManager,
) -> Dataset:
    """Process a chain file to create a chain dataset.

    A chain file is a file that contains a list of paths to other files. This function
    links the datasets obtained from those files into a single chain dataset that can be
    iterated over.
    """
    if path in dataset_manager:
        return dataset_manager.get_dataset(path)

    chain_files: Iterable[Path] = get_chain_paths(path)
    chain_dataset: Union[ChainDataset, ConcatDataset] = ConcatDataset(
        (process_file(file, load_file, dataset_manager) for file in chain_files)
    )
    dataset_manager.register_chain_dataset(path, chain_files, chain_dataset)
    return chain_dataset


def process_file(
    path: Path,
    load_file: Callable[[Path], Dataset],
    dataset_manager: DatasetManager,
) -> Dataset:
    """Process a file or chain file to create a dataset."""
    if "chain" in path.name:
        return process_chain_file(path, load_file, dataset_manager)

    return load_dataset(path, load_file, dataset_manager)


def get_dataset_generator(
    dataset_file: Optional[Path],
    load_dataset_file: DatasetLoader,
    dataset_manager: DatasetManager,
) -> Optional[Callable[[Dict], Dataset]]:
    """Get a dataset generator function."""
    if dataset_file is not None:

        def dataset_generator(_config) -> Dataset:
            return process_file(
                path=dataset_file,
                load_file=load_dataset_file,
                dataset_manager=dataset_manager,
            )

        return dataset_generator

    return None
