"""Manages dataset and parameter sets for b_hfl.

Dataset management and parameter management are handled by the DatasetManager and
ParameterManager, they are written to be independent of one another. Thus far the
hierarchical structure depends on the file system, but this could be changed to a
database or other hierarchical structure with the necesary work.
"""
import abc
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from flwr.common import NDArrays
from torch.utils.data import ChainDataset, Dataset, ConcatDataset


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


def get_eviction_dataset_manager(
    dataset_limit: int, eviction_proportion: float
) -> EvictionDatasetManager:
    """Get an eviction dataset manager."""
    return EvictionDatasetManager(dataset_limit, eviction_proportion)


class ParameterManager(ABC):
    """Abstract base class for parameter managers."""

    @abc.abstractmethod
    def get_parameters(self, path: Path) -> NDArrays:
        """Get a set of parameters from the parameter manager."""

    @abc.abstractmethod
    def set_parameters(self, path: Path, parameters: NDArrays) -> None:
        """Set a set of parameters in the parameter manager."""

    @abc.abstractmethod
    def save_parameters(self, path: Path, parameters: Optional[NDArrays]) -> None:
        """Save a set of parameters to a file."""

    @abc.abstractmethod
    def unload_parameters(self, path: Path, parameters: Optional[NDArrays]) -> None:
        """Unload the parameters of a given path."""

    @abc.abstractmethod
    def unload_children_parameters(self, paths: Iterable[Path]) -> None:
        """Unload the children parameters of a given path."""

    @abc.abstractmethod
    def __contains__(self, path: Path) -> bool:
        """Check if a set of parameters is in the parameter manager."""


class EvictionParameterManager(ABC):
    """Parameter manager that evicts parameters when it reaches a certain size."""

    def __init__(
        self,
        parameters_limit: int,
        save_parameters_to_file: Callable[[Path, NDArrays], None],
        eviction_proportion: float = 0.1,
    ) -> None:
        """Initialize the eviction parameter manager."""
        self.parameter_dict: Dict[Path, NDArrays] = {}
        self.parameter_limit = parameters_limit
        self.eviction_proportion = eviction_proportion
        self.save_parameters_to_file = save_parameters_to_file

    def get_parameters(self, path: Path) -> NDArrays:
        """Get a set of parameters from the parameter manager."""
        return self.parameter_dict[path]

    def set_parameters(self, path: Path, parameters: NDArrays) -> None:
        """Set a set of parameters in the parameter manager."""
        if path not in self:
            self._register_parameters(path, parameters)
        self.parameter_dict[path] = parameters

    def save_parameters(self, path: Path, parameters: Optional[NDArrays]) -> None:
        """Save a set of parameters to a file."""
        if parameters is not None:
            self.set_parameters(path, parameters)
        self.save_parameters_to_file(path, self.parameter_dict[path])

    def unload_parameters(self, path: Path, parameters: Optional[NDArrays]) -> None:
        """Unload the parameters of a given path."""
        self.save_parameters(path, parameters)
        self.parameter_dict.pop(path, None)

    def unload_children_parameters(self, paths: Iterable[Path]) -> None:
        """Unload the children parameters of a given path."""
        for path in paths:
            self.unload_parameters(path, None)

    def __contains__(self, path: Path) -> bool:
        """Check if a set of parameters is in the parameter manager."""
        return path in self.parameter_dict

    def _register_parameters(self, path: Path, parameters: NDArrays) -> NDArrays:
        """Register a set of parameters with the parameter manager."""
        if len(self.parameter_dict) < self.parameter_limit:
            self.parameter_dict[path] = parameters
            return parameters

        self._evict()
        self.parameter_dict[path] = parameters
        return parameters

    def _evict(self) -> None:
        """Evict a fraction of datasets from the dataset manager."""
        to_evict: List[Path] = [
            item
            for i, item in enumerate(self.parameter_dict.keys())
            if i < int(self.eviction_proportion * self.parameter_limit)
        ]
        for path in to_evict:
            self.unload_parameters(path, None)


def get_eviction_parameter_manager(
    parameters_limit: int,
    save_parameters_to_file: Callable[[Path, NDArrays], None],
    eviction_proportion: float,
) -> EvictionParameterManager:
    """Get an eviction dataset manager."""
    return EvictionParameterManager(
        parameters_limit, save_parameters_to_file, eviction_proportion
    )


def get_chain_paths(path: Path) -> Iterable[Path]:
    """Get the paths of the files in a chain file."""
    if path.suffix == ".csv":
        with open(path, "r") as file:
            return [Path(line.strip()) for line in file.readlines()]
    else:
        raise ValueError(f"File {path} is not a valid chain file.")


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
    path: Path, load_file: Callable[[Path], Dataset], dataset_manager: DatasetManager
) -> Dataset:
    """Process a file or chain file to create a dataset."""
    if "chain" in path.name:
        return process_chain_file(path, load_file, dataset_manager)

    return load_dataset(path, load_file, dataset_manager)


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


def load_parameters(
    path: Path,
    load_parameters_file: Callable[[Path], NDArrays],
    parameter_manager: ParameterManager,
) -> NDArrays:
    """Load a set of parameters from a file if it not already in manager.

    This function is used to load a set of parameters from a file. If the parameters are
    already in the parameter manager, they are returned from the parameter manager.
    Otherwise, the parameters are loaded from the file and registered with the parameter
    manager.
    """
    if path in parameter_manager:
        return parameter_manager.get_parameters(path)

    parameters: NDArrays = load_parameters_file(path)
    parameter_manager.set_parameters(path, parameters)
    return parameters
