"""Parameter management for B-HFL."""
import abc
from abc import ABC
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, cast

from flwr.common import NDArrays

from b_hfl.typing.common_types import ParametersLoader


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
    def cleanup(self):
        """Cleanup the parameter manager."""

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

    def cleanup(self) -> None:
        """Cleanup the parameter manager."""
        for path in self.parameter_dict:
            self.save_parameters(path, None)
        self.parameter_dict = {}

    def _evict(self) -> None:
        """Evict a fraction of datasets from the dataset manager."""
        to_evict: List[Path] = [
            item
            for i, item in enumerate(self.parameter_dict.keys())
            if i < int(self.eviction_proportion * self.parameter_limit)
        ]
        print("Evicting parameters")
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


def get_parameter_generator(
    parameter_file: Optional[Path],
    load_params_file: ParametersLoader,
    parameter_manager: ParameterManager,
) -> Optional[Callable[[Dict], NDArrays]]:
    """Get a parameter generator function."""
    if parameter_file is not None:

        def parameter_generator(_config: Dict) -> NDArrays:
            return load_parameters(
                path=cast(Path, parameter_file),
                load_parameters_file=load_params_file,
                parameter_manager=parameter_manager,
            )

        return parameter_generator

    return None
