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


class StateManager(ABC):
    """Abstract base class for state managers."""

    @abc.abstractmethod
    def get_state(self, path: Path) -> State:
        """Get a state from the state manager."""

    @abc.abstractmethod
    def set_state(self, path: Path, state: State) -> None:
        """Set state in the state manager."""

    @abc.abstractmethod
    def save_state(self, path: Path, state: Optional[State]) -> None:
        """Save state to a file."""

    @abc.abstractmethod
    def unload_state(self, path: Path, state: Optional[State]) -> None:
        """Unload the state of a given path."""

    @abc.abstractmethod
    def unload_children_state(self, paths: Iterable[Path]) -> None:
        """Unload the children state of a given path."""

    @abc.abstractmethod
    def cleanup(self):
        """Cleanup the state manager."""

    @abc.abstractmethod
    def __contains__(self, path: Path) -> bool:
        """Check if a state is in the state manager."""


class EvictionStateManager(StateManager):
    """Parameter manager that evicts state when it reaches a certain size."""

    def __init__(
        self,
        state_limit: int,
        save_state_to_file: Callable[[Path, State], None],
        eviction_proportion: float = 0.1,
    ) -> None:
        """Initialize the eviction parameter manager."""
        self.state_dict: Dict[Path, State] = {}
        self.state_limit = state_limit
        self.eviction_proportion = eviction_proportion
        self.save_state_to_file = save_state_to_file

    def get_state(self, path: Path) -> State:
        """Get a set of state from the parameter manager."""
        return self.state_dict[path]

    def set_state(self, path: Path, state: State) -> None:
        """Set a set of state in the parameter manager."""
        if path not in self:
            self._register_state(path, state)
        self.state_dict[path] = state

    def save_state(self, path: Path, state: Optional[State]) -> None:
        """Save a set of state to a file."""
        if state is not None:
            self.set_state(path, state)
        self.save_state_to_file(path, self.state_dict[path])

    def unload_state(self, path: Path, state: Optional[State]) -> None:
        """Unload the state of a given path."""
        self.save_state(path, state)
        self.state_dict.pop(path, None)

    def unload_children_state(self, paths: Iterable[Path]) -> None:
        """Unload the children state of a given path."""
        for path in paths:
            self.unload_state(path, None)

    def __contains__(self, path: Path) -> bool:
        """Check if a set of state is in the parameter manager."""
        return path in self.state_dict

    def _register_state(self, path: Path, state: State) -> State:
        """Register a set of state with the parameter manager."""
        if len(self.state_dict) < self.state_limit:
            self.state_dict[path] = state
            return state

        self._evict()
        self.state_dict[path] = state
        return state

    def cleanup(self) -> None:
        """Cleanup the parameter manager."""
        for path in self.state_dict:
            self.save_state(path, None)
        os.sync()
        self.state_dict = {}

    def _evict(self) -> None:
        """Evict a fraction of datasets from the dataset manager."""
        to_evict: List[Path] = [
            item
            for i, item in enumerate(self.state_dict.keys())
            if i < int(self.eviction_proportion * self.state_limit)
        ]
        print("Evicting state")
        for path in to_evict:
            self.unload_state(path, None)


def get_eviction_state_manager(
    state_limit: int,
    save_state_to_file: Callable[[Path, State], None],
    eviction_proportion: float,
) -> EvictionStateManager:
    """Get an eviction state manager."""
    return EvictionStateManager(state_limit, save_state_to_file, eviction_proportion)


def load_state(
    path: Path,
    load_state: Callable[[Path], State],
    state_manager: StateManager,
) -> State:
    """Load a set of parameters from a file if it not already in manager.

    This function is used to load a set of parameters from a file. If the parameters are
    already in the parameter manager, they are returned from the parameter manager.
    Otherwise, the parameters are loaded from the file and registered with the parameter
    manager.
    """
    if path in state_manager:
        return state_manager.get_state(path)

    state: State = load_state(path)
    state_manager.set_state(path, state)
    return state


def get_state_generator(
    state_file: Optional[Path],
    load_state_file: StateLoader,
    state_manager: StateManager,
) -> Optional[Callable[[Dict], State]]:
    """Get a state generator function."""
    if state_file is not None:

        def state_generator(_config: Dict) -> State:
            return load_state(
                path=state_file, load_state=load_state_file, state_manager=state_manager
            )

        return state_generator

    return None
