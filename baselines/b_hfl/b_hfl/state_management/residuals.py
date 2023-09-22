"""Residuals state management for residual connections between clients."""

from pathlib import Path
from typing import Any, Dict, Iterable

from b_hfl.schemas.file_system_schema import FolderHierarchy
from b_hfl.typing.common_types import FitRes


def get_residuals(
    residuals_manager: Dict[Any, Dict[Any, FitRes]],
    path_dict: FolderHierarchy,
    leaf_to_root: bool,
) -> Iterable[FitRes]:
    """Get the residuals for a given client."""
    client_id = path_dict.path / "leaf_to_root" / f"{leaf_to_root}"
    if client_id not in residuals_manager:
        residuals_manager[client_id] = {}
    return residuals_manager[client_id].values()


def send_residuals(
    residuals_manager: Dict[Any, Dict[Any, FitRes]],
    path_dict: FolderHierarchy,
    send_to: Path,
    residual: FitRes,
    leaf_to_root: bool,
) -> None:
    """Send the residuals from one client to another."""
    send_to_id = send_to / "leaf_to_root" / f"{leaf_to_root}"
    send_from_id = path_dict.path
    if send_to_id not in residuals_manager:
        residuals_manager[send_to_id] = {}
    residuals_manager[send_to_id][send_from_id] = residual
