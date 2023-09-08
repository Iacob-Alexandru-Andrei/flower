"""Pydantic schemas for the file system hierarchy."""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class ClientFolderHierarchy(BaseModel):
    """Pydantic schema for representing the relation between clients."""

    name: str
    children: List["ClientFolderHierarchy"] = []


ClientFolderHierarchy.update_forward_refs()


class FolderHierarchy(BaseModel):
    """Pydantic schema for the static structure of the file system hierarchy."""

    root: Path
    parent: Optional["FolderHierarchy"]
    parent_path: Optional[Path]
    path: Path
    children: List["FolderHierarchy"] = []


FolderHierarchy.update_forward_refs()


class ConfigFolderHierarchy(BaseModel):
    """Pydantic schema for configs in a file system."""

    path: Path
    on_fit_configs: List[BaseModel]
    on_evaluate_configs: List[BaseModel]
    children: List["ConfigFolderHierarchy"] = []


ConfigFolderHierarchy.update_forward_refs()
