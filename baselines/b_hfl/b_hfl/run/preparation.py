"""Partition the dataset in a hierarhical fashion."""
import shutil
from pathlib import Path

import hydra
from hydra.utils import call
from omegaconf import DictConfig

from b_hfl.utils.dataset_preparation import (
    child_map_to_file_hierarchy,
    hierarchy_to_edge_assingment,
)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def partition(cfg: DictConfig) -> None:
    """Partition the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    ## 1. print parsed config
    print("Partitioning the dataset")
    # Short-circuit if the dataset has already been partitioned
    if cfg.task.data.preparation.rebuild_partition is False:
        return

    src_folder = Path(cfg.task.data.preparation.src_partition_path)
    out_folder = Path(cfg.task.data.preparation.dataset_path)

    if out_folder.exists():
        shutil.rmtree(out_folder)

    if cfg.task.data.preparation.mkdir:
        out_folder.mkdir(parents=True, exist_ok=True)

    client_folder_hierarchy = hierarchy_to_edge_assingment(
        src_folder=src_folder,
        assign_flat=call(cfg.task.data.preparation.get_client_edge_assignment),
        assign_rec=call(cfg.task.data.preparation.get_node_edge_assignments),
    )
    child_map_to_file_hierarchy(
        logical_mapping=client_folder_hierarchy, in_root=src_folder, out_root=out_folder
    )


if __name__ == "__main__":
    partition()
