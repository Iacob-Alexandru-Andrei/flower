"""Partition the dataset in a hierarhical fashion."""
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset_preparation import (
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
    print(OmegaConf.to_yaml(cfg))
    # Short-circuit if the dataset has already been partitioned
    if cfg.task.data.preparation.rebuild_partition is False:
        return

    src_folder = Path(cfg.task.data.preparation.src_partition_path)
    out_folder = Path(cfg.task.data.preparation.dataset_path)

    if cfg.task.data.preparation.mkdir:
        out_folder.mkdir(parents=True, exist_ok=True)

    client_folder_hierarchy = hierarchy_to_edge_assingment(
        src_folder=src_folder,
        assign_flat=cfg.task.data.preparation.get_client_edge_assignment,
        assign_rec=cfg.task.data.preparation.get_node_edge_assignments,
    )
    child_map_to_file_hierarchy(
        logical_mapping=client_folder_hierarchy, in_root=src_folder, out_root=out_folder
    )


if __name__ == "__main__":
    partition()
