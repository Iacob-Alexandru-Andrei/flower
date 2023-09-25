"""Partition the dataset in a hierarhical fashion."""
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

default_femnist_recursive_fit_config = [
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

default_femnist_recursive_evaluate_config = [
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

if __name__ == "__main__":
    src_folder = Path(
        "/home/aai30/nfs-share/b_hfl/femnist_local/femnist/femnist/client_data_mappings/fed_natural"
    )
    # pylint: disable=line-too-long
    out_folder = Path(
        "/home/aai30/nfs-share/b_hfl/femnist_local/femnist/femnist/client_data_mappings/hierarchical_test"
    )

    child_map_to_file_hierarchy(
        ClientFolderHierarchy(
            **{
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
            }
        ),
        src_folder,
        out_folder,
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

    Path(cfg.task.data.preparation.src_partition_path)
    Path(cfg.task.data.preparation.dataset_path)
