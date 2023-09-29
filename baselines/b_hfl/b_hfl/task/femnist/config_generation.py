"""Generate the schema and client configs for the femnist task."""

from typing import Callable, List, Tuple

from omegaconf import DictConfig

from b_hfl.schema.client_schema import (
    ClientTestConfig,
    ClientTrainConfig,
    RecClientTestConf,
    RecClientTrainConf,
)
from b_hfl.schema.file_system_schema import ConfigFolderHierarchy, FolderHierarchy
from b_hfl.utils.dataset_preparation import get_configs_per_level
from b_hfl.utils.utils import get_children_at_given_level, get_parent_at_given_level


def get_level_config_generator(
    path_dict: FolderHierarchy, cfg: DictConfig
) -> Callable[
    [FolderHierarchy],
    Tuple[List[RecClientTrainConf], List[RecClientTestConf]],
]:
    """Generate config for each level of the hierarchy."""
    # Train param lists
    num_rounds: List[int] = cfg.task.client.client_train.num_rounds
    train_children: List[bool] = cfg.task.client.client_train.train_children
    fit_fraction: List[float] = cfg.task.client.client_train.fit_fraction
    train_chain: List[bool] = cfg.task.client.client_train.train_chain
    epochs: List[int] = cfg.task.client.client_train.epochs
    train_proxy: List[bool] = cfg.task.client.client_train.train_proxy
    track_parameter_changes: List[
        bool
    ] = cfg.task.client.client_train.track_parameter_changes

    # Residuals
    root_to_leaf_residuals: List[List[int]] = [
        [path_dict.max_level + 1 + level for level in node]
        for node in cfg.task.client.residuals.root_to_leaf_residuals
    ]

    leaf_to_root_residuals: List[
        List[int]
    ] = cfg.task.client.residuals.leaf_to_root_residuals

    # Node opt:
    anc_alpha: List[float] = cfg.task.client.node_opt.anc_alpha
    desc_alpha: List[float] = cfg.task.client.node_opt.desc_alpha

    # Test param lists
    test_children: List[bool] = cfg.task.client.client_test.test_children
    eval_fraction: List[float] = cfg.task.client.client_test.eval_fraction
    test_chain: List[bool] = cfg.task.client.client_test.test_chain
    test_proxy: List[bool] = cfg.task.client.client_test.test_proxy

    def recursive_descent_configs(
        path_dict: FolderHierarchy,
    ) -> Tuple[List[RecClientTrainConf], List[RecClientTestConf]]:
        level = path_dict.level
        all_children_at_level = [
            child.path
            for child_level in root_to_leaf_residuals[level]
            for child in get_children_at_given_level(path_dict, child_level)
        ]

        train_config = RecClientTrainConf(
            client_config=ClientTrainConfig(
                num_rounds=num_rounds[level],
                fit_fraction=fit_fraction[level],
                train_children=train_children[level],
                train_chain=train_chain[level],
                train_proxy=train_proxy[level],
                track_parameter_changes=track_parameter_changes[level],
                root_to_leaf_residuals=all_children_at_level,
                leaf_to_root_residuals=[
                    get_parent_at_given_level(path_dict, parent_level).path
                    for parent_level in leaf_to_root_residuals[level]
                ],
            ),
            dataloader_config={
                "batch_size": cfg.task.client.batch_size,
                "num_workers": cfg.task.client.num_workers,
                "shuffle": cfg.task.client.shuffle,
                "test": False,
            },
            run_config={
                "epochs": epochs[level],
                "client_learning_rate": cfg.task.client.client_learning_rate,
                "weight_decay": cfg.task.client.weight_decay,
            },
            parameter_config={},
            net_config={},
            state_loader_config={},
            state_generator_config={},
            anc_node_optimizer_config={"alpha": anc_alpha[level]},
            desc_node_optimizer_config={"alpha": desc_alpha[level]},
            get_parameters_config={},
            dataset_generator_config={},
        )
        test_config = RecClientTestConf(
            client_config=ClientTestConfig(
                eval_fraction=eval_fraction[level],
                test_children=test_children[level],
                test_chain=test_chain[level],
                test_proxy=test_proxy[level],
            ),
            dataloader_config={
                "batch_size": cfg.task.client.test_batch_size,
                "num_workers": cfg.task.client.test_num_workers,
                "shuffle": False,
                "test": True,
            },
            run_config={},
            parameter_config={},
            net_config={},
            state_loader_config={},
            state_generator_config={},
            anc_node_optimizer_config={"alpha": 0.5},
            desc_node_optimizer_config={"alpha": 0.5},
            get_parameters_config={},
            dataset_generator_config={},
        )
        return [train_config] * num_rounds[level], [test_config] * num_rounds[level]

    return recursive_descent_configs


def get_generate_level_configs(
    path_dict: FolderHierarchy, cfg: DictConfig
) -> ConfigFolderHierarchy:
    """Write the configs for each level of the hierarchy."""
    return get_configs_per_level(
        path_dict,
        get_level_config_generator(path_dict, cfg),
    )
