from os import path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast
import hydra
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import client
from b_hfl.schema.client_schema import (
    ClientTestConfig,
    ClientTrainConfig,
    RecClientTrainConf,
    RecClientTestConf,
)
from b_hfl.schema.file_system_schema import FolderHierarchy
from b_hfl.utils.dataset_preparation import get_configs_per_level
from b_hfl.node_optimizer.weighted_fed_avg import WeightedAvgConfig


class DataloaderConfig(BaseModel):
    """Pydantic schema for dataloader configuration."""

    batch_size: int
    num_workers: int
    shuffle: bool
    test: bool


class RunConfig(BaseModel):
    """Pydantic schema for run configuration."""

    epochs: int
    client_learning_rate: float
    weight_decay: float


class WeightedAvgClientFemnistTrainConf(RecClientTrainConf):
    """Pydantic schema for weighted average client train configuration."""

    dataloader_config: DataloaderConfig
    anc_node_optimizer_config: WeightedAvgConfig
    desc_node_optimizer_config: WeightedAvgConfig
    run_config: RunConfig


class WeightedAvgClientFemnistTestConf(RecClientTestConf):
    """Pydantic schema for weighted average client train configuration."""

    dataloader_config: DataloaderConfig
    anc_node_optimizer_config: WeightedAvgConfig
    desc_node_optimizer_config: WeightedAvgConfig


def get_recursive_client_train_configs() -> Callable:
    """Return a callable that returns a recursive client train config.

    For both validation and instantiation.
    """
    return lambda: WeightedAvgClientFemnistTrainConf


def get_recursive_client_test_configs() -> Callable:
    """Return a callable that returns a recursive client test config.

    For both validation and instantiation.
    """
    return lambda: WeightedAvgClientFemnistTestConf


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


def level_config_generator(
    path_dict: FolderHierarchy, cfg: DictConfig
) -> Callable[
    [FolderHierarchy],
    Tuple[
        List[WeightedAvgClientFemnistTestConf], List[WeightedAvgClientFemnistTestConf]
    ],
]:
    """Generate config for each level of the hierarchy.

    Parameters
    ----------
    path_dict : FolderHierarchy
        A dictionary that stores the path to each client.
    cfg : DictConfig
        An omegaconf object that stores the hydra config.

    Returns
    -------
    Callable[[FolderHierarchy], Tuple[List[WeightedAvgClientFemnistTestConf], Tuple[WeightedAvgClientFemnistTestConf]]]
        A function that returns a tuple of train and test configs for each level.
    """
    cnt_level = 0

    # Train param lists
    num_rounds: List[int] = cfg.task.client.client_train.num_rounds
    train_children: List[bool] = cfg.task.client.client_train.train_children
    fit_fraction: List[float] = cfg.task.client.client_train.fit_fraction
    train_chain: List[bool] = cfg.task.client.client_train.train_chain
    train_proxy: List[bool] = cfg.task.client.client_train.train_proxy

    # Residuals
    root_to_leaf_residuals: List[List[Path]] = [
        (
            [folder.path for folder in path_dict.levels_to_folder[-cast(int, idx) - 1]]
            if idx is not None
            else []
        )
        for idx in cfg.task.client.residuals.root_to_leaf_residuals
    ]

    leaf_to_root_residuals: List[List[Path]] = [
        (
            [folder.path for folder in path_dict.levels_to_folder[cast(int, idx)]]
            if idx is not None
            else []
        )
        for idx in cfg.task.client.residuals.leaf_to_root_residuals
    ]

    # Node opt:
    anc_alpha: List[float] = cfg.task.client.node_opt.anc_alpha
    desc_alpha: List[float] = cfg.task.client.node_opt.desc_alpha

    # Test param lists
    test_children: List[bool] = cfg.task.client.client_test.test_children
    eval_fraction: List[float] = cfg.task.client.client_test.eval_fraction
    test_chain: List[bool] = cfg.task.client.client_test.test_chain
    test_proxy: List[bool] = cfg.task.client.client_test.test_proxy
    level = 0

    def recursive_descent_configs(
        path_dict: FolderHierarchy,
    ) -> Tuple[
        List[WeightedAvgClientFemnistTestConf], List[WeightedAvgClientFemnistTestConf]
    ]:
        nonlocal level

        train_config = WeightedAvgClientFemnistTrainConf(
            client_config=ClientTrainConfig(
                num_rounds=num_rounds[level],
                fit_fraction=fit_fraction[level],
                train_children=train_children[level],
                train_chain=train_chain[level],
                train_proxy=train_proxy[level],
                root_to_leaf_residuals=root_to_leaf_residuals[level],
                leaf_to_root_residuals=[],
            ),
            dataloader_config=DataloaderConfig(
                batch_size=cfg.task.client.batch_size,
                num_workers=cfg.task.client.num_workers,
                shuffle=cfg.task.client.shuffle,
                test=False,
            ),
            run_config=RunConfig(
                epochs=1,
                client_learning_rate=cfg.task.client.client_learning_rate,
                weight_decay=cfg.task.client.weight_decay,
            ),
            parameter_config={},
            net_config={},
            state_loader_config={},
            state_generator_config={},
            anc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
            desc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
            get_parameters_config={},
            dataset_generator_config={},
        )
        level += 1


def generate_level_configs(path_dict: FolderHierarchy, cfg: DictConfig) -> None:
    default_femnist_recursive_fit_config = WeightedAvgClientFemnistTrainConf(
        client_config=ClientTrainConfig(
            num_rounds=1,
            fit_fraction=1.0,
            train_children=True,
            train_chain=False,
            train_proxy=False,
            root_to_leaf_residuals=[],
            leaf_to_root_residuals=[],
        ),
        dataloader_config=DataloaderConfig(
            batch_size=cfg.task.client.batch_size,
            num_workers=cfg.task.client.num_workers,
            shuffle=cfg.task.client.shuffle,
            test=False,
        ),
        run_config=RunConfig(
            epochs=1,
            client_learning_rate=cfg.task.client.client_learning_rate,
            weight_decay=cfg.task.client.weight_decay,
        ),
        parameter_config={},
        net_config={},
        state_loader_config={},
        state_generator_config={},
        anc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
        desc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
        get_parameters_config={},
        dataset_generator_config={},
    )

    default_femnist_recursive_evaluate_config = WeightedAvgClientFemnistTestConf(
        client_config=ClientTestConfig(
            eval_fraction=1.0,
            test_children=False,
            test_chain=True,
            test_proxy=False,
        ),
        dataloader_config=DataloaderConfig(
            batch_size=cfg.task.client.test_batch_size,
            num_workers=cfg.task.client.test_num_workers,
            shuffle=False,
            test=True,
        ),
        run_config={},
        parameter_config=None,
        net_config=None,
        state_loader_config=None,
        state_generator_config=None,
        anc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
        desc_node_optimizer_config=WeightedAvgConfig(alpha=0.5),
        get_parameters_config=None,
        dataset_generator_config=None,
    )
    level_config_generator = call(cfg)
    get_configs_per_level(path_dict, level_config_generator)
