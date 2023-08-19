"""Cerberus Config schema for the FEMNIST dataset."""
from typing import Any, Dict

client_train_schema_recursive: Dict[str, Any] = {
    "num_rounds": {"type": "integer", "required": True},
    "fit_fraction": {"type": "float", "required": True},
    "train_children": {"type": "boolean", "required": True},
    "train_chain": {"type": "boolean", "required": True},
    "train_proxy": {"type": "boolean", "required": True},
}

client_test_schema_recursive: Dict[str, Any] = {
    "eval_fraction": {"type": "float", "required": True},
    "test_children": {"type": "boolean", "required": True},
    "test_chain": {"type": "boolean", "required": True},
    "test_proxy": {"type": "boolean", "required": True},
}

dataloader_schema: Dict[str, Any] = {
    "batch_size": {"type": "integer", "required": True},
    "num_workers": {"type": "integer", "required": True},
    "shuffle": {"type": "boolean", "required": True},
    "test": {"type": "boolean", "required": True},
}


net_config_schema: Dict[str, Any] = {"type": "dict", "nullable": True}

parameter_config_schema: Dict[str, Any] = {"type": "dict", "nullable": True}

run_config_FEMNIST_schema: Dict[str, Any] = {
    "epochs": {"type": "integer", "required": True},
    "client_learning_rate": {"type": "float", "required": True},
    "weight_decay": {"type": "float", "required": True},
}

recursive_client_FEMNIST_train_schema: Dict[str, Any] = {
    "rounds": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "client_config": {
                    "type": "dict",
                    "required": True,
                    "schema": client_train_schema_recursive,
                },
                "dataloader_config": {
                    "type": "dict",
                    "required": True,
                    "schema": dataloader_schema,
                },
                "parameter_config": {
                    "type": "dict",
                    "required": True,
                    "nullable": True,
                    # "schema": parameter_config_schema,
                },
                "net_config": {
                    "type": "dict",
                    "required": True,
                    "nullable": True,
                    # "schema": net_config_schema,
                },
                "run_config": {
                    "type": "dict",
                    "required": True,
                    "schema": run_config_FEMNIST_schema,
                },
            },
        },
    }
}

recursive_client_FEMNIST_test_schema: Dict[str, Any] = {
    "rounds": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "client_config": {
                    "type": "dict",
                    "required": True,
                    "schema": client_test_schema_recursive,
                },
                "dataloader_config": {
                    "type": "dict",
                    "required": True,
                    "schema": dataloader_schema,
                },
                "parameter_config": {
                    "type": "dict",
                    "required": True,
                    "nullable": True,
                    # "schema": parameter_config_schema,
                },
                "net_config": {
                    "type": "dict",
                    "required": True,
                    "nullable": True,
                    # "schema": net_config_schema,
                },
                "run_config": {
                    "type": "dict",
                    "required": True,
                    "schema": run_config_FEMNIST_schema,
                },
            },
        },
    }
}


def get_recursive_FEMNIST_client_schema(test: bool) -> Dict[str, Any]:
    """Get the recursive client schema for a given config name."""
    if not test:
        return recursive_client_FEMNIST_train_schema
    else:
        return recursive_client_FEMNIST_test_schema
