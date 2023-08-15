"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from typing import Dict, List, Optional, Tuple

from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import wandb


class LoggingFedAvg(FedAvg):
    """Add per-client configs to FedAvg and W&B."""

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        if self.on_fit_config_fn is not None:
            client_true_ids = [
                client.get_properties({}, None)["true_id"]  # type: ignore
                for client in clients
            ]
            return [
                (
                    client,
                    FitIns(
                        parameters,
                        self.on_fit_config_fn(server_round, true_id),  # type: ignore
                    ),
                )
                for client, true_id in zip(clients, client_true_ids)
            ]
        else:
            fit_ins = FitIns(parameters, {})
            return [(client, fit_ins) for client in clients]

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict]]:
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result
        wandb.log({"centralised_loss": loss})

        return loss, metrics
