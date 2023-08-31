"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitIns, FitRes, GetPropertiesIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class LoggingFedAvg(FedAvg):
    """Add per-client configs to FedAvg and W&B."""

    def __init__(self, save_files: Callable[[int], None], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_files = save_files

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

        def get_properties(client):
            return client.get_properties(GetPropertiesIns({}), None)

        # Return client/config pairs
        if self.on_fit_config_fn is not None:
            client_true_ids = [
                get_properties(client).properties.properties["true_id"]  # type: ignore
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
        """Evaluate and log metrics to W&B."""
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result

        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit and save files."""
        res = super().aggregate_fit(server_round, results, failures)
        os.sync()
        print("Saving files for the round")
        self.save_files(server_round)
        return res
