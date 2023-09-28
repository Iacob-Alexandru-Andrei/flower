""""A deterministic client manager."""
import random
from logging import WARNING
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from b_hfl.utils.utils import hash_combine


class DeterministicClientManager(SimpleClientManager):
    """A deterministic client manager.

    Samples clients in the same order every time based on the seed.
    """

    def __init__(
        self,
        seed: int,
    ) -> None:
        super().__init__()
        self.sample_round = 0
        self.seed = seed

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        cids = list(self.clients)
        # Shuffle the list of clients
        random.seed(hash_combine(self.seed, self.sample_round))

        self.sample_round += 1

        available_cids = random.sample(cids, num_clients)

        if num_clients > len(available_cids):
            log(
                WARNING,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        client_list = [self.clients[cid] for cid in available_cids]
        print("Sampled the following clients: ", available_cids)
        return client_list
