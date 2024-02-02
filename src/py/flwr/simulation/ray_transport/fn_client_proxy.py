# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ray-based Flower ClientProxy implementation."""

from logging import ERROR
from typing import Dict, Optional, cast


from flwr import common
from flwr.client import Client, ClientFn
from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.node_state import NodeState
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

class FnClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn: ClientFn, cid: str):
        super().__init__(cid)
        self.client_fn = client_fn

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        return launch_and_get_properties(self.client_fn, self.cid, ins)
      

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        return launch_and_get_parameters(self.client_fn, self.cid, ins)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        return launch_and_fit(self.client_fn, self.cid, ins)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        return launch_and_evaluate(self.client_fn, self.cid, ins)

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)


# @ray.remote
def launch_and_get_properties(
    client_fn: ClientFn, cid: str, get_properties_ins: common.GetPropertiesIns
) -> common.GetPropertiesRes:
    """Exectue get_properties remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )


# @ray.remote
def launch_and_get_parameters(
    client_fn: ClientFn, cid: str, get_parameters_ins: common.GetParametersIns
) -> common.GetParametersRes:
    """Exectue get_parameters remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )


# @ray.remote
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: common.FitIns
) -> common.FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )


# @ray.remote
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: common.EvaluateIns
) -> common.EvaluateRes:
    """Exectue evaluate remotely."""
    client: Client = _create_client(client_fn, cid)
    return maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )


def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    # Materialize client
    return client_fn(cid)