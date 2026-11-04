"""Training backends as services (RFC #1146).

A deployment selects a backend service with ``training.backend: reef.train.remote:RemoteDeployment`` and
``training.options: {url, base, state_dir, token_env}``. ``contract`` is the HTTP contract, ``client`` the training
component that calls a service, and ``server`` the generic server that puts a ``TrainingBackend`` behind the
contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.component import TrainingComponent, TrainingComponentFactory
from reef.train.deployment import InProcessTrainingDeployment
from reef.train.remote.client import RemoteConfig


class RemoteDeployment(InProcessTrainingDeployment):
    """A training backend service as Reef's training backend. Reef's process runs no training."""

    component_type = "reef.train.remote:remote_component_factory"
    requires_local_model = False


class RemoteComponentFactory(TrainingComponentFactory):
    """Build the ``RemoteTrainingComponent`` that ``training.options`` describe."""

    kind = RemoteDeployment.component_type

    def config_type(self) -> type:
        return RemoteConfig

    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        from reef.train.remote.client import remote_component

        return remote_component(config, environ)


remote_component_factory = RemoteComponentFactory()

__all__ = ["RemoteComponentFactory", "RemoteDeployment", "remote_component_factory"]
