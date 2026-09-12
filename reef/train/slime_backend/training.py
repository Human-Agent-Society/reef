"""Slime training component that attaches to Reef-owned inference resources."""

from __future__ import annotations

import logging
from typing import Any

import ray

from reef.runtime.deployment import DeploymentResources, InferenceConnection
from reef.runtime.executor.failure import ExecutorFailedError, ExecutorFailure
from reef.train.slime_backend.reef_adapters.bridge import BridgePreparation, start_bridge
from reef.train.slime_backend.resources import INFERENCE_PROTOCOL, RayHealthProbe, SlimeDeploymentResources


class SlimeTrainingService:
    """Own training workers and the bridge, never supplied inference objects."""

    def __init__(
        self,
        args: Any,
        *,
        preparation: BridgePreparation,
        loss_family_config: object | None,
        actor_name: str,
        namespace: str,
        separate_inference: bool,
    ) -> None:
        self.args = args
        self.preparation = preparation
        self.loss_family_config = loss_family_config
        self.actor_name = actor_name
        self.namespace = namespace
        self.inference_protocol = INFERENCE_PROTOCOL if separate_inference else None
        self._bridge: Any = None
        self._started = False
        self._closed = False
        self._probe = RayHealthProbe()
        self._worker_failure: ExecutorFailure | None = None

    def start(self, resources: DeploymentResources, inference: InferenceConnection | None) -> None:
        if self._started or self._closed:
            raise RuntimeError("training service can only be started once")
        if not isinstance(resources, SlimeDeploymentResources):
            raise ValueError("Slime training requires its supplied deployment resources")
        serving = None
        placement_groups = None
        if self.inference_protocol is not None:
            if inference is None or inference.protocol != self.inference_protocol or not resources.placement_groups:
                raise ValueError("Slime training requires an existing inference connection and model reservations")
            serving = inference.control
            placement_groups = resources.placement_groups
        elif inference is not None or resources.placement_groups:
            raise ValueError("combined Slime compatibility mode cannot own supplied inference resources")
        self._started = True
        self._bridge = start_bridge(
            self.args,
            loss_family_config=self.loss_family_config,
            actor_name=self.actor_name,
            namespace=self.namespace,
            preparation=self.preparation,
            serving=serving,
            placement_groups=placement_groups,
            failure_listener=self if self.inference_protocol is not None else None,
        )

    def check_health(self) -> None:
        if self._worker_failure is not None:
            raise ExecutorFailedError(self._worker_failure)
        if self._bridge is None or self._closed:
            raise RuntimeError("training service is not running")
        # Actor construction can restore checkpoints and republish weights.
        # The launcher enforces training.ready-timeout for the whole startup;
        # a short health RPC timeout would interrupt a healthy recovery.
        result = ray.get(self._bridge.health.remote())
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(f"training bridge failed its startup health check: {result!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bridge is not None:
            try:
                ray.get(self._bridge.shutdown.remote(), timeout=90)
            except Exception:
                if self.inference_protocol is None:
                    raise
                logging.getLogger(__name__).exception("Bridge shutdown failed; retiring its owned process groups")
            finally:
                ray.kill(self._bridge, no_restart=True)

    def poll(self) -> None:
        if self._worker_failure is not None:
            raise ExecutorFailedError(self._worker_failure)
        if self._bridge is None or self._closed:
            raise RuntimeError("training service is not running")
        self._probe.poll(self._bridge, "health")

    def on_executor_failure(self, failure: ExecutorFailure) -> None:
        self._worker_failure = failure
