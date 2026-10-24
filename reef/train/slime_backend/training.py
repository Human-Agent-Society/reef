"""Slime training workers, initialized independently of the inference service."""

from __future__ import annotations

import logging
import math
from typing import Any

from reef.runtime.deployment import (
    NO_WEIGHT_TRANSFER_PROTOCOL,
    WEIGHT_FILES_PROTOCOL,
    DeploymentResources,
    NativeWeightTransferSession,
    TrainingService,
    WeightTransferSession,
)
from reef.runtime.executor import Executor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import TrainingBackend
from reef.train.deployment import NO_INFERENCE_BACKEND
from reef.train.slime_backend.reef_adapters.bridge import (
    BridgePreparation,
    create_train_groups,
    create_training_backend,
)
from reef.train.slime_backend.reef_adapters.train_groups import DEFAULT_TRAIN_RPC_TIMEOUT_S
from reef.train.slime_backend.resources import SlimeDeploymentResources


#: The receiver control protocol Slime's updaters attach to, per inference backend.
WEIGHT_TRANSFER_PROTOCOLS = {"sglang": "slime-sglang-control-v2", "vllm": "reef-vllm-control-v1"}


class SlimeTrainingService(TrainingService):
    """Own actor/critic workers and a sender attachment, never the coordinator.

    The workers send ``inference_backend``'s native protocol, else weight
    files, else ``reef-no-weight-transfer-v1``; ``inference.backend: none`` has
    only the last. With the no-transfer session, no updater connects to a receiver
    and no weights leave the workers: the backend records each version in the
    session's receiver instead.
    """

    def __init__(
        self,
        args: Any,
        *,
        preparation: BridgePreparation,
        loss_family_config: object | None,
        inference_backend: str = "sglang",
        train_rpc_timeout_s: float | None = None,
    ) -> None:
        if inference_backend == NO_INFERENCE_BACKEND:
            native: tuple[str, ...] = ()
        elif inference_backend in WEIGHT_TRANSFER_PROTOCOLS:
            native = (WEIGHT_TRANSFER_PROTOCOLS[inference_backend],)
        else:
            raise ValueError(
                f"Slime weight transfer has no receiver protocol for inference backend {inference_backend!r}"
            )
        self._native_protocols = native
        if train_rpc_timeout_s is None:
            train_rpc_timeout_s = DEFAULT_TRAIN_RPC_TIMEOUT_S
        elif (
            isinstance(train_rpc_timeout_s, bool)
            or not isinstance(train_rpc_timeout_s, (int, float))
            or not math.isfinite(train_rpc_timeout_s)
            or train_rpc_timeout_s <= 0
        ):
            raise ValueError("training.timeout-s must be a positive finite number")
        self.train_rpc_timeout_s = train_rpc_timeout_s
        self.args = args
        self.preparation = preparation
        self.loss_family_config = loss_family_config
        self._actor_group: Any = None
        self._critic_group: Any = None
        self._backend: TrainingBackend | None = None
        self._session_id: str | None = None
        # The receiver of a no-transfer session; None before attachment and for a native session.
        self._version_receiver: Executor | None = None
        #: Set by a file session: the coordinator delivers exported weight files and no native sender exists.
        self.delivers_weight_files = False
        self._started = False
        self._closed = False

    @property
    def weight_transfer_protocols(self) -> tuple[str, ...]:
        """The native transfer first, then weight files for a receiver that the workers cannot reach, then none.

        As files, full-weight training delivers Hugging Face checkpoints and
        LoRA training delivers each scenario's PEFT adapter.
        ``inference.backend: none`` has neither: its receiver takes only the no-transfer protocol.
        """
        files = (WEIGHT_FILES_PROTOCOL,) if self._native_protocols else ()
        # Last, so that a receiver with engines keeps its native protocol.
        return (*self._native_protocols, *files, NO_WEIGHT_TRANSFER_PROTOCOL)

    def start(self, resources: DeploymentResources) -> None:
        if self._started or self._closed:
            raise RuntimeError("training service can only be started once")
        if not isinstance(resources, SlimeDeploymentResources) or not resources.placement_groups:
            raise ValueError("Slime training requires its supplied model reservations")
        self._started = True
        self._actor_group, self._critic_group = create_train_groups(
            self.args,
            resources.placement_groups,
            rollout_manager=None,
            train_rpc_timeout_s=self.train_rpc_timeout_s,
        )
        # Ongoing observation belongs to the backend in the coordinator
        # process. Watching these driver-side handles would report intentional
        # release_train retirement as a deployment failure after serialization.

    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        """Attach the native sender, prepare file exports or a no-transfer session, after the workers are initialized."""
        if self._actor_group is None or self._closed:
            raise RuntimeError("start training workers before attaching their weight transport")
        rollout_manager: Any = None
        version_receiver: Executor | None = None
        if isinstance(session, NativeWeightTransferSession) and session.protocol in self._native_protocols:
            receiver = session.receiver
            if not isinstance(receiver, RayExecutor) or len(receiver.workers) != 1:
                raise ValueError("Slime weight transport requires one native Ray receiver")
            # The pinned native updaters consume one borrowed actor handle. Reef
            # already fenced/offloaded the receiver; attachment only configures
            # sender layout and transport, never its inference lifecycle.
            rollout_manager = receiver.workers[0]
        elif not isinstance(session, NativeWeightTransferSession) and session.protocol == WEIGHT_FILES_PROTOCOL:
            # No receiver handle: the workers only report their parallel layout, and export files on request.
            pass
        elif isinstance(session, NativeWeightTransferSession) and session.protocol == NO_WEIGHT_TRANSFER_PROTOCOL:
            # No updater connects and no weights leave the workers; they still report their
            # parallel layout, which the batch split reads, and the backend records versions here.
            version_receiver = session.receiver
        else:
            raise ValueError("Slime training received an incompatible weight transfer protocol")
        if self._session_id is not None:
            if self._session_id == session.session_id:
                return
            raise RuntimeError("training workers cannot reuse a different deployment's weight transfer session")
        for group in (self._actor_group, self._critic_group):
            if group is not None:
                group.set_rollout_manager(rollout_manager)
        self.delivers_weight_files = session.protocol == WEIGHT_FILES_PROTOCOL
        self._version_receiver = version_receiver
        self._session_id = session.session_id

    def backend(self) -> TrainingBackend:
        if self._actor_group is None or self._session_id is None or self._closed:
            raise RuntimeError("training backend requires initialized workers and attached weight transport")
        if self._backend is None:
            self._backend = create_training_backend(
                self.args,
                self._actor_group,
                self._critic_group,
                preparation=self.preparation,
                loss_family_config=self.loss_family_config,
                train_rpc_timeout_s=self.train_rpc_timeout_s,
                version_receiver=self._version_receiver,
                delivers_weight_files=self.delivers_weight_files,
            )
        return self._backend

    def check_health(self) -> None:
        self.poll()
        for group in (self._actor_group, self._critic_group):
            if group is not None:
                group.check_health()

    def poll(self) -> None:
        if self._actor_group is None or self._closed:
            raise RuntimeError("training service is not running")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for group in (self._critic_group, self._actor_group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
                    logging.getLogger(__name__).exception("Failed to release Slime training workers")
        if errors:
            raise errors[0]
