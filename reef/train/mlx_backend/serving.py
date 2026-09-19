"""The serving half of the in-process MLX runtime.

Reef splits a deployment into a training runtime and an inference runtime so
that the two can live in different processes. An MLX deployment keeps both in
one, but the split is still real here: exactly one of the two owns the model
parameters, the gate that decides when a request may read them, and the record
of which version is answering. That owner is this class, and training reaches
the engine only through it.

The gate is not a lock around a single call. Activation deliberately leaves it
closed and returns: between swapping the weights and Reef committing the new
head, a request would freeze the old artifact and be answered by the new
weights, which the weight surface rejects as a runtime-load mismatch. Reef
reopens it by committing. So the interface is an explicit ``hold``/``release``
pair rather than a context manager.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reef.runtime.interfaces import ActivatedModel, InferenceHandler, InferenceRuntime, RuntimeContractError
from reef.surface.base import WeightRuntime
from reef.train.mlx_backend.inference import MLXInferenceBackend

logger = logging.getLogger(__name__)


class MLXServingRuntime(InferenceRuntime, WeightRuntime):
    """Owns the resident model, the admission gate, and the served version."""

    def __init__(self, engine: Any, *, adapter_name: str = "reef-mlx", inference_timeout_s: float = 300.0) -> None:
        # The base owns the endpoint identity, the drain timeout and the
        # admission gate. Requests arrive here as calls rather than over the
        # wire, so the URL is nominal.
        super().__init__(base_url="mlx://local", inference_timeout_s=inference_timeout_s)
        self._engine = engine
        self._adapter_name = adapter_name
        self._backend: Any = None
        #: Trained snapshots awaiting Reef's select-or-reject.
        self._pending: dict[str, Any] = {}
        #: What the engine holds. Advances the moment weights are swapped.
        self._serving_runtime_load_id: str | None = engine.next_runtime_load_id()
        #: What Reef has made available to new requests. Lags the above between
        #: activation and the durable commit.
        self._committed_runtime_load_id: str | None = self._serving_runtime_load_id

    # -- The engine, for the training half that shares it.

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def inference_backend(self) -> MLXInferenceBackend:
        if self._backend is None:
            self._backend = MLXInferenceBackend(self)
        return self._backend

    @property
    def inference_handler(self) -> InferenceHandler:
        """Reef's name for the object that executes a request."""
        return self.inference_backend

    # -- Served identity.

    def serving_runtime_load_id(self) -> str | None:
        return self._serving_runtime_load_id

    def serving_adapter_name(self) -> str | None:
        # The adapter is resident in this process rather than addressed by
        # name over a wire, but naming it keeps the served identity explicit
        # in records and in the weight surface.
        return self._adapter_name

    def current_runtime_load_id(self) -> str | None:
        """The version Reef has actually made available to new inference.

        Between activation and Reef's durable commit this deliberately lags
        :meth:`serving_runtime_load_id`: the engine already holds the new
        weights, but no request may be admitted against them yet.
        """
        return self._committed_runtime_load_id

    # -- The gate.

    async def acquire_inference(self) -> Any:
        """Wait until this runtime may freeze its weights and serve a request."""
        return await self._inference_admission.acquire()

    @property
    def inference_admission_status(self) -> dict[str, Any]:
        """Whether serving is admitting, and what it is waiting on."""
        return dict(self._inference_admission.status)

    def hold(self) -> None:
        """Stop admitting requests and wait for in-flight generation to drain."""
        self._inference_admission.close(wait=True, timeout=self.inference_timeout_s)

    def release(self) -> None:
        """Admit requests again."""
        self._inference_admission.open()

    # -- Weight changes. The caller holds the gate around these.

    def swap_adapter(self, snapshot: Any) -> str:
        """Make a trained snapshot the resident weights, and mint its version."""
        self._engine.apply_adapter(snapshot)
        self._serving_runtime_load_id = self._engine.next_runtime_load_id()
        return self._serving_runtime_load_id

    def restore_checkpoint(self, artifact: Any) -> str:
        """Roll serving back to a published adapter."""
        local_path = artifact.local_path
        if local_path is None:
            raise RuntimeContractError("mlx rollback requires a materialized adapter")
        self.hold()
        try:
            self._engine.load_adapter(Path(local_path))
            self._serving_runtime_load_id = self._engine.next_runtime_load_id()
        finally:
            self.release()
        return self._serving_runtime_load_id

    def stage_candidate(self, candidate_id: str, snapshot: Any) -> None:
        """Hold a trained snapshot until Reef selects or rejects it."""
        self._pending[candidate_id] = snapshot

    def staged_candidate(self, candidate_id: str) -> Any:
        """The snapshot staged under this id, or None."""
        return self._pending.get(candidate_id)

    def discard_candidate(self, candidate_id: str) -> None:
        """Drop a rejected candidate; serving never saw its weights."""
        self._pending.pop(candidate_id, None)

    def activate_candidate(self, candidate: Any) -> ActivatedModel:
        """Make a selected candidate the weights that answer new requests."""
        snapshot = self._pending.pop(candidate.candidate_id, None)
        if snapshot is None:
            raise RuntimeContractError(f"no pending mlx candidate {candidate.candidate_id!r} to activate")
        # Admission stays closed past this method. Between swapping the weights
        # and Reef committing the new head, a request would freeze the old
        # artifact and then be answered by the new weights, which the weight
        # surface correctly rejects as a runtime-load mismatch.
        # ``commit_serving`` reopens once the commit is durable.
        self.hold()
        try:
            runtime_load_id = self.swap_adapter(snapshot)
        except BaseException:
            self.release()
            raise
        logger.info("activated mlx candidate %s at runtime load ID %s", candidate.candidate_id, runtime_load_id)
        return ActivatedModel(candidate_id=candidate.candidate_id, runtime_load_id=runtime_load_id)

    def commit_serving(self) -> None:
        """Publish the swapped weights to new requests, and reopen the gate.

        The second half of the activation handshake: Reef calls this once the
        publication is durable, which is the first moment a new request can
        safely freeze the new head.
        """
        self._committed_runtime_load_id = self._serving_runtime_load_id
        self.release()
