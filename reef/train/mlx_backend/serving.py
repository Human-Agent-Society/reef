"""The serving half of the in-process MLX runtime.

This runtime owns the engine's live weights, request admission and the
served version. The training half reaches the engine through it.

Activation swaps the weights and leaves admission closed. Reef's scheduler
reopens it after the durable commit: it calls ``acknowledge_publication``,
the training runtime's ``commit_candidate``, ``mark_published`` and
``resume_admission`` in that order. Until then, a request cannot freeze the
old head and be answered by the new weights.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from reef.artifact.artifact import Artifact
from reef.runtime.interfaces import (
    ActivatedModel,
    InferenceHandler,
    InferenceRuntime,
    ModelCandidate,
    RuntimeContractError,
    RuntimeLoadId,
    new_runtime_load_id_incarnation,
)
from reef.surface.weights import artifact_runtime_load_id
from reef.train.mlx_backend.contracts import AdapterSnapshot, ModelEngine
from reef.train.mlx_backend.inference import MLXInferenceHandler

logger = logging.getLogger(__name__)


class MLXServingRuntime(InferenceRuntime):
    """Serve one engine's adapter in this process, and hold trained candidates until Reef selects one."""

    def __init__(self, engine: ModelEngine, *, adapter_name: str = "reef-mlx", inference_timeout_s: float = 300.0):
        # Requests arrive as calls, not over a wire, so the URL is nominal.
        super().__init__(base_url="mlx://local", inference_timeout_s=inference_timeout_s)
        self._engine = engine
        self._adapter_name = adapter_name
        self._incarnation = new_runtime_load_id_incarnation()
        self._load_sequence = 0
        self._serving_runtime_load_id = self._next_runtime_load_id()
        self._staged: dict[str, AdapterSnapshot] = {}
        self._inference_handler = MLXInferenceHandler(engine, self)

    @property
    def engine(self) -> ModelEngine:
        return self._engine

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._inference_handler

    def _next_runtime_load_id(self) -> str:
        """A new version for the engine's weights, unique across restarts by its incarnation."""
        self._load_sequence += 1
        return str(RuntimeLoadId(self._incarnation, self._load_sequence))

    def serving_runtime_load_id(self) -> str:
        return self._serving_runtime_load_id

    def serving_adapter_name(self) -> str:
        return self._adapter_name

    @contextmanager
    def paused_admission(self) -> Iterator[None]:
        """Admit no request and wait for admitted ones to finish, for the block.

        On exit, admission is open again only if it was open before. Raises
        ``TimeoutError`` if admitted requests do not finish within
        ``inference_timeout_s``.
        """
        was_open = self.inference_admission_status["open"]
        try:
            self.pause_admission(wait=True, timeout=self.inference_timeout_s)
            yield
        finally:
            if was_open:
                self.resume_admission()

    # -- Weight changes

    def restore_checkpoint(self, artifact: Artifact) -> str:
        """Load a published adapter into the engine and return its new version."""
        local_path = artifact.local_path
        if local_path is None:
            raise RuntimeContractError("mlx rollback requires a materialized adapter")
        with self.paused_admission():
            self._engine.load_adapter(Path(local_path))
            self._serving_runtime_load_id = self._next_runtime_load_id()
        return self._serving_runtime_load_id

    def activate_checkpoint(self, artifact: Artifact) -> str:
        """Load a recovered checkpoint when the engine serves other weights.

        The engine keeps its weights in this process, so a restart loses
        them. An artifact that records the version the engine serves, or no
        version, needs no load.
        """
        checkpoint = artifact.materialize()
        recorded = artifact_runtime_load_id(checkpoint)
        if recorded is None or recorded == self._serving_runtime_load_id:
            return self._serving_runtime_load_id
        logger.info("loading recovered adapter %r into the mlx engine", artifact.ref.release_id)
        runtime_load_id = self.restore_checkpoint(checkpoint)
        # The release is final when Reef activates it.
        self.mark_published()
        return runtime_load_id

    def stage_candidate(self, candidate_id: str, snapshot: AdapterSnapshot) -> None:
        """Keep a trained adapter until Reef selects or rejects it."""
        self._staged[candidate_id] = snapshot

    def staged_candidate(self, candidate_id: str) -> AdapterSnapshot | None:
        return self._staged.get(candidate_id)

    def discard_candidate(self, candidate_id: str) -> None:
        """Drop a rejected candidate; serving never used its weights."""
        self._staged.pop(candidate_id, None)

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Make a selected candidate the live adapter; admission stays closed until Reef's commit."""
        snapshot = self._staged.pop(candidate.candidate_id, None)
        if snapshot is None:
            raise RuntimeContractError(f"no pending mlx candidate {candidate.candidate_id!r} to activate")
        self.pause_admission(wait=True, timeout=self.inference_timeout_s)
        try:
            self._engine.apply_adapter(snapshot)
        except BaseException:
            self.resume_admission()
            raise
        self._serving_runtime_load_id = self._next_runtime_load_id()
        logger.info(
            "activated mlx candidate %s at runtime load ID %s", candidate.candidate_id, self._serving_runtime_load_id
        )
        return ActivatedModel(candidate_id=candidate.candidate_id, runtime_load_id=self._serving_runtime_load_id)

    def shutdown(self) -> None:
        self._engine.close()


__all__ = ["MLXServingRuntime"]
