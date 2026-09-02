"""Cordis backend binding GEPA's archive to Reef's commit lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from reef.train.backend import PreparedStep
from reef.train.cordis_backend import CordisBackend
from reef.train.evaluation.contracts import SelectionDecision
from reef.train.types import TrainingBatch, TrainStepResult

from .archive import Archive

ARCHIVE_STATE_KEY = "gepa_archive"


class GEPABackend(CordisBackend):
    """Carry the search archive in algorithm state and mirror it post-commit."""

    def __init__(self, *, archive: Archive, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._archive = archive
        self._archive_before_step: Mapping[str, Any] | None = None

    def initial_state(self) -> Mapping[str, Any]:
        return {**super().initial_state(), ARCHIVE_STATE_KEY: self._archive.to_dict()}

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        self._restore_committed(state)
        before = self._archive.to_dict()
        self._archive_before_step = before
        try:
            prepared = super().prepare_step(batch, state, scenario_step)
        except BaseException:
            self._archive.restore(before)
            self._archive_before_step = None
            raise
        if prepared.outcome != "candidate":
            self._archive_before_step = None
        return replace(prepared, state=self._with_archive(prepared.state))

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        result = super().settle_step(prepared, decision)
        if result.state is None:
            raise TypeError("GEPA settlement requires mapping algorithm state")
        self._archive_before_step = None
        return replace(result, state=self._with_archive(result.state))

    def abort_step(self, prepared: PreparedStep) -> None:
        try:
            super().abort_step(prepared)
        finally:
            if self._archive_before_step is not None:
                self._archive.restore(self._archive_before_step)
                self._archive_before_step = None

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Refresh the JSON mirror only after Reef records ``state`` durably."""
        self._restore_committed(state)
        self._archive.persist()

    def _restore_committed(self, state: Mapping[str, Any]) -> None:
        archive_state = state.get(ARCHIVE_STATE_KEY)
        if archive_state is None:
            return
        if not isinstance(archive_state, Mapping):
            raise ValueError(f"{ARCHIVE_STATE_KEY} algorithm state must be a mapping")
        self._archive.restore(archive_state)

    def _with_archive(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {**state, ARCHIVE_STATE_KEY: self._archive.to_dict()}


__all__ = ["ARCHIVE_STATE_KEY", "GEPABackend"]
