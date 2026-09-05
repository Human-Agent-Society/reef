"""Bind the Meta-Harness population to Reef's transactional commit lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from reef.train.backend import PreparedStep
from reef.train.cordis_backend import CordisBackend, HarnessCandidate
from reef.train.evaluation.contracts import SelectionDecision
from reef.train.types import TrainingBatch, TrainStepResult

from .population import Population, PopulationStore, content_id


POPULATION_STATE_KEY = "meta_harness_population"
_LOG = logging.getLogger(__name__)


class MetaHarnessBackend(CordisBackend):
    """Stage population changes beside the composition and commit them together.

    The store's committed object is never mutated by prepare, evaluation, or
    settlement.  Those phases operate on a clone; ``commit_applied`` is the
    only transition that installs it and writes the JSON mirror.  A failed
    evaluation or settlement discards the clone, and a failed scenario commit
    never reaches the installation hook.
    """

    def __init__(self, *, population_store: PopulationStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._population_store = population_store

    def initial_state(self) -> Mapping[str, Any]:
        return {
            **super().initial_state(),
            POPULATION_STATE_KEY: self._population_store.committed.to_dict(),
        }

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        population_state = state.get(POPULATION_STATE_KEY, Population().to_dict())
        if not isinstance(population_state, Mapping):
            raise ValueError(f"{POPULATION_STATE_KEY} algorithm state must be a mapping")
        population = self._population_store.begin(population_state)
        entries = state.get("entries", ())
        if not isinstance(entries, (list, tuple)):
            self._population_store.abort()
            raise ValueError("Meta-Harness composition entries state must be a list")
        population.sync_served(entries, step=scenario_step)
        try:
            prepared = super().prepare_step(batch, state, scenario_step)
            if prepared.outcome == "candidate":
                prepared = self._bind_candidate(prepared, population)
            elif population.pending_id is not None:
                raise RuntimeError("Meta-Harness proposer left a pending candidate without a backend candidate")
            prepared = replace(prepared, state=self._with_population(prepared.state, population))
            if prepared.outcome != "candidate":
                # The result owns an immutable population snapshot from here.
                # Keeping the mutable clone resident would let a later commit
                # failure look like it advanced method-local state.
                self._population_store.abort()
            return prepared
        except BaseException:
            self._population_store.abort()
            raise

    def _bind_candidate(self, prepared: PreparedStep, population: Population) -> PreparedStep:
        candidate = prepared.candidate
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError("Meta-Harness requires the harness-evolution candidate type")
        candidate_id = content_id(candidate.candidate_entries)
        if population.pending_id is None:
            staged, is_new = population.stage_candidate(
                candidate.candidate_entries,
                parent_id=population.served.candidate_id,
                step=int(prepared.state["steps"]),
                hypothesis="External Reef proposer",
            )
            if not is_new:
                super().abort_step(prepared)
                population.record_attempt(
                    step=int(prepared.state["steps"]),
                    status="duplicate",
                    parent_id=population.served_id,
                    candidate_id=staged.candidate_id,
                )
                return PreparedStep.skipped(
                    state={**prepared.state, "entries": [dict(entry) for entry in candidate.current_entries]},
                    metrics={**prepared.metrics, "skipped": "proposal duplicates a retained candidate"},
                )
        if population.pending_id != candidate_id:
            super().abort_step(prepared)
            raise RuntimeError("Meta-Harness staged candidate does not match the composition Reef prepared")
        return prepared

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        try:
            result = super().settle_step(prepared, decision)
            if result.state is None:
                raise TypeError("Meta-Harness settlement requires mapping algorithm state")
            return replace(result, state=self._with_population(result.state, self._population_store.active))
        finally:
            # The returned state is the only speculative value allowed to
            # cross settlement.  The committed store changes only through
            # commit_applied after Reef's durable record exists.
            self._population_store.abort()

    def abort_step(self, prepared: PreparedStep) -> None:
        try:
            super().abort_step(prepared)
        finally:
            self._population_store.abort()

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Install committed population state, then refresh its derived mirror."""
        super().commit_applied(state)
        population_state = state.get(POPULATION_STATE_KEY)
        if not isinstance(population_state, Mapping):
            raise ValueError(f"{POPULATION_STATE_KEY} committed state must be a mapping")
        self._population_store.restore_committed(population_state)
        try:
            self._population_store.persist()
        except OSError as exc:
            # The scenario commit is already durable.  A mirror failure must
            # not report that commit as failed; recovery rewrites the mirror.
            _LOG.warning("could not refresh Meta-Harness population mirror: %s", exc)

    @staticmethod
    def _with_population(state: Mapping[str, Any], population: Population) -> dict[str, Any]:
        return {**state, POPULATION_STATE_KEY: population.to_dict()}


__all__ = ["POPULATION_STATE_KEY", "MetaHarnessBackend"]
