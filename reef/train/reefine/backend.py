"""Reefine step preparation; evaluation policy belongs to its candidate plugin."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace

from reef.core.requirements import parse_requires
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import EpisodeExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree.mutations import admit_mutations
from reef.harness.tree.render import render_composition
from reef.train.backend import PreparedStep
from reef.train.cordis_backend.backend import CordisBackend, HarnessCandidate
from reef.train.cordis_backend.contracts import ServedComposition, ServedCompositionConsumer
from reef.train.reefine.context import EvaluationContext
from reef.train.types import TrainingBatch


@dataclass(frozen=True, kw_only=True)
class ReefineCandidate(HarnessCandidate):
    context: EvaluationContext
    requires: tuple[Mapping[str, object], ...] = ()


class ReefineBackend(CordisBackend, ServedCompositionConsumer):
    """Use published entries as the base; pending entries are never the incumbent."""

    served_composition: ServedComposition | None = None

    def set_served_composition(self, composition: ServedComposition) -> None:
        self.served_composition = copy.deepcopy(composition)

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, object], scenario_step: int) -> PreparedStep:
        composition = self.served_composition
        if composition is None:
            composition = ServedComposition("unpublished-seed", tuple(copy.deepcopy(self._entries())))
        prepared = super().prepare_step(batch, {**state, "entries": composition.entries}, scenario_step)
        candidate = prepared.candidate
        if isinstance(candidate, HarnessCandidate):
            request_record = prepared.metrics.get("training_request")
            if isinstance(request_record, Mapping):
                requires = tuple(parse_requires(request_record.get("requires", ())))
            else:
                requires = ()
            prepared = replace(
                prepared,
                candidate=ReefineCandidate(
                    candidate_id=candidate.candidate_id,
                    metadata=candidate.metadata,
                    candidate_files=candidate.candidate_files,
                    current_files=candidate.current_files,
                    candidate_entries=candidate.candidate_entries,
                    current_entries=candidate.current_entries,
                    mutations=candidate.mutations,
                    evaluation_tasks=candidate.evaluation_tasks,
                    gate_tasks=candidate.gate_tasks,
                    recheck=candidate.recheck,
                    proposal_id=candidate.proposal_id,
                    record_dir=candidate.record_dir,
                    context=EvaluationContext(batch.request, composition),
                    requires=requires,
                ),
            )
        return prepared

    def prepare_reevaluation(self, prepared: PreparedStep) -> PreparedStep:
        candidate = prepared.candidate
        if not isinstance(candidate, ReefineCandidate) or self.served_composition is None:
            raise TypeError("Reefine re-evaluation requires a candidate and the serving composition")
        current = self.served_composition
        if self._model_resolver is not None:
            self._models = self._model_resolver.resolve()
            self._binding_nodes = self._models.served.compose_nodes(self._descriptor)
        entries, refusal = admit_mutations(current.entries, candidate.mutations, self._descriptor)
        if refusal is not None:
            raise ValueError(f"candidate no longer admitted against {current.release_id}: {refusal}")
        refreshed = replace(
            candidate,
            current_entries=current.entries,
            current_files=render_composition(self._nodes_from(current.entries), self._descriptor),
            candidate_entries=tuple(entries),
            candidate_files=render_composition(self._nodes_from(entries), self._descriptor),
            context=replace(candidate.context, current=current),
        )
        return super().prepare_reevaluation(replace(prepared, candidate=refreshed))

    @property
    def evaluation_models(self) -> ModelBindings:
        return self._models

    @property
    def episode_descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    @property
    def episode_binary(self) -> str:
        return self._binary

    @property
    def episode_executor(self) -> EpisodeExecutor:
        return self._executor

    @property
    def episode_timeout_seconds(self) -> float:
        return self._episode_timeout_s

    def episode_files(self, entries: tuple[Mapping[str, object], ...]) -> dict[str, str]:
        return self._render_for_episode(entries)

    def score_health(self, task: str, result: EpisodeResult) -> float:
        return self._score_episode.score_with_models(task, result, self._models)

    def show_evaluation_checks(self, checks: tuple[Mapping[str, object], ...]) -> None:
        if self._step_progress is not None:
            self._step_progress = replace(self._step_progress, checks=checks)
