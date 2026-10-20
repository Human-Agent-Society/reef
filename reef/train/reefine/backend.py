"""Reefine step preparation; evaluation policy belongs to its candidate plugin."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, replace

from reef.core.requirements import parse_requires
from reef.evaluation.harness import HarnessCandidate
from reef.evaluation.reefine import EvaluationContext
from reef.evaluation.reefine import ReefineCandidate as ReefineCandidate
from reef.evaluation.reefine import ReefineEvaluationBackend
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.harness_facts import harness_facts
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.step_result import design_sections
from reef.sandbox.process import EpisodeExecutor
from reef.train.backend import PreparedStep
from reef.train.cordis_backend.backend import CordisBackend
from reef.train.cordis_backend.contracts import ServedComposition, ServedCompositionConsumer
from reef.train.reefine.proposer import request_kinds
from reef.train.types import TrainingBatch


class ReefineBackend(CordisBackend, ServedCompositionConsumer, ReefineEvaluationBackend):
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
            notes = prepared.metrics.get("proposal_notes")
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
                    recheck=candidate.recheck,
                    proposal_id=candidate.proposal_id,
                    record_dir=candidate.record_dir,
                    context=EvaluationContext(batch.request, composition),
                    requires=requires,
                    usage=design_sections(notes)[1] if isinstance(notes, Mapping) else "",
                ),
            )
        return prepared

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

    def episode_files(self, entries: tuple[Mapping[str, object], ...]) -> dict[str, str]:
        return self._episode_evaluator.render(entries)

    @property
    def harness_description(self) -> Mapping[str, object]:
        facts = harness_facts(self._descriptor.name)
        return {
            "adapter": self._descriptor.name,
            "entry_kinds": list(request_kinds(self._descriptor.name)),
            "facts": None if facts is None else asdict(facts),
        }

    def show_evaluation_checks(self, checks: tuple[Mapping[str, object], ...]) -> None:
        if self._step_progress is not None:
            self._step_progress = replace(self._step_progress, checks=checks)
