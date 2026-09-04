"""The history-aware proposer and selection policy of Meta-Harness.

The proposer implements Reef's normal ``Proposer`` signature.  It gives one
configured model the complete committed population, the current trace batch,
and the adapter's available node vocabulary.  The response chooses any known
parent and returns one complete composition.  That composition is translated
back into Reef mutations against the composition currently being served.

The selector retains every unique, valid candidate and moves serving only on
a strict mean-score improvement.  Scores and parents are staged in the same
population transaction as the composition update; the backend makes them
durable together.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from reef.harness.descriptor import AdapterDescriptor
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.nodes import NODE_KINDS
from reef.harness.render import RenderError, render_composition
from reef.train.cordis_backend.strategies import Mutation, Proposer
from reef.train.evaluation.contracts import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import TraceSample

from .population import Population, PopulationStore, normalize_entries


SEARCH_MODES = ("full_history", "incumbent_only")

_SYSTEM_PROMPT = """You are the proposal policy in a Meta-Harness search.
Improve the fixed model's harness composition using the retained candidates,
their evaluation scores, and the latest recorded traces. Choose a parent,
change the harness rather than the underlying task or model, and return JSON
only. Do not copy task answers, credentials, or task-specific facts into the
composition. Prefer one falsifiable, reusable mechanism per proposal."""


def _entry_nodes(entries: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(entry["name"]), entry.get("config")) for entry in entries if not entry.get("disabled"))


def mutations_between(
    current: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
) -> tuple[Mutation, ...]:
    """Translate a complete target composition into Reef's mutation protocol.

    Updates are minimal while the target order is expressible by removals,
    in-place updates, and appended creates.  A real reorder is represented by
    an atomic remove-and-recreate sequence so the target tree remains exact.
    """
    before = normalize_entries(current)
    after = normalize_entries(target)
    if before == after:
        return ()
    before_by_id = {str(entry["id"]): entry for entry in before}
    after_by_id = {str(entry["id"]): entry for entry in after}
    retained = [str(entry["id"]) for entry in before if str(entry["id"]) in after_by_id]
    created = [str(entry["id"]) for entry in after if str(entry["id"]) not in before_by_id]
    target_ids = [str(entry["id"]) for entry in after]

    if target_ids != retained + created:
        return (
            *(Mutation("remove", str(entry["id"])) for entry in before),
            *(
                Mutation("create", str(entry["id"]), {key: value for key, value in entry.items() if key != "id"})
                for entry in after
            ),
        )

    mutations: list[Mutation] = [
        Mutation("remove", str(entry["id"])) for entry in before if str(entry["id"]) not in after_by_id
    ]
    for entry in after:
        entry_id = str(entry["id"])
        previous = before_by_id.get(entry_id)
        if previous is None:
            continue
        patch = {
            key: entry.get(key)
            for key in set(previous) | set(entry)
            if key != "id" and previous.get(key) != entry.get(key)
        }
        if patch:
            mutations.append(Mutation("update", entry_id, patch))
    mutations.extend(
        Mutation("create", entry_id, {key: value for key, value in after_by_id[entry_id].items() if key != "id"})
        for entry_id in created
    )
    return tuple(mutations)


class MetaHarnessProposer(Proposer):
    """One model-authored, full-composition proposal from retained history."""

    def __init__(
        self,
        *,
        store: PopulationStore,
        descriptor: AdapterDescriptor,
        tasks: Sequence[str],
        episode_repeats: int,
        mode: str = "full_history",
        model: str = "proposer",
        kinds: Sequence[str] = (),
        max_candidates: int = 0,
        max_target_episodes: int = 0,
        max_nodes: int = 32,
    ) -> None:
        if mode not in SEARCH_MODES:
            raise ValueError(f"Meta-Harness mode must be one of {SEARCH_MODES}")
        if not model:
            raise ValueError("Meta-Harness proposer model name must not be empty")
        for label, value in (("max_candidates", max_candidates), ("max_target_episodes", max_target_episodes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Meta-Harness {label} must be a non-negative integer")
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("Meta-Harness max_nodes must be a positive integer")
        available = _adapter_kinds(descriptor)
        selected = tuple(kinds) if kinds else available
        unknown = [kind for kind in selected if kind not in NODE_KINDS]
        unsupported = [kind for kind in selected if kind not in available]
        if unknown:
            raise ValueError(f"Meta-Harness node kinds are unknown: {unknown}")
        if unsupported:
            raise ValueError(f"adapter {descriptor.name!r} does not render Meta-Harness node kinds: {unsupported}")
        if not selected:
            raise ValueError(f"adapter {descriptor.name!r} exposes no evolvable Reef node kinds")
        self._store = store
        self._descriptor = descriptor
        self._tasks = tuple(str(task) for task in tasks)
        self._episode_repeats = episode_repeats
        self._mode = mode
        self._model = model
        self._available_kinds = available
        self._kinds = selected
        self._max_candidates = max_candidates
        self._max_target_episodes = max_target_episodes
        self._max_nodes = max_nodes
        self._evaluation_cost = 2 * len(self._tasks) * episode_repeats

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: ModelBindings,
        *,
        manifest: Any = None,
        rejected: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[Mutation, ...] | None:
        del nodes, manifest, rejected  # The committed population is the richer history surface.
        population = self._store.active
        step = population.proposer_calls + 1
        if self._max_candidates and population.generated_candidates >= self._max_candidates:
            population.record_attempt(step=step, status="budget_exhausted", budget="max_candidates")
            return None
        if self._max_target_episodes and population.episode_calls + self._evaluation_cost > self._max_target_episodes:
            population.record_attempt(step=step, status="budget_exhausted", budget="max_target_episodes")
            return None

        prompt = self._prompt(population, samples)
        population.proposer_calls += 1
        reply = self._binding(models).chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        audit = {
            "step": step,
            "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
        }
        try:
            proposal = _parse_proposal(reply)
            parent_id = proposal["parent_id"]
            parent = population.by_id(parent_id)
            if self._mode == "incumbent_only" and parent_id != population.served_id:
                raise ValueError("incumbent-only proposals must choose the served candidate")
            entries = self._validate_entries(proposal["entries"], parent.entries)
            candidate, is_new = population.stage_candidate(
                entries,
                parent_id=parent.candidate_id,
                step=step,
                hypothesis=proposal["hypothesis"],
                changes=proposal["changes"],
            )
            if not is_new:
                population.record_attempt(
                    **audit,
                    status="duplicate",
                    parent_id=parent_id,
                    candidate_id=candidate.candidate_id,
                )
                return None
            mutations = mutations_between(population.served.entries, candidate.entries)
            if not mutations:
                raise ValueError("proposal does not change the served composition")
        except (KeyError, TypeError, ValueError, RenderError) as exc:
            population.discard_pending()
            population.record_attempt(**audit, status="invalid", error=str(exc))
            return None

        population.record_attempt(
            **audit,
            status="proposed",
            parent_id=parent_id,
            candidate_id=candidate.candidate_id,
            hypothesis=candidate.hypothesis,
            changes=candidate.changes,
        )
        return mutations

    def _binding(self, models: ModelBindings) -> ModelBinding:
        try:
            return models[self._model]
        except KeyError as exc:
            raise ValueError(
                f"Meta-Harness proposer model {self._model!r} is not declared under evolution.models; "
                "the harness under test must not propose its own improvements"
            ) from exc

    def _validate_entries(
        self,
        value: Any,
        parent_entries: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("proposal entries must be a JSON list")
        entries = normalize_entries(value)
        if len(entries) > self._max_nodes:
            raise ValueError(f"proposal contains {len(entries)} nodes; max_nodes is {self._max_nodes}")
        for entry in entries:
            kind = str(entry["name"])
            if kind not in self._available_kinds:
                raise ValueError(
                    f"proposal uses node kind {kind!r}; adapter {self._descriptor.name!r} supports "
                    f"{self._available_kinds}"
                )
            NODE_KINDS[kind](None, entry.get("config"))
        fixed_before = [entry for entry in normalize_entries(parent_entries) if entry["name"] not in self._kinds]
        fixed_after = [entry for entry in entries if entry["name"] not in self._kinds]
        if fixed_after != fixed_before:
            raise ValueError(f"proposal may change only configured components {self._kinds}")
        # Adapter quirks and path collisions are part of the same admission
        # gate the backend applies.  Running the pure render here turns an
        # invalid proposal into committed attempt history instead of a crash.
        render_composition(_entry_nodes(entries), self._descriptor)
        return entries

    def _prompt(self, population: Population, samples: Sequence[TraceSample]) -> str:
        records = population.candidates if self._mode == "full_history" else [population.served]
        history = [
            {
                "candidate_id": candidate.candidate_id,
                "parent_id": candidate.parent_id,
                "outcome": candidate.outcome,
                "scores": None if candidate.scores is None else list(candidate.scores),
                "hypothesis": candidate.hypothesis,
                "changes": candidate.changes,
                "entries": [dict(entry) for entry in candidate.entries],
            }
            for candidate in records
        ]
        traffic = [
            {
                "source_agent_record_id": sample.source_agent_record_id,
                "score": sample.score,
                "feedback": sample.feedback,
                "payload": dict(sample.payload),
                "trajectory": [dict(event) for event in sample.trajectory],
            }
            for sample in samples
        ]
        surface = {
            "mode": self._mode,
            "served_candidate_id": population.served_id,
            "evaluation_tasks": list(self._tasks),
            "episode_repeats": self._episode_repeats,
            "adapter_node_kinds": list(self._available_kinds),
            "evolvable_node_kinds": list(self._kinds),
            "max_nodes": self._max_nodes,
            "population": history,
            "recent_traces": traffic,
        }
        return (
            "Propose exactly one complete Reef composition. In full_history mode you may branch from any "
            "candidate_id in population; in incumbent_only mode use served_candidate_id. Each entry must have "
            "a unique root-level id, a name from adapter_node_kinds, and that node kind's config. Entries whose "
            "name is not in evolvable_node_kinds must be copied unchanged from the selected parent. Return exactly "
            'this object: {"parent_id": "<full candidate id>", "hypothesis": "<falsifiable claim>", '
            '"changes": "<concise mechanism>", "entries": [<complete entry mappings>]}.\n\n'
            + json.dumps(surface, indent=2, sort_keys=True, ensure_ascii=False)
        )


class MetaHarnessSelector:
    """Retain every candidate and serve strict mean-score improvements."""

    def __init__(self, store: PopulationStore) -> None:
        self._store = store

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        del candidate
        population = self._store.active
        proposed = population.pending
        candidate_scores = _evaluation_scores(evaluation.metrics.get("candidate_scores", ()))
        current_scores = _evaluation_scores(evaluation.metrics.get("current_scores", ()))
        if len(candidate_scores) != len(current_scores):
            raise ValueError("Meta-Harness evaluation score vectors must have equal lengths")
        # The incumbent's number is the one from this batch, not the one it
        # scored when it won. Cordis interleaves candidate and current inside
        # each pairing so drift lands on both sides; comparing against a stored
        # score throws that away and ratchets the bar up to the incumbent's
        # luckiest run, which no later candidate has to beat on merit.
        candidate_mean = fmean(candidate_scores)
        incumbent_mean = fmean(current_scores)
        selected = candidate_mean > incumbent_mean
        population.record_decision(
            candidate_scores=candidate_scores,
            current_scores=current_scores,
            selected=selected,
        )
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="meta_harness",
            policy_version="1",
            reason=(
                f"candidate scored {candidate_mean:.4f} against committed incumbent {incumbent_mean:.4f} "
                f"over {len(candidate_scores)} episode results"
            ),
            evaluation=evaluation,
            metrics={
                "candidate_mean": candidate_mean,
                "incumbent_mean": incumbent_mean,
                "population_candidate_id": proposed.candidate_id,
                "population_parent_id": proposed.parent_id,
                "population_size": len(population.candidates),
                "proposer_calls": population.proposer_calls,
                "target_episode_calls": population.episode_calls,
            },
        )


def _adapter_kinds(descriptor: AdapterDescriptor) -> tuple[str, ...]:
    kinds = []
    if descriptor.config_targets:
        kinds.append("config")
    kinds.extend(kind for kind in NODE_KINDS if kind != "config" and kind in descriptor.node_paths)
    return tuple(kinds)


def _parse_proposal(reply: str) -> dict[str, Any]:
    text = reply.strip()
    if text.startswith("```"):
        first_line, separator, remainder = text.partition("\n")
        if separator and first_line.removeprefix("```").strip().lower() in ("", "json"):
            text = remainder
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("proposer response contains no JSON object") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"proposer response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("proposer response must be a JSON object")
    unknown = set(value) - {"parent_id", "hypothesis", "changes", "entries"}
    if unknown:
        raise ValueError(f"proposer response contains unknown fields: {sorted(unknown)}")
    parent = value.get("parent_id")
    entries = value.get("entries")
    if not isinstance(parent, str) or not parent:
        raise ValueError("proposer response requires a non-empty parent_id")
    if not isinstance(entries, list):
        raise ValueError("proposer response requires an entries list")
    return {
        "parent_id": parent,
        "hypothesis": str(value.get("hypothesis", "")),
        "changes": str(value.get("changes", "")),
        "entries": entries,
    }


def _evaluation_scores(values: Any) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("Meta-Harness evaluation requires non-empty score vectors")
    scores = tuple(0.0 if value is None else float(value) for value in values)
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("Meta-Harness evaluation scores must be finite")
    return scores


__all__ = ["SEARCH_MODES", "MetaHarnessProposer", "MetaHarnessSelector", "mutations_between"]
