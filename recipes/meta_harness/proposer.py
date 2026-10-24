"""The history-aware proposer of Meta-Harness.

The proposer gives one configured model the complete committed population,
the current trace batch, and the files the adapter takes. The reply chooses
any known parent and returns one complete tree, which the step validates and
states as a patch against the served tree. Scores reach the population when
the next step folds in the evaluation (``population_for_step``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.trajectories import recorded_payload, recorded_payloads, source_record_id
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import HarnessTree, TreeKind, TreePatch, locate, validate_tree
from reef.train.harness_backend.contract import ProposalContext, Proposer, TreeProposal

from .population import POPULATION_KEY, Population, population_for_step

SEARCH_MODES = ("full_history", "incumbent_only")

#: What each kind's file holds. The proposer is told this because the kind names alone do not describe the
#: shape, and on the first step the population carries no example to copy.
FILE_GUIDE = {
    TreeKind.CONFIG: "config/<target>.json: a JSON object merged into one of the adapter's config targets",
    TreeKind.RULES: "rules.md: the text every session reads, appended to the agent's rules file",
    TreeKind.AGENT_COMMAND: "commands/<name>.md: a prompt template",
    TreeKind.SKILL: "skills/<name>/SKILL.md: a skill",
    TreeKind.CODE_EXTENSION: "extensions/<name>.<ext>: code the harness loads, which must compile",
    TreeKind.NATIVE_TOOL: "native/tools/<name>.py: a native tool: run(args, workdir), then NAME and DESCRIPTION",
    TreeKind.NATIVE_HOOK: "native/hooks/<name>.py: a native hook: listen(payload, next), then NAME and EVENT",
    TreeKind.NATIVE_GRAPH: "native/graphs/<name>.json: the native loop's control flow",
    TreeKind.NATIVE_AGENT: "native/agents/<name>.json: one agent the native loop delegates to",
    TreeKind.NATIVE_LOOP: "native/loops/<name>.py: the native loop itself as code",
}

SYSTEM_PROMPT = """You are the proposal policy in a Meta-Harness search.
Improve the fixed model's harness tree using the retained candidates,
their evaluation scores, and the latest recorded traces. Choose a parent,
change the harness rather than the underlying task or model, and return JSON
only. Do not copy task answers, credentials, or task-specific facts into the
tree. Prefer one falsifiable, reusable mechanism per proposal."""


def adapter_kinds(descriptor: AdapterDescriptor) -> tuple[TreeKind, ...]:
    """The kinds ``descriptor``'s adapter renders, in layout order."""
    return tuple(
        kind
        for kind in TreeKind
        if (kind is TreeKind.CONFIG and descriptor.config_targets) or kind.value in descriptor.kind_paths
    )


@dataclass(frozen=True)
class MetaHarnessSettings:
    """The ``evolution.meta_harness`` section.

    Values: ``mode`` is one of ``SEARCH_MODES``; ``model`` names the binding
    that proposes, which must not be the model under test. ``kinds`` are the
    evolvable kinds, empty for every kind the adapter renders.
    ``max_candidates`` and ``max_target_episodes`` ≥ 0 are budgets, 0 for
    none; ``max_files`` ≥ 1 caps a proposed tree. ``evaluation_cost`` is the
    episodes one evaluation runs, both sides.
    Purpose: ``MetaHarnessRecipe`` parses it; ``MetaHarnessProposer`` runs by it.
    """

    evaluation_cost: int
    mode: str = "full_history"
    model: str = "proposer"
    kinds: tuple[TreeKind, ...] = ()
    max_candidates: int = 0
    max_target_episodes: int = 0
    max_files: int = 32

    def __post_init__(self) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"evolution.meta_harness.mode must be one of {SEARCH_MODES}")
        if not self.model:
            raise ValueError("evolution.meta_harness.model must not be empty")
        for label, value in (
            ("max_candidates", self.max_candidates),
            ("max_target_episodes", self.max_target_episodes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"evolution.meta_harness.{label} must be a non-negative integer")
        if isinstance(self.max_files, bool) or not isinstance(self.max_files, int) or self.max_files < 1:
            raise ValueError("evolution.meta_harness.max_files must be a positive integer")


class MetaHarnessProposer(Proposer):
    """One model-authored, complete-tree proposal from the retained population."""

    def __init__(self, settings: MetaHarnessSettings) -> None:
        self.settings = settings

    def propose(self, context: ProposalContext) -> TreeProposal:
        settings = self.settings
        available = adapter_kinds(context.descriptor)
        kinds = settings.kinds or available
        unsupported = [kind.value for kind in kinds if kind not in available]
        if unsupported:
            raise ValueError(f"adapter {context.descriptor.name!r} does not render Meta-Harness kinds {unsupported}")
        population = population_for_step(context.method_state, context.last_outcome)
        population.sync_served(context.tree, step=context.step)
        attempt = population.proposer_calls + 1
        if settings.max_candidates and population.generated_candidates >= settings.max_candidates:
            population.record_attempt(step=attempt, status="budget_exhausted", budget="max_candidates")
            return keep(population, {"status": "budget_exhausted", "budget": "max_candidates"})
        if settings.max_target_episodes and (
            population.episode_calls + settings.evaluation_cost > settings.max_target_episodes
        ):
            population.record_attempt(step=attempt, status="budget_exhausted", budget="max_target_episodes")
            return keep(population, {"status": "budget_exhausted", "budget": "max_target_episodes"})
        prompt = self.prompt(population, context, available, kinds)
        population.proposer_calls += 1
        reply = self.binding(context.models).chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        audit = {
            "step": attempt,
            "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
        }
        try:
            proposal = parse_proposal(reply)
            parent = population.by_id(proposal["parent_id"])
            if settings.mode == "incumbent_only" and parent.candidate_id != population.served_id:
                raise ValueError("incumbent-only proposals must choose the served candidate")
            tree = self.checked_tree(proposal["files"], parent.tree, context.descriptor, kinds)
            candidate, is_new = population.stage_candidate(
                tree,
                parent_id=parent.candidate_id,
                step=context.step,
                hypothesis=proposal["hypothesis"],
                changes=proposal["changes"],
            )
            if not is_new:
                population.record_attempt(
                    **audit, status="duplicate", parent_id=parent.candidate_id, candidate_id=candidate.candidate_id
                )
                return keep(population, {"status": "duplicate", "candidate_id": candidate.candidate_id})
            patch = context.tree.diff(tree)
            if not patch.changes:
                raise ValueError("proposal does not change the served tree")
        except (KeyError, TypeError, ValueError) as exc:
            population.discard_pending()
            population.record_attempt(**audit, status="invalid", error=str(exc))
            return keep(population, {"status": "invalid", "error": str(exc)})
        population.record_attempt(
            **audit,
            status="proposed",
            parent_id=parent.candidate_id,
            candidate_id=candidate.candidate_id,
            hypothesis=candidate.hypothesis,
            changes=candidate.changes,
        )
        notes = {"status": "proposed", "parent_id": parent.candidate_id, "candidate_id": candidate.candidate_id}
        return TreeProposal(patch, notes={**notes, "hypothesis": candidate.hypothesis}, method_state=state(population))

    def binding(self, models: ModelBindings) -> ModelBinding:
        try:
            return models[self.settings.model]
        except KeyError as exc:
            raise ValueError(
                f"Meta-Harness proposer model {self.settings.model!r} is not declared under evolution.models; "
                "the harness under test must not propose its own improvements"
            ) from exc

    def checked_tree(
        self, files: Any, parent: HarnessTree, descriptor: AdapterDescriptor, kinds: Sequence[TreeKind]
    ) -> HarnessTree:
        """The proposed tree, when it passes validation and changes only evolvable files; ValueError otherwise."""
        if not isinstance(files, Mapping):
            raise ValueError("proposal files must be a JSON object of path to text")
        tree = HarnessTree(dict(files))
        if len(tree.files) > self.settings.max_files:
            raise ValueError(f"proposal holds {len(tree.files)} files; max_files is {self.settings.max_files}")

        def fixed(of: HarnessTree) -> dict[str, str]:
            return {
                path: text
                for path, text in of.files.items()
                if (location := locate(path)) is None or location.kind not in kinds
            }

        if fixed(tree) != fixed(parent):
            raise ValueError(f"proposal may change only the files of kinds {[kind.value for kind in kinds]}")
        validation = validate_tree(tree, descriptor)
        if not validation.passed:
            raise ValueError(f"proposal fails validation: {validation.message()}")
        return tree

    def prompt(
        self,
        population: Population,
        context: ProposalContext,
        available: Sequence[TreeKind],
        kinds: Sequence[TreeKind],
    ) -> str:
        records = population.candidates if self.settings.mode == "full_history" else [population.served]
        history = [
            {
                "candidate_id": candidate.candidate_id,
                "parent_id": candidate.parent_id,
                "outcome": candidate.outcome,
                "scores": None if candidate.scores is None else list(candidate.scores),
                "hypothesis": candidate.hypothesis,
                "changes": candidate.changes,
                "files": dict(candidate.files),
            }
            for candidate in records
        ]
        surface = {
            "adapter": context.descriptor.name,
            "config_targets": sorted(context.descriptor.config_targets),
            "mode": self.settings.mode,
            "file_guide": {kind.value: FILE_GUIDE[kind] for kind in available},
            "served_candidate_id": population.served_id,
            "evaluation_tasks": list(context.evaluation_tasks),
            "evolvable_kinds": [kind.value for kind in kinds],
            "max_files": self.settings.max_files,
            "population": history,
            "recent_traces": traces(context.samples),
        }
        return (
            "Propose exactly one complete harness tree. Use file_guide for the paths the adapter takes and what "
            "each file holds; every file must pass the adapter's validation. In full_history mode you may branch "
            "from any candidate_id in population; in incumbent_only mode use served_candidate_id. Files whose kind "
            "is not in evolvable_kinds must be copied unchanged from the selected parent. Return exactly this "
            'object: {"parent_id": "<full candidate id>", "hypothesis": "<falsifiable claim>", "changes": '
            '"<concise mechanism>", "files": {"<path>": "<the whole text of the file>"}}.\n\n'
            + json.dumps(surface, indent=2, sort_keys=True, ensure_ascii=False)
        )


def traces(samples: Sequence[TrajectoryItem]) -> list[dict[str, Any]]:
    return [
        {
            "source_agent_record_id": source_record_id(sample),
            "score": sample.metadata.get("reward"),
            "feedback": sample.metadata.get("feedback"),
            "payload": dict(recorded_payload(sample)),
            "trajectory": [dict(event) for event in recorded_payloads(sample)],
        }
        for sample in samples
    ]


def state(population: Population) -> dict[str, Any]:
    return {POPULATION_KEY: population.to_dict()}


def keep(population: Population, notes: Mapping[str, Any]) -> TreeProposal:
    """A step that proposes nothing but keeps the attempt and the budget it recorded."""
    return TreeProposal(TreePatch({}), notes=dict(notes), method_state=state(population))


def parse_proposal(reply: str) -> dict[str, Any]:
    """The proposal object in the reply: ``parent_id``, ``hypothesis``, ``changes`` and ``files``."""
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
    unknown = set(value) - {"parent_id", "hypothesis", "changes", "files"}
    if unknown:
        raise ValueError(f"proposer response contains unknown fields: {sorted(unknown)}")
    parent = value.get("parent_id")
    files = value.get("files")
    if not isinstance(parent, str) or not parent:
        raise ValueError("proposer response requires a non-empty parent_id")
    if not isinstance(files, Mapping):
        raise ValueError("proposer response requires a files object")
    return {
        "parent_id": parent,
        "hypothesis": str(value.get("hypothesis", "")),
        "changes": str(value.get("changes", "")),
        "files": files,
    }


__all__ = ["SEARCH_MODES", "MetaHarnessProposer", "MetaHarnessSettings", "adapter_kinds", "parse_proposal"]
