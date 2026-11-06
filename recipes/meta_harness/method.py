"""Meta-Harness: full-history search over complete harness trees, scored on fixed evaluation tasks.

A training request starts each step: the deployment trains in ``manual`` mode,
and each ``POST /reef/train`` runs one step, its text an instruction the
proposer reads. The step gives one proposer model the search so far, read from
the scenario's history: the seed and every tree the method proposed, each with
its files, its parent, what the proposer said about it, its outcome and its
scores on the evaluation tasks; excerpts of the records of those runs; and the
parts of the adapter's tree. The reply names any candidate as its parent and
returns one complete tree. The step checks that tree and leaves it in
``workspace.harness`` as the candidate. A refused reply (invalid, or a
tree the history holds) leaves the head unchanged, so the step skips.

A deployment selects with the ``compare`` policy and ``statistic: mean``. The
incumbent's scores come from the score cache, not from new runs, so a
candidate must beat the mean the incumbent was measured at, as upstream's
frontier does.

The history holds the search. ``state/attempts.jsonl`` holds what the history
cannot: the replies that proposed no candidate.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.config import config_option
from reef.core.evaluation import mean_score
from reef.core.plugins import PluginSettings
from reef.core.prompts import untrusted_text
from reef.evaluation.harness import CandidateRecord, History, RunRecord
from reef.harness.adapters import get_adapter
from reef.harness.adapters.parts import TreeLayout
from reef.harness.tree import HarnessTree
from reef.harness.tree.checks import redact_secret_shaped
from reef.train.harness_backend import Method, Services, StepContext, Workspace

#: ``full_history`` branches from any candidate; ``incumbent_only``, the greedy control, from the served one.
MODES = ("full_history", "incumbent_only")
#: What happened to a candidate: the step builds on it, a decision selected it, decisions only rejected it, or none
#: decided it.
OUTCOMES = ("served", "selected", "rejected", "undecided")
#: The fields of a proposer reply.
REPLY_FIELDS = ("parent_id", "hypothesis", "changes", "files")
#: The method's file in ``state/``: one JSON line per refused reply, with the reason.
ATTEMPTS_FILE = "attempts.jsonl"
#: How many of the latest such replies a prompt shows.
RECENT_ATTEMPTS = 5
SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.md").read_text(encoding="utf-8").strip()
PROPOSAL_PROMPT = Path(__file__).with_name("proposal_prompt.md").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class MetaHarnessSettings(PluginSettings):
    """The config of ``MetaHarness``, from ``evolution.method.config``.

    Values: ``mode`` is one of ``MODES``. ``model`` names an
    ``evolution.models`` binding and is never ``served``. ``components`` are
    distinct parts of the adapter's tree, empty for every part.
    ``max_files`` ≥ 1, ``max_runs`` ≥ 0 and ``run_chars`` ≥ 1 bound one
    proposal and one prompt.
    Purpose: ``MetaHarness.from_config`` parses it, and the method reads it
    in each step.
    """

    mode: str = config_option(
        "full_history", help="full_history branches from any candidate; incumbent_only from the served one."
    )
    model: str = config_option("proposer", help="The evolution.models binding that proposes; never the served model.")
    components: tuple[str, ...] = config_option(
        (), help="The parts of the tree a proposal may change; none means every part."
    )
    max_files: int = config_option(32, help="The most files of one proposed tree.")
    max_runs: int = config_option(16, help="The most evaluation runs whose records one prompt shows.")
    run_chars: int = config_option(4000, help="The most characters one prompt shows of one run's records.")

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if not self.model.strip() or self.model == "served":
            raise ValueError(
                "model must name an evolution.models binding: the served model never proposes its harness"
            )
        if len(set(self.components)) != len(self.components):
            raise ValueError("components must not repeat a part")
        for label, value, least in (
            ("max_files", self.max_files, 1),
            ("max_runs", self.max_runs, 0),
            ("run_chars", self.run_chars, 1),
        ):
            if value < least:
                raise ValueError(f"{label} must be at least {least}")


@dataclass(frozen=True)
class Member:
    """One candidate a proposal may branch from.

    Values: ``record`` is the candidate's history record and ``tree`` its
    files, with ``tree.checksum() == record.tree``. ``outcome`` is one of
    ``OUTCOMES``: ``served`` when the tree is the head of the step; otherwise
    ``selected`` when a step's decision selected it, ``rejected`` when the
    decisions on it only rejected it, and ``undecided`` when no decision is on
    it. ``runs`` are runs of ``tree``: its latest one for each (task,
    repeat) of the evaluation tasks, in task order and then repeat order.
    Purpose: ``search_population`` makes them; the method shows them to the
    proposer model and checks the reply's parent against them, and the
    Terminal-Bench example lists them in its summary.
    """

    record: CandidateRecord
    tree: HarnessTree
    outcome: str
    runs: tuple[RunRecord, ...]

    def __post_init__(self) -> None:
        if self.tree.checksum() != self.record.tree:
            raise ValueError("Member.tree must be the tree its record names")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Member.outcome must be one of {', '.join(OUTCOMES)}")
        if any(task_run.tree != self.record.tree for task_run in self.runs):
            raise ValueError("Member.runs must be runs of the member's tree")

    @property
    def parent(self) -> str | None:
        """The tree this method's proposer branched from; for another tree, its record's parent: the head it was
        proposed against, ``None`` for the seed."""
        parent = self.record.notes.get("parent_id")
        return parent if isinstance(parent, str) else self.record.parent

    @property
    def mean_score(self) -> float | None:
        """The mean score of the ``runs`` that did not fail, as ``compare`` counts it (``mean_score``); ``None``
        before the first run, or when every run failed."""
        return mean_score(task_run.score for task_run in self.runs)


class MetaHarness(Method):
    """Propose one complete tree per step from the whole search history.

    Values: ``settings`` is the method's config.
    Purpose: a deployment names it in ``evolution.method``; the backend calls
    ``step`` once per step.
    """

    def __init__(self, settings: MetaHarnessSettings | None = None) -> None:
        self.settings = MetaHarnessSettings() if settings is None else settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(MetaHarnessSettings.parse(config, plugin_class=cls))

    def step(self, context: StepContext) -> None:
        workspace, services, settings = context.workspace, context.services, self.settings
        host = workspace.host()
        layout = get_adapter(str(host["adapter"])).layout
        parts = settings.components or tuple(layout.parts)
        unsupported = [part for part in parts if part not in layout.parts]
        if unsupported:
            raise ValueError(f"the {host['adapter']} harness tree has no parts {unsupported}")
        if settings.model not in services.models:
            raise ValueError(
                f"the Meta-Harness proposer model {settings.model!r} is not declared under evolution.models; "
                "the harness under test must not propose its own improvements"
            )
        history = workspace.history()
        head = workspace.tree().checksum()
        tasks = workspace.tasks()
        members = search_population(
            history, head, workspace.step, tasks, incumbent_only=settings.mode == "incumbent_only"
        )
        task_ids = {task: f"t{number}" for number, task in enumerate(tasks, start=1)}
        search_state: dict[str, Any] = {
            "mode": settings.mode,
            "adapter": host["adapter"],
            "config_targets": host["config_targets"],
            "parts": host["parts"],
            "evolvable_parts": list(parts),
            "max_files": settings.max_files,
            "served_candidate_id": head,
            "evaluation_tasks": {task_id: task for task, task_id in task_ids.items()},
            "population": [member_json(member, task_ids) for member in members],
            "runs": self.run_excerpts(history, members, task_ids),
            "attempts": read_attempts(workspace)[-RECENT_ATTEMPTS:],
        }
        request = workspace.request()
        if request is not None:
            search_state["instruction"] = request.text
        state_json = json.dumps(search_state, indent=2, sort_keys=True, ensure_ascii=False)
        reply = services.models[settings.model].chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{PROPOSAL_PROMPT}\n\n{untrusted_text(state_json, 'search state')}"},
            ]
        )
        try:
            proposal = parse_proposal(reply)
            parent = next((member for member in members if member.record.tree == proposal["parent_id"]), None)
            if parent is None:
                raise ValueError(f"parent_id must name a candidate of the population ({settings.mode} mode)")
            tree = self.checked_tree(proposal["files"], parent.tree, layout, parts, services)
            # The head is a committed tree of the history, so a reply that returns it repeats a candidate too.
            if any(
                record.tree == tree.checksum() and not uncommitted(record, workspace.step)
                for record in history.candidates.values()
            ):
                raise ValueError(f"the tree repeats candidate {tree.checksum()}")
        except ValueError as exc:
            append_attempt(workspace, {"step": workspace.step, "status": "refused", "error": str(exc)})
            workspace.write_output(
                f"The proposer's reply was refused: {exc}", {"status": "refused", "error": str(exc)}
            )
            return
        workspace.write_tree(tree)
        notes = {
            "status": "proposed",
            "parent_id": parent.record.tree,
            "hypothesis": proposal["hypothesis"],
            "changes": proposal["changes"],
        }
        summary = f"Hypothesis: {proposal['hypothesis']}\nChanges: {proposal['changes']}\nParent: {parent.record.tree}"
        workspace.write_output(summary, notes)

    def checked_tree(
        self,
        files: Mapping[str, Any],
        parent: HarnessTree,
        layout: TreeLayout,
        parts: Sequence[str],
        services: Services,
    ) -> HarnessTree:
        """The proposed tree when it passes the adapter's validation and changes only files of ``parts`` of
        ``parent``; ``ValueError`` otherwise."""
        tree = HarnessTree(dict(files))
        if len(tree.files) > self.settings.max_files:
            raise ValueError(f"the tree holds {len(tree.files)} files; max_files is {self.settings.max_files}")
        if fixed_files(tree, layout, parts) != fixed_files(parent, layout, parts):
            raise ValueError(f"a proposal may change only the files of the parts {list(parts)}")
        validation = services.check(tree)
        if not validation.passed:
            raise ValueError(f"the tree fails validation: {validation.message()}")
        return tree

    def run_excerpts(
        self, history: History, members: Sequence[Member], task_ids: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """The evaluation runs a prompt shows, each with an excerpt of its records: at most ``max_runs``.

        The members take turns, the served member first and the others newest
        first, and each gives its failed runs and then its lowest scores
        first, so the prompt shows where every candidate falls short.
        """
        served = [member for member in members if member.outcome == "served"]
        others = [member for member in reversed(members) if member.outcome != "served"]
        worst_first = (
            sorted(member.runs, key=lambda task_run: (not task_run.score.is_failed, task_run.score.value or 0.0))
            for member in (*served, *others)
        )
        shown = (task_run for turn in itertools.zip_longest(*worst_first) for task_run in turn if task_run is not None)
        limit = self.settings.run_chars
        return [
            {
                "candidate_id": task_run.tree,
                "task": task_ids[task_run.task],
                "repeat": task_run.repeat,
                "score": task_run.score.to_json(),
                "failure": (
                    None
                    if task_run.failure is None
                    else clipped(f"{task_run.failure.stage}: {task_run.failure.cause}", limit)
                ),
                "record": excerpt(history.run_dir(task_run.run_id), limit),
            }
            for task_run in itertools.islice(shown, self.settings.max_runs)
        ]


def search_population(
    history: History, head: str, step: int, tasks: Sequence[str], *, incumbent_only: bool = False
) -> tuple[Member, ...]:
    """The candidates a proposal at ``step`` may branch from, in the order the history first saw them.

    ``head`` is the checksum of the tree the step builds on, which the history
    holds; it is always a member. In full history the seed and every tree the
    method proposed before ``step`` are members too. A tree that an earlier
    attempt of ``step`` proposed is not: that attempt never committed.
    Outcomes read the final decision of each step before ``step``, the latest
    one the history holds for it. The runs are those on ``tasks``.
    """
    decisions = {decision.step: decision for decision in history.decisions if decision.step < step}
    order = {task: index for index, task in enumerate(tasks)}
    members = []
    for record in history.candidates.values():
        searched = record.origin == "seed" or (record.origin == "method" and not uncommitted(record, step))
        if record.tree != head and (incumbent_only or not searched):
            continue
        selections = [decision.selected for decision in decisions.values() if decision.tree == record.tree]
        if record.tree == head:
            outcome = "served"
        elif any(selections):
            outcome = "selected"
        elif selections:
            outcome = "rejected"
        else:
            outcome = "undecided"
        latest = {(task_run.task, task_run.repeat): task_run for task_run in history.runs_of(record.tree)}
        task_runs = sorted(
            (task_run for (task, _), task_run in latest.items() if task in order),
            key=lambda task_run: (order[task_run.task], task_run.repeat),
        )
        members.append(Member(record, history.tree(record.tree), outcome, tuple(task_runs)))
    return tuple(members)


def uncommitted(record: CandidateRecord, step: int) -> bool:
    """Whether ``record`` is a tree that an earlier attempt of ``step`` proposed.

    The history keeps every attempt, and that attempt never committed: a retry
    of ``step`` neither shows the tree nor refuses it as known.
    """
    return record.origin == "method" and record.step >= step


def member_json(member: Member, task_ids: Mapping[str, str]) -> dict[str, Any]:
    """What a prompt shows of one member; ``task_ids`` names each evaluation task."""
    scores: dict[str, list[float | str]] = {}
    for task_run in member.runs:
        scores.setdefault(task_ids[task_run.task], []).append(task_run.score.to_json())
    return {
        "candidate_id": member.record.tree,
        "parent_id": member.parent,
        "origin": member.record.origin,
        "outcome": member.outcome,
        "hypothesis": str(member.record.notes.get("hypothesis", "")),
        "changes": str(member.record.notes.get("changes", "")),
        "mean_score": member.mean_score,
        "scores": scores,
        "files": dict(member.tree.files),
    }


def fixed_files(tree: HarnessTree, layout: TreeLayout, parts: Sequence[str]) -> dict[str, str]:
    """The files of ``tree`` that a proposal must keep: every file whose part is not in ``parts``."""
    return {
        path: text
        for path, text in tree.files.items()
        if (location := layout.locate(path)) is None or location.part.name not in parts
    }


def excerpt(directory: Path, limit: int) -> str:
    """The text files under ``directory``, each after a line that names it, cut to ``limit`` characters.

    Links and files that are not UTF-8 text are left out, and
    credential-shaped literals are redacted. A missing directory has no files.
    """
    parts = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        parts.append(f"--- {path.relative_to(directory).as_posix()} ---\n{text}")
    return clipped(redact_secret_shaped("\n".join(parts)), limit)


def clipped(text: str, limit: int) -> str:
    """``text`` when it has at most ``limit`` characters; otherwise its first and last characters, ``limit`` in
    all, around a line that says how many were left out."""
    if len(text) <= limit:
        return text
    head = limit // 2
    return f"{text[:head]}\n[... {len(text) - limit} characters left out ...]\n{text[len(text) - (limit - head) :]}"


def read_attempts(workspace: Workspace) -> list[dict[str, Any]]:
    """Every reply that proposed no candidate, oldest first, from ``state/``."""
    path = workspace.state / ATTEMPTS_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_attempt(workspace: Workspace, attempt: Mapping[str, Any]) -> None:
    """Add one reply that proposed no candidate to ``state/``."""
    with (workspace.state / ATTEMPTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(attempt), sort_keys=True) + "\n")


def parse_proposal(reply: str) -> dict[str, Any]:
    """The proposal object in the reply: ``parent_id``, ``hypothesis``, ``changes`` and ``files``.

    The object is the text from the first ``{`` to the last ``}``, so a
    Markdown fence or words around it do not matter. ``ValueError`` when the
    reply holds no such object.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("proposer response contains no JSON object")
    try:
        value = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"proposer response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("proposer response must be a JSON object")
    unknown = set(value) - set(REPLY_FIELDS)
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


__all__ = ["MODES", "Member", "MetaHarness", "MetaHarnessSettings", "parse_proposal", "search_population"]
