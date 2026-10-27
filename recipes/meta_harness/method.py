"""Meta-Harness: full-history search over complete harness trees.

Each step gives one proposer model the search so far, read from the
scenario's history: the seed and every tree the method proposed, each with its
files, its parent, what the proposer said about it, its outcome and its scores
on the evaluation tasks; excerpts of the episode records and of the step's
traces; and the files the adapter takes. The reply names any candidate as its
parent and returns one complete tree. The step checks that tree and leaves it
in ``workspace.harness`` as the candidate. A reply that is invalid, or that
repeats a tree the history holds, leaves the head unchanged, so the step skips.

A deployment selects with the ``compare`` policy and ``statistic: mean``. The
incumbent's scores come from the score cache, not from new episodes, so a
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
from statistics import fmean
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.prompts import untrusted_text
from reef.evaluation.harness import CandidateRecord, EpisodeRecord, History
from reef.harness.tree import HarnessTree, TreeKind, locate
from reef.harness.tree.checks import redact_secret_shaped
from reef.train.harness_backend import Method, StepContext, StepResult, Tools, Workspace

#: ``full_history`` branches from any candidate; ``incumbent_only``, the greedy control, from the served one.
MODES = ("full_history", "incumbent_only")
#: What happened to a candidate: the step builds on it, a decision selected it, decisions only rejected it, or none
#: decided it.
OUTCOMES = ("served", "selected", "rejected", "undecided")
#: The fields of a proposer reply.
REPLY_FIELDS = ("parent_id", "hypothesis", "changes", "files")
#: The method's file in ``state/``: one JSON line per reply that proposed no candidate.
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
    distinct tree kinds, empty for every kind the adapter renders.
    ``max_candidates`` and ``max_target_episodes`` are budgets ≥ 0, where 0
    sets no limit. ``max_files`` ≥ 1, ``max_traces`` ≥ 0 and
    ``trace_chars`` ≥ 1 bound one proposal and one prompt.
    Purpose: ``MetaHarness.from_config`` parses it, and the method reads it
    in each step.
    """

    mode: str = config_option(
        "full_history", help="full_history branches from any candidate; incumbent_only from the served one."
    )
    model: str = config_option("proposer", help="The evolution.models binding that proposes; never the served model.")
    components: tuple[str, ...] = config_option(
        (), help="The tree kinds a proposal may change; none means every kind the adapter renders."
    )
    max_candidates: int = config_option(0, help="The most candidates the method proposes. 0 sets no limit.")
    max_target_episodes: int = config_option(
        0, help="Propose no candidate once the evaluations ran this many episodes. 0 sets no limit."
    )
    max_files: int = config_option(32, help="The most files of one proposed tree.")
    max_traces: int = config_option(16, help="The most traces whose record files one prompt shows.")
    trace_chars: int = config_option(4000, help="The most characters one prompt shows of one trace's record files.")

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if not self.model.strip() or self.model == "served":
            raise ValueError(
                "model must name an evolution.models binding: the served model never proposes its harness"
            )
        kinds = {kind.value for kind in TreeKind}
        unknown = [kind for kind in self.components if kind not in kinds]
        if unknown:
            raise ValueError(f"components must be tree kinds ({', '.join(sorted(kinds))}), got {unknown}")
        if len(set(self.components)) != len(self.components):
            raise ValueError("components must not repeat a kind")
        for label, value, least in (
            ("max_candidates", self.max_candidates, 0),
            ("max_target_episodes", self.max_target_episodes, 0),
            ("max_files", self.max_files, 1),
            ("max_traces", self.max_traces, 0),
            ("trace_chars", self.trace_chars, 1),
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
    it. ``episodes`` are episodes of ``tree``: its latest one for each (task,
    repeat) of the evaluation tasks, in task order and then repeat order.
    Purpose: ``search_population`` makes them; the method shows them to the
    proposer model and checks the reply's parent against them, and the
    Terminal-Bench example lists them in its summary.
    """

    record: CandidateRecord
    tree: HarnessTree
    outcome: str
    episodes: tuple[EpisodeRecord, ...]

    def __post_init__(self) -> None:
        if self.tree.checksum() != self.record.tree:
            raise ValueError("Member.tree must be the tree its record names")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Member.outcome must be one of {', '.join(OUTCOMES)}")
        if any(episode.tree != self.record.tree for episode in self.episodes):
            raise ValueError("Member.episodes must be episodes of the member's tree")

    @property
    def parent(self) -> str | None:
        """The tree this method's proposer branched from; for another tree, its record's parent: the head it was
        proposed against, ``None`` for the seed."""
        parent = self.record.notes.get("parent_id")
        return parent if isinstance(parent, str) else self.record.parent

    @property
    def mean_score(self) -> float | None:
        """The mean score of ``episodes``, an episode that could not run as 0, as ``compare`` counts it; ``None``
        before the first episode."""
        if not self.episodes:
            return None
        return fmean(0.0 if episode.score is None else episode.score for episode in self.episodes)


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

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        settings = self.settings
        host = workspace.host()
        available = tuple(TreeKind(kind) for kind in host["kinds"])
        kinds = tuple(TreeKind(kind) for kind in settings.components) or available
        unsupported = [kind.value for kind in kinds if kind not in available]
        if unsupported:
            raise ValueError(f"adapter {host['adapter']!r} does not render the Meta-Harness components {unsupported}")
        try:
            proposer = tools.models[settings.model]
        except KeyError as exc:
            raise ValueError(
                f"the Meta-Harness proposer model {settings.model!r} is not declared under evolution.models; "
                "the harness under test must not propose its own improvements"
            ) from exc
        history = workspace.history()
        budget = self.spent_budget(history, workspace.step)
        if budget is not None:
            return StepResult(f"The {budget} budget is spent.", {"status": "budget_exhausted", "budget": budget})
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
            "file_guide": host["kinds"],
            "evolvable_kinds": [kind.value for kind in kinds],
            "max_files": settings.max_files,
            "served_candidate_id": head,
            "evaluation_tasks": {task_id: task for task, task_id in task_ids.items()},
            "population": [member_json(member, task_ids) for member in members],
            "traces": self.traces(workspace, history, members, task_ids),
            "attempts": read_attempts(workspace)[-RECENT_ATTEMPTS:],
        }
        request = workspace.request()
        if request is not None:
            search_state["instruction"] = request.text
        prompt = "\n\n".join(
            (
                PROPOSAL_PROMPT,
                untrusted_text(json.dumps(search_state, indent=2, sort_keys=True, ensure_ascii=False), "search state"),
            )
        )
        reply = proposer.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        try:
            proposal = parse_proposal(reply)
            parent = next((member for member in members if member.record.tree == proposal["parent_id"]), None)
            if parent is None:
                raise ValueError(f"parent_id must name a candidate of the population ({settings.mode} mode)")
            tree = self.checked_tree(proposal["files"], parent.tree, kinds, tools)
        except ValueError as exc:
            append_attempt(workspace, {"step": workspace.step, "status": "invalid", "error": str(exc)})
            return StepResult(f"The proposer's reply was invalid: {exc}", {"status": "invalid", "error": str(exc)})
        candidate = tree.checksum()
        # The head is a committed tree of the history, so it is known too.
        known = {record.tree for record in history.candidates.values() if not uncommitted(record, workspace.step)}
        if candidate in known:
            append_attempt(
                workspace,
                {
                    "step": workspace.step,
                    "status": "duplicate",
                    "candidate_id": candidate,
                    "parent_id": parent.record.tree,
                },
            )
            return StepResult(
                f"The reply repeats candidate {candidate}.", {"status": "duplicate", "candidate_id": candidate}
            )
        workspace.write_tree(tree)
        notes = {
            "status": "proposed",
            "parent_id": parent.record.tree,
            "hypothesis": proposal["hypothesis"],
            "changes": proposal["changes"],
        }
        summary = f"Hypothesis: {proposal['hypothesis']}\nChanges: {proposal['changes']}\nParent: {parent.record.tree}"
        return StepResult(summary, notes)

    def spent_budget(self, history: History, step: int) -> str | None:
        """The budget the search spent before ``step``: ``max_candidates`` or ``max_target_episodes``, ``None``
        while both last.

        The counts read the history's records of the steps before ``step``. An
        earlier attempt of ``step`` itself never committed, so a retry of the
        step does not count it.
        """
        settings = self.settings
        proposed = sum(
            1 for record in history.candidates.values() if record.origin == "method" and not uncommitted(record, step)
        )
        if settings.max_candidates and proposed >= settings.max_candidates:
            return "max_candidates"
        episodes = sum(1 for episode in history.episodes if episode.step < step)
        if settings.max_target_episodes and episodes >= settings.max_target_episodes:
            return "max_target_episodes"
        return None

    def checked_tree(
        self, files: Mapping[str, Any], parent: HarnessTree, kinds: Sequence[TreeKind], tools: Tools
    ) -> HarnessTree:
        """The proposed tree when it passes the adapter's validation and changes only files of ``kinds`` of
        ``parent``; ``ValueError`` otherwise."""
        tree = HarnessTree(dict(files))
        if len(tree.files) > self.settings.max_files:
            raise ValueError(f"the tree holds {len(tree.files)} files; max_files is {self.settings.max_files}")
        if fixed_files(tree, kinds) != fixed_files(parent, kinds):
            raise ValueError(f"a proposal may change only the files of kinds {[kind.value for kind in kinds]}")
        validation = tools.check(tree)
        if not validation.passed:
            raise ValueError(f"the tree fails validation: {validation.message()}")
        return tree

    def traces(
        self, workspace: Workspace, history: History, members: Sequence[Member], task_ids: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        """The traces a prompt shows, each with an excerpt of its record files: at most ``max_traces``.

        The step's own traces come first. Then the members' episodes take
        turns: the served member first and the others newest first, and each
        member gives its lowest scores first, an episode that could not run
        before any score.
        """
        max_traces = self.settings.max_traces
        shown: list[dict[str, Any]] = [
            {
                "source": "traffic",
                "score": trace.score,
                "record": excerpt(trace.directory, self.settings.trace_chars),
            }
            for trace in workspace.traces()[:max_traces]
        ]
        served = [member for member in members if member.outcome == "served"]
        others = [member for member in reversed(members) if member.outcome != "served"]
        turns = itertools.zip_longest(
            *(
                sorted(member.episodes, key=lambda episode: (episode.score is not None, episode.score or 0.0))
                for member in (*served, *others)
            )
        )
        for episode in itertools.chain.from_iterable(turns):
            if len(shown) >= max_traces:
                break
            if episode is None:
                continue
            failure = None if episode.failure is None else f"{episode.failure.stage}: {episode.failure.cause}"
            shown.append(
                {
                    "source": "evaluation",
                    "candidate_id": episode.tree,
                    "task": task_ids[episode.task],
                    "repeat": episode.repeat,
                    "score": episode.score,
                    "failure": None if failure is None else clipped(failure, self.settings.trace_chars),
                    "record": excerpt(history.episode_dir(episode.episode), self.settings.trace_chars),
                }
            )
        return shown


def search_population(
    history: History, head: str, step: int, tasks: Sequence[str], *, incumbent_only: bool = False
) -> tuple[Member, ...]:
    """The candidates a proposal at ``step`` may branch from, in the order the history first saw them.

    ``head`` is the checksum of the tree the step builds on, which the history
    holds; it is always a member. In full history the seed and every tree the
    method proposed before
    ``step`` are members too. A tree that an earlier attempt of ``step``
    proposed is not: that attempt never committed. Outcomes read the final
    decision of each step before ``step``, the latest one the history holds
    for it. The episodes are those on ``tasks``.
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
        latest = {(episode.task, episode.repeat): episode for episode in history.episodes_of(record.tree)}
        episodes = sorted(
            (episode for (task, _), episode in latest.items() if task in order),
            key=lambda episode: (order[episode.task], episode.repeat),
        )
        members.append(Member(record, history.tree(record.tree), outcome, tuple(episodes)))
    return tuple(members)


def uncommitted(record: CandidateRecord, step: int) -> bool:
    """Whether ``record`` is a tree that an earlier attempt of ``step`` proposed.

    The history keeps every attempt, and that attempt never committed: a retry
    of ``step`` neither counts the tree, nor shows it, nor refuses it as known.
    """
    return record.origin == "method" and record.step >= step


def member_json(member: Member, task_ids: Mapping[str, str]) -> dict[str, Any]:
    """What a prompt shows of one member; ``task_ids`` names each evaluation task."""
    scores: dict[str, list[float | None]] = {}
    for episode in member.episodes:
        scores.setdefault(task_ids[episode.task], []).append(episode.score)
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


def fixed_files(tree: HarnessTree, kinds: Sequence[TreeKind]) -> dict[str, str]:
    """The files of ``tree`` that a proposal must keep: every file whose kind is not in ``kinds``."""
    return {
        path: text
        for path, text in tree.files.items()
        if (location := locate(path)) is None or location.kind not in kinds
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

    A Markdown code fence around the object is removed. ``ValueError`` when
    the reply holds no such object.
    """
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
