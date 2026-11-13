"""SkillLibrary: procedural skills that an agent induces from its scored runs, kept only when a replay shows they help.

Each step reads the scored runs among the step's traces. An extractor model writes at most one candidate skill per
run: a procedure, and a check that the agent runs before it trusts its result, with a known bad case on which the
check fails. When the run's task is one of the step's evaluation tasks, the step runs the task with and without the
candidate (``Services.evaluate``) and keeps the candidate only when it scores higher. A run whose task cannot be
replayed adds its candidate unverified, and only from a success. When a failed run implicates a skill of the library,
the step runs the task without that skill. A library that does better without it gives the skill a loss, one that does
worse without it gives a win, and a tie gives neither. A skill whose losses pass its wins by ``max_losses`` leaves the
library. The library is the method's state, ``state/skills.json``, and each skill is served as
``skills/sl-<slug>/SKILL.md``. The README describes the method and its sources.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from string import Template
from typing import Any, Self

import yaml

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.prompts import untrusted_text
from reef.core.tasks.harbor import task_instruction
from reef.core.trajectories import atif_messages
from reef.evaluation.harness.history import RunRecord
from reef.harness.tree import HarnessTree
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.train.harness_backend import Method, Services, StepContext, Trace, Workspace

PROMPTS = Path(__file__).parent / "prompts"
#: The extractor's prompt: ``$score`` and ``$success_score`` say how the run did, ``$library`` lists the library's
#: skills and ``$trace`` is the run.
INDUCE_PROMPT = Template((PROMPTS / "induce.md").read_text(encoding="utf-8"))
#: The library's file in ``state/``.
LIBRARY_FILE = "skills.json"
#: The tree directory of the method's skills; the method leaves every other file of the tree as it is.
SKILL_PREFIX = "sl-"
SKILL_ROOT = f"skills/{SKILL_PREFIX}"
#: The Agent Skills format allows a skill name of at most 64 characters and a description of at most 1024.
MAX_SLUG_CHARS = 64 - len(SKILL_PREFIX)
MAX_DESCRIPTION_CHARS = 1024
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: The most characters of one run that a prompt holds; a longer run loses the middle of its transcript.
MAX_TRACE_CHARS = 30_000
#: The text fields of a skill that the extractor writes.
SKILL_FIELDS = ("name", "description", "procedure", "check", "known_bad")


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` when ``condition`` is False; return ``None`` when it is True."""
    if not condition:
        raise ValueError(message)


def check_line(label: str, line: object) -> None:
    """Refuse a skill ``label`` that is not a non-empty line without leading, trailing or repeated whitespace."""
    require(
        isinstance(line, str) and bool(line) and line == " ".join(line.split()),
        f"a skill {label} must be one non-empty line",
    )


def check_text(label: str, text: object) -> None:
    """Refuse a skill ``label`` that is not a non-empty text without leading or trailing whitespace."""
    require(
        isinstance(text, str) and bool(text) and text == text.strip(),
        f"a skill {label} must be non-empty text without leading or trailing whitespace",
    )


def check_count(label: str, count: object, least: int) -> None:
    """Refuse a skill ``label`` that is not an integer (a bool is not one) of at least ``least``."""
    require(
        not isinstance(count, bool) and isinstance(count, int) and count >= least,
        f"a skill's {label} must be an integer of at least {least}",
    )


@dataclass(frozen=True)
class Skill:
    """One procedural skill of the library.

    Values: ``slug`` matches ``SLUG`` in at most ``MAX_SLUG_CHARS`` characters. ``name`` and ``description`` are
    non-empty single lines without leading, trailing or repeated whitespace; the description has at most
    ``MAX_DESCRIPTION_CHARS`` characters. ``procedure``, ``check`` and ``known_bad`` are non-empty texts without
    leading or trailing whitespace. Every text encodes as UTF-8. ``verified`` is True when a replay of a task showed
    that the skill's current text scores higher than the library without it. ``wins`` ≥ 0 and ``losses`` ≥ 0 count
    the replays that the skill won and lost. ``step`` ≥ 1 is the step that wrote its current text.
    Purpose: ``state/skills.json`` holds the library, oldest first. Each step writes each skill into the tree as the
    skill at ``path``, with the text ``skill()``.
    """

    slug: str
    name: str
    description: str
    procedure: str
    check: str
    known_bad: str
    verified: bool
    wins: int
    losses: int
    step: int

    def __post_init__(self) -> None:
        require(
            isinstance(self.slug, str) and SLUG.fullmatch(self.slug) is not None and len(self.slug) <= MAX_SLUG_CHARS,
            f"a skill slug must match {SLUG.pattern} in at most {MAX_SLUG_CHARS} characters",
        )
        check_line("name", self.name)
        check_line("description", self.description)
        require(
            len(self.description) <= MAX_DESCRIPTION_CHARS,
            f"a skill description has at most {MAX_DESCRIPTION_CHARS} characters",
        )
        check_text("procedure", self.procedure)
        check_text("check", self.check)
        check_text("known_bad", self.known_bad)
        # A JSON escape can make a lone surrogate, which has no UTF-8 form: UnicodeEncodeError is a ValueError.
        "".join((self.name, self.description, self.procedure, self.check, self.known_bad)).encode("utf-8")
        require(isinstance(self.verified, bool), "a skill's verified flag must be a bool")
        check_count("wins", self.wins, 0)
        check_count("losses", self.losses, 0)
        check_count("step", self.step, 1)

    @property
    def path(self) -> str:
        """The tree path of the skill: ``skills/sl-<slug>/SKILL.md``."""
        return f"{SKILL_ROOT}{self.slug}/SKILL.md"

    @property
    def text(self) -> str:
        """The texts the extractor wrote, one per line, for the credential and instruction screens."""
        return "\n".join((self.name, self.description, self.procedure, self.check, self.known_bad))

    def skill(self) -> str:
        """The skill's ``SKILL.md``: ``name`` and ``description`` frontmatter, then the procedure and the check."""
        header = yaml.safe_dump(
            {"name": SKILL_PREFIX + self.slug, "description": self.description},
            sort_keys=False,
            allow_unicode=True,
            width=4096,
        )
        return (
            f"---\n{header}---\n\n# {self.name}\n\n{self.procedure}\n\n"
            f"## Check before you trust the result\n\n{self.check}\n\n"
            f"The check must fail on this known bad result: {self.known_bad}\n"
        )


@dataclass(frozen=True)
class SkillLibrarySettings(PluginSettings):
    """The config of ``SkillLibrary``.

    Values: ``extract_model`` is a non-empty model binding name: ``served`` or an ``evolution.models`` entry.
    ``success_score`` is finite and ``min_margin`` is a finite number ≥ 0. ``repeats`` ≥ 0, where 0 takes the
    evaluation's repeats. ``max_runs_per_step`` ≥ 0 and ``max_skills`` ≥ 0, where 0 sets no limit.
    ``max_ablations_per_step`` ≥ 0 and ``max_losses`` ≥ 1.
    Purpose: ``SkillLibrary.from_config`` parses it from ``evolution.method.config``; each step reads it.
    """

    extract_model: str = config_option("served", help="The model that writes a candidate skill from a run.")
    success_score: float = config_option(1.0, help="A run scored at least this is a success.")
    repeats: int = config_option(0, help="Runs per tree in a replay. 0 takes the evaluation's repeats.")
    min_margin: float = config_option(
        0.0, help="A replay keeps a candidate when its mean score beats the library's by more than this."
    )
    keep_unverified: bool = config_option(
        True, help="Keep a candidate from a successful run whose task cannot be replayed, marked unverified."
    )
    max_runs_per_step: int = config_option(0, help="The most runs that one step learns from. 0 sets no limit.")
    max_ablations_per_step: int = config_option(
        2, help="The most skills that one step replays without, after failed runs implicate them."
    )
    max_losses: int = config_option(2, help="A skill leaves when its losses pass its wins by this many.")
    max_skills: int = config_option(30, help="The most skills the library keeps. 0 sets no limit.")

    def __post_init__(self) -> None:
        require(
            bool(self.extract_model.strip()), "extract_model must name a model: served or an evolution.models entry"
        )
        require(math.isfinite(self.success_score), "success_score must be a finite number")
        require(
            math.isfinite(self.min_margin) and self.min_margin >= 0,
            "min_margin must be a finite number of at least 0",
        )
        require(
            min(self.repeats, self.max_runs_per_step, self.max_ablations_per_step, self.max_skills) >= 0,
            "repeats, max_runs_per_step, max_ablations_per_step and max_skills must be at least 0",
        )
        require(self.max_losses >= 1, "max_losses must be at least 1")


@dataclass(frozen=True)
class ReplayOutcome:
    """The result of replaying one task on two trees, a challenger and a reference.

    Values: ``challenger`` and ``reference`` are the mean scores of the completed runs of each tree, ``None`` when no
    run of that tree completed or the replay was refused. ``decided`` is True exactly when both means exist.
    ``better`` is True exactly when ``decided`` and ``challenger`` beats ``reference`` by more than the replay's
    margin.
    Purpose: ``Replay.compare`` makes one; the step reads it to keep a candidate or to count a skill's win or loss.
    """

    challenger: float | None
    reference: float | None
    decided: bool
    better: bool

    def __post_init__(self) -> None:
        require(
            self.decided == (self.challenger is not None and self.reference is not None),
            "a replay is decided exactly when both trees have a mean score",
        )
        require(self.decided or not self.better, "an undecided replay cannot be better")

    @classmethod
    def of_means(cls, challenger: float | None, reference: float | None, margin: float) -> ReplayOutcome:
        """The outcome of two mean scores: undecided when either is ``None``, else better when ``challenger`` beats
        ``reference`` by more than ``margin``."""
        if challenger is None or reference is None:
            return cls(challenger, reference, decided=False, better=False)
        return cls(challenger, reference, decided=True, better=challenger > reference + margin)

    def notes(self) -> dict[str, float | bool | None]:
        """The outcome for the step's notes."""
        return {"challenger": self.challenger, "reference": self.reference, "better": self.better}


class Replay:
    """The step's replays: runs of one evaluation task on two trees, within the step's run budget.

    Values: ``tasks`` maps the text that a run's harness was asked (the prompt, or the instruction of a task
    directory) to its evaluation task; it is empty when the deployment scores no runs. ``repeats`` ≥ 1 when
    ``tasks`` is not empty. ``comparisons`` counts the ``compare`` calls. ``refusal`` is why the services refused a
    replay, ``None`` before any refusal; after a refusal no task is replayable.
    Purpose: ``SkillLibrary.step`` makes one per step and asks it for each run's task and each comparison.
    """

    def __init__(self, context: StepContext, settings: SkillLibrarySettings) -> None:
        workspace = context.workspace
        self.services: Services = context.services
        evaluation_repeats = workspace.evaluation_repeats()
        self.repeats = settings.repeats or evaluation_repeats
        self.margin = settings.min_margin
        if evaluation_repeats > 0:
            self.tasks = replay_tasks(workspace)
        else:
            self.tasks = {}
        self.comparisons = 0
        self.refusal: str | None = None

    def task_of(self, asked: str) -> str | None:
        """The evaluation task whose text is ``asked``; ``None`` when there is none or a replay was refused."""
        if self.refusal is not None:
            return None
        return self.tasks.get(asked)

    def compare(self, task: str, challenger: HarnessTree, reference: HarnessTree) -> ReplayOutcome:
        """Run ``task`` ``repeats`` times on each tree and compare the mean scores of their completed runs.

        A refusal of the services (no run budget left, or a tree that fails validation) gives an undecided outcome
        and stops every later replay of the step.
        """
        self.comparisons += 1
        try:
            challenger_mean, reference_mean = [
                mean_score(self.services.evaluate(tree, [task], repeats=self.repeats))
                for tree in (challenger, reference)
            ]
        except (RuntimeError, ValueError) as exc:
            self.refusal = str(exc)
            return ReplayOutcome(None, None, decided=False, better=False)
        return ReplayOutcome.of_means(challenger_mean, reference_mean, self.margin)


class LibraryStep:
    """One step's pass over its scored runs: the library as it changes, and what happened to each run.

    Values: ``library`` is the library, oldest first: at construction the one in ``state/skills.json`` (empty when
    the file is absent), and after each ``learn`` the library with that run's ablations and candidate applied.
    ``others`` are the tree's files outside ``SKILL_ROOT``. ``events`` holds one notes entry per learned run, in
    learning order. ``ablations`` ≥ 0 counts the step's ablation replays, at most ``max_ablations_per_step``.
    Purpose: ``SkillLibrary.step`` makes one per step, calls ``learn`` per run, then ``retire``, ``drop`` and
    ``write``.
    """

    def __init__(self, context: StepContext, settings: SkillLibrarySettings) -> None:
        self.workspace, self.services, self.settings = context.workspace, context.services, settings
        self.extractor = self.services.models[settings.extract_model]
        self.library_path = self.workspace.state / LIBRARY_FILE
        self.library = (
            [Skill(**row) for row in json.loads(self.library_path.read_text(encoding="utf-8"))]
            if self.library_path.is_file()
            else []
        )
        self.others = {
            path: text for path, text in self.workspace.tree().files.items() if not path.startswith(SKILL_ROOT)
        }
        self.replay = Replay(context, settings)
        self.events: list[dict[str, Any]] = []
        self.ablations = 0

    def tree(self, skills: Sequence[Skill]) -> HarnessTree:
        """The tree of ``others`` plus ``skills``."""
        return tree_of(self.others, skills)

    def prior(self, slug: str) -> Skill | None:
        """The library's skill ``slug``; ``None`` when it has none."""
        return next((skill for skill in self.library if skill.slug == slug), None)

    def without(self, slug: str) -> list[Skill]:
        """The library without its skill ``slug``, in library order."""
        return [skill for skill in self.library if skill.slug != slug]

    def learn(self, trace: Trace) -> None:
        """Ask the extractor about one scored run, replay the skills it implicates, then offer its candidate.

        A run below ``success_score`` whose task is replayable gets the ablations and a verified candidate; the
        run's event records the result.
        """
        score = run_score(trace)
        event: dict[str, Any] = {"record": trace.record, "score": score}
        self.events.append(event)
        try:
            fields, implicated = induced(self.extractor.chat([{"role": "user", "content": self.prompt(trace, score)}]))
        except ValueError as exc:
            event["result"] = f"skipped: the extractor's reply is not the asked JSON: {exc}"
            return
        failed = score < self.settings.success_score
        task = self.replay.task_of(str(trace.task))
        replay_task = task if failed else None
        self.ablate(replay_task, implicated, event)
        self.offer(fields, replay_task, failed, event)

    def prompt(self, trace: Trace, score: float) -> str:
        """The extractor's prompt for a run scored ``score``, with the library's current skills."""
        listing = "\n".join(f"- {skill.name}: {skill.description}" for skill in self.library) or "(no skills yet)"
        return INDUCE_PROMPT.substitute(
            score=f"{score:g}",
            success_score=f"{self.settings.success_score:g}",
            library=untrusted_text(listing, "skill library"),
            trace=trace_text(trace),
        )

    def ablate(self, replay_task: str | None, implicated: Sequence[str], event: dict[str, Any]) -> None:
        """Replay ``replay_task`` without each implicated skill of the library, once per slug; none without a task."""
        if replay_task is None:
            return
        for blamed in dict.fromkeys(slug_of(name) for name in implicated):
            self.ablate_one(replay_task, blamed, event)

    def ablate_one(self, task: str, blamed: str, event: dict[str, Any]) -> None:
        """Replay ``task`` without the skill ``blamed`` and count the replay as its loss, its win or neither.

        Nothing runs when the library has no skill ``blamed`` or the step has used its ablations.
        """
        if self.prior(blamed) is None or self.ablations >= self.settings.max_ablations_per_step:
            return
        self.ablations += 1
        outcome = self.replay.compare(task, self.tree(self.without(blamed)), self.tree(self.library))
        event.setdefault("ablations", []).append({"skill": blamed, **outcome.notes()})
        self.credit(blamed, outcome)

    def credit(self, blamed: str, outcome: ReplayOutcome) -> None:
        """Count an ablation of ``blamed``: a loss when the library did better without it, a win when it did better
        with it, each by more than the replay's margin. An undecided replay or a tie counts for neither."""
        # A tie, most often a task that fails with and without the skill, shows nothing about the skill. Counted as a
        # win, it hid later losses, so a skill that tied often could not leave after the policy it states changed.
        helped = ReplayOutcome.of_means(outcome.reference, outcome.challenger, self.replay.margin).better
        if not (outcome.better or helped):
            return
        self.library = [
            replay_counted(skill, lost=outcome.better) if skill.slug == blamed else skill for skill in self.library
        ]

    def offer(
        self, fields: dict[str, str] | None, replay_task: str | None, failed: bool, event: dict[str, Any]
    ) -> None:
        """Build the run's candidate from ``fields``, screen it and keep it when ``decide`` says so."""
        if fields is None:
            event["result"] = "no skill: the run teaches nothing new"
            return
        candidate = self.candidate(fields, event)
        if candidate is None:
            return
        event["skill"] = candidate.slug
        kept, event["result"] = self.decide(candidate, replay_task, failed, event)
        if kept is not None:
            self.library = [*self.without(kept.slug), kept]

    def candidate(self, fields: dict[str, str], event: dict[str, Any]) -> Skill | None:
        """The unverified skill of ``fields``, with the counts of the library's skill of its slug; ``None``, with the
        event's result, when the skill is invalid, carries a credential or an override, or fails validation."""
        slug = slug_of(fields["name"])
        prior = self.prior(slug)
        wins, losses = (0, 0) if prior is None else (prior.wins, prior.losses)
        try:
            candidate = Skill(
                slug=slug,
                name=fields["name"],
                description=fields["description"],
                procedure=fields["procedure"],
                check=fields["check"],
                known_bad=fields["known_bad"],
                verified=False,
                wins=wins,
                losses=losses,
                step=self.workspace.step,
            )
        except ValueError as exc:
            event["result"] = f"refused: {exc}"
            return None
        refusal = self.refusal(candidate)
        if refusal is not None:
            event["result"] = refusal
            return None
        return candidate

    def refusal(self, candidate: Skill) -> str | None:
        """Why the library cannot serve ``candidate``; ``None`` when it can."""
        # The tree is served to every session: a credential would leak, and an override would speak to every run.
        if secret_shaped(candidate.text) or directive_shaped(candidate.text):
            return "refused: the skill carries a credential or an instruction override"
        validation = self.services.check(self.tree([*self.without(candidate.slug), candidate]))
        if not validation.passed:
            return f"refused: the tree fails validation: {validation.message()}"
        return None

    def decide(
        self, candidate: Skill, replay_task: str | None, failed: bool, event: dict[str, Any]
    ) -> tuple[Skill | None, str]:
        """The skill to keep (``None`` for none) and the event's result.

        A replayable failed run keeps the candidate when its replay scores higher. A success, which leaves no room
        for a replay to show a gain on its task, keeps it unverified when ``keep_unverified`` is set. A failed run
        that cannot be replayed keeps nothing.
        """
        if replay_task is not None:
            return self.verify(candidate, replay_task, event)
        if failed:
            return None, "rejected: the task cannot be replayed and the run is not a success"
        if self.settings.keep_unverified:
            return candidate, f"{self.action(candidate.slug)}, unverified"
        return None, "not kept: only verified skills are kept, and a successful run is not replayed"

    def verify(self, candidate: Skill, task: str, event: dict[str, Any]) -> tuple[Skill | None, str]:
        """Replay ``task`` with and without ``candidate``: the candidate marked verified with one more win when it
        scores higher, else ``None``; and the event's result."""
        outcome = self.replay.compare(
            task, self.tree([*self.without(candidate.slug), candidate]), self.tree(self.library)
        )
        event["replay"] = outcome.notes()
        if not outcome.better:
            return None, "rejected: the replay did not score higher with the skill"
        return replace(candidate, verified=True, wins=candidate.wins + 1), f"{self.action(candidate.slug)}, verified"

    def action(self, slug: str) -> str:
        """``revised`` when the library has a skill ``slug``, else ``added``."""
        return "added" if self.prior(slug) is None else "revised"

    def retire(self) -> list[str]:
        """Remove the skills whose losses pass their wins by ``max_losses``; return their slugs."""
        retired = [skill.slug for skill in self.library if skill.losses - skill.wins >= self.settings.max_losses]
        self.library = [skill for skill in self.library if skill.slug not in retired]
        return retired

    def drop(self) -> list[str]:
        """Remove the weakest skills past ``max_skills``; return their slugs."""
        limit = self.settings.max_skills or len(self.library)
        # The weakest leave first: unverified before verified, then by wins minus losses, then the oldest.
        ranked = sorted(self.library, key=lambda skill: (skill.verified, skill.wins - skill.losses, skill.step))
        dropped = [skill.slug for skill in ranked[: max(0, len(self.library) - limit)]]
        self.library = [skill for skill in self.library if skill.slug not in dropped]
        return dropped

    def notes(self, traces: int, runs: int, retired: list[str], dropped: list[str]) -> dict[str, Any]:
        """The step's notes: the counts, the events, the retired and dropped slugs, and any replay refusal."""
        notes: dict[str, Any] = {
            "traces": traces,
            "runs": runs,
            "comparisons": self.replay.comparisons,
            "events": self.events,
            "retired": retired,
            "dropped": dropped,
            "skills": len(self.library),
            "verified": sum(1 for skill in self.library if skill.verified),
        }
        if self.replay.refusal is not None:
            notes["replay_refusal"] = self.replay.refusal
        return notes

    def write(self, notes: dict[str, Any]) -> None:
        """Write the library as the tree's skills and ``state/skills.json``, and the step's output; when the tree
        fails validation, write only the output and leave the library as it was."""
        candidate_tree = self.tree(self.library)
        validation = self.services.check(candidate_tree)
        if not validation.passed:
            summary = f"the skills fail validation, so the library stays as it was: {validation.message()}"
            self.workspace.write_output(summary, {**notes, "problems": list(validation.problems)})
            return
        self.workspace.write_tree(candidate_tree)
        rows = [asdict(skill) for skill in self.library]
        self.library_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        kept = sum(1 for event in self.events if str(event.get("result", "")).startswith(("added", "revised")))
        summary = (
            f"learned from {notes['runs']} scored run(s): {kept} skill(s) kept, {len(notes['retired'])} retired and "
            f"{len(notes['dropped'])} dropped; the library holds {len(self.library)} ({notes['verified']} verified)"
        )
        self.workspace.write_output(summary, notes)


class SkillLibrary(Method):
    """A library of procedural skills that each step grows from scored runs and checks by replaying their tasks.

    Values: ``settings`` is the method's config.
    Purpose: a deployment names ``recipes.skill_library:SkillLibrary`` in ``evolution.method``; the backend calls
    ``step`` once per step.
    """

    def __init__(self, settings: SkillLibrarySettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(SkillLibrarySettings.parse(config, plugin_class=cls))

    def step(self, context: StepContext) -> None:
        """Learn from each scored run among the traces, then write the library as the tree's ``sl-*`` skills.

        Per run, lowest score first and in trace order within a score: when the run failed and implicates skills of
        the library, replay its task without each of them (at most ``max_ablations_per_step`` per step); then replay
        its task with the run's candidate skill and keep the candidate when it scores higher. A replay needs a failed
        run, its task among the step's evaluation tasks and a scorer. A candidate from a success, or from a run that
        cannot be replayed, joins unverified only from a success and only when ``keep_unverified`` is set. Skills
        whose losses pass their wins by ``max_losses`` leave, then the weakest leave past ``max_skills``. The notes
        say what happened to every run.
        """
        update = LibraryStep(context, self.settings)
        traces = context.workspace.traces()
        runs = scored_runs(traces, self.settings.max_runs_per_step)
        for trace in runs:
            update.learn(trace)
        retired = update.retire()
        dropped = update.drop()
        update.write(update.notes(len(traces), len(runs), retired, dropped))


def replay_tasks(workspace: Workspace) -> dict[str, str]:
    """The workspace's evaluation tasks by the text a run's harness was asked: the task itself, or the instruction of
    its task directory."""
    directories = bool(workspace.host().get("tasks_are_directories"))
    return {(task_instruction(Path(task)) if directories else task): task for task in workspace.tasks()}


def run_score(trace: Trace) -> float:
    """The score of a scored run."""
    return float(trace.score or 0.0)


def scored_runs(traces: Sequence[Trace], limit: int) -> list[Trace]:
    """The traces with a task and a score, lowest score first and in trace order within a score; the first ``limit``
    of them, or all when ``limit`` is 0."""
    # A replay can show a gain only where the library fails, so the failed runs get the step's run budget first.
    runs = sorted((trace for trace in traces if trace.task is not None and trace.score is not None), key=run_score)
    if limit:
        return runs[:limit]
    return runs


def replay_counted(skill: Skill, lost: bool) -> Skill:
    """``skill`` with one more loss when ``lost``, else with one more win."""
    return replace(skill, losses=skill.losses + 1) if lost else replace(skill, wins=skill.wins + 1)


def mean_score(records: Sequence[RunRecord]) -> float | None:
    """The mean score of the runs among ``records`` that completed, ``None`` when none did."""
    values = [record.score.value for record in records if record.score.value is not None]
    return sum(values) / len(values) if values else None


def tree_of(others: Mapping[str, str], library: Sequence[Skill]) -> HarnessTree:
    """The tree of the files outside the method's skills, plus one ``sl-*`` skill per skill of ``library``."""
    return HarnessTree({**others, **{skill.path: skill.skill() for skill in library}})


def trace_text(trace: Trace) -> str:
    """What the extractor reads of a run, fenced as data.

    It is one JSON object: the task, the transcript without the system prompt, the final reply, the score, the
    feedback and the failure, each left out when empty. Past ``MAX_TRACE_CHARS`` characters, the text loses its
    middle.
    """
    view = {
        "task": trace.task,
        "transcript": transcript_of(trace),
        "final_reply": trace.reply,
        "score": trace.score,
        "feedback": trace.feedback,
        "failure": trace.failure,
    }
    text = json.dumps({key: value for key, value in view.items() if value not in (None, "", [])}, ensure_ascii=False)
    return untrusted_text(cut_middle(text), "trace")


def transcript_of(trace: Trace) -> list[dict[str, Any]]:
    """The run's chat messages without the system prompt; empty when the run has no trajectory."""
    trajectory = trace.trajectory()
    if trajectory is None:
        return []
    return [message for message in atif_messages(trajectory) if message["role"] != "system"]


def cut_middle(text: str) -> str:
    """``text`` when it has at most ``MAX_TRACE_CHARS`` characters; else its first and last ``MAX_TRACE_CHARS // 2``
    characters around a marker that counts the cut characters."""
    if len(text) <= MAX_TRACE_CHARS:
        return text
    half = MAX_TRACE_CHARS // 2
    return f"{text[:half]} [... {len(text) - 2 * half} characters cut ...] {text[-half:]}"


def induced(reply: str) -> tuple[dict[str, str] | None, list[str]]:
    """The skill fields and the implicated skill names of an extractor reply
    ``{"skill": {...} | null, "implicated": [...]}``; ``ValueError`` for a reply of another form.

    The fields are the ``SKILL_FIELDS`` texts, stripped; a list of non-empty strings becomes one Markdown bullet
    per string. A name and a description become single lines, and a description longer than
    ``MAX_DESCRIPTION_CHARS`` is cut. A skill that lacks a field, its check or its known bad
    case included, is a reply of another form. A missing ``implicated`` is an empty list.
    """
    value = json_object(reply)
    implicated = value.get("implicated", [])
    require(
        isinstance(implicated, list) and all(isinstance(name, str) for name in implicated),
        "implicated is not a list of skill names",
    )
    return skill_fields(value.get("skill")), implicated


def skill_fields(row: object) -> dict[str, str] | None:
    """The ``SKILL_FIELDS`` texts of the reply's skill object, as ``induced`` describes; ``None`` for a null skill."""
    if row is None:
        return None
    if not isinstance(row, dict):
        raise ValueError("the skill is not an object")
    fields = {key: field_text(row.get(key)) for key in SKILL_FIELDS}
    missing = [key for key in SKILL_FIELDS if not fields[key]]
    require(not missing, f"the skill lacks {', '.join(missing)}")
    fields["name"] = " ".join(fields["name"].split())
    fields["description"] = " ".join(fields["description"].split())[:MAX_DESCRIPTION_CHARS].rstrip()
    return fields


def field_text(written: object) -> str:
    """A field the extractor wrote, stripped: a text, or a list of bullets (``bullet_text``); empty for another
    value."""
    # A model may write the procedure as a list of bullets instead of one Markdown text.
    text = bullet_text(written) if isinstance(written, list) else written
    return text.strip() if isinstance(text, str) else ""


def bullet_text(lines: list[Any]) -> str | None:
    """One Markdown bullet per line of a non-empty list of non-empty strings; ``None`` for another list."""
    if not lines or not all(isinstance(line, str) and line.strip() for line in lines):
        return None
    return "\n".join(bullet(line) for line in lines)


def bullet(line: str) -> str:
    """``line`` stripped, as a Markdown bullet: kept when it starts with ``-``, else prefixed with ``- ``."""
    return line.strip() if line.lstrip().startswith("-") else f"- {line.strip()}"


def json_object(reply: str) -> dict[str, Any]:
    """The JSON object in a model reply, from the first ``{`` to the last ``}`` after the reasoning, which ends at the
    reply's first ``</think>``; ``ValueError`` when the reply holds no JSON object."""
    # Reasoning comes first and ends at its first closing tag; a later "</think>" is text inside the answer.
    _, tag, after = reply.partition("</think>")
    answer = after if tag else reply
    start = answer.find("{")
    if start < 0:
        raise ValueError("it holds no JSON object")
    # The text runs from a "{" to a "}", so it parses only as an object; without a "}" after the "{" it is empty.
    try:
        value: dict[str, Any] = json.loads(answer[start : answer.rfind("}") + 1])
    # A reply nested past the interpreter's recursion limit cannot parse: it is a reply of another form.
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"it is not JSON ({exc})") from None
    return value


def slug_of(name: str) -> str:
    """The slug of a skill name: its letters and digits as lowercase ASCII, in groups joined by hyphens and cut to
    ``MAX_SLUG_CHARS``; a name without any is named by the start of the SHA-256 digest of its UTF-8 bytes, where a
    lone surrogate (a JSON escape can make one) passes through."""
    letters = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    slug = "-".join(re.findall(r"[a-z0-9]+", letters))[:MAX_SLUG_CHARS].strip("-")
    return slug or hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()[:16]


__all__ = ["ReplayOutcome", "Skill", "SkillLibrary", "SkillLibrarySettings"]
