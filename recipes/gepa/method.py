"""GEPA's reflective search as a harness evolution method, one GEPA iteration per step.

``GEPA`` runs the part of an iteration that GEPA calls the reflective
mutation. It samples a parent from the archive's Pareto front, takes the
parent's minibatch, reflects on one component with the feedback that the
minibatch produced, runs the child on the same minibatch, and accepts only a
strict improvement. The accepted child is the step's candidate: the method
writes it to ``harness/``.

The rest of the iteration is GEPA's validation pass, and the step's
evaluation runs it. The ``compare`` policy with ``statistic: mean`` scores
the child on every evaluation task and publishes it only when its mean is
strictly above the served tree's. The served tree's scores come from the
history's score cache, so they are the scores it was validated with, as
GEPA keeps them. The next step reads the validation and the decision from
the history and folds them into the archive (``archive_for_step``) before it
samples the next parent.

The parent's minibatch costs no episodes when the parent is the served
tree: the step's traces are that tree's own run, scores included. Any other
parent and every child run through ``tools.evaluate``. The archive counts
metric calls the way GEPA does, whatever the cache saves: the seed's
validation, every parent and child minibatch, and every accepted candidate's
validation. So a budget means what it means upstream. With
``evolution.evaluation.repeats`` above 1, a validation pass costs one call
per episode, and a task's validation score is the mean of its repeats.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import ConfiguredPlugin, PluginSettings
from reef.evaluation.harness import History
from reef.harness.tree import HarnessTree, TreeKind
from reef.train.harness_backend import Method, StepContext, StepResult, Tools, Trace, Workspace

from .archive import Archive
from .components import EVOLVABLE_KINDS, candidate_tree, texts_of
from .reflection import extract_new_text, render_prompt

#: GEPA's reflective dataset keys. The names are load bearing: the copied
#: prompt tells the reflection model to read them.
INPUTS_KEY = "Inputs"
OUTPUTS_KEY = "Generated Outputs"
FEEDBACK_KEY = "Feedback"
#: The archive's file in the method's ``state/`` directory.
ARCHIVE_FILE = "archive.json"


class Feedback(ConfiguredPlugin, ABC):
    """What the reflection model reads about one minibatch example.

    Purpose: a deployment names a subclass in the ``feedback`` setting of
    ``GEPA``; ``ScoreFeedback`` is the default. ``feedback`` returns the
    sentence that the example's reflective record shows.
    """

    @abstractmethod
    def feedback(self, task: str, output: str, score: float) -> str:
        """The sentence about ``output``, the answer to ``task`` that scored ``score``."""


class ScoreFeedback(Feedback):
    """GEPA's default feedback: the score restated as text.

    It is weak on purpose. A benchmark that can say why an answer was wrong
    should say so through its own ``Feedback``, because that sentence is the
    whole signal that reflection gets.
    Purpose: ``GEPA`` uses it when its settings name no ``feedback``.
    """

    def feedback(self, task: str, output: str, score: float) -> str:
        return f"This response scored {score:.3f}. A better response would score higher."


@dataclass(frozen=True)
class Example:
    """One minibatch example as reflection sees it: input, output, grade.

    Values: ``task`` is the prompt, ``output`` the final answer the parent's
    run gave, and ``score`` its grade, 0.0 for a run that gave none.
    ``error`` says why the run failed, empty when it did not; it replaces
    the feedback, because the failure is the only feedback there is.
    Purpose: ``GEPA.minibatch`` makes one per minibatch task, and
    ``GEPA.records`` renders them into the reflective dataset.
    """

    task: str
    output: str
    score: float
    error: str = ""


@dataclass(frozen=True)
class GEPASettings(PluginSettings):
    """The configuration of ``GEPA``.

    Values: ``minibatch_size`` > 0 is the number of examples a parent and its
    child are compared on. ``seed`` seeds the archive's random generator.
    ``skip_perfect_score`` skips an iteration whose parent scored at least
    ``perfect_score`` on every example. ``max_metric_calls`` > 0 is the
    metric-call budget, ``None`` for none. ``components`` names the tree
    kinds GEPA may rewrite, a non-empty subset of ``EVOLVABLE_KINDS``.
    ``trainset_size`` > 0, when set, plans each iteration over a training set
    of that size in GEPA's epoch-shuffled order (``Archive.plan``), so a
    driver that sends the planned problems as traffic walks GEPA's own order.
    ``reflection_model`` names the model binding that reflects; the served
    model reflects when the deployment declares no model of that name.
    ``feedback`` is a plugin setting that names a ``Feedback``, ``None`` for
    ``ScoreFeedback``.
    Purpose: ``GEPA.from_config`` parses it from ``evolution.method.config``;
    ``GEPA.step`` runs by it.
    """

    minibatch_size: int = config_option(3, help="The examples a parent and its child are compared on.")
    seed: int = config_option(0, help="The seed of the random generator that draws parents and minibatches.")
    skip_perfect_score: bool = config_option(
        True, help="Skip an iteration whose parent scored perfect_score on every example."
    )
    perfect_score: float = config_option(1.0, help="The score that counts as perfect.")
    max_metric_calls: int | None = config_option(None, help="The metric-call budget. None sets no budget.")
    components: tuple[str, ...] = config_option(("rules", "skill"), help="The tree kinds GEPA may rewrite.")
    trainset_size: int | None = config_option(
        None, help="Plan each iteration over a training set of this size, in GEPA's order."
    )
    reflection_model: str = config_option(
        "reflection", help="The model that reflects. The served model reflects when none has this name."
    )
    feedback: str | Mapping[str, Any] | None = config_option(
        None, help="The Feedback that explains each example to the reflection model. None restates the score."
    )

    def __post_init__(self) -> None:
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        evolvable = [kind.value for kind in EVOLVABLE_KINDS]
        if not self.components or any(component not in evolvable for component in self.components):
            raise ValueError(f"components must name kinds from {', '.join(evolvable)}")
        if self.max_metric_calls is not None and self.max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be positive when set")
        if self.trainset_size is not None and self.trainset_size <= 0:
            raise ValueError("trainset_size must be positive when set")

    @property
    def kinds(self) -> tuple[TreeKind, ...]:
        """``components`` as tree kinds."""
        return tuple(TreeKind(component) for component in self.components)


class GEPA(Method):
    """GEPA's reflective search, one iteration per step.

    Values: ``settings`` is the configuration, and ``feedback`` writes the
    feedback sentence of each reflective record.
    Purpose: a deployment names the class in ``evolution.method``, and the
    backend calls ``step`` once per step. The search lives in the archive,
    ``state/archive.json``, which the backend commits with each step.
    """

    def __init__(self, settings: GEPASettings | None = None, feedback: Feedback | None = None) -> None:
        if feedback is not None and not isinstance(feedback, Feedback):
            raise TypeError("GEPA feedback must inherit Feedback")
        self.settings = GEPASettings() if settings is None else settings
        self.feedback = ScoreFeedback() if feedback is None else feedback

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        settings = GEPASettings.parse(config, plugin_class=cls)
        feedback = (
            None if settings.feedback is None else Feedback.from_setting(settings.feedback, setting="GEPA.feedback")
        )
        return cls(settings, feedback)

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        path = workspace.state / ARCHIVE_FILE
        tasks = workspace.tasks()
        head = workspace.tree()
        archive = archive_for_step(
            path.read_text(encoding="utf-8") if path.is_file() else None,
            workspace.history(),
            tasks,
            texts_of(head, self.settings.kinds),
            # The evaluation's episodes per task and tree: 0 without a scorer, and GEPA's count needs at least 1.
            repeats=max(1, workspace.evaluation_repeats()),
            rng_seed=self.settings.seed,
        )
        summary, record = self.iterate(archive, workspace, tools, head, len(tasks))
        path.write_text(json.dumps(archive.to_state(), indent=2) + "\n", encoding="utf-8")
        return StepResult(summary, {**record, "metric_calls": archive.metric_calls})

    def iterate(
        self, archive: Archive, workspace: Workspace, tools: Tools, head: HarnessTree, valset_size: int
    ) -> tuple[str, dict[str, Any]]:
        """One GEPA iteration on ``archive``: the step's summary and the iteration's record.

        An accepted child goes to ``harness/``. Every other ending leaves
        ``harness/`` as it is, so the backend skips the step.
        """
        settings = self.settings
        if settings.max_metric_calls is not None and archive.metric_calls >= settings.max_metric_calls:
            reason = f"the budget of {settings.max_metric_calls} metric calls is spent"
            return reason, {"stopped": reason}
        if settings.trainset_size is None:
            parent = archive.take_parent(valset_size)
        else:
            parent = int(archive.plan(settings.trainset_size, valset_size, settings.minibatch_size)["parent"])
        # A trace without a task (a request with no user prompt) is no example: there is nothing to run again.
        traces = [(task, trace) for trace in workspace.traces() if (task := trace.task)][: settings.minibatch_size]
        if not traces:
            return skipped("the step's traces hold no task to run")
        minibatch = self.minibatch(archive, parent, traces, tools, head)
        if settings.skip_perfect_score and all(example.score >= settings.perfect_score for example in minibatch):
            return skipped("the parent scored perfectly on the minibatch")
        parent_texts = archive.candidates[parent].texts
        keys = sorted(parent_texts)
        if not keys:
            return skipped("the served tree holds no evolvable file")
        component = archive.next_component(parent, keys)
        prompt = render_prompt(parent_texts[component], self.records(minibatch))
        reflection = tools.models.get(settings.reflection_model, tools.models.served)
        reply = reflection.chat([{"role": "user", "content": prompt}])
        new_text = extract_new_text(reply)
        # The step's calls.json keeps the prompt and the reply. The record leaves them out: it goes into the archive,
        # which every step commits whole, and into the step's notes.
        record: dict[str, Any] = {
            "parent": parent,
            "served": archive.served,
            "component": component,
            "parent_scores": [example.score for example in minibatch],
            "child_scores": None,
            "accepted": False,
            "candidate": None,
        }
        if not new_text.strip():
            archive.log_proposal(record)
            return f"the reflection on {component} wrote no new text", record
        child_texts = {**parent_texts, component: new_text}
        child = candidate_tree(head, child_texts)
        validation = tools.check(child)
        if not validation.passed:
            record["problems"] = list(validation.problems)
            archive.log_proposal(record)
            return f"the child fails validation: {validation.message()}", record
        episodes = tools.evaluate(child, [example.task for example in minibatch])
        archive.charge(len(episodes))
        child_scores = task_scores([episode.score for episode in episodes], repeats=1)
        record["child_scores"] = child_scores
        parent_total, child_total = sum(record["parent_scores"]), sum(child_scores)
        if child_total <= parent_total:
            # The one ending GEPA counts as a spent iteration: a child that lost the minibatch comparison.
            archive.reject()
            archive.log_proposal(record)
            return f"the child scored {child_total:g} on the minibatch and its parent {parent_total:g}", record
        if child == head:
            # The child differs from its parent but not from the served tree: there is no candidate to evaluate.
            archive.log_proposal(record)
            return "the child holds the texts of the served tree", record
        record["accepted"] = True
        record["candidate"] = archive.add(child_texts, parent, child_scores, workspace.step)
        archive.log_proposal(record)
        workspace.write_tree(child)
        return (
            f"rewrote {component} of candidate {parent}: the child scored {child_total:g} on the minibatch "
            f"and its parent {parent_total:g}, so candidate {record['candidate']} goes to validation"
        ), record

    def minibatch(
        self, archive: Archive, parent: int, traces: Sequence[tuple[str, Trace]], tools: Tools, head: HarnessTree
    ) -> list[Example]:
        """The parent's run on the tasks of ``traces``, charged as GEPA charges its evaluation of the parent.

        The served parent's run is the traces themselves: free of episodes,
        not of budget. Any other parent's tree runs the tasks as episodes.
        """
        if parent == archive.served:
            archive.charge(len(traces))
            return [
                Example(task, trace.reply or "", 0.0 if trace.score is None else trace.score, trace.failure or "")
                for task, trace in traces
            ]
        tasks = [task for task, _ in traces]
        episodes = tools.evaluate(candidate_tree(head, archive.candidates[parent].texts), tasks)
        archive.charge(len(episodes))
        return [
            Example(
                task,
                episode.reply or "",
                0.0 if episode.score is None else episode.score,
                "" if episode.failure is None else episode.failure.cause,
            )
            for task, episode in zip(tasks, episodes, strict=True)
        ]

    def records(self, minibatch: Sequence[Example]) -> list[dict[str, Any]]:
        """GEPA's reflective dataset: one record per minibatch example."""
        return [
            {
                INPUTS_KEY: example.task,
                OUTPUTS_KEY: example.output,
                FEEDBACK_KEY: example.error or self.feedback.feedback(example.task, example.output, example.score),
            }
            for example in minibatch
        ]


def archive_for_step(
    text: str | None,
    history: History,
    tasks: Sequence[str],
    served_texts: Mapping[str, str],
    *,
    repeats: int,
    rng_seed: int = 0,
) -> Archive:
    """The archive as a step starts: the committed one, its pending candidate settled, its served candidate current.

    ``text`` is the committed ``archive.json``, ``None`` before the first
    step. The pending candidate's validation pass is the evaluation of the
    step that proposed it: ``history`` holds its scores on ``tasks``, the
    scores of the tree that step served, and the step's decision. The
    evaluation runs ``repeats`` ≥ 1 episodes per task and tree, so a task's
    score is the mean of its repeats, and a failed episode reads as a zero,
    never as a gap. A pending candidate whose step reached no decision (the
    backend skipped the step) is dropped.

    ``served_texts`` are the evolvable texts of the tree the step serves. A
    tree that does not hold the served candidate's texts moved without the
    method: a first boot, an operator edit or a rollback. It becomes a new
    parentless candidate, never a descendant of the last served one, and it
    costs GEPA's seed validation, one metric call per episode:
    ``len(tasks) * repeats``. A driver that plans an iteration reads the same
    inputs, so it draws the plan the method draws.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    archive = Archive.from_state(None if text is None else json.loads(text), rng_seed=rng_seed)
    if archive.pending_step is not None:
        decision = history.decision(archive.pending_step)
        if decision is None:
            archive.clear_pending()
        else:
            archive.settle(
                task_scores(history.scores(decision.tree, tasks, repeats=repeats), repeats=repeats),
                task_scores(history.scores(decision.head, tasks, repeats=repeats), repeats=repeats),
                decision.selected,
                repeats=repeats,
            )
    if archive.served is None or archive.candidates[archive.served].texts != dict(served_texts):
        # GEPA scores its seed on the validation set before its first iteration; the step's evaluation runs that
        # pass later, at the same price.
        archive.seed(served_texts)
        archive.charge(len(tasks) * repeats)
    return archive


def task_scores(values: Sequence[float | None], *, repeats: int) -> list[float]:
    """One score per task from ``values``, task by task and repeat by repeat: each task's mean over its ``repeats``
    values, with a failed episode read as zero, never as missing."""
    zeroed = [0.0 if value is None else float(value) for value in values]
    return [sum(zeroed[start : start + repeats]) / repeats for start in range(0, len(zeroed), repeats)]


def skipped(reason: str) -> tuple[str, dict[str, Any]]:
    """An iteration that ends before reflection: ``reason`` is its summary, and its record says it skipped."""
    return reason, {"skipped": reason}


__all__ = [
    "ARCHIVE_FILE",
    "GEPA",
    "Feedback",
    "GEPASettings",
    "ScoreFeedback",
    "archive_for_step",
]
