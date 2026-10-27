"""GEPA's reflective mutation as the step's proposer.

One GEPA iteration is split across the step. ``GEPAProposer`` owns the half
GEPA calls the reflective mutation: sample a parent from the archive's Pareto
front, evaluate it on a minibatch, reflect on one component with the feedback
that minibatch produced, evaluate the child on the same minibatch, and accept
only a strict improvement. The other half is the full validation pass: the
step's evaluation runs every ``evolution.tasks`` prompt on both trees, so its
per-task scores are exactly GEPA's valset evaluation. The next step folds
them into the archive (``Archive.settle``) from ``context.last_outcome``,
before it samples the next parent, and ``selection.GEPASelection`` decides
the step on the archive's stored means.

The parent's minibatch costs no episodes when the parent is the served tree:
the step's job data already holds that tree's real recorded traffic, scores included.
Any other parent and every child run through ``context.trials``. The
archive's metric-call count follows GEPA's accounting regardless (the seed's
validation, every parent and child minibatch, every accepted candidate's
validation), so a budget means what it means upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.plugins import ConfiguredPlugin
from reef.core.trajectories import recorded_payload
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import TreeKind, TreePatch
from reef.train.harness_backend.contract import ProposalContext, Proposer, StepOutcome, TreeProposal

from . import components, reflection
from .archive import Archive

#: GEPA's reflective dataset keys. The names are load bearing: they are what
#: the copied prompt tells the reflection model to read.
INPUTS_KEY = "Inputs"
OUTPUTS_KEY = "Generated Outputs"
FEEDBACK_KEY = "Feedback"

#: The method state key that holds the archive.
ARCHIVE_KEY = "archive"


class Feedback(ConfiguredPlugin, ABC):
    """The ``evolution.feedback`` hook: what the reflection model is told.

    Purpose: ``feedback(task, output, score)`` returns the sentence the
    reflective record shows for one minibatch example; ``GEPARecipe`` builds
    the configured class, ``ScoreFeedback`` when none is named.
    """

    @abstractmethod
    def feedback(self, task: str, output: str, score: float) -> str: ...


class ScoreFeedback(Feedback):
    """Explain a result using its numeric score."""

    def feedback(self, task: str, output: str, score: float) -> str:
        return default_feedback(task, output, score)


def default_feedback(task: str, output: str, score: float) -> str:
    """The score restated as text, for a benchmark with no richer signal.

    GEPA's own default. It is weak on purpose: a benchmark that can say
    *why* an answer was wrong should say so through ``evolution.feedback``,
    because that sentence is the entire signal reflection gets.
    """
    return f"This response scored {score:.3f}. A better response would score higher."


@dataclass(frozen=True)
class Example:
    """One minibatch example as reflection sees it: input, output, grade.

    Values: ``task`` is the prompt, ``output`` the final answer the episode
    gave, and ``score`` its grade. ``error`` is set when the episode produced
    no score; it replaces the feedback hook, because the failure is the only
    feedback there is.
    Purpose: the proposer makes one per minibatch episode and renders them
    into the reflective record.
    """

    task: str
    output: str
    score: float
    error: str = ""


@dataclass(frozen=True)
class GEPASettings:
    """The ``evolution.gepa`` section.

    Values: ``minibatch_size`` > 0 is the number of examples a parent and its
    child are compared on. ``rng_seed`` seeds the archive's generator.
    ``skip_perfect_score`` skips an iteration whose parent scored at least
    ``perfect_score`` on every example. ``max_metric_calls`` is the budget,
    ``None`` for none. ``kinds`` are the evolvable kinds, a non-empty subset
    of ``components.EVOLVABLE_KINDS``. ``trainset_size``, when set, makes the
    proposer plan each iteration over a training set of that size, GEPA's
    epoch-shuffled order (``Archive.plan``), so a driver that sends the
    planned problems as traffic walks GEPA's own order.
    Purpose: ``GEPARecipe`` parses it; ``GEPAProposer`` runs by it.
    """

    minibatch_size: int = 3
    rng_seed: int = 0
    skip_perfect_score: bool = True
    perfect_score: float = 1.0
    max_metric_calls: int | None = None
    kinds: tuple[TreeKind, ...] = (TreeKind.RULES, TreeKind.SKILL)
    trainset_size: int | None = None

    def __post_init__(self) -> None:
        if self.minibatch_size <= 0:
            raise ValueError("evolution.gepa.minibatch_size must be positive")
        if not self.kinds or any(kind not in components.EVOLVABLE_KINDS for kind in self.kinds):
            evolvable = ", ".join(kind.value for kind in components.EVOLVABLE_KINDS)
            raise ValueError(f"evolution.gepa.components must name kinds from {evolvable}")
        if self.trainset_size is not None and self.trainset_size <= 0:
            raise ValueError("evolution.gepa.trainset_size must be positive when set")


def archive_for_step(
    method_state: Mapping[str, Any], last_outcome: StepOutcome | None, *, rng_seed: int = 0
) -> Archive:
    """The archive as a step starts: the committed one, with the evaluation of its pending candidate folded in.

    The pending candidate's validation pass is the outcome of the step that
    proposed it. A pending candidate whose step never reached an evaluation
    (the backend skipped it after the proposal) has none and is dropped. A
    driver that plans the next iteration reads the committed method state and
    the outcome the same way, so it draws the plan the proposer draws.
    """
    archive = Archive.from_state(method_state.get(ARCHIVE_KEY), rng_seed=rng_seed)
    if archive.pending is None:
        return archive
    if last_outcome is not None and last_outcome.step == archive.pending_step:
        archive.settle(
            scores(last_outcome.candidate_scores), scores(last_outcome.current_scores), last_outcome.selected
        )
    else:
        archive.clear_pending()
    return archive


def scores(values: Sequence[float | None]) -> list[float]:
    """Per-task scores with failed episodes read as zero, never as missing."""
    return [0.0 if value is None else float(value) for value in values]


class GEPAProposer(Proposer):
    """GEPA's reflective mutation, one iteration per step.

    ``feedback`` writes the reflective records; ``reflection_model`` names
    the binding that reflects, the served model when the deployment declares
    none of that name.
    """

    def __init__(
        self, settings: GEPASettings, *, feedback: Feedback | None = None, reflection_model: str = "reflection"
    ) -> None:
        if feedback is not None and not isinstance(feedback, Feedback):
            raise TypeError("GEPA feedback must inherit Feedback")
        self.settings = settings
        self.feedback = ScoreFeedback() if feedback is None else feedback
        self.reflection_model = reflection_model

    def propose(self, context: ProposalContext) -> TreeProposal:
        if context.trials is None:
            raise ValueError("GEPA runs its minibatches as trial episodes; this step offers none")
        settings = self.settings
        archive = archive_for_step(context.method_state, context.last_outcome, rng_seed=settings.rng_seed)
        valset_size = len(context.evaluation_tasks)
        served_texts = components.texts_of(context.tree, settings.kinds)
        sync(archive, served_texts, valset_size)
        if settings.max_metric_calls is not None and archive.metric_calls >= settings.max_metric_calls:
            return keep(archive, {"stopped": f"the budget of {settings.max_metric_calls} metric calls is spent"})
        if settings.trainset_size is not None:
            parent = int(archive.plan(settings.trainset_size, valset_size, settings.minibatch_size)["parent"])
        else:
            parent = archive.take_parent(valset_size)
        parent_texts = archive.candidates[parent].texts
        minibatch = self.minibatch(context, archive, parent, parent_texts)
        if not minibatch:
            return keep(archive, {"skipped": "the step's traffic holds no task to run"})
        if settings.skip_perfect_score and all(example.score >= settings.perfect_score for example in minibatch):
            return keep(archive, {"skipped": "the parent scored perfectly on the minibatch"})
        keys = sorted(parent_texts)
        if not keys:
            return keep(archive, {"skipped": "the served tree holds no evolvable file"})
        key = archive.next_component(parent, keys)
        records = self.records(minibatch)
        prompt = reflection.render_prompt(parent_texts[key], records)
        reply = self.binding(context.models).chat([{"role": "user", "content": prompt}])
        new_text = reflection.extract_new_text(reply)
        row: dict[str, Any] = {
            "parent": parent,
            "served": archive.served,
            "component": key,
            "minibatch": records,
            "parent_scores": [example.score for example in minibatch],
            "prompt": prompt,
            "reply": reply,
            "child_scores": None,
            "accepted": False,
            "candidate": None,
        }
        if not new_text.strip():
            archive.log_proposal(row)
            return keep(archive, row)
        child_texts = {**parent_texts, key: new_text}
        child = components.candidate_tree(context.tree, child_texts)
        episodes = context.trials.run(child, [example.task for example in minibatch])
        archive.charge(len(episodes))
        child_scores = [0.0 if episode.score is None else float(episode.score) for episode in episodes]
        row["child_scores"] = child_scores
        if sum(child_scores) <= sum(example.score for example in minibatch):
            # The one case GEPA counts as a spent iteration: a child that lost the minibatch comparison.
            archive.reject()
            archive.log_proposal(row)
            return keep(archive, row)
        # The patch is stated against the served tree, so a child that differs from its parent but not from what
        # is being served has nothing to apply; it is dropped rather than left pending.
        patch = components.patch_to(served_texts, child_texts)
        if not patch.changes:
            archive.log_proposal(row)
            return keep(archive, row)
        row["accepted"] = True
        row["candidate"] = archive.add(child_texts, parent, child_scores, context.step)
        archive.log_proposal(row)
        return TreeProposal(patch, notes=summary(row), method_state={ARCHIVE_KEY: archive.to_state()})

    def minibatch(
        self, context: ProposalContext, archive: Archive, parent: int, parent_texts: Mapping[str, str]
    ) -> list[Example]:
        """The parent's minibatch: free from the step's traffic, or run again on the parent's tree."""
        batch = traffic(context.samples)[: self.settings.minibatch_size]
        if parent == archive.served:
            # Free of episodes, but not of budget: upstream evaluates the parent on the minibatch and counts it, so
            # the traffic that stands in for that evaluation counts the same.
            archive.charge(len(batch))
            return [Example(task, output, float(sample.metadata.get("reward"))) for sample, (task, output) in batch]
        if context.trials is None:
            raise ValueError("GEPA runs its minibatches as trial episodes; this step offers none")
        tasks = [task for _, (task, _) in batch]
        episodes = context.trials.run(components.candidate_tree(context.tree, parent_texts), tasks)
        archive.charge(len(episodes))
        return [
            Example(
                task,
                episode.reply or "",
                0.0 if episode.score is None else float(episode.score),
                "" if episode.failure is None else episode.failure.cause,
            )
            for task, episode in zip(tasks, episodes, strict=True)
        ]

    def records(self, minibatch: Sequence[Example]) -> list[dict[str, Any]]:
        """The reflective dataset: one record per minibatch example."""
        return [
            {
                INPUTS_KEY: example.task,
                OUTPUTS_KEY: example.output,
                FEEDBACK_KEY: example.error or self.feedback.feedback(example.task, example.output, example.score),
            }
            for example in minibatch
        ]

    def binding(self, models: ModelBindings) -> ModelBinding:
        """The reflection model, falling back to the model under test.

        GEPA's reflection LM is normally a stronger model than the one being
        optimized, but it does not have to be: a deployment that declares no
        ``evolution.models.reflection`` reflects with the served model, which
        is what makes the method runnable with no second endpoint at all.
        """
        try:
            return models[self.reflection_model]
        except KeyError:
            return models.served


def sync(archive: Archive, served_texts: Mapping[str, str], valset_size: int) -> int:
    """Reconcile the archive with the tree the step serves; the served candidate's index.

    A tree the archive does not recognise means the tree moved without the
    method: a first boot, an operator edit, or a rollback. Whatever it is, it
    becomes a new parentless candidate rather than being mistaken for a
    descendant of the archive's last served one.
    """
    index = archive.served
    if index is None or archive.candidates[index].texts != dict(served_texts):
        index = archive.seed(served_texts)
        # GEPA scores its seed on the validation set before the first iteration and charges it then; the step
        # runs that pass during the first evaluated step, but the budget is the same.
        archive.charge(valset_size)
    return index


def keep(archive: Archive, notes: Mapping[str, Any]) -> TreeProposal:
    """A step that proposes nothing but keeps what the iteration spent and recorded."""
    return TreeProposal(TreePatch({}), notes=summary(notes), method_state={ARCHIVE_KEY: archive.to_state()})


def summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """What the step's notes keep of an iteration: everything but the prompt and the minibatch, which the archive
    keeps in full."""
    return {key: value for key, value in row.items() if key not in ("prompt", "minibatch")}


def traffic(samples: Sequence[TrajectorySample]) -> list[tuple[TrajectorySample, tuple[str, str]]]:
    """Each sample as its prompt and the answer the served tree gave.

    A recorded request with no user message carries no task to run again, so
    it is no minibatch example and is dropped here.
    """
    pairs = [
        (sample, (prompt_of(recorded_payload(sample)), response_of(recorded_payload(sample)))) for sample in samples
    ]
    return [(sample, texts) for sample, texts in pairs if texts[0]]


def prompt_of(payload: Mapping[str, Any]) -> str:
    for message in payload.get("messages", ()):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return text_of(message.get("content"))
    return ""


def response_of(payload: Mapping[str, Any]) -> str:
    """The answer the served tree gave, from either recorded shape.

    A buffered reply is recorded as the provider's chat completion, with the
    message under ``choices``; a streamed reply is recorded as the message
    the proxy assembled from the chunks, under ``message`` beside the raw
    stream body. Pi streams, so the second shape is the common one.
    """
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return ""
    choices = response.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
    else:
        message = response.get("message")
    return text_of(message.get("content")) if isinstance(message, Mapping) else ""


def text_of(content: Any) -> str:
    """Message content as text, whether it is a string or typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping) and "text" in part)
    return ""


__all__ = [
    "ARCHIVE_KEY",
    "Feedback",
    "GEPAProposer",
    "GEPASettings",
    "ScoreFeedback",
    "archive_for_step",
    "default_feedback",
    "sync",
]
