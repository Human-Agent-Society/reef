"""The two seams GEPA plugs into: the proposer and the selection policy.

One GEPA iteration is split across the mechanism's step. ``GEPAProposer``
owns the half GEPA calls the reflective mutation: sample a parent from the
archive's Pareto front, evaluate it on a minibatch, reflect on one component
with the feedback that minibatch produced, evaluate the child on the same
minibatch, and accept only a strict improvement. ``GEPASelector`` owns the
other half, the full validation pass: the mechanism has already run every
``evolution.tasks`` prompt on both compositions by the time ``decide`` is
called, so the per-task scores in its evaluation are exactly GEPA's valset
evaluation, and recording them is what updates the fronts and moves the
served composition.

Two things are cheaper here than upstream and one is dearer. The parent's
minibatch is free when the parent is the served composition: the mechanism
already batched that composition's real recorded traffic, scores included,
so the proposer runs no episodes at all for it. Any other parent has to be
re-run, because no traffic was ever served from it. Dearer: the mechanism
re-runs the served composition over the whole validation set every evaluated
step, so an accepted proposal costs ``2 x |tasks|`` episodes where GEPA pays
``|valset|``. The archive counts what the method actually spends, which is
what a budget comparison against upstream reads.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Protocol

from reef.harness.descriptor import AdapterDescriptor
from reef.harness.episode import EpisodeError, EpisodeResult, run_episode
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.render import render_composition
from reef.harness.trajectory import TrajectoryError
from reef.train.cordis_backend.strategies import EpisodeScorer, Mutation
from reef.train.evaluation.contracts import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import TraceSample

from . import components, reflection
from .archive import Archive

#: The episode budget the evolution backend uses for its own scoring runs;
#: the proposer's episodes are the same work and get the same ceiling.
EPISODE_TIMEOUT_S = 600.0

#: GEPA's reflective dataset keys. The names are load bearing: they are what
#: the copied prompt tells the reflection model to read.
INPUTS_KEY = "Inputs"
OUTPUTS_KEY = "Generated Outputs"
FEEDBACK_KEY = "Feedback"


class Feedback(Protocol):
    """The ``evolution.feedback`` hook: what the reflection model is told."""

    def __call__(self, task: str, output: str, score: float) -> str: ...


class EpisodeRunner(Protocol):
    """``reef.harness.episode.run_episode``, as an injectable interface."""

    def __call__(
        self,
        descriptor: AdapterDescriptor,
        files: Mapping[str, str],
        prompt: str,
        *,
        binary: str | None = None,
        timeout: float = EPISODE_TIMEOUT_S,
    ) -> EpisodeResult: ...


def default_feedback(task: str, output: str, score: float) -> str:
    """The score restated as text, for a benchmark with no richer signal.

    GEPA's own default. It is weak on purpose: a benchmark that can say
    *why* an answer was wrong should say so through ``evolution.feedback``,
    because that sentence is the entire signal reflection gets.
    """
    return f"This response scored {score:.3f}. A better response would score higher."


@dataclass(frozen=True)
class _Example:
    """One minibatch example as reflection sees it: input, output, grade."""

    task: str
    output: str
    score: float
    #: Set when the episode never produced a trajectory; it replaces the
    #: feedback hook, because the failure is the only feedback there is.
    error: str = ""


class GEPAProposer:
    """GEPA's reflective mutation, as the evolution mechanism's ``propose``."""

    def __init__(
        self,
        *,
        archive: Archive,
        descriptor: AdapterDescriptor,
        binary: str | None,
        score_episode: EpisodeScorer,
        feedback: Feedback,
        minibatch_size: int,
        rng_seed: int,
        skip_perfect_score: bool,
        perfect_score: float,
        max_metric_calls: int | None,
        kinds: Sequence[str],
        reflection_model: str = "reflection",
        episode_runner: EpisodeRunner = run_episode,
    ) -> None:
        self._archive = archive
        self._descriptor = descriptor
        self._binary = binary
        self._score_episode = score_episode
        self._feedback = feedback
        self._minibatch_size = minibatch_size
        self._rng = random.Random(rng_seed)
        self._skip_perfect_score = skip_perfect_score
        self._perfect_score = perfect_score
        self._max_metric_calls = max_metric_calls
        self._kinds = tuple(kinds)
        self._reflection_model = reflection_model
        self._run_episode = episode_runner

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: ModelBindings,
    ) -> tuple[Mutation, ...] | None:
        archive = self._archive
        served_texts = components.texts_of(nodes, self._kinds)
        self._sync(served_texts)

        if self._max_metric_calls is not None and archive.metric_calls >= self._max_metric_calls:
            return None

        parent = archive.select_parent(self._rng)
        parent_texts = archive.candidates[parent].texts
        minibatch = self._minibatch(nodes, parent, parent_texts, samples, models)
        if not minibatch:
            return None
        if self._skip_perfect_score and all(example.score >= self._perfect_score for example in minibatch):
            return None

        keys = sorted(parent_texts)
        if not keys:
            return None
        key = archive.next_component(parent, keys)
        new_text = self._reflect(parent_texts[key], minibatch, models)
        if not new_text.strip():
            return None

        child_texts = {**parent_texts, key: new_text}
        child_scores = [self._score(nodes, child_texts, example.task, models)[0] for example in minibatch]
        if sum(child_scores) <= sum(example.score for example in minibatch):
            archive.reject()
            return None

        # The proposal is stated against the served tree, so a child that
        # differs from its parent but not from what is being served has
        # nothing to apply; it is dropped rather than left pending.
        proposal = components.mutations(served_texts, child_texts)
        if not proposal:
            return None
        archive.add(child_texts, parent, child_scores)
        return proposal

    def _sync(self, served_texts: Mapping[str, str]) -> int:
        """Reconcile the archive with the tree the mechanism is actually serving.

        A tree the archive does not recognise means the composition moved
        without the method: a first boot, an operator edit, or a rollback.
        Whatever it is, it becomes a new parentless candidate rather than
        being mistaken for a descendant of the archive's last served one.
        """
        archive = self._archive
        index = archive.served
        if index is None or archive.candidates[index].texts != dict(served_texts):
            return archive.seed(served_texts)
        return index

    def _minibatch(
        self,
        nodes: tuple[tuple[str, object], ...],
        parent: int,
        parent_texts: Mapping[str, str],
        samples: Sequence[TraceSample],
        models: ModelBindings,
    ) -> list[_Example]:
        """The parent's minibatch: free from traffic, or re-run on the parent."""
        batch = _traffic(samples)[: self._minibatch_size]
        if parent == self._archive.served:
            return [_Example(task, output, float(sample.score)) for sample, (task, output) in batch]
        examples = []
        for _, (task, _) in batch:
            score, output, error = self._score(nodes, parent_texts, task, models)
            examples.append(_Example(task, output, score, error))
        return examples

    def _score(
        self,
        nodes: tuple[tuple[str, object], ...],
        texts: Mapping[str, str],
        task: str,
        models: ModelBindings,
    ) -> tuple[float, str, str]:
        """Run one task on a composition of ``texts`` and charge one metric call."""
        files = render_composition(
            (*_substituted(nodes, texts), *models.served.compose_nodes(self._descriptor)),
            self._descriptor,
        )
        self._archive.charge(1)
        try:
            result = self._run_episode(self._descriptor, files, task, binary=self._binary, timeout=EPISODE_TIMEOUT_S)
        except (EpisodeError, TrajectoryError) as error:
            return 0.0, "", str(error)
        return float(self._score_episode(task, result)), _episode_output(result), ""

    def _reflect(self, current_text: str, minibatch: Sequence[_Example], models: ModelBindings) -> str:
        records = [
            {
                INPUTS_KEY: example.task,
                OUTPUTS_KEY: example.output,
                FEEDBACK_KEY: example.error or self._feedback(example.task, example.output, example.score),
            }
            for example in minibatch
        ]
        prompt = reflection.render_prompt(current_text, records)
        reply = self._binding(models).chat([{"role": "user", "content": prompt}])
        return reflection.extract_new_text(reply)

    def _binding(self, models: ModelBindings) -> ModelBinding:
        """The reflection model, falling back to the model under test.

        GEPA's reflection LM is normally a stronger model than the one being
        optimized, but it does not have to be: a deployment that declares no
        ``evolution.models.reflection`` reflects with the served model, which
        is what makes the method runnable with no second endpoint at all.
        """
        try:
            return models[self._reflection_model]
        except KeyError:
            return models.served


class GEPASelector:
    """GEPA's valset pass and Pareto update, as the selection policy.

    Selection is strict mean improvement over the served composition, which
    keeps the served tree equal to the archive's best candidate - GEPA's
    ``best_idx``. The fronts still record everything, so a candidate that
    loses on the mean can go on parenting proposals through its specialties.
    """

    def __init__(self, archive: Archive) -> None:
        self._archive = archive

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        archive = self._archive
        candidate_scores = _scores(evaluation.metrics.get("candidate_scores", ()))
        served_scores = _scores(evaluation.metrics.get("current_scores", ()))

        pending = archive.pending
        parent = None
        if pending is not None:
            archive.record_validation(pending, candidate_scores)
            archive.charge(len(candidate_scores))
            parent = archive.candidates[pending].parent
        served = archive.served
        if served is not None and archive.candidates[served].val_scores is None:
            # The seed's own validation arrives with the first evaluated
            # step; the mechanism ran those episodes, so the budget pays.
            archive.record_validation(served, served_scores)
            archive.charge(len(served_scores))

        # GEPA scores every candidate on the validation set once and keeps
        # that number; the mechanism's fresh re-run of the served tree is
        # redundant work, not a second opinion. Comparing against the stored
        # mean is what keeps the served tree equal to the archive's best.
        candidate_mean = fmean(candidate_scores) if candidate_scores else 0.0
        served_mean = archive.mean_val(served) if served is not None else 0.0
        selected = candidate_mean > served_mean
        if selected and pending is not None:
            archive.serve(pending)
        archive.clear_pending()

        fronts = archive.fronts()
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="gepa",
            policy_version="1",
            reason=(
                f"candidate scored {candidate_mean:.4f} against {served_mean:.4f} served "
                f"over {len(candidate_scores)} validation tasks"
            ),
            evaluation=evaluation,
            metrics={
                "candidate_val_mean": candidate_mean,
                "served_val_mean": served_mean,
                "parent": parent,
                "archive_size": len(archive.candidates),
                "front_size": len({index for front in fronts.values() for index in front}),
                "metric_calls": archive.metric_calls,
            },
        )


def _scores(values: Any) -> list[float]:
    """Per-task scores with failed episodes read as zero, never as missing."""
    return [0.0 if value is None else float(value) for value in values]


def _traffic(samples: Sequence[TraceSample]) -> list[tuple[TraceSample, tuple[str, str]]]:
    """Each sample as its prompt and the answer the served composition gave.

    A recorded request with no user message carries no task to re-run, so it
    is not a minibatch example at all and is dropped here.
    """
    pairs = [(sample, (_prompt_of(sample.payload), _response_of(sample.payload))) for sample in samples]
    return [(sample, texts) for sample, texts in pairs if texts[0]]


def _prompt_of(payload: Mapping[str, Any]) -> str:
    for message in payload.get("messages", ()):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _text(message.get("content"))
    return ""


def _response_of(payload: Mapping[str, Any]) -> str:
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    return _text(message.get("content")) if isinstance(message, Mapping) else ""


def _text(content: Any) -> str:
    """Message content as text, whether it is a string or typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping) and "text" in part)
    return ""


def _episode_output(result: EpisodeResult) -> str:
    """The last assistant text in a trajectory, across both event shapes.

    Harnesses log either a bare chat message or one wrapped in a typed
    event; reflection only needs the text the agent finally produced, so
    both are read and anything else reads as no output at all.
    """
    for event in reversed(result.trajectory):
        message = event.get("message") if isinstance(event.get("message"), Mapping) else event
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            text = _text(message.get("content"))
            if text:
                return text
    return ""


def _substituted(
    nodes: Sequence[tuple[str, object]],
    texts: Mapping[str, str],
) -> tuple[tuple[str, Any], ...]:
    """The served tree with the candidate's texts swapped into its nodes.

    Only the evolvable nodes change: a composition is more than its
    instructions, and the config and extension nodes that make episodes
    reach a model have to travel with every candidate the proposer runs.
    """
    rendered: list[tuple[str, Any]] = []
    for kind, config in nodes:
        options = config if isinstance(config, Mapping) else {}
        key = components.component_key(kind, options)
        if key is not None and key in texts:
            rendered.append((kind, {**options, "text": texts[key]}))
        else:
            rendered.append((kind, config))
    return tuple(rendered)
