from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.artifact_ref import RuntimeLoadSpan


@dataclass(frozen=True)
class TrainingBatch:
    """Base type for every batch flowing from a processor to a preparer.

    Carries only the batch identity; each subclass adds its concrete payload
    (tokenized policy samples, grouped comparison sets, raw recorded traces).
    Concrete subclasses provide wiring safety between processors and backend
    algorithms or local artifact backends.
    """

    batch_id: str


@dataclass(frozen=True)
class PolicySample:
    """One scored rollout, tokenized for policy training.

    A sample normally represents one model call. Shared trajectory shaping may
    assemble multiple ordered calls into one sample before a training recipe
    decides whether it supports that representation.

    The first five fields are what every policy objective needs. The rest are
    optional and stay at their defaults when a method has no use for them:

    * ``action_mask`` — per response token, ``1`` for a model-generated (action)
      token and ``0`` for an environment/observation token. Methods that
      propagate advantage over actions only set it; ``loss_mask`` still
      selects which tokens receive gradient, and for a single-turn response
      the two masks are identical. Empty means "no observation boundaries
      declared".
    * ``rollout_created_at`` — wall-clock time the rollout was recorded, so a
      backend can report queue age (train time - created at). With
      ``runtime_load_id`` (the producing version, from the serving
      ``artifact_ref``) it also gives policy lag.
    * ``turn_count`` — number of ordered inference calls represented by this
      sample. Values greater than one mark a multi-turn trajectory. This is
      Reef-side provenance and is not part of the Slime training payload.
    * ``topk_indices`` / ``topk_log_probs`` — the generation-time top-K vocab
      ids and log-probs per response token, present when the serving backend
      captures them (``capture_topk``). Objectives that compare the rollout
      distribution against another model's use them as the index set.
    * ``extras`` — per-sample channels a method's processor attaches for its
      own loss family to read in ``shape_sample_row``. Nothing shared reads
      them.

    ``rollout_log_probs`` are the ``logπrollout`` behaviour proxy; importance
    ratios ``exp(logπθ - logπrollout)`` are formed from them at train time.
    """

    source_agent_record_id: str
    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    reward: float
    runtime_load_id: str | None = None
    action_mask: tuple[int, ...] = ()
    rollout_created_at: float | None = None
    turn_count: int = 1
    topk_indices: tuple[tuple[int, ...], ...] = ()
    topk_log_probs: tuple[tuple[float, ...], ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    runtime_load_spans: tuple[RuntimeLoadSpan, ...] = ()

    @property
    def is_multi_turn(self) -> bool:
        """Whether shared trajectory assembly joined multiple model calls."""
        return self.turn_count > 1


@dataclass(frozen=True)
class PolicyBatch(TrainingBatch):
    samples: tuple[PolicySample, ...]


@dataclass(frozen=True)
class GroupedPolicyBatch(TrainingBatch):
    comparison_sets: tuple[tuple[PolicySample, ...], ...]


def policy_samples(batch: TrainingBatch) -> tuple[PolicySample, ...]:
    """Flatten either policy batch shape into its ordered samples.

    Grouped batches are flattened in comparison-set order. Batch types that
    carry no policy samples (e.g. trace batches) are a caller error.
    """
    if isinstance(batch, PolicyBatch):
        return batch.samples
    if isinstance(batch, GroupedPolicyBatch):
        return tuple(sample for group in batch.comparison_sets for sample in group)
    raise TypeError(f"batch type {type(batch).__name__} carries no policy samples")


@dataclass(frozen=True)
class TraceSample:
    """One recorded exchange, exactly as served, with its reported score.

    A report referencing one request batches as a sample whose ``payload``
    is that request. A report referencing several batches as one sample
    whose ``trajectory`` holds every referenced payload in reference order
    and whose ``payload`` is the last of them, so single-payload consumers
    keep seeing the exchange that carries the full conversation.
    ``feedback`` is the report's feedback field, verbatim. ``score`` is
    ``None`` for a sample batched from recorded traffic without a report.
    """

    source_agent_record_id: str
    payload: Mapping[str, Any]
    score: float | None
    feedback: str | Mapping[str, Any] | None = None
    trajectory: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TraceBatch(TrainingBatch):
    samples: tuple[TraceSample, ...]
