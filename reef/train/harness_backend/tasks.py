"""The tasks each harness step evaluates on.

A step evaluates the candidate and the current tree on the same tasks: the
seed tasks (``evolution.tasks``), or the eval split of the scenario's
imported tasks as the step reads it, then the prompts promoted from failing
traffic. Promotion is
off by default. When it is on, the method's ``Promoter`` picks the prompts
(every trace's user prompt without one), and Reef screens out credentials and
instruction overrides, skips duplicates, and caps the total and each tagged
client's share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.trajectories import recorded_payload, source_record_id
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.storage.samples import TrainerSamples
from reef.train.harness_backend.contract import Promoter, SampleSource
from reef.train.harness_backend.manifest import FailureManifest
from reef.train.harness_backend.state import HarnessState


def prompt_of(sample: TrajectorySample) -> str | None:
    """The trace's last user message, the prompt an evaluation would run; ``None`` for a tool-only turn."""
    messages = recorded_payload(sample).get("messages")
    if not isinstance(messages, Sequence):
        return None
    for message in reversed(list(messages)):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, Sequence) and not isinstance(content, str):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
            )
            if text.strip():
                return text
    return None


def source_of(sample: TrajectorySample) -> SampleSource:
    """Where a sample came from: its record and its client's tag (``x-reef-tag-client``, else the session tag)."""
    metadata = recorded_payload(sample).get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    client = "untagged"
    for key in ("client", "session"):
        value = tags.get(key) if isinstance(tags, Mapping) else None
        if isinstance(value, str) and value.strip():
            client = value
            break
    return SampleSource(record=source_record_id(sample), client=client)


@dataclass(frozen=True)
class StepTasks:
    """The tasks one step evaluates, and the promotion state that goes with them.

    Values: ``tasks`` is the step's seed tasks, then every promoted prompt not
    among them, without duplicates. ``promoted_tasks`` and
    ``promoted_clients`` are the state's promotion fields after this step.
    ``screened`` ≥ 0 counts the candidate prompts the screens refused.
    Purpose: ``EvaluationTasks.for_step`` makes it; the backend evaluates on
    ``tasks`` and carries the rest into ``HarnessState``.
    """

    tasks: tuple[str, ...]
    promoted_tasks: tuple[str, ...]
    promoted_clients: Mapping[str, int]
    screened: int = 0


class EvaluationTasks:
    """The source of every step's evaluation tasks.

    ``seed_tasks`` are ``evolution.tasks``; with ``eval_samples`` set, each step
    reads the eval split of the imported tasks in that sample store instead. With ``promote_failures``, a step
    promotes up to ``max_promoted`` prompts in total and ``max_promoted_per_client``
    per tagged client, chosen by ``promoter`` (every trace's prompt without one).
    """

    def __init__(
        self,
        *,
        seed_tasks: tuple[str, ...],
        eval_samples: TrainerSamples | None = None,
        promote_failures: bool = False,
        promoter: Promoter | None = None,
        max_promoted: int = 50,
        max_promoted_per_client: int = 5,
    ) -> None:
        if not seed_tasks and eval_samples is None:
            raise ValueError("harness evolution requires a non-empty task set")
        if max_promoted < 0 or max_promoted_per_client < 0:
            raise ValueError("max_promoted_tasks and max_promoted_per_client must be at least 0")
        self.seed_tasks = seed_tasks
        self.eval_samples = eval_samples
        self.promote_failures = promote_failures
        self.promoter = promoter
        self.max_promoted = max_promoted
        self.max_promoted_per_client = max_promoted_per_client

    def seed(self) -> tuple[str, ...]:
        """The seed tasks as of now: with ``eval_samples``, the eval split of its imported tasks, in arrival order."""
        if self.eval_samples is None:
            return self.seed_tasks
        return tuple(
            str(stored.sample.task_path)
            for stored in self.eval_samples.imported_tasks(split="eval")
            if isinstance(stored.sample, TaskSample)
        )

    def for_step(
        self, samples: tuple[TrajectorySample, ...], state: HarnessState, manifest: FailureManifest | None
    ) -> StepTasks:
        """The tasks this step evaluates, after promoting this step's prompts when promotion is on."""
        seed = self.seed()
        promoted = list(state.promoted_tasks)
        counts = dict(state.promoted_clients)
        screened = 0
        if self.promote_failures:
            candidates = (
                [prompt for sample in samples if (prompt := prompt_of(sample)) is not None]
                if self.promoter is None
                else list(self.promoter.promote(samples, manifest=manifest))
            )
            clients = {prompt: source_of(sample).client for sample in samples if (prompt := prompt_of(sample))}
            seen = set(seed) | set(promoted)
            for prompt in candidates:
                if len(promoted) >= self.max_promoted:
                    break
                if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
                    continue
                if secret_shaped(prompt) or directive_shaped(prompt):
                    screened += 1
                    continue
                # Untagged traffic has no identity to count under, so only a tagged client meets the per-client cap.
                client = clients.get(prompt, "untagged")
                if self.max_promoted_per_client and client != "untagged":
                    if counts.get(client, 0) >= self.max_promoted_per_client:
                        continue
                    counts[client] = counts.get(client, 0) + 1
                promoted.append(prompt)
                seen.add(prompt)
        tasks = (*seed, *(task for task in promoted if task not in frozenset(seed)))
        return StepTasks(tasks, tuple(promoted), counts, screened)


__all__ = ["EvaluationTasks", "StepTasks", "prompt_of", "source_of"]
