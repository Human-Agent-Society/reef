"""The tasks each harness step evaluates on.

A step evaluates the candidate and the head tree on the same tasks: the
configured tasks (``evolution.evaluation.tasks``, or a task manifest), or the
eval split of the scenario's imported tasks as the step reads it, then the prompts collected
from failing traffic. Collection is off by default. When it is on, a
``TaskCollector`` picks the prompts (every trace's user prompt without one), and
Reef screens out credentials and instruction overrides, skips duplicates, and
caps the total and each tagged client's share.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.plugins import ConfiguredPlugin
from reef.core.trajectories import recorded_payload
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.storage.samples import TrainerSamples


class TaskCollector(ConfiguredPlugin, ABC):
    """Choose which trace prompts become permanent evaluation tasks.

    ``collect`` receives a step's trace samples and returns the prompts to
    collect. Reef still removes duplicates, screens for credentials and
    instruction overrides, and caps them.
    Purpose: a deployment names the class in ``evolution.evaluation.collector``;
    ``EvaluationTasks`` calls it in a step that collects failures.
    """

    @abstractmethod
    def collect(self, samples: tuple[TrajectorySample, ...]) -> Sequence[str]:
        """The prompts this step should collect into the evaluation tasks."""


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


def client_of(sample: TrajectorySample) -> str:
    """The client a sample came from: its ``x-reef-tag-client`` tag, else its session tag, else ``"untagged"``."""
    metadata = recorded_payload(sample).get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    for key in ("client", "session"):
        value = tags.get(key) if isinstance(tags, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value
    return "untagged"


@dataclass(frozen=True)
class StepTasks:
    """The tasks one step evaluates, and the collection state that goes with them.

    Values: ``tasks`` is the step's seed tasks, then every collected prompt not
    among them, without duplicates. ``collected_tasks`` are the collected
    prompts after this step, in collection order, and ``collected_clients``
    counts them per tagged client. ``screened`` ≥ 0 counts the candidate
    prompts the screens refused.
    Purpose: ``EvaluationTasks.for_step`` makes it; the backend evaluates on
    ``tasks`` and carries the rest into its state.
    """

    tasks: tuple[str, ...]
    collected_tasks: tuple[str, ...]
    collected_clients: Mapping[str, int]
    screened: int = 0


class EvaluationTasks:
    """The source of every step's evaluation tasks.

    ``seed_tasks`` are the configured tasks; with ``eval_samples`` set, each
    step reads the eval split of the imported tasks in that sample store
    instead. With ``collect_failures``, a step collects up to
    ``max_collected`` prompts in total and ``max_collected_per_client`` per
    tagged client, chosen by ``collector`` (every trace's prompt without one).
    """

    def __init__(
        self,
        *,
        seed_tasks: tuple[str, ...],
        eval_samples: TrainerSamples | None = None,
        collect_failures: bool = False,
        collector: TaskCollector | None = None,
        max_collected: int = 50,
        max_collected_per_client: int = 5,
    ) -> None:
        if max_collected < 0 or max_collected_per_client < 0:
            raise ValueError("max_collected_tasks and max_collected_per_client must be at least 0")
        self.seed_tasks = seed_tasks
        self.eval_samples = eval_samples
        self.collect_failures = collect_failures
        self.collector = collector
        self.max_collected = max_collected
        self.max_collected_per_client = max_collected_per_client

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
        self,
        samples: tuple[TrajectorySample, ...],
        collected_tasks: Sequence[str],
        collected_clients: Mapping[str, int],
    ) -> StepTasks:
        """The tasks this step evaluates, after collecting this step's prompts when collection is on.

        ``collected_tasks`` and ``collected_clients`` are the collection state the previous step committed.
        """
        seed = self.seed()
        collected = list(collected_tasks)
        counts = dict(collected_clients)
        screened = 0
        if self.collect_failures:
            candidates = (
                [prompt for sample in samples if (prompt := prompt_of(sample)) is not None]
                if self.collector is None
                else list(self.collector.collect(samples))
            )
            clients = {prompt: client_of(sample) for sample in samples if (prompt := prompt_of(sample))}
            seen = set(seed) | set(collected)
            for prompt in candidates:
                if len(collected) >= self.max_collected:
                    break
                if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
                    continue
                if secret_shaped(prompt) or directive_shaped(prompt):
                    screened += 1
                    continue
                # Untagged traffic has no identity to count under, so only a tagged client meets the per-client cap.
                client = clients.get(prompt, "untagged")
                if self.max_collected_per_client and client != "untagged":
                    if counts.get(client, 0) >= self.max_collected_per_client:
                        continue
                    counts[client] = counts.get(client, 0) + 1
                collected.append(prompt)
                seen.add(prompt)
        tasks = (*seed, *(task for task in collected if task not in frozenset(seed)))
        return StepTasks(tasks, tuple(collected), counts, screened)


__all__ = ["EvaluationTasks", "StepTasks", "TaskCollector", "client_of", "prompt_of"]
