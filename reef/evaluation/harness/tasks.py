"""The tasks each harness step evaluates on.

An :class:`EvaluationTask` is one task with its id: an instruction with an
optional answer, or a Harbor task directory, whose verifier holds its answer.
A step evaluates the candidate and the head tree on the same tasks: the
configured tasks (``evolution.evaluation.tasks``, ``task_file``,
``harbor_tasks``, or a task manifest), or the eval split of the scenario's
imported tasks as the step reads it, then the instructions collected from
failing traffic. Collection is off by default. When it is on, a
``TaskCollector`` picks the instructions (every trace's last user message
without one), and Reef screens out credentials and instruction overrides,
skips duplicates, and caps the total and each tagged client's share.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.plugins import ConfiguredPlugin
from reef.core.trajectories import recorded_payload
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.storage.samples import TrainerSamples
from reef.storage.tasks import files_digest

#: The fields of one instruction task, in ``evolution.evaluation.tasks`` and each line of ``task_file``.
TASK_FIELDS = ("id", "instruction", "answer", "metadata")


@dataclass(frozen=True)
class EvaluationTask:
    """One task an evaluation runs, with what its scorer reads besides the run.

    Values: ``id`` is a non-empty name, unique among one step's tasks. Exactly
    one of ``instruction`` (non-empty text, which the adapter's ``{task}`` slot
    receives) and ``directory`` (a Harbor task directory) is set. ``answer`` is
    ``None`` or the expected answer of an instruction task; a directory task has
    none, because its verifier holds it. ``metadata`` maps names to JSON values
    for the scorer. A Harbor task's id is its directory's path, and an
    instruction without a configured id has ``instruction_id(instruction)``.
    Purpose: the evaluation settings, the imported tasks and the step's
    collection make them; the evaluator runs ``text`` and gives the task to the
    ``TaskScorer``. A method sees each task :meth:`without_answer`.
    """

    id: str
    instruction: str | None = None
    directory: Path | None = None
    answer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("an evaluation task needs a non-empty id")
        if (self.instruction is None) == (self.directory is None):
            raise ValueError(f"evaluation task {self.id!r} holds exactly one of an instruction and a directory")
        if self.instruction is not None and (not isinstance(self.instruction, str) or not self.instruction.strip()):
            raise ValueError(f"evaluation task {self.id!r} needs a non-empty instruction")
        if self.directory is not None and not isinstance(self.directory, Path):
            raise TypeError(f"evaluation task {self.id!r} names its directory as a Path")
        if self.answer is not None and (self.directory is not None or not isinstance(self.answer, str)):
            raise ValueError(f"evaluation task {self.id!r}: only an instruction task has an answer, and it is text")
        if not isinstance(self.metadata, Mapping) or not all(isinstance(key, str) for key in self.metadata):
            raise TypeError(f"evaluation task {self.id!r} needs metadata that maps names to values")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"evaluation task {self.id!r} needs metadata of JSON values: {exc}") from exc

    @classmethod
    def of_instruction(cls, instruction: str) -> Self:
        """The instruction as a task without an answer, named by ``instruction_id``."""
        return cls(instruction_id(instruction), instruction=instruction)

    @classmethod
    def of_directory(cls, directory: Path | str) -> Self:
        """The Harbor task directory as a task, named by its path."""
        return cls(str(directory), directory=Path(directory))

    @classmethod
    def parse(cls, value: object, *, setting: str) -> Self:
        """One instruction task from its configured mapping (``TASK_FIELDS``); ``ValueError`` names ``setting``."""
        if not isinstance(value, Mapping):
            raise ValueError(f"{setting} must be a mapping with id and instruction")
        unknown = set(value) - set(TASK_FIELDS)
        if unknown:
            raise ValueError(f"{setting} takes {', '.join(TASK_FIELDS)}, not {', '.join(sorted(map(str, unknown)))}")
        task_id, instruction, answer = value.get("id"), value.get("instruction"), value.get("answer")
        if not isinstance(task_id, str) or not isinstance(instruction, str):
            raise ValueError(f"{setting} needs an id and an instruction, each text")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"{setting}.answer must be text: quote a number")
        try:
            return cls(task_id, instruction=instruction, answer=answer, metadata=value.get("metadata", {}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{setting}: {exc}") from exc

    @property
    def text(self) -> str:
        """What the run plays: the instruction, or the task directory's path."""
        return self.instruction if self.instruction is not None else str(self.directory)

    def key(self) -> str:
        """What identifies the task's content for the score cache: the digest of a directory's files, or of the
        instruction, answer and metadata, so that a changed answer is a new task."""
        if self.directory is not None:
            return files_digest(self.directory)
        content = {"instruction": self.instruction, "answer": self.answer, "metadata": self.metadata}
        return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def without_answer(self) -> Self:
        """The task as a method sees it: no answer and no metadata, so a proposal cannot copy them into the tree."""
        return replace(self, answer=None, metadata={})

    def to_json(self) -> dict[str, str]:
        """The task as the workspace's ``evaluation.json`` lists it: its id, and its instruction or directory."""
        if self.directory is not None:
            return {"id": self.id, "directory": str(self.directory)}
        return {"id": self.id, "instruction": self.text}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        """The inverse of :meth:`to_json`."""
        if "directory" in value:
            return cls(str(value["id"]), directory=Path(value["directory"]))
        return cls(str(value["id"]), instruction=value["instruction"])


def instruction_id(instruction: str) -> str:
    """The id of an instruction that no configuration names: ``instruction-`` and a prefix of its SHA-256."""
    return "instruction-" + hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]


def read_task_file(path: Path) -> tuple[EvaluationTask, ...]:
    """The instruction tasks of a JSONL file, one ``TASK_FIELDS`` object per non-blank line, in file order.

    ``ValueError`` names the file and line of an unreadable or invalid task, and of a repeated id.
    """
    tasks: list[EvaluationTask] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read the task file {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not a JSON object: {exc}") from exc
        tasks.append(EvaluationTask.parse(value, setting=f"{path}:{number}"))
    return unique_tasks(tasks, source=str(path))


def unique_tasks(tasks: Sequence[EvaluationTask], *, source: str) -> tuple[EvaluationTask, ...]:
    """``tasks`` as a tuple; ``ValueError`` names ``source`` when two of them share an id."""
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"{source} names the task id {task.id!r} twice")
        seen.add(task.id)
    return tuple(tasks)


class TaskCollector(ConfiguredPlugin, ABC):
    """Choose which trace instructions become permanent evaluation tasks.

    ``collect`` receives a step's trace samples and returns the instructions to
    collect. Reef still removes duplicates, screens for credentials and
    instruction overrides, and caps them.
    Purpose: a deployment names the class in ``evolution.evaluation.collector``;
    ``EvaluationTasks`` calls it in a step that collects failures.
    """

    @abstractmethod
    def collect(self, samples: tuple[TrajectorySample, ...]) -> Sequence[str]:
        """The instructions this step should collect into the evaluation tasks."""


def instruction_of(sample: TrajectorySample) -> str | None:
    """The trace's last user message, the instruction an evaluation would run; ``None`` for a tool-only turn."""
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

    Values: ``tasks`` is the step's seed tasks, then each collected instruction
    whose task is not among them, with distinct ids. ``collected_tasks`` are the
    collected instructions after this step, in collection order, and
    ``collected_clients`` counts them per tagged client. ``screened`` ≥ 0
    counts the candidate instructions the screens refused.
    Purpose: ``EvaluationTasks.for_step`` makes it; the backend evaluates on
    ``tasks`` and carries the rest into its state.
    """

    tasks: tuple[EvaluationTask, ...]
    collected_tasks: tuple[str, ...]
    collected_clients: Mapping[str, int]
    screened: int = 0


class EvaluationTasks:
    """The source of every step's evaluation tasks.

    ``seed_tasks`` are the configured tasks, with distinct ids; with
    ``samples`` set, each step reads the eval split of the scenario's imported
    tasks in those samples instead. With ``collect_failures``, a step collects
    up to ``max_collected`` instructions in total and
    ``max_collected_per_client`` per tagged client, chosen by ``collector``
    (every trace's last user message without one).
    """

    def __init__(
        self,
        *,
        seed_tasks: tuple[EvaluationTask, ...],
        samples: TrainerSamples | None = None,
        collect_failures: bool = False,
        collector: TaskCollector | None = None,
        max_collected: int = 50,
        max_collected_per_client: int = 5,
    ) -> None:
        if max_collected < 0 or max_collected_per_client < 0:
            raise ValueError("max_collected_tasks and max_collected_per_client must be at least 0")
        self.seed_tasks = unique_tasks(seed_tasks, source="the seed tasks")
        self.samples = samples
        self.collect_failures = collect_failures
        self.collector = collector
        self.max_collected = max_collected
        self.max_collected_per_client = max_collected_per_client

    def seed(self) -> tuple[EvaluationTask, ...]:
        """The seed tasks as of now: the eval split of the imported tasks when they are set, in arrival order."""
        if self.samples is None:
            return self.seed_tasks
        return tuple(
            EvaluationTask.of_directory(stored.sample.task_path)
            for stored in self.samples.imported_tasks(split="eval")
            if isinstance(stored.sample, TaskSample)
        )

    def for_step(
        self,
        samples: tuple[TrajectorySample, ...],
        collected_tasks: Sequence[str],
        collected_clients: Mapping[str, int],
    ) -> StepTasks:
        """The tasks this step evaluates, after collecting this step's instructions when collection is on.

        ``collected_tasks`` and ``collected_clients`` are the collection state the previous step committed.
        """
        seed = self.seed()
        collected = list(collected_tasks)
        counts = dict(collected_clients)
        screened = 0
        if self.collect_failures:
            candidates = (
                [instruction for sample in samples if (instruction := instruction_of(sample)) is not None]
                if self.collector is None
                else list(self.collector.collect(samples))
            )
            clients = {instruction: client_of(sample) for sample in samples if (instruction := instruction_of(sample))}
            seen = {task.text for task in seed} | set(collected)
            for instruction in candidates:
                if len(collected) >= self.max_collected:
                    break
                if not isinstance(instruction, str) or not instruction.strip() or instruction in seen:
                    continue
                if secret_shaped(instruction) or directive_shaped(instruction):
                    screened += 1
                    continue
                # Untagged traffic has no identity to count under, so only a tagged client meets the per-client cap.
                client = clients.get(instruction, "untagged")
                if self.max_collected_per_client and client != "untagged":
                    if counts.get(client, 0) >= self.max_collected_per_client:
                        continue
                    counts[client] = counts.get(client, 0) + 1
                collected.append(instruction)
                seen.add(instruction)
        seed_texts = frozenset(task.text for task in seed)
        tasks = (*seed, *(EvaluationTask.of_instruction(text) for text in collected if text not in seed_texts))
        return StepTasks(tasks, tuple(collected), counts, screened)


__all__ = [
    "EvaluationTask",
    "EvaluationTasks",
    "StepTasks",
    "TaskCollector",
    "client_of",
    "instruction_id",
    "instruction_of",
    "read_task_file",
]
