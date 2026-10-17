"""Persist generation outcomes and Harbor tasks independently of their rollout.

Generation documents own task metadata. Harbor directories contain only Harbor
files, and the split manifest is rebuilt from published generation outcomes.
"""

import fcntl
import os
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from pydantic import TypeAdapter, ValidationError

from reef.core.errors import ReefError
from reef.core.tasks import GeneratedTask, HarborTaskError, TaskSplit, read_harbor_task, write_harbor_task
from reef.core.tasks.split import MANIFEST_VERSION, extend_split_by_source

STORE_VERSION = 1
STORE_FILE = "store.json"
LOCK_FILE = ".lock"
STAGING_DIRECTORY = ".staging"
TASK_DIRECTORY = "tasks"
GENERATION_DIRECTORY = "generations"
MANIFEST_FILE = "split-manifest.json"
GENERATION_PREFIX = "gen-"
SplitName = Literal["train", "eval"]


class TaskStoreError(ReefError):
    """The task store cannot be opened or its persisted state cannot be used."""


class TaskConflict(TaskStoreError):
    """A task name already identifies different task content."""


@dataclass(frozen=True)
class StoreSettings:
    """The persisted format and immutable split configuration."""

    version: int
    eval_fraction: float
    seed: int


@dataclass(frozen=True)
class CandidateOutcome:
    """One validated candidate's publication or rejection."""

    index: int
    split: SplitName | None = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.index < 0 or (self.split is None) == (not self.error):
            raise ValueError("candidate outcome requires a non-negative index and either a split or rejection reason")


@dataclass(frozen=True)
class GenerationState:
    """Recoverable candidates and validation outcomes for one generation job."""

    generation: int
    source_record_ids: tuple[str, ...]
    candidates: tuple[GeneratedTask, ...] | None = None
    outcomes: tuple[CandidateOutcome, ...] = ()
    is_complete: bool = False

    def __post_init__(self) -> None:
        indices = [outcome.index for outcome in self.outcomes]
        if len(indices) != len(set(indices)) or any(index >= len(self.candidates or ()) for index in indices):
            raise ValueError("generation outcomes must identify distinct saved candidates")
        if self.is_complete and (self.candidates is None or self.pending_indices):
            raise ValueError("a complete generation must have an outcome for every candidate")

    def candidate(self, *, index: int) -> GeneratedTask:
        """Resolve a candidate position from a persisted checkpoint."""
        if self.candidates is None or not 0 <= index < len(self.candidates):
            raise TaskStoreError(f"generation {self.generation} has no candidate {index}")
        return self.candidates[index]

    @property
    def pending_indices(self) -> tuple[int, ...]:
        """Candidate positions whose validation has not completed."""
        completed = {outcome.index for outcome in self.outcomes}
        return tuple(index for index in range(len(self.candidates or ())) if index not in completed)


@dataclass(frozen=True)
class StoredTask:
    """A published task and its fixed split, independent of execution status."""

    generated: GeneratedTask
    generation: int
    split: SplitName
    path: Path

    @property
    def source_record_ids(self) -> tuple[str, ...]:
        """Original session references, excluding the author's recorded call."""
        return tuple(
            reference
            for reference in self.generated.task.source_agent_record_ids
            if reference not in self.designer_record_ids
        )

    @property
    def designer_record_ids(self) -> tuple[str, ...]:
        """Authoring receipts kept separately from split sources."""
        receipt = self.generated.task.metadata.get("designer_record_id")
        return (receipt,) if isinstance(receipt, str) and receipt else ()


SETTINGS_DOCUMENT = TypeAdapter(StoreSettings)
GENERATION_DOCUMENT = TypeAdapter(GenerationState)
SPLIT_DOCUMENT = TypeAdapter(dict[str, object])


class TaskStore(ABC):
    """Store the task-generation checkpoints used by the existing worker."""

    @abstractmethod
    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create a generation or return its checkpoint on retry."""

    @abstractmethod
    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        """Save generated candidates before validation or rollout starts."""

    @abstractmethod
    def generation(self, *, generation: int) -> GenerationState:
        """Read the checkpoint of an existing generation."""

    @abstractmethod
    def publish_task(self, *, generation: int, index: int) -> StoredTask:
        """Save a validated task and fix its split before execution."""

    @abstractmethod
    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        """Record a candidate's terminal validation failure."""

    @abstractmethod
    def complete_generation(self, *, generation: int) -> None:
        """Mark a generation complete after every candidate has an outcome."""

    @abstractmethod
    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        """List published tasks, optionally restricted to a split."""

    @abstractmethod
    def counts(self) -> dict[str, int]:
        """Count tasks by publication state and split."""

    @abstractmethod
    def close(self) -> None:
        """Release the store's writer ownership."""


class FileTaskStore(TaskStore):
    """A single-writer file store with atomic generation checkpoints.

    Task files are published first. The generation checkpoint then makes the
    task visible. Interrupted file publication can be repeated from the saved
    candidate without another authoring call.
    """

    def __init__(self, *, root: Path, eval_fraction: float, seed: int) -> None:
        TaskSplit(train=(), eval=(), eval_fraction=eval_fraction, seed=seed)
        self.root = root
        self.task_root = root / TASK_DIRECTORY
        self.manifest_path = root / MANIFEST_FILE
        self.settings = StoreSettings(version=STORE_VERSION, eval_fraction=eval_fraction, seed=seed)
        self.lock = threading.RLock()
        self.generations: dict[int, GenerationState] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer_lock = (root / LOCK_FILE).open("a+b")
        try:
            fcntl.flock(self.writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.writer_lock.close()
            raise TaskStoreError(f"task store {root} already has a writer") from error
        try:
            self.load()
        except (OSError, TaskStoreError, ValidationError, HarborTaskError) as error:
            self.close()
            raise TaskStoreError(f"cannot open task store {root}: {error}") from error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def load(self) -> None:
        """Validate format and published task files, then rebuild the manifest."""
        settings_path = self.root / STORE_FILE
        if settings_path.exists():
            settings = SETTINGS_DOCUMENT.validate_json(settings_path.read_bytes())
            if settings.version != STORE_VERSION:
                raise TaskStoreError(f"unsupported task store version {settings.version}; expected {STORE_VERSION}")
            if settings != self.settings:
                raise TaskStoreError("task store split configuration differs from the saved eval fraction or seed")
        else:
            self.write_atomic(path=settings_path, content=SETTINGS_DOCUMENT.dump_json(self.settings))
        for staging in (self.root / STAGING_DIRECTORY, self.task_root / STAGING_DIRECTORY):
            if staging.exists():
                shutil.rmtree(staging)
        for path in sorted((self.root / GENERATION_DIRECTORY).glob(f"{GENERATION_PREFIX}*.json")):
            generation = GENERATION_DOCUMENT.validate_json(path.read_bytes())
            self.generations[generation.generation] = generation
        for saved in self.tasks():
            if read_harbor_task(path=saved.path).digest != saved.generated.task.digest:
                raise TaskStoreError(f"published task {saved.path} has a different digest from its generation")
        self.write_manifest()

    def generation(self, *, generation: int) -> GenerationState:
        """Return the persisted generation checkpoint."""
        with self.lock:
            return self.generations[generation]

    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create a checkpoint, or recover an earlier attempt of the same job."""
        if generation < 0:
            raise ValueError("generation must be non-negative")
        with self.lock:
            existing = self.generations.get(generation)
            if existing is not None:
                if existing.source_record_ids != source_record_ids:
                    raise TaskStoreError(f"generation {generation} already belongs to different source records")
                return existing
            state = GenerationState(generation=generation, source_record_ids=source_record_ids)
            self.save_generation(state=state)
            return state

    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        """Checkpoint candidates once, before any validation is attempted."""
        with self.lock:
            state = self.generations[generation]
            if state.candidates is not None:
                if state.candidates != candidates:
                    raise TaskStoreError(f"generation {generation} already has different candidates")
                return state
            state = replace(state, candidates=candidates)
            self.save_generation(state=state)
            return state

    def publish_task(self, *, generation: int, index: int) -> StoredTask:
        """Publish Harbor files and their split in a recoverable order."""
        with self.lock:
            self.require_writer()
            state = self.generations[generation]
            generated = state.candidate(index=index)
            outcomes = {outcome.index: outcome for outcome in state.outcomes}
            if index in outcomes and outcomes[index].error:
                raise TaskStoreError(f"candidate {index} of generation {generation} was rejected")
            published = {saved.generated.task.name: saved for saved in self.tasks()}
            existing = published.get(generated.task.name)
            if existing is not None and existing.generated.task.digest != generated.task.digest:
                raise TaskConflict(
                    f"task {generated.task.name!r} has a different digest: "
                    f"{existing.generated.task.digest} != {generated.task.digest}"
                )
            if index in outcomes:
                self.write_manifest()
                return published[generated.task.name]
            if existing is not None:
                split = existing.split
            else:
                tentative = StoredTask(
                    generated=generated,
                    generation=generation,
                    split="train",
                    path=self.task_root / generated.task.name,
                )
                previous = self.split()
                assigned = extend_split_by_source(
                    previous=previous,
                    previous_sources={name: saved.source_record_ids for name, saved in published.items()},
                    sources={generated.task.name: tentative.source_record_ids},
                )
                split = "eval" if generated.task.name in assigned.eval else "train"
                path = write_harbor_task(task=generated.task, root=self.task_root)
                self.sync_task(path=path)
            saved = StoredTask(
                generated=generated, generation=generation, split=split, path=self.task_root / generated.task.name
            )
            outcome = CandidateOutcome(index=index, split=split)
            self.save_generation(state=replace(state, outcomes=(*state.outcomes, outcome)))
            self.write_manifest()
            return saved

    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        """Persist one terminal rejection without changing existing placements."""
        if not reason:
            raise ValueError("task rejection requires a reason")
        with self.lock:
            state = self.generations[generation]
            state.candidate(index=index)
            outcome = CandidateOutcome(index=index, error=reason)
            previous = next((entry for entry in state.outcomes if entry.index == index), None)
            if previous is not None:
                if previous != outcome:
                    raise TaskStoreError(f"candidate {index} of generation {generation} already has another outcome")
                return
            self.save_generation(state=replace(state, outcomes=(*state.outcomes, outcome)))

    def complete_generation(self, *, generation: int) -> None:
        """Make source completion durable after all task outcomes are saved."""
        with self.lock:
            state = self.generations[generation]
            if state.candidates is None or state.pending_indices:
                raise TaskStoreError(f"generation {generation} has unfinished candidates")
            self.save_generation(state=replace(state, is_complete=True))

    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        """Return tasks whose publication is recorded in a generation document."""
        with self.lock:
            saved_by_name: dict[str, StoredTask] = {}
            for state in self.generations.values():
                for outcome in state.outcomes:
                    if outcome.split is None:
                        continue
                    generated = state.candidate(index=outcome.index)
                    saved_by_name.setdefault(
                        generated.task.name,
                        StoredTask(
                            generated=generated,
                            generation=state.generation,
                            split=outcome.split,
                            path=self.task_root / generated.task.name,
                        ),
                    )
            return tuple(saved for saved in saved_by_name.values() if split is None or saved.split == split)

    def counts(self) -> dict[str, int]:
        """Count published train/eval tasks and rejected candidates."""
        with self.lock:
            counts = Counter(
                f"{'available' if saved.split == 'train' else 'held_out'}/{saved.split}" for saved in self.tasks()
            )
            counts["rejected"] = sum(
                bool(outcome.error) for state in self.generations.values() for outcome in state.outcomes
            )
            counts["candidate"] = sum(len(state.pending_indices) for state in self.generations.values())
            return dict(counts)

    def split(self) -> TaskSplit:
        """Build the cumulative split from authoritative publication outcomes."""
        return TaskSplit(
            train=tuple(saved.generated.task.name for saved in self.tasks(split="train")),
            eval=tuple(saved.generated.task.name for saved in self.tasks(split="eval")),
            seed=self.settings.seed,
            eval_fraction=self.settings.eval_fraction,
        )

    def write_manifest(self) -> None:
        """Publish the cumulative manifest consumed by evaluation tooling."""
        with self.lock:
            split = self.split()
            document = {
                "version": MANIFEST_VERSION,
                "seed": split.seed,
                "eval_fraction": split.eval_fraction,
                "train": list(split.train),
                "eval": list(split.eval),
            }
            self.write_atomic(path=self.manifest_path, content=SPLIT_DOCUMENT.dump_json(document))

    def save_generation(self, *, state: GenerationState) -> None:
        """Replace the generation document before exposing its new state."""
        path = self.root / GENERATION_DIRECTORY / f"{GENERATION_PREFIX}{state.generation:05d}.json"
        self.write_atomic(path=path, content=GENERATION_DOCUMENT.dump_json(state))
        self.generations[state.generation] = state

    def write_atomic(self, *, path: Path, content: bytes) -> None:
        """Publish one durable file, leaving the previous checkpoint on failure."""
        with self.lock:
            self.require_writer()
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = self.root / STAGING_DIRECTORY
            staging.mkdir(exist_ok=True)
            partial = staging / uuid.uuid4().hex
            try:
                with partial.open("wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(partial, path)
                self.sync_directory(path=path.parent)
                if path.parent != self.root:
                    self.sync_directory(path=self.root)
            finally:
                partial.unlink(missing_ok=True)

    def sync_task(self, *, path: Path) -> None:
        """Flush task files and directories before publishing their metadata."""
        directories = [path]
        for entry in path.rglob("*"):
            if entry.is_dir():
                directories.append(entry)
            else:
                with entry.open("rb") as stream:
                    os.fsync(stream.fileno())
        for directory in reversed(directories):
            self.sync_directory(path=directory)
        self.sync_directory(path=path.parent)

    def sync_directory(self, *, path: Path) -> None:
        """Flush directory entries after publishing files or task directories."""
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def require_writer(self) -> None:
        """Reject late writes after worker shutdown releases store ownership."""
        if self.writer_lock.closed:
            raise TaskStoreError(f"task store {self.root} is closed; reopen it before writing")

    def close(self) -> None:
        """Release the process-level writer lock after any active write finishes."""
        with self.lock:
            self.writer_lock.close()
