"""Split tasks into a train, an eval and a test split by the records they came from.

Two tasks made from a shared agent record go to the same split, so nothing
the train split saw reappears, reworded, in an evaluation. :func:`assign_splits`
places a task by the tasks it shares records with, its parents and a keyed hash
of its name, so a task that shares no record has its split known before play,
and a manifest that lists a task keeps it there as tasks come and go;
:func:`split_by_source` splits a closed set once. The result is written as a
manifest the evaluation reads; only reports read the test split.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reef.core.errors import ReefError
from reef.core.tasks.harbor import TASK_NAME_PATTERN, HarborTask, HarborTaskError, read_harbor_task

SplitName = Literal["train", "eval", "test"]
STAGING_DIRECTORY = ".staging"
#: The keys of each manifest version; version 2 adds the test split.
MANIFEST_KEYS: dict[int, tuple[str, ...]] = {
    1: ("version", "seed", "eval_fraction", "train", "eval"),
    2: ("version", "seed", "eval_fraction", "test_fraction", "train", "eval", "test"),
}


class TaskSplitError(ReefError):
    """A split request or manifest that cannot be honored."""


@dataclass(frozen=True)
class TaskSplit:
    """Task names in each split, sorted, with the parameters that produced them."""

    train: tuple[str, ...]
    eval: tuple[str, ...]
    seed: int
    eval_fraction: float
    test: tuple[str, ...] = ()
    test_fraction: float = 0.0

    def __post_init__(self) -> None:
        train = checked_split("train", self.train)
        eval_ = checked_split("eval", self.eval)
        test = checked_split("test", self.test)
        if set(train) & set(eval_):
            raise TaskSplitError("a task cannot be in both the train split and the eval split")
        if set(test) & (set(train) | set(eval_)):
            raise TaskSplitError("a task cannot be in both the test split and another split")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TaskSplitError("seed must be an integer")
        for label, fraction in (("eval_fraction", self.eval_fraction), ("test_fraction", self.test_fraction)):
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
                raise TaskSplitError(f"{label} must be a number between 0 and 1")
        if self.eval_fraction + self.test_fraction > 1:
            raise TaskSplitError("eval_fraction and test_fraction must add up to at most 1")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "eval", eval_)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "eval_fraction", float(self.eval_fraction))
        object.__setattr__(self, "test_fraction", float(self.test_fraction))

    def split_of(self, name: str) -> SplitName | None:
        """The split that lists ``name``, or None when none does."""
        if name in self.train:
            return "train"
        if name in self.eval:
            return "eval"
        if name in self.test:
            return "test"
        return None


def checked_split(split: str, names: object) -> tuple[str, ...]:
    """One split as a sorted tuple of distinct task names."""
    if isinstance(names, str) or not isinstance(names, Iterable):
        raise TaskSplitError(f"{split} must be a sequence of task names")
    listed = tuple(names)
    if any(not isinstance(name, str) or not TASK_NAME_PATTERN.fullmatch(name) or ".." in name for name in listed):
        raise TaskSplitError(
            f"{split} must be a sequence of task names (a task name matches {TASK_NAME_PATTERN.pattern})"
        )
    if len(set(listed)) != len(listed):
        raise TaskSplitError(f"{split} lists a task twice")
    return tuple(sorted(listed))


def split_by_source(sources: Mapping[str, Iterable[str]], *, eval_fraction: float, seed: int) -> TaskSplit:
    """Assign each group of tasks that share a source record to one split; the seed fixes the draw.

    ``sources`` maps a task name to the agent record ids it was made from.
    Groups are drawn in seeded random order into the eval split until it
    holds at least ``eval_fraction`` of the tasks, then the rest train. This
    splits a closed set once: a task's split depends on the other tasks, so
    a set that grows is split with :func:`assign_splits`.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TaskSplitError("seed must be an integer")
    if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, (int, float)) or not 0 <= eval_fraction <= 1:
        raise TaskSplitError("eval_fraction must be a number between 0 and 1")
    if not isinstance(sources, Mapping) or any(not isinstance(name, str) or not name for name in sources):
        raise TaskSplitError("sources must map non-empty task names to their record ids")
    record_ids_by_task: dict[str, tuple[str, ...]] = {}
    for name, record_ids in sources.items():
        try:
            listed = tuple(record_ids)
        except TypeError:
            raise TaskSplitError(f"sources of {name!r} must be a sequence of non-empty record ids") from None
        if isinstance(record_ids, str) or any(not isinstance(record_id, str) or not record_id for record_id in listed):
            raise TaskSplitError(f"sources of {name!r} must be a sequence of non-empty record ids")
        record_ids_by_task[name] = listed
    names = list(record_ids_by_task)
    order = sorted(task_groups(record_ids_by_task), key=lambda group: group[0])
    random.Random(seed).shuffle(order)
    # ceil with a small slack absorbs float noise such as 7.000000000000001; any positive fraction holds a group.
    target = 0 if eval_fraction == 0 else max(1, math.ceil(eval_fraction * len(names) - 1e-9))
    held: list[str] = []
    for group in order:
        if len(held) >= target:
            break
        held.extend(group)
    eval_names = frozenset(held)
    return TaskSplit(
        train=tuple(sorted(name for name in names if name not in eval_names)),
        eval=tuple(sorted(eval_names)),
        seed=seed,
        eval_fraction=float(eval_fraction),
    )


def hashed_split(seed: int, name: str, *, eval_fraction: float, test_fraction: float) -> SplitName:
    """HMAC-SHA256 keyed by the seed over the name, read as a number in [0, 1): test, then eval, then train."""
    digest = hmac.new(str(seed).encode("ascii"), name.encode("utf-8"), hashlib.sha256).digest()
    # The top 53 bits, so the number is exact as a float and never rounds up to 1.
    position = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
    if position < test_fraction:
        return "test"
    if position < test_fraction + eval_fraction:
        return "eval"
    return "train"


def assign_splits(
    tasks: Sequence[HarborTask],
    *,
    seed: int,
    eval_fraction: float,
    test_fraction: float = 0.0,
    pinned: TaskSplit | None = None,
) -> TaskSplit:
    """Place each task by its pin, the tasks it shares source records with and its parents, else by a keyed hash.

    A task ``pinned`` lists keeps its split. The other tasks of a group that
    shares source records take the splits of the group's pinned tasks and of
    every eval or test parent, a task among ``tasks`` whose digest a member
    lists in ``parents``: one split, they take it; more than one, they are
    left out of every split; none, :func:`hashed_split` of the group's first
    name decides. A group with a parent no task among ``tasks`` has is left
    out too, since that parent may be an eval or test task. So a task that
    shares no source record goes where its own name hashes, known before play,
    while a task not yet pinned can move when a task sharing its records
    arrives. A pinned name that is not among ``tasks`` is not listed.
    """
    parameters = TaskSplit((), (), seed, eval_fraction, (), test_fraction)  # checks the seed and the fractions
    if pinned is not None and not isinstance(pinned, TaskSplit):
        raise TaskSplitError("pinned must be a TaskSplit")
    tasks_by_name: dict[str, HarborTask] = {}
    for task in tasks:
        if not isinstance(task, HarborTask):
            raise TaskSplitError("tasks must be HarborTask values")
        if task.name in tasks_by_name:
            raise TaskSplitError(f"two tasks share the name {task.name!r}")
        tasks_by_name[task.name] = task
    placed: dict[str, SplitName | None] = {}
    if pinned is not None:
        pinned_names: dict[SplitName, tuple[str, ...]] = {
            "train": pinned.train,
            "eval": pinned.eval,
            "test": pinned.test,
        }
        placed = {name: split for split, names in pinned_names.items() for name in names if name in tasks_by_name}
    name_by_digest = {task.digest: name for name, task in tasks_by_name.items()}
    pending: list[tuple[list[str], set[str]]] = []
    for group in task_groups({name: task.source_agent_record_ids for name, task in tasks_by_name.items()}):
        if all(name in placed for name in group):
            continue
        parent_digests = {parent for name in group for parent in tasks_by_name[name].parents}
        if not parent_digests <= name_by_digest.keys():
            placed.update({name: None for name in group if name not in placed})
            continue
        parent_names = {name_by_digest[digest] for digest in parent_digests}
        pending.append((group, parent_names - set(group)))
    # A group waits for its parents' groups. When none can go, parents cross between groups made together: no
    # order places them, so their new tasks are left out.
    while pending:
        waiting: list[tuple[list[str], set[str]]] = []
        for group, parent_names in pending:
            if not parent_names <= placed.keys():
                waiting.append((group, parent_names))
                continue
            # A train parent imposes nothing: a Designer shown train tasks may write a task for any split.
            imposed = {placed[name] for name in group if name in placed} | {
                placed[name] for name in parent_names if placed[name] in ("eval", "test")
            }
            if len(imposed) > 1:
                split: SplitName | None = None
            elif imposed:
                split = imposed.pop()
            else:
                split = hashed_split(
                    seed, group[0], eval_fraction=parameters.eval_fraction, test_fraction=parameters.test_fraction
                )
            placed.update({name: split for name in group if name not in placed})
        if len(waiting) == len(pending):
            placed.update({name: None for group, _ in waiting for name in group if name not in placed})
            break
        pending = waiting
    return TaskSplit(
        train=tuple(name for name, split in placed.items() if split == "train"),
        eval=tuple(name for name, split in placed.items() if split == "eval"),
        seed=seed,
        eval_fraction=parameters.eval_fraction,
        test=tuple(name for name, split in placed.items() if split == "test"),
        test_fraction=parameters.test_fraction,
    )


def write_split_manifest(path: Path, split: TaskSplit) -> None:
    """Write the split as JSON: version 2 when it has a test split, else version 1 as older readers expect."""
    document: dict[str, object] = {
        "version": 1,
        "seed": split.seed,
        "eval_fraction": split.eval_fraction,
        "train": list(split.train),
        "eval": list(split.eval),
    }
    if split.test or split.test_fraction:
        document.update(version=2, test_fraction=split.test_fraction, test=list(split.test))
    # The real file, so a manifest published through a symlink changes behind the link instead of replacing it.
    target = Path(os.path.realpath(path))
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    # Written whole under .staging beside the manifest, then replaced into place: a reader sees the old manifest
    # or the new one, never a torn one, and a writer killed midway leaves nothing where the manifest lives.
    partial = target.parent / STAGING_DIRECTORY / f"{target.name}.{uuid.uuid4().hex}"
    try:
        if not target.name:
            raise TaskSplitError(f"cannot write split manifest {path}: it names no file")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(text, encoding="utf-8")
        if target.exists():
            os.chmod(partial, target.stat().st_mode)
        os.replace(partial, target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise TaskSplitError(f"cannot write split manifest {path}: {exc}") from exc


def read_split_manifest(path: Path) -> TaskSplit:
    """Read a manifest :func:`write_split_manifest` wrote; a version 1 manifest has an empty test split."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskSplitError(f"cannot read split manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise TaskSplitError(f"{path} is not a version 1 or 2 split manifest")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version not in MANIFEST_KEYS:
        raise TaskSplitError(f"{path} is not a version 1 or 2 split manifest")
    unknown_keys = sorted(key for key in document if key not in MANIFEST_KEYS[version])
    if unknown_keys:
        raise TaskSplitError(f"{path} carries keys reef did not write: {', '.join(unknown_keys)}")
    missing_keys = [key for key in MANIFEST_KEYS[version] if key not in document]
    if missing_keys:
        raise TaskSplitError(f"{path} is not a version 1 or 2 split manifest: it lacks {', '.join(missing_keys)}")
    for split in ("train", "eval", "test"):
        if not isinstance(document.get(split, []), list):
            raise TaskSplitError(f"{path}: {split} must be a list of task names")
    seed = document["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TaskSplitError(f"{path}: seed must be an integer")
    eval_fraction, test_fraction = document["eval_fraction"], document.get("test_fraction", 0.0)
    for label, fraction in (("eval_fraction", eval_fraction), ("test_fraction", test_fraction)):
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise TaskSplitError(f"{path}: {label} must be a number between 0 and 1")
    try:
        return TaskSplit(
            train=tuple(document["train"]),
            eval=tuple(document["eval"]),
            seed=seed,
            eval_fraction=eval_fraction,
            test=tuple(document.get("test", [])),
            test_fraction=test_fraction,
        )
    except TaskSplitError as exc:
        raise TaskSplitError(f"{path}: {exc}") from exc


def task_groups(record_ids_by_task: Mapping[str, Iterable[str]]) -> list[list[str]]:
    """Connected components of tasks over shared record ids, each sorted by name."""
    parent: dict[str, str] = {name: name for name in record_ids_by_task}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    owner: dict[str, str] = {}
    for name in sorted(record_ids_by_task):
        for record_id in record_ids_by_task[name]:
            first = owner.setdefault(record_id, name)
            parent[find(name)] = find(first)
    members: dict[str, list[str]] = {}
    for name in sorted(record_ids_by_task):
        members.setdefault(find(name), []).append(name)
    return [sorted(group) for group in members.values()]


def manifest_task_paths(manifest_path: Path, root: Path, split: str) -> tuple[Path, ...]:
    """The task directories one split of a manifest names under ``root``, each read back before it is trusted."""
    if split not in ("train", "eval", "test"):
        raise TaskSplitError(f"split must be 'train', 'eval' or 'test', not {split!r}")
    manifest = read_split_manifest(manifest_path)
    if split == "test" and not manifest.test and not manifest.test_fraction:
        raise TaskSplitError(f"split must be 'train' or 'eval' in {manifest_path}: it holds no test split")
    names = {"train": manifest.train, "eval": manifest.eval, "test": manifest.test}[split]
    paths: list[Path] = []
    for name in names:
        path = Path(os.path.abspath(Path(root) / name))
        try:
            read_harbor_task(path)
        except HarborTaskError as exc:
            raise TaskSplitError(f"{manifest_path}: {split} task {name!r}: {exc}") from exc
        paths.append(path)
    return tuple(paths)
