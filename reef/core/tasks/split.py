"""Split generated tasks into a train split and an eval split by the records they came from.

Two tasks made from a shared agent record go to the same split, so nothing
the train split saw reappears, reworded, in the gate. The split is a
function of the task names, their sources and a seed, and the result is
written as a manifest the gate reads.
"""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from reef.core.errors import ReefError
from reef.core.tasks.harbor import TASK_NAME_PATTERN, HarborTaskError, read_harbor_task

MANIFEST_VERSION = 1
STAGING_DIRECTORY = ".staging"
MANIFEST_KEYS = ("version", "seed", "eval_fraction", "train", "eval")


class TaskSplitError(ReefError):
    """A split request or manifest that cannot be honored."""


@dataclass(frozen=True)
class TaskSplit:
    """Task names in each split, sorted, with the parameters that produced them."""

    train: tuple[str, ...]
    eval: tuple[str, ...]
    seed: int
    eval_fraction: float

    def __post_init__(self) -> None:
        train = checked_split("train", self.train)
        eval_ = checked_split("eval", self.eval)
        if set(train) & set(eval_):
            raise TaskSplitError("a task cannot be in both the train split and the eval split")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TaskSplitError("seed must be an integer")
        fraction = self.eval_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise TaskSplitError("eval_fraction must be a number between 0 and 1")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "eval", eval_)
        object.__setattr__(self, "eval_fraction", float(fraction))


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
    holds at least ``eval_fraction`` of the tasks, then the rest train.
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


def write_split_manifest(path: Path, split: TaskSplit) -> None:
    """Write the split as JSON; the gate reads the eval split and the trainer the train split from here."""
    document = {
        "version": MANIFEST_VERSION,
        "seed": split.seed,
        "eval_fraction": split.eval_fraction,
        "train": list(split.train),
        "eval": list(split.eval),
    }
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
    """Read a manifest :func:`write_split_manifest` wrote."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskSplitError(f"cannot read split manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise TaskSplitError(f"{path} is not a version {MANIFEST_VERSION} split manifest")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != MANIFEST_VERSION:
        raise TaskSplitError(f"{path} is not a version {MANIFEST_VERSION} split manifest")
    unknown_keys = sorted(key for key in document if key not in MANIFEST_KEYS)
    if unknown_keys:
        raise TaskSplitError(f"{path} carries keys reef did not write: {', '.join(unknown_keys)}")
    for split in ("train", "eval"):
        if not isinstance(document.get(split), list):
            raise TaskSplitError(f"{path}: {split} must be a list of task names")
    seed, fraction = document.get("seed"), document.get("eval_fraction")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TaskSplitError(f"{path}: seed must be an integer")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TaskSplitError(f"{path}: eval_fraction must be a number between 0 and 1")
    try:
        return TaskSplit(
            train=tuple(document["train"]), eval=tuple(document["eval"]), seed=seed, eval_fraction=fraction
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
    if split not in ("train", "eval"):
        raise TaskSplitError(f"split must be 'train' or 'eval', not {split!r}")
    manifest = read_split_manifest(manifest_path)
    names = manifest.train if split == "train" else manifest.eval
    paths: list[Path] = []
    for name in names:
        path = Path(os.path.abspath(Path(root) / name))
        try:
            read_harbor_task(path)
        except HarborTaskError as exc:
            raise TaskSplitError(f"{manifest_path}: {split} task {name!r}: {exc}") from exc
        paths.append(path)
    return tuple(paths)
