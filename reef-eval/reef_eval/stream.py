"""Task streams for continual-learning evaluation.

A :class:`Stream` runs an ordered list of Harbor tasks under one agent and
carries a state directory between them. Each task runs in a fresh
container with the directory mounted at ``$REEF_EVAL_STATE_DIR``; whatever the
agent writes there is visible in the next task. reef-eval never reads the
contents; an agent that ignores the directory is the stateless
baseline.

The directory is snapshotted after every task and restored before the
next, so every task starts from a known state and a crashed stream
resumes cleanly. Episode keys cover the task list up to each position:
appending tasks extends a finished stream, while editing an earlier task
re-runs everything after it.

Rows land in the Lab's store tagged ``stream`` and ``position``. See
:func:`reef_eval.metrics.learning_curve`, :func:`reef_eval.metrics.transfer`, and
:func:`reef_eval.metrics.forgetting` for the matching metrics.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reef_eval.budget import Budget
from reef_eval.types import Row, Tags

if TYPE_CHECKING:
    from reef_eval.lab import Lab

logger = logging.getLogger("reef_eval")


class Stream:
    """An ordered task sequence run under one carried agent state.

    ``name`` identifies the stream: re-running the same name (with the
    same setup) resumes it, and a new name starts fresh with empty state.
    """

    def __init__(self, name: str, tasks: Sequence[str]):
        if not tasks:
            raise ValueError("a stream needs at least one task")
        self.name = name
        self.tasks = list(tasks)

    async def run(
        self,
        lab: Lab,
        agent: dict[str, Any],
        *,
        tags: Tags | None = None,
        budget: Budget | dict[str, Any] | float | int | str | None = None,
        **overrides: Any,
    ) -> list[Row]:
        """Run every task in order, carrying state; returns one Row each.

        ``agent``, ``budget``, and ``**overrides`` mean the same as on
        :meth:`reef_eval.lab.Lab.run` and apply to every task. ``tags`` are
        recorded on every row, plus ``stream`` and ``position``.

        Tasks run sequentially because each one's starting state is the
        previous one's ending state. Tasks that already have a stored row
        are skipped, with their snapshots standing in for re-execution.
        Distinct streams can run concurrently; the same stream must not
        run concurrently with itself.
        """
        tags = dict(tags or {})
        budget = Budget.coerce(budget)
        variant = self._variant(agent, tags, budget, overrides)
        root = self._root(lab, variant)
        live = root / "state"
        snapshots = root / "snapshots"

        rows: list[Row] = []
        prev_snap: Path | None = None
        for position, task in enumerate(self.tasks):
            key = self._key(position, task, variant)
            # Named by the same prefix digest as the key, so a snapshot is
            # only ever reused by a stream whose history up to here matches.
            snap = snapshots / f"{position:03d}-{self._prefix(position, variant)}"
            existing = lab.store.get(key)
            if existing is not None:
                logger.info("skip %s (already recorded)", key)
                rows.append(existing)
                prev_snap = snap
                continue
            self._reset_state(live, prev_snap, init=snapshots / "init")
            logger.info("[%d/%d] %s", position + 1, len(self.tasks), task)
            row = await lab.run(
                task,
                agent,
                tags={**tags, "stream": self.name, "position": position},
                budget=budget,
                key=key,
                state_dir=str(live),
                **overrides,
            )
            _copy_dir(live, snap)
            rows.append(row)
            prev_snap = snap
        return rows

    def state_root(
        self,
        lab: Lab,
        agent: dict[str, Any],
        *,
        tags: Tags | None = None,
        budget: Budget | dict[str, Any] | float | int | str | None = None,
        **overrides: Any,
    ) -> Path:
        """Where this stream keeps its state, given the same arguments as
        :meth:`run`.

        ``<state_root>/state`` is the live directory (write here before
        the first run to seed the stream) and
        ``<state_root>/snapshots/<position>-<digest>`` is each task's ending
        state. The digest covers the task list up to that position and also
        appears in the episode's key, so a snapshot pairs with its row.
        """
        variant = self._variant(
            agent, dict(tags or {}), Budget.coerce(budget), overrides
        )
        return self._root(lab, variant)

    def _root(self, lab: Lab, variant: str) -> Path:
        return lab.root / "streams" / f"{self.name}-{variant}"

    def _prefix(self, position: int, variant: str) -> str:
        """Digest of the task list up to *position*, under this setup.

        Both the episode key and the snapshot name are built from it, so the
        two always agree: appending tasks leaves earlier prefixes untouched
        and reuses their snapshots, while editing a position changes every
        prefix after it.
        """
        from reef_eval.lab import Lab

        return Lab._digest({"variant": variant, "prefix": self.tasks[: position + 1]})

    def _variant(
        self,
        agent: dict[str, Any],
        tags: Tags,
        budget: Budget | None,
        overrides: dict[str, Any],
    ) -> str:
        """One digest per setup, so the same stream name under two agents
        (or budgets) keeps separate state and keys."""
        from reef_eval.lab import Lab

        return Lab._digest(
            {
                "agent": agent,
                "tags": tags,
                "budget": None if budget is None else budget.to_tags(),
                "overrides": overrides,
            }
        )

    def _key(self, position: int, task: str, variant: str) -> str:
        """Episode key covering the task list up to this position, so
        appends extend a stream while edits invalidate what follows."""
        short = task.rstrip("/").split("/")[-1]
        return f"{self.name}@{position:03d}:{short}:{self._prefix(position, variant)}"

    @staticmethod
    def _reset_state(live: Path, prev_snap: Path | None, *, init: Path) -> None:
        """Set the live directory to the previous task's snapshot (or, for
        position 0, to the seed captured as the init snapshot)."""
        if prev_snap is None:
            if not init.is_dir():
                live.mkdir(parents=True, exist_ok=True)
                _copy_dir(live, init)
            prev_snap = init
        elif not prev_snap.is_dir():
            # The previous task has a row but no snapshot: the run crashed
            # right after the row was stored. Nothing ran since, so the
            # live directory is its ending state, so recover the snapshot.
            if not live.is_dir():
                raise RuntimeError(
                    f"cannot resume stream: snapshot {prev_snap} is missing and "
                    f"there is no live state at {live} to recover it from"
                )
            logger.warning("recovering missing snapshot %s from live state", prev_snap)
            _copy_dir(live, prev_snap)
        if live.exists():
            shutil.rmtree(live)
        _copy_dir(prev_snap, live)


def _copy_dir(src: Path, dst: Path) -> None:
    """Copy a directory via a sibling temp dir and a rename, so a crash
    never leaves a half-written copy at *dst*."""
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    # copytree preserves the source directory timestamp. Stamp the completed
    # snapshot so tools can reliably distinguish a newly written history from
    # an older snapshot for the same stream position.
    os.utime(dst, None)
