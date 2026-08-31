"""Failure manifest for harness evolution steps.

Each candidate step observes how the evaluation episodes of the composition
it commits fail: which task, at which stage, and why. ``advance`` folds the
step's observations into the previous step's manifest and classes every
fingerprint as new, persisting, or fixed. The manifest travels in the
algorithm state through the commit log, so the next step's proposer reads
structured evidence of what actually failed instead of prose.

Causes are normalized before fingerprinting because episode errors embed
volatile absolute paths and line numbers; a fingerprint must survive a
process restart and a fresh temporary directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MANIFEST_KIND = "reef-failure-manifest/1"

#: Failure stages, each tied to one observation locus: ``launch`` (the
#: episode could not run), ``trajectory`` (the session files could not be
#: read back), ``exit`` (the binary ran and exited nonzero).
STAGES = ("launch", "trajectory", "exit")

_ABSOLUTE_PATH = re.compile(r"(?:/[\w.+-]+){2,}")
#: Long unbroken identifier runs are opaque values (credentials, hashes,
#: request ids), never words a failure mode is named by; masking them keeps
#: key material out of the persisted cause (#476) and the fingerprint stable.
_OPAQUE_RUN = re.compile(r"[A-Za-z0-9_-]{16,}")
_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_CAUSE_LIMIT = 200


def normalize_cause(text: str) -> str:
    """Strip the volatile parts of an error text so equal failures collide.

    Absolute path runs become ``<path>``, long identifier runs become
    ``<id>``, and digit runs become ``<n>``, in that order so a path's own
    tokens never leak; whitespace collapses and the result is truncated to
    a bounded length.
    """
    text = _ABSOLUTE_PATH.sub("<path>", text)
    text = _OPAQUE_RUN.sub("<id>", text)
    text = _DIGIT_RUN.sub("<n>", text)
    return _WHITESPACE.sub(" ", text).strip()[:_CAUSE_LIMIT]


def fingerprint(task: str, stage: str, cause: str) -> str:
    """The stable identity of one failure mode across steps and restarts."""
    key = "\x1f".join((task, stage, normalize_cause(cause)))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FailureObservation:
    """One failed episode as the evaluation saw it, before any diffing."""

    task: str
    stage: str
    cause: str

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task, "stage": self.stage, "cause": self.cause}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureObservation:
        return cls(task=str(value["task"]), stage=str(value["stage"]), cause=str(value["cause"]))


@dataclass(frozen=True)
class FailureRecord:
    """One fingerprinted failure mode and its consecutive-step streak.

    ``count`` accumulates over consecutive steps that observed the
    fingerprint; ``first_seen_step`` and ``last_seen_step`` bound the streak.
    ``cause`` is stored normalized, so a record never carries a volatile
    path.
    """

    fingerprint: str
    task: str
    stage: str
    cause: str
    count: int
    first_seen_step: int
    last_seen_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "task": self.task,
            "stage": self.stage,
            "cause": self.cause,
            "count": self.count,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureRecord:
        return cls(
            fingerprint=str(value["fingerprint"]),
            task=str(value["task"]),
            stage=str(value["stage"]),
            cause=str(value["cause"]),
            count=int(value["count"]),
            first_seen_step=int(value["first_seen_step"]),
            last_seen_step=int(value["last_seen_step"]),
        )


@dataclass(frozen=True)
class FailureManifest:
    """The failures the composition committed at ``step`` showed.

    ``entries`` are the failures observed at this step; ``fixed`` are the
    previous step's entries that did not reappear, streaks frozen. A
    fingerprint that reappears after it was fixed restarts as new with a
    fresh streak.
    """

    step: int
    entries: tuple[FailureRecord, ...]
    fixed: tuple[FailureRecord, ...] = ()

    @property
    def new(self) -> tuple[FailureRecord, ...]:
        """Entries first observed at this step."""
        return tuple(record for record in self.entries if record.first_seen_step == self.step)

    @property
    def persisting(self) -> tuple[FailureRecord, ...]:
        """Entries carried over from an earlier step."""
        return tuple(record for record in self.entries if record.first_seen_step < self.step)

    def to_state(self) -> dict[str, Any]:
        return {
            "kind": MANIFEST_KIND,
            "step": self.step,
            "entries": [record.to_dict() for record in self.entries],
            "fixed": [record.to_dict() for record in self.fixed],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FailureManifest:
        kind = state.get("kind")
        if kind != MANIFEST_KIND:
            raise ValueError(f"unknown failure manifest kind: {kind!r}")
        return cls(
            step=int(state["step"]),
            entries=tuple(FailureRecord.from_dict(value) for value in state["entries"]),
            fixed=tuple(FailureRecord.from_dict(value) for value in state["fixed"]),
        )


def advance(
    previous: FailureManifest | None,
    step: int,
    observations: Sequence[FailureObservation],
) -> FailureManifest:
    """Fold one step's observations into the previous step's manifest.

    An observed fingerprint already in ``previous`` keeps its first-seen
    step and extends its streak; a fresh one starts a streak at this step; a
    previous entry that was not observed moves to ``fixed``. Entries sort by
    fingerprint so serialization is deterministic.
    """
    observed: dict[str, FailureRecord] = {}
    for observation in observations:
        key = fingerprint(observation.task, observation.stage, observation.cause)
        record = observed.get(key)
        if record is not None:
            observed[key] = dataclasses.replace(record, count=record.count + 1)
            continue
        observed[key] = FailureRecord(
            fingerprint=key,
            task=observation.task,
            stage=observation.stage,
            cause=normalize_cause(observation.cause),
            count=1,
            first_seen_step=step,
            last_seen_step=step,
        )
    fixed: list[FailureRecord] = []
    if previous is not None:
        for record in previous.entries:
            current = observed.get(record.fingerprint)
            if current is None:
                fixed.append(record)
            else:
                observed[record.fingerprint] = dataclasses.replace(
                    current,
                    count=record.count + current.count,
                    first_seen_step=record.first_seen_step,
                )
    return FailureManifest(
        step=step,
        entries=tuple(sorted(observed.values(), key=lambda record: record.fingerprint)),
        fixed=tuple(sorted(fixed, key=lambda record: record.fingerprint)),
    )
