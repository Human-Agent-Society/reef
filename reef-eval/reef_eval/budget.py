"""Budget: how much an episode is allowed to spend.

A run is bounded by whichever resource is scarce: wall-clock time, judge
evaluations (submissions), or tokens. reef-eval models all three the same
way. Each dimension is set on the run, delivered to the agent as a
``REEF_EVAL_*`` environment variable (:meth:`Budget.to_env`), and recorded as a
``budget_*`` tag (:meth:`Budget.to_tags`). The actual spend comes back as
``used_*`` columns.

Enforcement differs by dimension. ``time_h`` is hard: it becomes the
container timeout, so the episode is killed at the deadline and the
verifier still grades the best submission so far. ``max_submissions`` is
enforced by the judge up to the task's own ``judge_config.json`` ceiling;
a lower per-run value is a signal the agent is asked to honor, because
Harbor cannot inject env into the judge sidecar. ``max_tokens`` is a soft
signal: reef-eval cannot halt a black-box harness mid-generation, so it passes
the limit to the agent and records the true spend regardless.

Set the scarce dimension and leave the rest ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNIT_HOURS = {"s": 1 / 3600, "m": 1 / 60, "h": 1.0, "d": 24.0}


def parse_duration_hours(text: str) -> float:
    """A human duration as hours: ``2h``, ``30m``, ``90s``, ``1d``. A bare
    number is read as hours (``0.5`` is ``30m``)."""
    t = str(text).strip().lower()
    if not t:
        raise ValueError("empty duration")
    if t[-1].isalpha():
        if t[-1] not in _UNIT_HOURS:
            raise ValueError(f"unknown duration unit {t[-1]!r} (use s/m/h/d)")
        return float(t[:-1]) * _UNIT_HOURS[t[-1]]
    return float(t)  # bare number = hours


@dataclass(frozen=True)
class Budget:
    time_h: float | None = None
    max_submissions: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("time_h", "max_submissions", "max_tokens"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"budget {name} must be positive, got {v!r}")

    @property
    def is_empty(self) -> bool:
        return not any(
            v is not None for v in (self.time_h, self.max_submissions, self.max_tokens)
        )

    def timeout_sec(self) -> float | None:
        """The wall-clock budget as seconds: the container or ``--local``
        timeout."""
        return None if self.time_h is None else self.time_h * 3600.0

    def to_env(self) -> dict[str, str]:
        """Budget signals for the agent's container, mirroring ``$BUDGET_SEC``.

        A harness or method reads these to pace itself; ignoring them only
        means the run spends more, which ``used_*`` will show.
        """
        env: dict[str, str] = {}
        if self.time_h is not None:
            env["REEF_EVAL_BUDGET_SEC"] = repr(self.timeout_sec())
        if self.max_submissions is not None:
            env["REEF_EVAL_MAX_SUBMISSIONS"] = str(self.max_submissions)
        if self.max_tokens is not None:
            env["REEF_EVAL_MAX_TOKENS"] = str(self.max_tokens)
        return env

    def to_tags(self) -> dict[str, Any]:
        """The budget as grouping tags. ``budget`` stays the hours value for
        back-compat with existing scaling queries; the rest are explicit."""
        tags: dict[str, Any] = {}
        if self.time_h is not None:
            tags["budget"] = self.time_h
            tags["budget_time_h"] = self.time_h
        if self.max_submissions is not None:
            tags["budget_max_submissions"] = self.max_submissions
        if self.max_tokens is not None:
            tags["budget_max_tokens"] = self.max_tokens
        return tags

    @classmethod
    def coerce(
        cls, value: Budget | dict[str, Any] | str | float | int | None
    ) -> Budget | None:
        """Accept a Budget, a dict of its fields, a bare number (= hours), or a
        duration string (``"30m"``, ``"2h"``)."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise TypeError(f"cannot coerce {value!r} to Budget")
        if isinstance(value, (int, float)):
            return cls(time_h=float(value))
        if isinstance(value, str):
            return cls(time_h=parse_duration_hours(value))
        if isinstance(value, dict):
            allowed = {"time_h", "max_submissions", "max_tokens"}
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unknown budget fields: {sorted(unknown)}")
            return cls(**value)
        raise TypeError(f"cannot coerce {value!r} to Budget")
