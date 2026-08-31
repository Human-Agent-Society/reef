"""Core data types shared across reef_eval.

The vocabulary is deliberately tiny:

- An :class:`EpisodeSpec` says what to run (a Harbor task + an agent).
- An :class:`EpisodeResult` is what an executor hands back.
- A :class:`TracePoint` is one judge-scored submission from *during* an
  episode.
- A :class:`Row` is what actually lands in the results store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

Tags = dict[str, Any]
Rewards = dict[str, float | int]
Usage = dict[str, float | int]


@dataclass(frozen=True)
class EpisodeSpec:
    """What to run: a Harbor task under a given agent configuration.

    ``task`` is a Harbor task directory path or a registry id
    (e.g. ``"some-benchmark/some-task"``). ``agent`` uses Harbor's own
    ``AgentConfig`` field names verbatim (``name``, ``model_name``,
    ``import_path``, ...), with no translation layer on top.
    ``overrides`` are passed through to Harbor's ``TrialConfig`` for anything
    else (verifier/environment settings, timeout multipliers, ...).
    """

    task: str
    agent: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TracePoint:
    """One judge-scored submission from inside an episode.

    ``t`` is seconds since the episode started (or any monotonic offset the
    task convention defines). ``data`` carries anything else the score log
    recorded (snapshot path, trade id, ...).
    """

    t: float
    score: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    """What an executor returns for one episode.

    ``rewards`` is the trusted score (from Harbor's verifier, or a fake in
    tests). ``trace`` is the judge's per-submission score log, recovered
    from the episode's artifacts. ``uri`` points at the provenance (the
    Harbor trial directory) so every stored number stays auditable.

    ``agent_exit_code`` is the agent's exit status when it was not 0. A
    non-zero exit is not by itself a failed episode, since spending the whole
    time budget ends with the container being killed, so it is recorded as a
    fact rather than as ``error``.

    ``usage`` is what the episode actually spent, the measured side of a
    budget: ``n_input_tokens``, ``n_output_tokens``, ``n_cache_tokens``
    (the counts the agent adapter reads off the harness's own accounting),
    ``n_total_tokens`` (input plus output) and ``n_submissions`` (evals
    used). Every key is optional; the
    Lab records whatever is present as ``used_*`` columns.
    """

    rewards: Rewards
    uri: str | None = None
    trace: tuple[TracePoint, ...] = ()
    error: str | None = None
    usage: Usage = field(default_factory=dict)
    agent_exit_code: int | None = None


@dataclass(frozen=True)
class Row:
    """One record in the results store. ``kind`` is one of:

    - ``episode``: the verifier's final score (one per episode)
    - ``trace``: one judge-scored submission (many per episode)

    ``kind`` is an open string, not an enum: future evaluation regimes add
    new kinds (with their own key shapes) without touching this schema.
    """

    key: str
    kind: str
    task: str
    tags: Tags
    rewards: Rewards
    uri: str | None = None
    created_at: float = field(default_factory=time.time)
