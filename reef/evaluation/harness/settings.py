"""The ``evolution.evaluation`` section: the tasks, the scorer, the rollouts and the publication policy.

Example::

    evolution:
      evaluation:
        tasks: ["[fib] Compute fib(90) and reply with the number alone."]
        scorer: harness.scoring:AnswerScorer
        policy: compare
        statistic: wins
        repeats: 1
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import Settings, config_option
from reef.evaluation.harness.policy import POLICIES, STATISTICS


@dataclass(frozen=True)
class EvaluationSettings(Settings):
    """The parsed ``evolution.evaluation`` section of a harness evolution deployment.

    Values: at most one task source is set: ``tasks`` (prompts or task
    directories), ``task_manifest`` with ``tasks_root``, or ``task_store``;
    one is required unless ``policy`` is ``none``. ``scorer`` and ``collector``
    are plugin settings (a ``package.module:Class`` reference, or a mapping
    with ``module`` and ``config``); the recipe requires a scorer unless
    ``policy`` is ``none``. ``policy`` is one of ``POLICIES`` and ``statistic`` one of
    ``STATISTICS``. ``min_margin`` ≥ 0, ``floor_score`` > 0 and
    ``rollout_timeout_s`` > 0 are finite numbers, ``repeats`` ≥ 1, and every
    count ≥ 0. ``collector`` needs ``collect_failures``. ``agent_host`` is a
    non-empty host name or address: an agent that plays a Harbor task in the
    task's container reaches Reef's model route there.
    Purpose: the harness recipe parses it and builds the evaluator, the task
    source and the publication policy from it.
    """

    tasks: tuple[str, ...] = config_option((), help="The evaluation tasks: prompts, or task directories.")
    task_manifest: str | None = config_option(None, help="A split manifest whose eval side names the tasks.")
    tasks_root: str | None = config_option(None, help="The directory of the task manifest's task directories.")
    task_store: str | None = config_option(None, help="A task store whose eval split each step evaluates.")
    scorer: str | Mapping[str, Any] | None = config_option(None, help="The RolloutScorer that scores each rollout.")
    policy: str = config_option("compare", help="The publication policy: compare, floor or none.")
    statistic: str = config_option("wins", help="What compare compares: wins (pairwise) or mean.")
    min_margin: float = config_option(
        0.0, help="The margin a candidate must exceed: wins minus losses, or the mean score difference."
    )
    floor_score: float = config_option(1.0, help="The score every floor rollout must reach.")
    repeats: int = config_option(1, help="The rollouts per task and tree.")
    rollout_timeout_s: float = config_option(600.0, help="The seconds one rollout may run.")
    agent_host: str = config_option(
        "host.docker.internal", help="The address at which an agent in a Harbor task's container reaches Reef."
    )
    forbid_residue: bool = config_option(False, help="Score a rollout that leaves files behind 0.")
    collect_failures: bool = config_option(False, help="Add failing traffic prompts to the evaluation tasks.")
    collector: str | Mapping[str, Any] | None = config_option(None, help="The TaskCollector that picks the prompts.")
    max_collected_tasks: int = config_option(50, help="The most prompts collection adds in total; 0 adds none.")
    max_collected_per_client: int = config_option(
        5, help="The most prompts collection adds per tagged client; 0 sets no per-client limit."
    )

    def __post_init__(self) -> None:
        sources = [
            name
            for name, present in (
                ("tasks", bool(self.tasks)),
                ("task_manifest", self.task_manifest is not None),
                ("task_store", self.task_store is not None),
            )
            if present
        ]
        if len(sources) > 1:
            raise ValueError(f"evaluation takes one task source, not {' and '.join(sources)}")
        if (self.task_manifest is None) != (self.tasks_root is None):
            raise ValueError("evaluation.task_manifest and evaluation.tasks_root go together")
        if not isinstance(self.agent_host, str) or not self.agent_host.strip():
            raise ValueError("evaluation.agent_host must be a non-empty host name or address")
        if self.policy not in POLICIES:
            raise ValueError(f"evaluation.policy must be one of {', '.join(POLICIES)}")
        if self.policy != "none" and not sources:
            raise ValueError("evaluation needs tasks, a task_manifest or a task_store unless its policy is none")
        if self.statistic not in STATISTICS:
            raise ValueError(f"evaluation.statistic must be one of {', '.join(STATISTICS)}")
        for label, number in (
            ("min_margin", self.min_margin),
            ("floor_score", self.floor_score),
            ("rollout_timeout_s", self.rollout_timeout_s),
        ):
            if not math.isfinite(number):
                raise ValueError(f"evaluation.{label} must be a finite number")
        if self.min_margin < 0:
            raise ValueError("evaluation.min_margin must be at least 0")
        if self.floor_score <= 0:
            raise ValueError("evaluation.floor_score must be positive")
        if self.repeats < 1:
            raise ValueError("evaluation.repeats must be at least 1")
        if self.rollout_timeout_s <= 0:
            raise ValueError("evaluation.rollout_timeout_s must be positive")
        for label, value in (
            ("max_collected_tasks", self.max_collected_tasks),
            ("max_collected_per_client", self.max_collected_per_client),
        ):
            if value < 0:
                raise ValueError(f"evaluation.{label} must be at least 0")
        if self.collector is not None and not self.collect_failures:
            raise ValueError("evaluation.collector applies only with evaluation.collect_failures")


__all__ = ["EvaluationSettings"]
