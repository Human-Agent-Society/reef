"""The ``evolution.evaluation`` section: the tasks, the scorer, the episodes and the publication policy.

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
    one is required unless ``policy`` is ``none``. ``scorer`` and ``promote``
    are plugin settings (a ``package.module:Class`` reference, or a mapping
    with ``module`` and ``config``); the recipe requires a scorer unless
    ``policy`` is ``none``. ``policy`` is one of ``POLICIES`` and ``statistic`` one of
    ``STATISTICS``. ``min_margin`` ≥ 0, ``floor_score`` > 0 and
    ``episode_timeout_s`` > 0 are finite numbers, ``repeats`` ≥ 1, and every
    count ≥ 0. ``recheck_every`` needs the ``compare`` policy, and
    ``promote`` needs ``promote_failures``.
    Purpose: the harness recipe parses it and builds the evaluator, the task
    source and the publication policy from it.
    """

    tasks: tuple[str, ...] = config_option((), help="The evaluation tasks: prompts, or task directories.")
    task_manifest: str | None = config_option(None, help="A split manifest whose eval side names the tasks.")
    tasks_root: str | None = config_option(None, help="The directory of the task manifest's task directories.")
    task_store: str | None = config_option(None, help="A task store whose eval split each step evaluates.")
    scorer: str | Mapping[str, Any] | None = config_option(None, help="The EpisodeScorer that scores each episode.")
    policy: str = config_option("compare", help="The publication policy: compare, floor or none.")
    statistic: str = config_option("wins", help="What compare compares: wins (pairwise) or mean.")
    min_margin: float = config_option(
        0.0, help="The margin a candidate must exceed: wins minus losses, or the mean score difference."
    )
    floor_score: float = config_option(1.0, help="The score every floor episode must reach.")
    repeats: int = config_option(1, help="The episodes per task and tree.")
    episode_timeout_s: float = config_option(600.0, help="The seconds one episode may run.")
    forbid_residue: bool = config_option(False, help="Score an episode that leaves files behind as not run.")
    recheck_every: int = config_option(
        0, help="Every this many steps, compare the latest publish with the tree it replaced. 0 turns it off."
    )
    promote_failures: bool = config_option(False, help="Add failing traffic prompts to the evaluation tasks.")
    promote: str | Mapping[str, Any] | None = config_option(None, help="The Promoter that picks the prompts.")
    max_promoted_tasks: int = config_option(50, help="The most prompts promotion adds in total; 0 adds none.")
    max_promoted_per_client: int = config_option(
        5, help="The most prompts promotion adds per tagged client; 0 sets no per-client limit."
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
        if self.policy not in POLICIES:
            raise ValueError(f"evaluation.policy must be one of {', '.join(POLICIES)}")
        if self.policy != "none" and not sources:
            raise ValueError("evaluation needs tasks, a task_manifest or a task_store unless its policy is none")
        if self.statistic not in STATISTICS:
            raise ValueError(f"evaluation.statistic must be one of {', '.join(STATISTICS)}")
        for label, number in (
            ("min_margin", self.min_margin),
            ("floor_score", self.floor_score),
            ("episode_timeout_s", self.episode_timeout_s),
        ):
            if not math.isfinite(number):
                raise ValueError(f"evaluation.{label} must be a finite number")
        if self.min_margin < 0:
            raise ValueError("evaluation.min_margin must be at least 0")
        if self.floor_score <= 0:
            raise ValueError("evaluation.floor_score must be positive")
        if self.repeats < 1:
            raise ValueError("evaluation.repeats must be at least 1")
        if self.episode_timeout_s <= 0:
            raise ValueError("evaluation.episode_timeout_s must be positive")
        for label, value in (
            ("recheck_every", self.recheck_every),
            ("max_promoted_tasks", self.max_promoted_tasks),
            ("max_promoted_per_client", self.max_promoted_per_client),
        ):
            if value < 0:
                raise ValueError(f"evaluation.{label} must be at least 0")
        if self.recheck_every and self.policy != "compare":
            raise ValueError("evaluation.recheck_every compares two trees, so it needs the compare policy")
        if self.promote is not None and not self.promote_failures:
            raise ValueError("evaluation.promote applies only with evaluation.promote_failures")


__all__ = ["EvaluationSettings"]
