"""OpenClaw-RL's candidate evaluation: a style probe with a regression check.

:class:`OpenClawRLCandidateEvaluation` decides whether Reef publishes a trained
candidate. ``evaluate`` runs the candidate on a fixed set of GSM8K problems. It
scores each reply with the benchmark's ``student_violations`` criterion and
reports ``clean_rate``: the fraction of replies that are not empty and have no
style violation. ``decide`` comes from
:class:`~reef.evaluation.evaluators.RegressionCheckMixin`. It selects a candidate
while ``clean_rate`` stays within ``regression_margin`` of the best selected
score. Otherwise serving keeps the last selected weights.

The probe problems do not change during a run, so a change in the score comes
from the candidate, not from the problems. Only the in-process MLX training
runtime can probe an unpublished candidate.

Configure the plugin in the deployment:

    evaluation:
      module: recipes.openclawrl.candidate_evaluator:OpenClawRLCandidateEvaluation
      config:
        probe_size: 8            # probe problems, 1 to 10
        max_tokens: 320          # generation budget for one probe reply
        regression_margin: 0.17  # allowed drop below the best selected clean_rate

``max_tokens`` must leave room for a full reply with its arithmetic. A smaller
budget cuts replies short, and a cut reply can fail the shown-work check because
of its length, not its style. One probe problem is worth ``1 / probe_size`` of
``clean_rate``. A ``regression_margin`` above that value accepts a candidate that
fails one more problem than the best.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from recipes.openclawrl.examples.openclawrl.user_sim.personas import student_violations
from reef.core.config import config_option
from reef.core.evaluation import EvaluationResult, UpdateCandidate
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.evaluation.config import ConfiguredEvaluationPlugin
from reef.evaluation.evaluators import RegressionCheckMixin
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.train.mlx_backend.runtime import MLXRuntime

logger = logging.getLogger(__name__)

EVALUATOR_NAME = "openclawrl-style-probe"
EVALUATOR_VERSION = "1"
CLEAN_RATE_METRIC = "clean_rate"

# Fixed GSM8K problems with gold answers; a run probes the first ``probe_size``.
# Each candidate generates one reply per problem on the engine thread, between
# training and selection, so the set stays small.
PROBE_PROBLEMS: tuple[tuple[str, str], ...] = (
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "10",
    ),
    (
        "Betty is saving for a $100 wallet. She has only half the money she needs. Her parents give her $15, and her grandparents give twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "624",
    ),
    (
        "Julie is reading a 120-page book. Yesterday she read 12 pages and today she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "42",
    ),
    (
        "Ken created a care package to send to his brother. He placed a box on a scale, then poured jelly beans in to bring the weight to 2 pounds. Then he added brownies to triple the weight, then another 2 pounds of jelly beans, then doubled it with gummy worms. What was the final weight of the box of goodies, in pounds?",
        "16",
    ),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "3"),
    (
        "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150% of what he paid for it. How much profit did he make?",
        "70000",
    ),
    (
        "Kylar went to the store to buy glasses. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "64",
    ),
    (
        "Toula bought 3 dozen donuts at $68 per dozen, 2 dozen mini cupcakes at $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
        "694",
    ),
)

#: The probe request, in the student's words. It asks for the working but says
#: nothing about style, so the score shows the style that the policy learned,
#: not compliance with this instruction.
PROBE_INSTRUCTION = (
    "Here is my homework problem. Solve it and show the full arithmetic, step by step, then give the final number."
)


@dataclass(frozen=True)
class OpenClawRLCandidateEvaluationSettings(PluginSettings):
    """The config of :class:`OpenClawRLCandidateEvaluation`."""

    probe_size: int = config_option(8, help=f"The probe problems, 1 to {len(PROBE_PROBLEMS)}.")
    max_tokens: int = config_option(320, help="The generation budget in tokens for one probe reply.")
    regression_margin: float = config_option(
        0.17, help="The allowed drop of clean_rate below the best selected clean_rate."
    )

    def __post_init__(self) -> None:
        if not 1 <= self.probe_size <= len(PROBE_PROBLEMS):
            raise ValueError(f"probe_size must be between 1 and {len(PROBE_PROBLEMS)}, got {self.probe_size}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.regression_margin < 0:
            raise ValueError(f"regression_margin must be non-negative, got {self.regression_margin}")


class OpenClawRLCandidateEvaluation(RegressionCheckMixin, ConfiguredEvaluationPlugin):
    """The style probe as ``evaluate`` and the regression check on ``clean_rate`` as ``decide``."""

    def __init__(self, runtime: MLXRuntime, settings: OpenClawRLCandidateEvaluationSettings) -> None:
        super().__init__(metric=CLEAN_RATE_METRIC, margin=settings.regression_margin)
        self.runtime = runtime
        self.max_tokens = settings.max_tokens
        self.probe = PROBE_PROBLEMS[: settings.probe_size]
        # The prompts are rendered once, so every candidate gets the same tokens.
        self.prompts = [
            runtime.engine.render_prompt([{"role": "user", "content": f"{PROBE_INSTRUCTION}\n\n{question}"}])
            for question, _ in self.probe
        ]

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        replies = self.runtime.probe_candidate(candidate.candidate_id, self.prompts, max_tokens=self.max_tokens)
        violations = [student_violations(reply) for reply in replies]
        # Clean does not check the gold answer: correctness adds sampling noise
        # to a small probe. ``gold_rate`` reports it separately.
        clean_count = sum(1 for reply, found in zip(replies, violations, strict=True) if reply.strip() and not found)
        answered_count = sum(1 for reply in replies if reply.strip())
        correct_count = sum(
            1 for (_, gold), reply in zip(self.probe, replies, strict=True) if gold in reply.replace(",", "")
        )
        probe_count = len(replies)
        return EvaluationResult(
            evaluator=EVALUATOR_NAME,
            evaluator_version=EVALUATOR_VERSION,
            metrics={
                CLEAN_RATE_METRIC: clean_count / probe_count,
                "answered_rate": answered_count / probe_count,
                "gold_rate": correct_count / probe_count,
                # The distance from the criterion; ``clean_rate`` is all or
                # nothing per reply. An empty reply fails only the shown-work
                # check, so ``answered_rate``, not this, shows replies that stop.
                "mean_violations": sum(len(found) for found in violations) / probe_count,
                "clean_count": clean_count,
                "probe_count": probe_count,
            },
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        runtime: InferenceRuntime,
        training_runtime: TrainingRuntime,
        scenario: str,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the plugin on the training runtime.

        The probe applies a candidate's weights before Reef selects it, and
        serving never sees an unselected candidate. Only the in-process MLX
        runtime can do this. Other training runtimes are refused, so a
        deployment never runs without the regression check.
        """
        if not isinstance(training_runtime, MLXRuntime):
            raise PluginConfigError(
                f"{EVALUATOR_NAME} needs the mlx training runtime to probe an unpublished candidate; "
                f"got {type(training_runtime).__name__}"
            )
        settings = OpenClawRLCandidateEvaluationSettings.parse(config, plugin_class=cls)
        logger.info(
            "%s active for scenario %s: %d probe problems, regression-check margin %.2f",
            EVALUATOR_NAME,
            scenario,
            settings.probe_size,
            settings.regression_margin,
        )
        return cls(training_runtime, settings)


__all__ = ["OpenClawRLCandidateEvaluation", "OpenClawRLCandidateEvaluationSettings"]
