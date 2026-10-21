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

The style probe has no tools, so it cannot see a policy that stops answering
and only calls tools. ``reply_probe_file`` adds a reply probe for that. The file
holds harness conversations that stop where the policy should answer in text:
the user asked for the homework, and the policy has read the file. ``evaluate``
reports ``reply_rate``, the fraction of these conversations that the candidate
answers with text and no tool call. ``decide`` refuses a candidate whose
``reply_rate`` drops more than ``reply_margin`` below the best selected
``reply_rate``, before it checks ``clean_rate``:

    evaluation:
      config:
        reply_probe_file: recipes/openclawrl/examples/openclawrl/reply_probe.json
        reply_max_tokens: 512    # generation budget for one reply-probe answer
        reply_margin: 0.25       # allowed drop below the best selected reply_rate
        min_reply_rate: 0.75     # lowest reply_rate a candidate may have, also right after a restart

The file is JSON: ``{"tools": [...], "conversations": [[message, ...], ...],
"answers": [...]}``, with the tools and messages in the format that the harness
sends. ``answers`` is optional and gives each conversation's gold answer; a
reply then counts only when its visible text contains that number, so an
incoherent reply does not count.

With ``answers``, ``evaluate`` also reports ``reply_clean_rate``: the fraction
of the conversations that the candidate answers in text with the gold answer
and no ``student_violations``. This is the student's acceptance criterion on
a first reply, in the harness context. ``regression_metric:
reply_clean_rate`` makes the regression check use it instead of
``clean_rate``:

    evaluation:
      config:
        regression_metric: reply_clean_rate
        regression_margin: 0.07  # with 16 conversations, accepts a drop of one answer
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from recipes.openclawrl.examples.openclawrl.user_sim.personas import student_violations
from reef.core.config import config_option
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.evaluation.config import ConfiguredEvaluationPlugin
from reef.evaluation.evaluators import RegressionCheckMixin
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.train.mlx_backend.runtime import MLXRuntime

logger = logging.getLogger(__name__)

EVALUATOR_NAME = "openclawrl-style-probe"
EVALUATOR_VERSION = "2"
CLEAN_RATE_METRIC = "clean_rate"
REPLY_RATE_METRIC = "reply_rate"
REPLY_CLEAN_RATE_METRIC = "reply_clean_rate"
REGRESSION_METRICS = (CLEAN_RATE_METRIC, REPLY_CLEAN_RATE_METRIC)

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
        0.17, help="The allowed drop of the regression metric below the best selected value."
    )
    regression_metric: str = config_option(
        CLEAN_RATE_METRIC,
        help="The metric of the regression check: clean_rate, or reply_clean_rate, which needs reply-probe answers.",
    )
    reply_probe_file: str = config_option(
        "", help="A JSON file of conversations where the policy should answer in text; empty disables the reply probe."
    )
    reply_max_tokens: int = config_option(512, help="The generation budget in tokens for one reply-probe answer.")
    reply_margin: float = config_option(
        0.25, help="The allowed drop of reply_rate below the best selected reply_rate."
    )
    min_reply_rate: float = config_option(
        0.0, help="The lowest reply_rate a selected candidate may have, whatever the best selected one was."
    )

    def __post_init__(self) -> None:
        if not 1 <= self.probe_size <= len(PROBE_PROBLEMS):
            raise ValueError(f"probe_size must be between 1 and {len(PROBE_PROBLEMS)}, got {self.probe_size}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.regression_margin < 0:
            raise ValueError(f"regression_margin must be non-negative, got {self.regression_margin}")
        if self.regression_metric not in REGRESSION_METRICS:
            raise ValueError(f"regression_metric must be one of {REGRESSION_METRICS}, got {self.regression_metric!r}")
        if self.reply_max_tokens <= 0:
            raise ValueError(f"reply_max_tokens must be positive, got {self.reply_max_tokens}")
        if self.reply_margin < 0:
            raise ValueError(f"reply_margin must be non-negative, got {self.reply_margin}")
        if not 0 <= self.min_reply_rate <= 1:
            raise ValueError(f"min_reply_rate must be between 0 and 1, got {self.min_reply_rate}")


def load_reply_probe(path: pathlib.Path) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[str]]:
    """Read the tools, conversations and gold answers of a reply-probe file; raise ``PluginConfigError`` for a malformed one."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginConfigError(f"cannot read reply_probe_file {str(path)!r}: {exc}") from exc
    tools = content.get("tools") if isinstance(content, dict) else None
    conversations = content.get("conversations") if isinstance(content, dict) else None
    if not isinstance(tools, list) or not isinstance(conversations, list) or not conversations:
        raise PluginConfigError(
            f"reply_probe_file {str(path)!r} needs a 'tools' list and a non-empty 'conversations' list"
        )
    if any(not isinstance(messages, list) or not messages for messages in conversations):
        raise PluginConfigError(
            f"every conversation in reply_probe_file {str(path)!r} must be a non-empty message list"
        )
    answers = content.get("answers", [])
    if not isinstance(answers, list) or (answers and len(answers) != len(conversations)):
        raise PluginConfigError(f"reply_probe_file {str(path)!r}: 'answers' must give one answer per conversation")
    return tools, conversations, [str(answer) for answer in answers]


def visible_reply(generated: str) -> str:
    """The text of a generated turn that the client reads as the reply: what follows any reasoning."""
    return generated.split("</think>")[-1]


def answers_in_text(generated: str, *, opens_reasoning: bool = False, answer: str = "") -> bool:
    """Whether a generated turn answers in text: visible text after any reasoning, no tool call, and the answer.

    When the chat template opened the reasoning block, a turn that never closes it has no visible text:
    the client reads all of it as reasoning. With ``answer``, the visible text must contain that number,
    so an incoherent reply does not count.
    """
    if opens_reasoning and "</think>" not in generated:
        return False
    visible = visible_reply(generated)
    if not visible.strip() or "<tool_call>" in visible:
        return False
    return not answer or re.search(rf"(?<![\d.]){re.escape(answer)}(?![\d])", visible.replace(",", "")) is not None


class OpenClawRLCandidateEvaluation(RegressionCheckMixin, ConfiguredEvaluationPlugin):
    """The style and reply probes as ``evaluate``; the regression checks on ``reply_rate`` and ``regression_metric`` as ``decide``."""

    def __init__(self, runtime: MLXRuntime, settings: OpenClawRLCandidateEvaluationSettings) -> None:
        super().__init__(metric=settings.regression_metric, margin=settings.regression_margin)
        self.runtime = runtime
        self.max_tokens = settings.max_tokens
        self.probe = PROBE_PROBLEMS[: settings.probe_size]
        # The prompts are rendered once, so every candidate gets the same tokens.
        self.prompts = [
            runtime.engine.render_prompt([{"role": "user", "content": f"{PROBE_INSTRUCTION}\n\n{question}"}])
            for question, _ in self.probe
        ]
        self.reply_max_tokens = settings.reply_max_tokens
        self.reply_margin = settings.reply_margin
        self.min_reply_rate = settings.min_reply_rate
        #: The best reply_rate of a selected candidate; ``None`` until the first.
        self.best_reply_rate: float | None = None
        self.reply_prompts: list[list[int]] = []
        self.reply_opens_reasoning: list[bool] = []
        self.reply_answers: list[str] = []
        if settings.reply_probe_file.strip():
            tools, conversations, self.reply_answers = load_reply_probe(
                pathlib.Path(settings.reply_probe_file.strip())
            )
            self.reply_prompts = [runtime.engine.render_prompt(messages, tools=tools) for messages in conversations]
            self.reply_opens_reasoning = [runtime.engine.opens_reasoning(prompt) for prompt in self.reply_prompts]
        if settings.regression_metric == REPLY_CLEAN_RATE_METRIC and not self.reply_answers:
            raise PluginConfigError(
                f"regression_metric {REPLY_CLEAN_RATE_METRIC} needs a reply_probe_file with 'answers'"
            )

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
        reply_metrics: dict[str, float | int] = {}
        if self.reply_prompts:
            answers = self.runtime.probe_candidate(
                candidate.candidate_id, self.reply_prompts, max_tokens=self.reply_max_tokens
            )
            golds = self.reply_answers or [""] * len(answers)
            replied = [
                answers_in_text(answer, opens_reasoning=opens, answer=gold)
                for answer, opens, gold in zip(answers, self.reply_opens_reasoning, golds, strict=True)
            ]
            reply_count = sum(replied)
            reply_metrics = {
                REPLY_RATE_METRIC: reply_count / len(answers),
                "reply_count": reply_count,
                "reply_probe_count": len(answers),
            }
            if self.reply_answers:
                # The student's acceptance criterion on a first reply in the
                # harness context: the correct answer, the work shown, no style marker.
                reply_clean_count = sum(
                    1
                    for answer, ok in zip(answers, replied, strict=True)
                    if ok and not student_violations(visible_reply(answer))
                )
                reply_metrics[REPLY_CLEAN_RATE_METRIC] = reply_clean_count / len(answers)
                reply_metrics["reply_clean_count"] = reply_clean_count
        return EvaluationResult(
            evaluator=EVALUATOR_NAME,
            evaluator_version=EVALUATOR_VERSION,
            metrics={
                **reply_metrics,
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

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        """Refuse a candidate whose ``reply_rate`` regressed, then apply the ``regression_metric`` check."""
        if not self.reply_prompts:
            return super().decide(candidate, evaluation)
        reply_rate = float(evaluation.metrics[REPLY_RATE_METRIC])
        # The best selected rate lives in this process; the floor also holds after a restart.
        if reply_rate < self.min_reply_rate:
            return SelectionDecision(
                selected=False,
                policy="regression-check",
                reason=(
                    f"{REPLY_RATE_METRIC} {reply_rate:g} is below min_reply_rate {self.min_reply_rate:g}; "
                    "holding the last selected weights"
                ),
                evaluation=evaluation,
                metrics={"metric": REPLY_RATE_METRIC, "value": reply_rate, "floor": self.min_reply_rate},
            )
        best = self.best_reply_rate
        if best is not None and reply_rate < best - self.reply_margin:
            return SelectionDecision(
                selected=False,
                policy="regression-check",
                reason=(
                    f"{REPLY_RATE_METRIC} {reply_rate:g} regressed past margin {self.reply_margin:g} below best "
                    f"{best:g}; holding the last selected weights"
                ),
                evaluation=evaluation,
                metrics={"metric": REPLY_RATE_METRIC, "value": reply_rate, "best": best, "margin": self.reply_margin},
            )
        decision = super().decide(candidate, evaluation)
        if decision.selected:
            self.best_reply_rate = reply_rate if best is None else max(best, reply_rate)
        return decision

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
