"""OpenClaw-RL's candidate evaluation: a style probe under a regression gate.

Reef's default plugin is ``BackendAlwaysSelectPlugin``, which publishes every
trained step.
That is what lets OpenClaw-RL's stream keep its post-adaptation collapse: a few
steps past adaptation the policy stops answering and loops on its tools, and
because every step reaches serving the drift compounds instead of rolling back
(the verdicts show pre-adaptation rejects as style violations on healthy
replies, post-adaptation rejects as empty "no-reply" turns).

One cohesive plugin, :class:`OpenClawRLCandidateEvaluationPlugin`, does both
halves. Its ``evaluate`` is the OpenClaw-RL-specific probe — each candidate runs
on a fixed, pinned set of GSM8K openings, and every reply is scored with the
benchmark's own ``student_violations`` criterion (a non-empty reply that shows
its working in plain prose, no markdown), reporting a ``clean_rate``. Its
``decide`` comes from reef's generic
:class:`~reef.train.evaluation.RegressionGateMixin`: select a candidate only
while that rate has not regressed below the best seen — so the stream is free to
climb during adaptation and, once it peaks, a step that makes the policy answer
worse is held out of serving.

The probe set is pinned on purpose: a probe that resampled its problems would
measure the problems, not the candidate.

Wire it into a deployment's config:

    evaluation:
      module: recipes.openclawrl.candidate_evaluator:build
      config:
        probe_size: 8            # pinned GSM8K openings to probe
        max_tokens: 96           # per-probe generation budget
        regression_margin: 0.17  # tolerated dip below the running best
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.train.evaluation import EvaluationResult, RegressionGateMixin, UpdateCandidate

logger = logging.getLogger(__name__)

_EVALUATOR = "openclawrl-style-probe"
_VERSION = "1"
_METRIC = "clean_rate"

# A fixed, pinned held-out probe: canonical GSM8K questions with gold answers.
# Held constant for the life of a run so a score reflects the candidate, not a
# freshly sampled problem. Kept small — this generates once per candidate, on
# the engine thread, between training and selection.
_PROBE: tuple[tuple[str, str], ...] = (
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

# What the probe asks for: a plainly-written, worked answer — the same shape the
# acceptance criterion rewards, so the score tracks the behaviour that collapses.
_INSTRUCTION = (
    "Solve this math problem. Show the full arithmetic, step by step, then give the final "
    "number. Write plainly in complete sentences — do not use bold, headings, bullet "
    "points, or numbered lists."
)


def _load_criterion() -> Any:
    """The benchmark's own ``student_violations``, loaded from the user_sim package.

    Reused rather than re-implemented so the probe scores a reply exactly as the
    judge that produced the run's verdicts does; if the criterion moves, so does
    the probe.
    """
    personas = Path(__file__).parent / "examples" / "openclawrl" / "user_sim" / "personas.py"
    spec = importlib.util.spec_from_file_location("openclawrl_personas", personas)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the OpenClaw-RL style criterion from {personas}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: personas.py defines a dataclass, whose processing
    # looks the module up in sys.modules by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.student_violations


class OpenClawRLCandidateEvaluationPlugin(RegressionGateMixin):
    """The whole OpenClaw-RL candidate evaluation in one class.

    ``evaluate`` is the OpenClaw-RL-specific probe — a pinned GSM8K set scored
    with the benchmark's ``student_violations`` criterion, run through the
    runtime's ``probe_candidate``. ``decide`` comes from
    :class:`~reef.train.evaluation.RegressionGateMixin`: publish a candidate only
    while its ``clean_rate`` has not regressed below the best seen, so the stream
    climbs during adaptation and a step that makes the policy answer worse is held
    out of serving. The probe set is pinned so a score reflects the candidate, not
    a freshly sampled problem.
    """

    def __init__(self, runtime: Any, *, probe_size: int, max_tokens: int, regression_margin: float) -> None:
        super().__init__(metric=_METRIC, margin=regression_margin)
        self._runtime = runtime
        self._max_tokens = int(max_tokens)
        self._probe = _PROBE[: max(1, int(probe_size))]
        self._violations = _load_criterion()
        # Prompts rendered once — the ruler is fixed for the run.
        self._prompts = [
            runtime.engine.render_prompt([{"role": "user", "content": f"{_INSTRUCTION}\n\n{question}"}])
            for question, _ in self._probe
        ]

    def _clean(self, reply: str) -> bool:
        """A reply counts as clean when it is non-empty and has no style violation.

        This is the judge's own reward condition minus the gold-answer check:
        the collapse is an empty or markdown-laden reply, and folding in
        correctness would only add sampling noise to a small probe. The
        gold-answer hit rate is reported alongside for visibility.
        """
        return bool(reply.strip()) and not self._violations(reply)

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        replies = self._runtime.probe_candidate(candidate.candidate_id, self._prompts, max_tokens=self._max_tokens)
        clean = sum(1 for reply in replies if self._clean(reply))
        answered = sum(1 for reply in replies if reply.strip())
        correct = sum(
            1 for (_, gold), reply in zip(self._probe, replies, strict=True) if gold in reply.replace(",", "")
        )
        total = len(replies)
        return EvaluationResult(
            evaluator=_EVALUATOR,
            evaluator_version=_VERSION,
            metrics={
                _METRIC: clean / total if total else 0.0,
                "answered_rate": answered / total if total else 0.0,
                "gold_rate": correct / total if total else 0.0,
                "n_clean": clean,
                "n_total": total,
            },
        )


def build(
    config: Mapping[str, Any],
    *,
    runtime: Any,
    scenario: str,
    environ: Mapping[str, str],
) -> OpenClawRLCandidateEvaluationPlugin:
    """Factory for ``evaluation.module``: the OpenClaw-RL probe + regression gate.

    Requires a runtime that can probe an unpublished candidate — the in-process
    MLX runtime's ``probe_candidate``. A runtime without it (e.g. the Ray
    training bridge, whose candidates live in a separate process) is refused
    here rather than silently degrading to no gate.
    """
    if not callable(getattr(runtime, "probe_candidate", None)):
        raise ValueError(
            f"{_EVALUATOR} needs a runtime that can probe an unpublished candidate; "
            f"{type(runtime).__name__} does not provide probe_candidate()"
        )
    plugin = OpenClawRLCandidateEvaluationPlugin(
        runtime,
        probe_size=int(config.get("probe_size", 8)),
        max_tokens=int(config.get("max_tokens", 96)),
        regression_margin=float(config.get("regression_margin", 0.17)),
    )
    logger.info(
        "%s active for scenario %s: %d pinned probes, regression-gate margin %.2f",
        _EVALUATOR,
        scenario,
        len(plugin._probe),
        plugin._margin,
    )
    return plugin


__all__ = ["OpenClawRLCandidateEvaluationPlugin", "build"]
