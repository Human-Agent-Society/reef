"""OpenClaw-RL next-state binary reward recipe."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from recipes.openclawrl.processor import OpenClawRLProcessor
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field

_logger = logging.getLogger(__name__)

# Settings retired with the SGLang teacher prefill (the Megatron teacher pass
# owns scoring now). Accepted and ignored for one release so existing configs
# keep validating; a non-default value logs a warning.
_RETIRED_FIELDS = {"prm_teacher_timeout_s": 900.0, "prm_logprob_temperature": 1.0}


@dataclass(frozen=True, kw_only=True)
class OpenClawRLRecipe(WeightTrainingRecipe):
    """Reproduces upstream OpenClaw-RL's binary-RL configuration on reef.

    The whole method runs server-side: the processor correlates main-turn
    records into sessions by preferring a harness-supplied session tag and
    falling back to trace matching, judges each turn by its next state against
    the PRM's sglang server, and batches judgments directly — no external
    grader. The step preparer then applies the verbatim upstream top-K select
    objective.

    Empty ``prm_url`` is correlate-only mode: sessions resolve, nothing
    trains, records stay compactable. ``batch_size`` must equal the slime
    driver's ``--global-batch-size`` (each turn-sample is its own rollout).
    Teacher scoring lives trainer-side: the Megatron teacher pass computes
    exact log-probs on the trainer's own tempered scale.
    """

    name: str = "openclawrl"
    batch_size: int = config_field(16)
    session_ttl_s: float = config_field(900.0)
    prm_url: str = config_field("")  # the PRM's sglang server; empty = correlate-only
    prm_tokenizer_path: str = config_field("")
    prm_m: int = config_field(3)  # judge votes per turn
    prm_temperature: float = config_field(0.6)
    prm_max_tokens: int = config_field(8192)
    prm_timeout_s: float = config_field(120.0)
    prm_concurrency: int = config_field(8)  # turns judged at once
    prm_max_hint_candidates: int = config_field(3)
    # Deprecated, ignored (see _RETIRED_FIELDS).
    prm_teacher_timeout_s: float = config_field(900.0)
    prm_logprob_temperature: float = config_field(1.0)
    # Upstream's OPENCLAW_PRM_RECORD_FILE: the judged population, one JSON
    # line per batch. Empty disables it.
    prm_record_file: str = config_field("")

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        # The paper objective is the verbatim upstream top-K select loss
        # (recipes/openclawrl/slime), not the plain pg surrogate.
        return WeightTrainingSpec(
            step_preparer="openclawrl",
            loss_family="openclawrl",
            processor=OpenClawRLProcessor,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        if self.prm_url and not self.prm_tokenizer_path:
            raise ValueError("OpenClaw-RL judging requires prm_tokenizer_path next to prm_url")
        for name, default in _RETIRED_FIELDS.items():
            if getattr(self, name) != default:
                _logger.warning("openclawrl: %s is deprecated and ignored; remove it from the config", name)
