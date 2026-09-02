"""The pinned upstream quickstart adapter, driven by a tracked model."""

from __future__ import annotations

from typing import Any, cast

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, DefaultDataInst

from .data import RULES_SEED

OFFICIAL_COMPONENT = "system_prompt"
OFFICIAL_SEED_CANDIDATE = {OFFICIAL_COMPONENT: RULES_SEED}


class OfficialAIMEAdapter(DefaultAdapter):
    """Upstream's default adapter with a tracked, batch-capable model in place of its private LM."""

    def __init__(self, model: Any, *, max_workers: int = 1) -> None:
        super().__init__(model=model, max_litellm_workers=max_workers)
        self._lm = model

    @property
    def usage(self) -> Any:
        """The tracked model's ledger, so both arms report usage the same way."""
        return self._lm.usage

    def evaluate(self, batch, candidate, capture_traces=False):
        # The upstream scorer reads the optional context while building feedback
        # for an incorrect answer; AIME 2025 rows omit it.
        normalized = [{**example, "additional_context": example.get("additional_context") or {}} for example in batch]
        return super().evaluate(cast(list[DefaultDataInst], normalized), candidate, capture_traces)
