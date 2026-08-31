"""OpenEvolve configuration builder."""

from __future__ import annotations

from typing import Any


def openevolve_config(model: str, api_base: str | None = None) -> dict[str, Any]:
    """Return a small, reproducible OpenEvolve configuration."""
    llm: dict[str, Any] = {
        "primary_model": model,
        "primary_model_weight": 1.0,
        "temperature": 0.7,
        "max_tokens": 16_000,
        "timeout": 120,
    }
    if api_base:
        llm["api_base"] = api_base
    return {
        "max_iterations": 100,
        "checkpoint_interval": 10,
        "llm": llm,
        "prompt": {
            "system_message": (
                "Improve the program's circle packing. Preserve its JSON output "
                "contract; the evaluator returns the reef-eval judge's score."
            )
        },
        "database": {
            "population_size": 30,
            "archive_size": 15,
            "num_islands": 2,
            "elite_selection_ratio": 0.2,
            "exploitation_ratio": 0.7,
        },
        "evaluator": {"timeout": 60, "parallel_evaluations": 1},
        "diff_based_evolution": True,
        "max_code_length": 20_000,
    }
