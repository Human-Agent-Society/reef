"""Build optional scenario experiment providers from deployment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.observability.base import ExperimentTracker, NullExperimentTracker
from reef.observability.tracing import TracingConfig
from reef.observability.wandb import WandbConfig, WandbExperimentTracker
from reef.storage.observer import RecordObserver


def build_experiment_tracker(
    wandb: object,
    *,
    model: str | None,
    training_config: Mapping[str, Any],
) -> ExperimentTracker:
    config = WandbConfig.from_mapping(wandb)
    if not config.active:
        return NullExperimentTracker()
    return WandbExperimentTracker(config, model=model, training_config=training_config)


def build_record_observer(tracing: object, *, environ: Mapping[str, str] | None = None) -> RecordObserver | None:
    """The record observer for ``observability.tracing``, or ``None`` when tracing is off.

    The OpenTelemetry SDK loads only when enabled. ``environ`` supplies the
    ``REEF_TRACING_AUTHORIZATION`` fallback for the credential.
    """
    config = TracingConfig.from_mapping(tracing, environ=environ)
    if not config.enabled:
        return None
    try:
        from reef.observability.open_telemetry import OpenTelemetryRecordObserver
    except ImportError as exc:
        raise ValueError(
            "observability.tracing.enabled requires the OpenTelemetry SDK: install reef-infra[opentelemetry]"
        ) from exc
    return OpenTelemetryRecordObserver(config)


__all__ = ["build_experiment_tracker", "build_record_observer"]
