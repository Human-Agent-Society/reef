"""Optional W&B provider for Reef's scenario experiment contract.

Importing this module does not import :mod:`wandb`; the dependency is loaded
only when an enabled tracker binds its first scenario.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef, encode_artifact_ref
from reef.observability.base import (
    ExperimentLogger,
    ExperimentTracker,
    RollbackExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)

logger = logging.getLogger(__name__)

#: ``event.metrics`` key carrying a training job's per-optimizer-step metric
#: dicts (the Slime backend fills it; see ``worker_hooks.WORKER_STEP_METRICS_KEY``).
OPTIMIZER_STEPS_KEY = "train_steps"
#: Run-summary key of the run-wide optimizer-step counter behind ``step/*`` (``step/step``).
OPTIMIZER_STEP_COUNTER = "reef/optimizer_steps"

WandbMode = Literal["online", "offline", "disabled"]
_SECRET_KEY_PARTS = ("api_key", "apikey", "credential", "password", "secret", "token")
_UNSAFE_TRAINING_FIELDS = frozenset({"slime_flags", "wandb"})
_METRIC_NAMESPACE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class WandbConfig:
    """Validated first-class ``observability.wandb`` configuration."""

    enabled: bool = False
    project: str = "reef"
    entity: str | None = None
    group_prefix: str | None = None
    name_prefix: str | None = None
    tags: tuple[str, ...] = ()
    mode: WandbMode = "disabled"
    directory: str | None = None
    upload_checkpoints: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> WandbConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("observability.wandb must be a mapping")
        allowed = {
            "enabled",
            "project",
            "entity",
            "group_prefix",
            "name_prefix",
            "tags",
            "mode",
            "directory",
            "upload_checkpoints",
        }
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(f"unknown observability.wandb settings: {', '.join(unknown)}")

        enabled = _boolean(value.get("enabled", False), "observability.wandb.enabled")
        mode = _optional_string(value.get("mode"), "observability.wandb.mode") or ("online" if enabled else "disabled")
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("observability.wandb.mode must be online, offline, or disabled")
        return cls(
            enabled=enabled,
            project=_optional_string(value.get("project"), "observability.wandb.project") or "reef",
            entity=_optional_string(value.get("entity"), "observability.wandb.entity"),
            group_prefix=_optional_string(value.get("group_prefix"), "observability.wandb.group_prefix"),
            name_prefix=_optional_string(value.get("name_prefix"), "observability.wandb.name_prefix"),
            tags=_tags(value.get("tags", ())),
            mode=mode,  # type: ignore[arg-type]
            directory=_optional_string(value.get("directory"), "observability.wandb.directory"),
            upload_checkpoints=_boolean(
                value.get("upload_checkpoints", False),
                "observability.wandb.upload_checkpoints",
            ),
        )

    @property
    def active(self) -> bool:
        return self.enabled and self.mode != "disabled"


class _WandbScenarioLogger(ExperimentLogger):
    """Stable scenario proxy whose underlying W&B run rotates on rollback."""

    def __init__(
        self,
        tracker: WandbExperimentTracker,
        *,
        scenario: str,
        recipe: str,
        source_artifact_ref: ArtifactRef,
        run_segment: int,
    ) -> None:
        self._tracker = tracker
        self._scenario = scenario
        self._recipe = recipe
        self._source_artifact_ref = source_artifact_ref
        self._run_segment = run_segment
        self._event_counts: dict[str, int] = {}
        self._defined_namespaces: set[str] = set()

    def log(self, metrics: Mapping[str, Any], *, namespace: str) -> None:
        self._tracker._record_component(self, metrics, namespace)

    def _context(self) -> TrainingExperimentContext:
        return TrainingExperimentContext(
            scenario=self._scenario,
            recipe=self._recipe,
            step=self._run_segment,
            source_artifact_ref=self._source_artifact_ref,
            run_segment=self._run_segment,
        )

    def _rebind(self, *, recipe: str, source_artifact_ref: ArtifactRef, run_segment: int) -> None:
        self._recipe = recipe
        if run_segment != self._run_segment:
            self._event_counts = {}
            self._defined_namespaces = set()
        self._source_artifact_ref = source_artifact_ref
        self._run_segment = run_segment

    def _rotate(self, *, source_artifact_ref: ArtifactRef, run_segment: int) -> None:
        self._source_artifact_ref = source_artifact_ref
        self._run_segment = run_segment
        self._event_counts = {}
        self._defined_namespaces = set()

    def _next_event(self, namespace: str, summary: Any) -> int:
        current = self._event_counts.get(namespace)
        if current is None:
            recovered = None
            if summary is not None:
                try:
                    recovered = summary.get(f"reef/{namespace}_events")
                except Exception:
                    recovered = None
            current = recovered if isinstance(recovered, int) and recovered >= 0 else 0
        self._event_counts[namespace] = current + 1
        return current


class WandbExperimentTracker(ExperimentTracker):
    """W&B observer for Reef's provider-neutral training commit events."""

    def __init__(
        self,
        config: WandbConfig,
        *,
        model: str | None,
        training_config: Mapping[str, Any],
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._training_config = _safe_mapping(training_config, skip=_UNSAFE_TRAINING_FIELDS)
        self._client = client
        self._runs: dict[str, Any] = {}
        self._scenario_loggers: dict[str, _WandbScenarioLogger] = {}
        self._failed_run_ids: set[str] = set()
        self._lock = RLock()

    def bind_scenario(
        self,
        *,
        scenario: str,
        recipe: str,
        source_artifact_ref: ArtifactRef,
        run_segment: int,
    ) -> ExperimentLogger:
        with self._lock:
            logger = self._scenario_loggers.get(scenario)
            if logger is None:
                logger = _WandbScenarioLogger(
                    self,
                    scenario=scenario,
                    recipe=recipe,
                    source_artifact_ref=source_artifact_ref,
                    run_segment=run_segment,
                )
                self._scenario_loggers[scenario] = logger
            else:
                logger._rebind(
                    recipe=recipe,
                    source_artifact_ref=source_artifact_ref,
                    run_segment=run_segment,
                )
            if self.config.active:
                context = logger._context()
                run_id = self._run_id(context)
                if run_id not in self._runs:
                    self._initialize_run(run_id, context)
            return logger

    def correlation_metrics(self, context: TrainingExperimentContext) -> Mapping[str, Any]:
        if not self.config.active:
            return {}
        self.bind_scenario(
            scenario=context.scenario,
            recipe=context.recipe,
            source_artifact_ref=context.source_artifact_ref,
            run_segment=context.run_segment,
        )
        metrics: dict[str, Any] = {
            "experiment/provider": "wandb",
            "experiment/project": self.config.project,
            "experiment/group": _scenario_group(self.config.group_prefix, context.scenario),
            "experiment/run_id": self._run_id(context),
        }
        if self.config.entity is not None:
            metrics["experiment/entity"] = self.config.entity
        return metrics

    def record(self, event: TrainingExperimentEvent) -> None:
        with self._lock:
            self._record(event)

    def record_rollback(self, event: RollbackExperimentEvent) -> None:
        """Finish the current curve and immediately open its next run segment."""
        if not self.config.active:
            return
        run_id = stable_run_id(
            project=self.config.project,
            entity=self.config.entity,
            group=_scenario_group(self.config.group_prefix, event.scenario),
            name_prefix=self.config.name_prefix,
            scenario=event.scenario,
            recipe=event.recipe,
            run_segment=event.run_segment,
        )
        with self._lock:
            run = self._runs.pop(run_id, None)
            if run is not None:
                try:
                    summary = getattr(run, "summary", None)
                    if summary is not None:
                        summary.update(
                            {
                                "reef/ended_by": "rollback",
                                "reef/rollback_step": event.step,
                                "reef/rollback_from_release_id": event.source_artifact_ref.release_id,
                                "reef/rollback_target_release_id": event.target_release_id,
                                "reef/current_release_id": event.produced_artifact_ref.release_id,
                            }
                        )
                    run.finish()
                except Exception as exc:
                    logger.warning("W&B rollback finalization failed (%s)", type(exc).__name__)
            scenario_logger = self._scenario_loggers.get(event.scenario)
            if scenario_logger is not None:
                scenario_logger._rotate(
                    source_artifact_ref=event.produced_artifact_ref,
                    run_segment=event.step,
                )
                context = scenario_logger._context()
                next_run_id = self._run_id(context)
                if next_run_id not in self._runs:
                    self._initialize_run(next_run_id, context)

    def _record(self, event: TrainingExperimentEvent) -> None:
        if not self.config.active:
            return
        run_id = self._run_id(event.context)
        run = self._runs.get(run_id)
        if run is None:
            run = self._initialize_run(run_id, event.context)
        if run is None:
            return
        self._update_run_config(run, event.context)
        metadata = self._event_metadata(event, run_id)
        self._record_optimizer_steps(run, event)
        values: dict[str, Any] = {
            "train/step": event.context.run_step,
            "reef/step": event.context.step,
            **_numeric_metrics(event.metrics),
            **{f"reef/{key}": value for key, value in metadata.items() if value is not None and key != "step"},
        }
        try:
            run.log(values)
            summary = getattr(run, "summary", None)
            if summary is not None:
                summary.update({f"reef/{key}": value for key, value in metadata.items() if value is not None})
            if event.checkpoint_path and self.config.upload_checkpoints:
                self._upload_checkpoint(run, event)
        except Exception as exc:
            logger.warning("W&B logging failed (%s); training will continue", type(exc).__name__)

    def close(self) -> None:
        with self._lock:
            runs, self._runs = tuple(self._runs.values()), {}
            self._scenario_loggers = {}
        for run in runs:
            try:
                run.finish()
            except Exception as exc:  # noqa: PERF203
                logger.warning("W&B shutdown failed (%s)", type(exc).__name__)

    def _record_optimizer_steps(self, run: Any, event: TrainingExperimentEvent) -> None:
        """Log every optimizer step of the job as its own row under ``step/*``.

        A backend that trains several optimizer steps per job (Slime cuts a
        batch into ``--global-batch-size`` steps, times epochs) hands them
        over as ``metrics["train_steps"]``, a list of per-step metric dicts.
        Each becomes one row keyed by ``step/step``, a run-wide optimizer-step
        counter kept in the run summary, so the curve survives restarts and
        resumed segments (it replaces the backend's own per-step counter,
        which need not be monotonic across jobs of varying length); the
        job-level row that follows keeps its ``train/*`` keys unchanged.
        """
        steps = event.metrics.get(OPTIMIZER_STEPS_KEY)
        if not isinstance(steps, Sequence) or isinstance(steps, str | bytes) or not steps:
            return
        summary = getattr(run, "summary", None)
        start = 0
        if summary is not None:
            try:
                start = int(summary.get(OPTIMIZER_STEP_COUNTER) or 0)
            except (TypeError, ValueError, AttributeError):
                start = 0
        logged = 0
        try:
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                values = {f"step/{key.removeprefix('train/')}": value for key, value in _numeric_metrics(step).items()}
                values["step/step"] = start + logged
                values["reef/step"] = event.context.step
                run.log(values)
                logged += 1
            if summary is not None:
                summary[OPTIMIZER_STEP_COUNTER] = start + logged
        except Exception as exc:
            logger.warning("W&B optimizer-step logging failed (%s); training will continue", type(exc).__name__)

    def _initialize_run(self, run_id: str, context: TrainingExperimentContext) -> Any | None:
        if run_id in self._failed_run_ids:
            return None
        try:
            client = self._client
            if client is None:
                import wandb

                client = wandb
                self._client = client
            directory = None
            if self.config.directory:
                directory = str(Path(self.config.directory).expanduser())
                Path(directory).mkdir(parents=True, exist_ok=True)
            run = client.init(
                id=run_id,
                project=self.config.project,
                entity=self.config.entity,
                group=_scenario_group(self.config.group_prefix, context.scenario),
                name=_run_name(self.config.name_prefix or context.scenario, context.run_segment),
                tags=list(self.config.tags),
                mode=self.config.mode,
                dir=directory,
                resume="allow",
                reinit="create_new",
                config={
                    "reef": {
                        "scenario": context.scenario,
                        "recipe": context.recipe,
                        "backend": context.backend,
                        "model": self._model,
                        "run_segment": context.run_segment,
                        "source_artifact": encode_artifact_ref(context.source_artifact_ref),
                    },
                    "backend": _safe_mapping(context.backend_config or {}),
                    "training": dict(self._training_config),
                },
            )
            run.define_metric("train/step")
            run.define_metric("train/*", step_metric="train/step")
            run.define_metric("step/step")
            run.define_metric("step/*", step_metric="step/step")
            self._runs[run_id] = run
            return run
        except Exception as exc:
            self._failed_run_ids.add(run_id)
            logger.warning("W&B initialization failed (%s); training will continue", type(exc).__name__)
            return None

    def _record_component(
        self,
        scenario_logger: _WandbScenarioLogger,
        metrics: Mapping[str, Any],
        namespace: str,
    ) -> None:
        if not self.config.active:
            return
        namespace = _metric_namespace(namespace)
        numeric_metrics = _numeric_metrics(metrics)
        if not numeric_metrics:
            return
        with self._lock:
            context = scenario_logger._context()
            run_id = self._run_id(context)
            run = self._runs.get(run_id)
            if run is None:
                run = self._initialize_run(run_id, context)
            if run is None:
                return
            event_step = scenario_logger._next_event(namespace, getattr(run, "summary", None))
            values = {
                f"{namespace}/event": event_step,
                **{f"{namespace}/{key}": value for key, value in numeric_metrics.items()},
            }
            try:
                if namespace not in scenario_logger._defined_namespaces:
                    run.define_metric(f"{namespace}/event")
                    run.define_metric(f"{namespace}/*", step_metric=f"{namespace}/event")
                    scenario_logger._defined_namespaces.add(namespace)
                run.log(values)
                summary = getattr(run, "summary", None)
                if summary is not None:
                    summary[f"reef/{namespace}_events"] = event_step + 1
            except Exception as exc:
                logger.warning("W&B %s logging failed (%s); execution will continue", namespace, type(exc).__name__)

    def _update_run_config(self, run: Any, context: TrainingExperimentContext) -> None:
        config = getattr(run, "config", None)
        if config is None:
            return
        values = {
            "reef": {
                "scenario": context.scenario,
                "recipe": context.recipe,
                "backend": context.backend,
                "model": self._model,
                "run_segment": context.run_segment,
                "source_artifact": encode_artifact_ref(context.source_artifact_ref),
            },
            "backend": _safe_mapping(context.backend_config or {}),
        }
        try:
            config.update(values, allow_val_change=True)
        except Exception as exc:
            logger.warning("W&B run config update failed (%s); execution will continue", type(exc).__name__)

    def _run_id(self, context: TrainingExperimentContext) -> str:
        return stable_run_id(
            project=self.config.project,
            entity=self.config.entity,
            group=_scenario_group(self.config.group_prefix, context.scenario),
            name_prefix=self.config.name_prefix,
            scenario=context.scenario,
            recipe=context.recipe,
            run_segment=context.run_segment,
        )

    def _event_metadata(self, event: TrainingExperimentEvent, run_id: str) -> dict[str, Any]:
        source = event.context.source_artifact_ref
        produced = event.produced_artifact_ref
        source_weight = event.source_runtime_load_id
        if source_weight is None and isinstance(source, LiveWeightArtifactRef):
            source_weight = source.runtime_load_id
        produced_weight = event.produced_runtime_load_id
        if produced_weight is None and isinstance(produced, LiveWeightArtifactRef):
            produced_weight = produced.runtime_load_id
        return {
            "step": event.context.step,
            "run_segment": event.context.run_segment,
            "run_step": event.context.run_step,
            "run_id": run_id,
            "scenario": event.context.scenario,
            "recipe": event.context.recipe,
            "backend": event.context.backend,
            "training_job_id": event.training_job_id,
            "outcome": event.outcome,
            "source_release_id": source.release_id,
            "produced_release_id": produced.release_id,
            "source_runtime_load_id": source_weight,
            "produced_runtime_load_id": produced_weight,
            "checkpoint_path": event.checkpoint_path,
        }

    def _upload_checkpoint(self, run: Any, event: TrainingExperimentEvent) -> None:
        client = self._client
        if client is None or event.checkpoint_path is None:
            return
        identity = event.training_job_id or event.produced_artifact_ref.release_id
        artifact = client.Artifact(name=f"reef-checkpoint-{_artifact_name(identity)}", type="model")
        artifact.add_dir(event.checkpoint_path)
        run.log_artifact(artifact)


def stable_run_id(
    *,
    project: str,
    entity: str | None,
    group: str | None,
    name_prefix: str | None,
    scenario: str,
    recipe: str | None,
    run_segment: int,
) -> str:
    identity = json.dumps(
        {
            "entity": entity,
            "group": group,
            "name_prefix": name_prefix,
            "project": project,
            "recipe": recipe,
            "scenario": scenario,
            "run_segment": run_segment,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"reef-wandb-v2\0" + identity).hexdigest()[:32]


def _numeric_metrics(value: Mapping[str, Any], prefix: str = "") -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for raw_key, item in value.items():
        key = f"{prefix}/{raw_key}" if prefix else str(raw_key)
        if isinstance(item, Mapping):
            metrics.update(_numeric_metrics(item, key))
        elif isinstance(item, Real) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number):
                metrics[key] = int(item) if isinstance(item, int) else number
    return metrics


def _safe_mapping(value: Mapping[str, Any], *, skip: frozenset[str] = frozenset()) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        lowered = key.lower()
        if lowered in skip or any(part in lowered for part in _SECRET_KEY_PARTS):
            continue
        prepared = _safe_value(item)
        if prepared is not None:
            safe[key] = prepared
    return safe


def _safe_value(value: Any) -> Any | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        prepared = [_safe_value(item) for item in value]
        return [item for item in prepared if item is not None]
    return None


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip() or None


def _tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("observability.wandb.tags must be a list of strings")
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("observability.wandb.tags must be a list of non-empty strings")
        tags.append(item.strip())
    return tuple(dict.fromkeys(tags))


def _artifact_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _run_name(base: str, run_segment: int) -> str:
    return f"{base}-segment-{run_segment}"


def _scenario_group(prefix: str | None, scenario: str) -> str:
    return scenario if prefix is None else f"{prefix}/{scenario}"


def _metric_namespace(value: str) -> str:
    if not isinstance(value, str) or _METRIC_NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError("experiment metric namespace must start with a letter and contain only letters, numbers, ._-")
    return value


__all__ = ["WandbConfig", "WandbExperimentTracker", "stable_run_id"]
