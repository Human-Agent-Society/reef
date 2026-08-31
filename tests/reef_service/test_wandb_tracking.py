from __future__ import annotations

import dataclasses

import pytest

from reef.core.artifact_ref import LiveWeightArtifactRef
from reef.observability.base import RollbackExperimentEvent, TrainingExperimentContext, TrainingExperimentEvent
from reef.observability.wandb import WandbConfig, WandbExperimentTracker
from reef.service.deploy.settings import service_settings_from_config


class _StubRun:
    def __init__(self, *, fail_log: bool = False, config=None) -> None:
        self.fail_log = fail_log
        self.config = _StubConfig(config or {})
        self.defined: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, object] = {}
        self.artifacts: list[object] = []
        self.finished = False

    def define_metric(self, *args, **kwargs) -> None:
        self.defined.append((args, kwargs))

    def log(self, metrics, *, step=None) -> None:
        if self.fail_log:
            raise ConnectionError("offline test failure")
        self.logged.append((dict(metrics), step))

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class _StubClient:
    def __init__(self, *, fail_init: bool = False, fail_log: bool = False) -> None:
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[_StubRun] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise ConnectionError("offline test failure")
        run = _StubRun(fail_log=self.fail_log, config=kwargs.get("config"))
        self.runs.append(run)
        return run


class _StubConfig(dict):
    def update(self, *args, allow_val_change=False, **kwargs) -> None:
        del allow_val_change
        super().update(*args, **kwargs)


def _context(
    *,
    scenario: str = "math",
    step: int = 7,
    run_segment: int = 0,
    run_step: int = 0,
) -> TrainingExperimentContext:
    return TrainingExperimentContext(
        scenario=scenario,
        recipe="openclawrl",
        step=step,
        source_artifact_ref=LiveWeightArtifactRef(
            artifact_id="math",
            version="artifact:7",
            parent_version="artifact:6",
            weight_version="engine:7",
        ),
        run_segment=run_segment,
        run_step=run_step,
        backend="ExampleTrainingBackend",
        backend_config={"runtime": "custom", "optimizer": "adamw"},
    )


def _event(
    *,
    scenario: str = "math",
    step: int = 7,
    run_segment: int = 0,
    run_step: int = 0,
) -> TrainingExperimentEvent:
    return TrainingExperimentEvent(
        context=_context(scenario=scenario, step=step, run_segment=run_segment, run_step=run_step),
        produced_artifact_ref=LiveWeightArtifactRef(
            artifact_id="math",
            version="artifact:8",
            parent_version="artifact:7",
            weight_version="engine:8",
        ),
        metrics={"train/loss": 0.25, "reward": 0.8, "selection": {"score": 1.0, "label": "ok"}},
        outcome="committed",
        training_job_id="job-7",
        source_weight_version="engine:7",
        produced_weight_version="engine:8",
        checkpoint_path="/checkpoints/hf/7",
    )


def _tracker(client: _StubClient, **overrides) -> WandbExperimentTracker:
    values = {
        "enabled": True,
        "project": "reef-tests",
        "entity": "has",
        "group_prefix": "pg",
        "name_prefix": "smoke",
        "tags": ("cpu", "stub"),
        "mode": "offline",
    }
    values.update(overrides)
    return WandbExperimentTracker(
        WandbConfig(**values),
        model="Qwen/test",
        training_config={
            "global_batch_size": 2,
            "lr": 1e-6,
            "slime_flags": "--wandb-key secret",
            "api_token": "secret",
        },
        client=client,
    )


@pytest.mark.unit
def test_generic_wandb_provider_records_committed_backend_event() -> None:
    client = _StubClient()
    tracker = _tracker(client)
    event = _event()

    correlation = tracker.correlation_metrics(event.context)
    tracker.record(event)

    assert correlation == {
        "experiment/provider": "wandb",
        "experiment/project": "reef-tests",
        "experiment/group": "pg/math",
        "experiment/run_id": client.init_calls[0]["id"],
        "experiment/entity": "has",
    }
    init = client.init_calls[0]
    assert init["mode"] == "offline"
    assert init["resume"] == "allow"
    assert init["reinit"] == "create_new"
    assert init["group"] == "pg/math"
    assert init["name"] == "smoke-segment-0"
    assert init["config"] == {
        "reef": {
            "scenario": "math",
            "recipe": "openclawrl",
            "backend": None,
            "model": "Qwen/test",
            "run_segment": 0,
            "source_artifact": {
                "kind": "weight",
                "artifact_id": "math",
                "version": "artifact:7",
                "parent_version": "artifact:6",
                "weight_version": "engine:7",
            },
        },
        "backend": {},
        "training": {"global_batch_size": 2, "lr": 1e-6},
    }
    assert client.runs[0].config["reef"]["backend"] == "ExampleTrainingBackend"
    assert client.runs[0].config["backend"] == {"runtime": "custom", "optimizer": "adamw"}
    logged, step = client.runs[0].logged[0]
    assert step is None
    assert logged["train/step"] == 0
    assert logged["reef/step"] == 7
    assert logged["train/loss"] == pytest.approx(0.25)
    assert logged["selection/score"] == pytest.approx(1.0)
    assert "selection/label" not in logged
    assert logged["reef/source_artifact_version"] == "artifact:7"
    assert logged["reef/produced_artifact_version"] == "artifact:8"
    assert logged["reef/source_weight_version"] == "engine:7"
    assert logged["reef/produced_weight_version"] == "engine:8"
    assert logged["reef/training_job_id"] == "job-7"
    assert logged["reef/checkpoint_path"] == "/checkpoints/hf/7"
    assert client.runs[0].artifacts == []

    tracker.close()
    assert client.runs[0].finished is True


@pytest.mark.unit
def test_run_identity_is_stable_across_restart_and_separate_per_scenario() -> None:
    clients = [_StubClient(), _StubClient()]
    for client in clients:
        _tracker(client).record(_event())

    assert clients[0].init_calls[0]["id"] == clients[1].init_calls[0]["id"]
    assert clients[0].init_calls[0]["resume"] == clients[1].init_calls[0]["resume"] == "allow"

    tracker = _tracker(clients[0])
    tracker.record(_event(scenario="coding"))
    assert clients[0].init_calls[0]["id"] != clients[0].init_calls[1]["id"]
    assert clients[0].init_calls[0]["group"] == "pg/math"
    assert clients[0].init_calls[1]["group"] == "pg/coding"


@pytest.mark.unit
def test_rollback_finishes_one_curve_and_next_commit_starts_another_run() -> None:
    client = _StubClient()
    tracker = _tracker(client)
    tracker.record(_event(step=7, run_segment=0, run_step=3))
    first_run = client.runs[0]

    tracker.record_rollback(
        RollbackExperimentEvent(
            scenario="math",
            recipe="openclawrl",
            step=8,
            run_segment=0,
            source_artifact_ref=_event().produced_artifact_ref,
            produced_artifact_ref=_context().source_artifact_ref,
            target_artifact_version="artifact:6",
        )
    )

    assert first_run.finished is True
    assert first_run.summary["reef/ended_by"] == "rollback"
    assert first_run.summary["reef/rollback_step"] == 8
    tracker.record(_event(step=9, run_segment=8, run_step=0))

    assert len(client.runs) == 2
    assert client.init_calls[0]["group"] == client.init_calls[1]["group"] == "pg/math"
    assert client.init_calls[0]["id"] != client.init_calls[1]["id"]
    assert client.init_calls[1]["name"] == "smoke-segment-8"
    assert client.runs[1].logged[0][1] is None


@pytest.mark.unit
def test_recipe_and_processor_share_the_scenario_logger_without_wandb_imports() -> None:
    client = _StubClient()
    tracker = _tracker(client)
    logger = tracker.bind_scenario(
        scenario="math",
        recipe="openclawrl",
        source_artifact_ref=_context().source_artifact_ref,
        run_segment=0,
    )

    logger.log({"accepted": 2, "quality": {"mean": 0.75}, "label": "ignored"}, namespace="processor")
    logger.log({"temperature": 0.6}, namespace="recipe")

    assert len(client.runs) == 1
    processor, processor_step = client.runs[0].logged[0]
    recipe, recipe_step = client.runs[0].logged[1]
    assert processor_step is None and recipe_step is None
    assert processor == {
        "processor/event": 0,
        "processor/accepted": 2,
        "processor/quality/mean": pytest.approx(0.75),
    }
    assert recipe == {"recipe/event": 0, "recipe/temperature": pytest.approx(0.6)}

    tracker.record_rollback(
        RollbackExperimentEvent(
            scenario="math",
            recipe="openclawrl",
            step=8,
            run_segment=0,
            source_artifact_ref=_event().produced_artifact_ref,
            produced_artifact_ref=_context().source_artifact_ref,
            target_artifact_version="artifact:6",
        )
    )
    logger.log({"accepted": 1}, namespace="processor")

    assert len(client.runs) == 2
    assert client.init_calls[1]["name"] == "smoke-segment-8"
    assert client.runs[1].logged[0][0]["processor/event"] == 0


@pytest.mark.unit
def test_disabled_and_failed_tracking_never_fail_training() -> None:
    disabled_client = _StubClient(fail_init=True)
    disabled = WandbExperimentTracker(
        WandbConfig(enabled=True, mode="disabled"),
        model=None,
        training_config={},
        client=disabled_client,
    )
    disabled.record(_event())
    assert disabled.correlation_metrics(_context()) == {}
    assert disabled_client.init_calls == []

    failing_client = _StubClient(fail_init=True)
    failing = _tracker(failing_client, mode="online")
    failing.record(_event())
    failing.record(_event(step=8))
    assert len(failing_client.init_calls) == 1

    logging_client = _StubClient(fail_log=True)
    logging_tracker = _tracker(logging_client)
    logging_tracker.record(_event())
    assert len(logging_client.init_calls) == 1


@pytest.mark.unit
def test_config_surface_is_generic_and_rejects_credentials() -> None:
    assert WandbConfig.from_mapping(None).active is False
    assert WandbConfig.from_mapping({}).active is False
    default_group = WandbExperimentTracker(
        WandbConfig(enabled=True, mode="offline"),
        model=None,
        training_config={},
    ).correlation_metrics(_context())
    assert default_group["experiment/group"] == "math"

    settings = service_settings_from_config(
        {
            "reef": {"recipe": "pg"},
            "training": {
                "wandb": {
                    "enabled": True,
                    "project": "reef-smoke",
                    "tags": ["baseline", "cpu", "baseline"],
                    "mode": "offline",
                }
            },
        }
    )
    config = WandbConfig.from_mapping(settings.wandb_config)
    assert config.project == "reef-smoke"
    assert config.tags == ("baseline", "cpu")
    assert config.active is True

    with pytest.raises(ValueError, match=r"unknown training\.wandb settings"):
        WandbConfig.from_mapping({"enabled": True, "api_key": "must-not-be-accepted"})
    with pytest.raises(ValueError, match=r"unknown training\.wandb settings: group, name"):
        WandbConfig.from_mapping({"group": "legacy", "name": "legacy"})


@pytest.mark.unit
def test_wandb_logs_every_optimizer_step_of_a_job_before_the_job_row() -> None:
    client = _StubClient()
    tracker = _tracker(client)
    first = dataclasses.replace(
        _event(step=1, run_step=0),
        metrics={
            "train/loss": 0.25,
            "train_steps": [{"train/loss": 0.5, "train/step": 0}, {"train/loss": 0.4, "train/step": 1}],
        },
    )
    second = dataclasses.replace(
        _event(step=2, run_step=1),
        metrics={"train/loss": 0.2, "train_steps": [{"train/loss": 0.3}, {"train/loss": 0.25}, {"train/loss": 0.22}]},
    )

    tracker.record(first)
    tracker.record(second)

    run = client.runs[0]
    rows = [metrics for metrics, _ in run.logged]
    step_rows = [row for row in rows if "step/step" in row]
    job_rows = [row for row in rows if "train/step" in row]
    # Two jobs, five optimizer steps in a run-wide sequence, each job's rows before its summary row.
    assert [row["step/step"] for row in step_rows] == [0, 1, 2, 3, 4]
    assert [row["step/loss"] for row in step_rows] == pytest.approx([0.5, 0.4, 0.3, 0.25, 0.22])
    assert [row["reef/step"] for row in step_rows] == [1, 1, 2, 2, 2]
    assert rows.index(job_rows[0]) == 2 and rows.index(job_rows[1]) == 6
    assert "train_steps" not in job_rows[0] and "step/loss" not in job_rows[0]
    assert run.summary["reef/optimizer_steps"] == 5
    assert (("step/*",), {"step_metric": "step/step"}) in run.defined


@pytest.mark.unit
def test_wandb_optimizer_step_counter_resumes_from_the_run_summary() -> None:
    client = _StubClient()
    tracker = _tracker(client)
    tracker.record(_event(step=1))  # initializes the run
    client.runs[0].summary["reef/optimizer_steps"] = 40  # a resumed run that logged 40 steps before restart

    tracker.record(dataclasses.replace(_event(step=2, run_step=1), metrics={"train_steps": [{"train/loss": 0.1}]}))

    step_rows = [m for m, _ in client.runs[0].logged if "step/step" in m]
    assert [row["step/step"] for row in step_rows] == [40]
    assert client.runs[0].summary["reef/optimizer_steps"] == 41
