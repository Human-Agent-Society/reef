"""The Slime component runs each job in a driver process of its own, and the driver trains the job."""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from tests.slime_backend.test_slime_component import batch_for, parameters, worker_args
from tests.support.slime_component import finished

from reef.core.job_data import JobData, TaskSample
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.train.local_jobs import JOB_FILE
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend import driver
from reef.train.slime_backend.component import (
    DRIVER_FILE,
    DRIVER_RESULT_FILE,
    NO_ROLLOUT_ENGINES,
    SERVING_LOG,
    SlimeSettings,
    SlimeStart,
    SlimeTrainingComponent,
    settings_to_json,
)
from reef.train.slime_backend.rollout_group import SERVING_FILE

SETTINGS = SlimeSettings(
    arguments=("--lr=1e-06",),
    loss_family="sft",
    recipe="recipes.test:Recipe",
    execution={"execution": {"training": "ray"}},
    ray_address="ray://head:10001",
    namespace="test",
)
#: A driver that records one checkpoint of rollout 0 and the job's metrics, as ``SlimeJobRunner`` does.
SAVING_DRIVER = """
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
settings = json.loads((directory / "driver.json").read_text())
start = directory.parent.parent / "reef-starts" / "rollout_0"
start.mkdir(parents=True)
(start / "start.json").write_text(json.dumps(
    {"rollout_id": 0, "objective": None, "algorithm_state": {}, "schedule_state": None}))
(directory / "checkpoints.json").write_text(json.dumps({"checkpoints": [
    {"step": 2, "path": str(start), "release_path": str(start), "metrics": {"train/loss": 0.5}}]}))
(directory / "driver-result.json").write_text(json.dumps({"metrics": {"arguments": settings["arguments"]}}))
"""


class Component(SlimeTrainingComponent):
    """The component with a driver that runs ``script`` on the job's directory."""

    script = SAVING_DRIVER

    def driver_command(self, directory: Path) -> list[str]:
        return [sys.executable, "-c", self.script, str(directory)]


@pytest.fixture
def parsed(tmp_path, monkeypatch):
    """Slime's parse of the deployment's arguments, without Slime: the arguments it got, in ``calls``."""
    calls: list[tuple[str, ...]] = []

    def prepare(arguments, **options):
        calls.append(tuple(arguments))
        return worker_args(tmp_path, megatron_lora_rank=0, hf_checkpoint=str(tmp_path / "model")), None, None

    monkeypatch.setattr(slime_component, "prepare_slime_arguments", prepare)
    monkeypatch.setattr(slime_component, "prepare_checkpoint_storage", lambda args, retention: None)
    # The LoRA module loads Megatron's LoRA stack; the component reads only whether the actor trains one.
    lora = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.lora")
    lora.megatron_lora_enabled = lambda args: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, lora.__name__, lora)
    return calls


def sft_job(component: SlimeTrainingComponent, job_id: str = "job-0", start: str | None = None) -> TrainingJob:
    return TrainingJob(job_id, "slime", start or component.base_start, batch_for(0), parameters(TrainingMethod("sft")))


@pytest.mark.unit
def test_the_component_parses_the_arguments_at_deployment_and_starts_nothing(tmp_path, parsed):
    component = Component(SETTINGS)
    try:
        # One parse, as a job that starts no rollout engines; a malformed option fails here.
        assert parsed == [("--lr=1e-06", NO_ROLLOUT_ENGINES)]
        assert component.held_rollout_id is None
    finally:
        component.close()


@pytest.mark.unit
def test_a_jobs_driver_reports_its_checkpoints_and_the_next_job_continues_them(tmp_path, parsed):
    component = Component(SETTINGS)
    try:
        component.submit(sft_job(component))
        status = finished(component, "job-0")
        assert status.state == "finished", status.error
        assert [(checkpoint.step, checkpoint.metrics) for checkpoint in status.checkpoints] == [
            (2, {"train/loss": 0.5})
        ]
        # The driver started from the deployment's settings.
        assert status.metrics == {"arguments": ["--lr=1e-06"]}
        directory = component.job_directory("job-0")
        assert read_json(directory / DRIVER_FILE) == settings_to_json(SETTINGS)
        # The next job continues the rollout the driver saved, not the untrained weights.
        assert component.held_rollout_id == 0
        with pytest.raises(ValueError, match="continues the weights of rollout 0"):
            component.submit(sft_job(component, "job-1"))
        component.submit(sft_job(component, "job-1", status.checkpoints[0].path))
    finally:
        component.close()


@pytest.mark.unit
def test_a_failed_driver_fails_its_job_with_the_end_of_its_output(tmp_path, parsed):
    component = Component(SETTINGS)
    component.script = "import sys; print('Megatron could not load the checkpoint'); sys.exit(3)"
    try:
        component.submit(sft_job(component))
        status = finished(component, "job-0")
        assert status.state == "failed"
        assert "exited with 3" in status.error and "Megatron could not load the checkpoint" in status.error
    finally:
        component.close()


@pytest.mark.unit
def test_a_cancel_stops_the_driver_and_a_close_leaves_the_job_to_resume(tmp_path, parsed):
    component = Component(SETTINGS)
    component.script = (
        "import sys, time; from pathlib import Path; (Path(sys.argv[1]) / 'started').touch(); time.sleep(600)"
    )
    try:
        component.submit(sft_job(component))
        directory = component.job_directory("job-0")
        deadline = time.monotonic() + 20
        while not (directory / "started").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        component.cancel("job-0")
        status = finished(component, "job-0")
        assert (status.state, status.error) == ("failed", "cancelled")
    finally:
        component.close()
    # A job whose driver a close stopped has not ended: a restarted component runs its driver again.
    component = Component(SETTINGS)
    component.script = (
        "import sys, time; from pathlib import Path; (Path(sys.argv[1]) / 'started').touch(); time.sleep(600)"
    )
    try:
        component.submit(sft_job(component, "job-1"))
        directory = component.job_directory("job-1")
        deadline = time.monotonic() + 20
        while not (directory / "started").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        component.close()
    assert component.status("job-1").state == "running"
    component = Component(SETTINGS)
    try:
        assert finished(component, "job-1").state == "finished"
    finally:
        component.close()


@pytest.mark.unit
def test_closing_the_component_disposes_the_serving_session_that_a_driver_left(tmp_path, parsed):
    component = Component(SETTINGS)
    root = component.root
    # No session: closing starts no disposal.
    component.dispose_command = lambda: [sys.executable, "-c", "raise SystemExit(5)"]  # type: ignore[method-assign]
    component.close()
    component = Component(SETTINGS)
    write_json(root / SERVING_FILE, {"session": "rollout:abc", "lease": "0" * 32})
    component.dispose_command = lambda: [  # type: ignore[method-assign]
        sys.executable,
        "-c",
        f"import pathlib; print('disposed'); pathlib.Path({str(root / SERVING_FILE)!r}).unlink()",
    ]
    component.close()
    assert not (root / SERVING_FILE).exists()
    assert "disposed" in (root / SERVING_LOG).read_text()


class Runner:
    """The runner a driver starts: records its settings, the job it ran, and its close."""

    def __init__(self, settings: SlimeSettings, calls: dict[str, Any], serving: bool = False) -> None:
        calls["settings"] = settings
        calls["serving"] = serving
        self.calls = calls

    def run(self, job: TrainingJob, directory: Path) -> dict[str, Any]:
        self.calls["job"] = job
        return {"train/loss": 0.25}

    def close(self) -> None:
        self.calls["closed"] = True


def job_directory(root: Path, data: JobData, rollout: bool) -> Path:
    directory = root / "reef-jobs" / "job-0"
    start = root / "reef-starts" / "base"
    start.mkdir(parents=True)
    SlimeStart(None, None, {}, None).write(start)
    job_parameters = {
        **parameters(TrainingMethod("sft")),
        **({"rollout": {"rollouts_per_task": 2}} if rollout else {}),
    }
    write_json(directory / JOB_FILE, {"start": str(start), "data": data.to_dict(), "parameters": job_parameters})
    write_json(directory / DRIVER_FILE, settings_to_json(SETTINGS))
    return directory


@pytest.mark.unit
@pytest.mark.parametrize(("task_job", "session"), [(False, False), (True, False), (False, True)])
def test_the_driver_uses_the_serving_session_for_a_task_job_and_while_one_runs(
    tmp_path, monkeypatch, task_job, session
):
    calls: dict[str, Any] = {}
    monkeypatch.setattr(driver, "start_slime_runner", lambda settings, *, serving: Runner(settings, calls, serving))
    (tmp_path / "add").mkdir()
    data = JobData("tasks", (TaskSample(tmp_path / "add"),)) if task_job else batch_for(0)
    directory = job_directory(tmp_path, data, task_job)
    if session:
        write_json(tmp_path / SERVING_FILE, {"session": "rollout:abc", "lease": "0" * 32})

    driver.main([str(directory)])

    arguments = calls["settings"].arguments
    # A trajectory job without a session starts no rollout engines; a job on the session keeps the
    # deployment's options, which the session's engines started with.
    serving = task_job or session
    assert calls["serving"] is serving
    assert arguments == (("--lr=1e-06",) if serving else ("--lr=1e-06", NO_ROLLOUT_ENGINES))
    assert calls["job"].job_id == "job-0" and calls["closed"]
    assert json.loads((directory / DRIVER_RESULT_FILE).read_text()) == {"metrics": {"train/loss": 0.25}}


@pytest.mark.unit
def test_a_failed_job_still_closes_the_driver(tmp_path, monkeypatch):
    calls: dict[str, Any] = {}

    class Failing(Runner):
        def run(self, job, directory):
            raise RuntimeError("the workers died")

    monkeypatch.setattr(driver, "start_slime_runner", lambda settings, *, serving: Failing(settings, calls))
    directory = job_directory(tmp_path, batch_for(0), False)
    with pytest.raises(RuntimeError, match="the workers died"):
        driver.main([str(directory)])
    assert calls["closed"] and not (directory / DRIVER_RESULT_FILE).exists()
