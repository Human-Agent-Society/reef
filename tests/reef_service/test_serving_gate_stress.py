"""Temporary: repeat the serving-gate test under the full suite's load, so CI catches its rare hang."""

import threading

import pytest
from tests.reef_service.test_training_server import (
    test_status_reports_the_running_job_the_finished_jobs_and_the_served_adapter as served_adapter_test,
)
from tests.support.hang_report import install_timeline


@pytest.mark.unit
@pytest.mark.parametrize("repeat", range(25))
@pytest.mark.parametrize("per_scenario_adapters", [False, True], ids=["shared-weights", "per-scenario-adapters"])
def test_serving_gate_repeats(tmp_path, monkeypatch, per_scenario_adapters, repeat) -> None:
    install_timeline(monkeypatch)
    # Before the fix, a worker held more than 100 reef threads of earlier tests' unclosed dispatchers.
    reef_threads = [thread.name for thread in threading.enumerate() if thread.name.startswith("reef-")]
    assert len(reef_threads) < 60, f"{len(reef_threads)} reef threads before the test: {sorted(reef_threads)}"
    served_adapter_test(tmp_path, monkeypatch, per_scenario_adapters)
