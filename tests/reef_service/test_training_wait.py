"""The training thread's wait policy: when may it sleep forever?

Regression suite for a dropped wire: #289 introduced
``DataProcessor.derivation_pending`` so judgments landing asynchronously
wake the training thread on a bounded poll, but the dispatcher-side
polling was lost in a merge and never reached main — on a quiet workload
a landed judgment sat unabsorbed until the next record happened to
arrive. These tests pin the wait policy itself, and the status surface
that makes a wrongly sleeping thread visible (issue #344): the last
drain time, the ready-but-undrained flag, and its bounded warning.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

import reef.dispatcher as dispatcher_module
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.dispatcher import _DERIVATION_POLL_SECONDS, _STORAGE_RETRY_SECONDS, _UNDRAINED_WARNING_SECONDS, Dispatcher
from reef.recipe import Recipe

pytestmark = pytest.mark.unit


def _dispatcher() -> Dispatcher:
    root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
    initial = root / "initial"
    initial.mkdir()
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(initial, root=root / "repository"),
    )


def _bind(
    dispatcher: Dispatcher,
    *,
    pending: bool,
    batch_ready: bool = False,
    processor_status: dict | None = None,
    runtime=None,
    last_committed_step: dict | None = None,
) -> None:
    processor = SimpleNamespace(derivation_pending=lambda: pending)
    trainer = SimpleNamespace(
        processor=processor,
        batch_ready=lambda: batch_ready,
        processor_status=lambda: dict(processor_status or {}),
    )
    scenario = SimpleNamespace(
        trainer=trainer,
        scenario_step=0,
        runtime=runtime,
        commit_status={"scenario_step": 0, "last_committed_step": last_committed_step},
    )
    dispatcher._registry = SimpleNamespace(
        training_scenario_name="s",
        training_status_scenario_names=("s",),
        get_optional=lambda name: scenario if name == "s" else None,
        preload_errors=(),
    )


def test_no_training_scenario_sleeps_until_the_next_accept() -> None:
    dispatcher = _dispatcher()
    assert dispatcher._training_wait_timeout() is None


def test_quiet_processor_sleeps_until_the_next_accept() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False)
    assert dispatcher._training_wait_timeout() is None


def test_pending_derivation_polls_on_a_bounded_interval() -> None:
    # The regression: judgments land on the worker without a new record
    # ever setting the ready event; sleeping forever here stalls training
    # on a quiet workload.
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=True)
    assert dispatcher._training_wait_timeout() == _DERIVATION_POLL_SECONDS


def test_blocked_storage_keeps_the_tighter_cadence() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=True)
    dispatcher._training.storage_status = {"state": "blocked"}
    assert dispatcher._training_wait_timeout() == min(_STORAGE_RETRY_SECONDS, _DERIVATION_POLL_SECONDS)


def test_status_exposes_the_last_drain_time_and_ready_state() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False, batch_ready=True)
    drained_at = time.time()
    dispatcher._training.last_drain = drained_at
    status = dispatcher.training_status
    assert status["last_drain_at"] == drained_at
    assert status["scenarios"]["s"]["batch_ready"] is True


def test_status_exposes_the_last_committed_step_outcome() -> None:
    dispatcher = _dispatcher()
    committed = {
        "step": 3,
        "recorded_at": 1_756_400_000.0,
        "metrics": {"skipped": "no proposal", "selection": {"reason": "candidate lost"}},
    }
    _bind(dispatcher, pending=False, last_committed_step=committed)

    assert dispatcher.training_status["scenarios"]["s"]["last_committed_step"] == committed


def test_status_keeps_the_published_version_until_inference_reopens(monkeypatch) -> None:
    class Runtime:
        open = False
        current = "engine:old"

        @property
        def inference_admission_status(self):
            return {"open": self.open, "active": 0}

        @staticmethod
        def serving_runtime_load_id():
            return "engine:new"

        def current_runtime_load_id(self):
            return self.current

    monkeypatch.setattr(dispatcher_module, "TrainingRuntime", Runtime)
    dispatcher = _dispatcher()
    _bind(
        dispatcher,
        pending=False,
        runtime=Runtime(),
    )

    status = dispatcher.training_status["scenarios"]["s"]
    assert status["current_runtime_load_id"] == "engine:old"
    assert status["inference_admission"] == {"open": False, "active": 0}

    runtime = dispatcher._registry.get_optional("s").runtime
    runtime.current = "engine:new"
    runtime.open = True
    status = dispatcher.training_status["scenarios"]["s"]
    assert status["current_runtime_load_id"] == "engine:new"
    assert status["inference_admission"] == {"open": True, "active": 0}


def test_status_exposes_terminal_processor_outcomes() -> None:
    dispatcher = _dispatcher()
    outcome = {
        "failed_steps": [
            {
                "step": 1,
                "reason": "mixed_release_ids",
                "release_ids": ["old", "new"],
            }
        ]
    }
    _bind(dispatcher, pending=False, processor_status=outcome)

    assert dispatcher.training_status["scenarios"]["s"]["processor"] == outcome


def test_status_reports_build_failure_instead_of_raising() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False)

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher._registry.get_optional("s").trainer.batch_ready = fail

    status = dispatcher.training_status

    assert status["scenarios"] == {}
    assert status["error"] == "s: RuntimeError: processor status is unavailable"

    dispatcher._registry.get_optional("s").trainer.batch_ready = lambda: False
    recovered = dispatcher.training_status
    assert recovered["error"] is None
    assert "s" in recovered["scenarios"]


def test_status_reports_training_and_build_failures() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False)
    dispatcher._record_training_error("s", "RuntimeError: backend failed")

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher._registry.get_optional("s").trainer.batch_ready = fail

    assert dispatcher.training_status["error"] == (
        "s: RuntimeError: backend failed\ns: RuntimeError: processor status is unavailable"
    )


def test_status_reports_an_unexpectedly_stopped_training_thread() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False)
    stopped = Thread(target=lambda: None)
    stopped.start()
    stopped.join()
    dispatcher._training.thread = stopped

    assert dispatcher.training_status["error"] == "s: RuntimeError: training thread stopped unexpectedly"


def test_training_loop_reports_recovery_failures_outside_a_training_attempt() -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False)
    dispatcher._training.ready.set()

    def fail_training() -> bool:
        raise RuntimeError("training failed")

    def fail_reload(name: str) -> None:
        assert name == "s"
        raise RuntimeError("scenario reload failed")

    dispatcher._process_training = fail_training
    dispatcher._registry.reload = fail_reload
    dispatcher._run_training()

    assert dispatcher.training_status["error"] == "s: RuntimeError: scenario reload failed"


def test_a_ready_batch_undrained_past_the_bound_warns_once(caplog) -> None:
    # The stall signature: a batch is ready, the thread sleeps. Status reads
    # must say so out loud, and exactly once per stall.
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False, batch_ready=True)
    dispatcher._training.last_drain = time.time() - _UNDRAINED_WARNING_SECONDS - 1
    with caplog.at_level(logging.WARNING, logger="reef.dispatcher"):
        for _ in range(2):
            assert dispatcher.training_status["scenarios"]["s"]["batch_ready"] is True
    assert sum("undrained" in record.message for record in caplog.records) == 1
    # A completed drain re-arms the warning for the next stall.
    dispatcher._record_training_drain()
    assert dispatcher._training.undrained_warned is False


def test_a_freshly_drained_ready_batch_does_not_warn(caplog) -> None:
    dispatcher = _dispatcher()
    _bind(dispatcher, pending=False, batch_ready=True)
    dispatcher._training.last_drain = time.time()
    with caplog.at_level(logging.WARNING, logger="reef.dispatcher"):
        assert dispatcher.training_status["scenarios"]["s"]["batch_ready"] is True
    assert not caplog.records
