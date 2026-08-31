from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest

from recipes.tttd.examples.tttd.harness.run_controller import (
    ReefTrainingStatusClient,
    ScenarioTrainingFailure,
    ScenarioTrainingStatus,
    TTTDRunController,
    TTTDRunIdentity,
    TTTDRunStateError,
    TTTDRunStateStore,
)


class _Archive:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def state_dict(self):
        return {"steps": list(self.steps)}

    def load_state_dict(self, value):
        self.steps = list(value["steps"])

    @property
    def candidates(self):
        return tuple(SimpleNamespace(reward=float(step + 1)) for step in self.steps)


class _Harness:
    def __init__(self, events, rollouts_per_step=4) -> None:
        self.archive = _Archive()
        self.events = events
        self.rollouts_per_step = rollouts_per_step

    def run_step(self, step):
        self.events.append(("run", step))
        self.archive.steps.append(step)
        return tuple(f"step-{step}-rollout-{index}" for index in range(self.rollouts_per_step))


class _StatusReader:
    def __init__(self, statuses, events=None) -> None:
        self.statuses = deque(statuses)
        self.events = events
        self.last = None

    def scenario_status(self, scenario):
        assert scenario == "smoke"
        if self.statuses:
            self.last = self.statuses.popleft()
        if self.events is not None:
            self.events.append(("status", None if self.last is None else self.last.scenario_step))
        return self.last


def _identity():
    return TTTDRunIdentity(
        scenario="smoke",
        model="qwen",
        recipe="tttd",
        inference_path="/v1/chat/completions",
        instruction_sha256="instruction-hash",
        groups_per_step=2,
        rollouts_per_group=2,
        max_new_tokens=64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        enable_thinking=True,
        exploration=1.0,
        invalid_reward=0.0,
    )


def _status(step, version):
    return ScenarioTrainingStatus(step, version, batch_ready=False)


@pytest.mark.unit
def test_two_by_two_smoke_waits_for_each_training_commit(tmp_path) -> None:
    events = []
    harness = _Harness(events)
    reader = _StatusReader(
        [
            _status(0, "v0"),  # initial restore check
            _status(0, "v0"),  # step 0 precondition
            _status(0, "v0"),
            _status(1, "v1"),  # step 0 commit
            _status(1, "v1"),  # step 1 precondition
            _status(1, "v1"),
            _status(2, "v2"),  # step 1 commit
        ],
        events,
    )
    store = TTTDRunStateStore(tmp_path / "state.json", _identity())
    controller = TTTDRunController(
        harness,
        reader,
        store,
        poll_interval_s=0.01,
        sleep=lambda _seconds: events.append(("sleep", None)),
        emit=lambda event: events.append((event["event"], event.get("step"))),
    )

    outcome = controller.run(2)

    assert len(outcome.results) == 8
    assert (outcome.start_step, outcome.next_step, outcome.runtime_load_id) == (0, 2, "v2")
    assert events.index(("tttd_step_committed", 0)) < events.index(("run", 1))
    assert store.load()["phase"] == "committed"
    assert store.load()["archive"] == {"steps": [0, 1]}


@pytest.mark.unit
def test_commit_event_reports_archive_progress(tmp_path) -> None:
    events = []
    controller = TTTDRunController(
        _Harness([]),
        _StatusReader([_status(0, "v0"), _status(0, "v0"), _status(1, "v1")]),
        TTTDRunStateStore(tmp_path / "state.json", _identity()),
        poll_interval_s=0.01,
        sleep=lambda _seconds: None,
        emit=events.append,
    )

    controller.run(1)

    assert events[-1] == {
        "event": "tttd_step_committed",
        "step": 0,
        "next_step": 1,
        "runtime_load_id": "v1",
        "archive_size": 1,
        "archive_best_reward": 1.0,
    }


@pytest.mark.unit
def test_pending_archive_finishes_recovered_training_before_resuming(tmp_path) -> None:
    store = TTTDRunStateStore(tmp_path / "state.json", _identity())
    store.save_pending(next_step=1, previous_runtime_load_id="v0", archive={"steps": [0]})
    harness = _Harness([])
    reader = _StatusReader([_status(0, "v0"), _status(1, "v1")])
    controller = TTTDRunController(harness, reader, store, poll_interval_s=0.01, sleep=lambda _seconds: None)

    outcome = controller.run(1)

    assert outcome.results == ()
    assert harness.archive.steps == [0]
    assert store.load()["phase"] == "committed"
    assert store.load()["runtime_load_id"] == "v1"


@pytest.mark.unit
def test_commit_waits_for_serving_version_reconciliation(tmp_path) -> None:
    events = []
    harness = _Harness(events)
    reader = _StatusReader(
        [
            _status(0, "v0"),  # initial restore check
            _status(0, "v0"),  # step 0 precondition
            _status(1, "v0"),  # durable head committed; runtime status is stale
            _status(1, "v1"),  # runtime acknowledges the new serving weights
        ],
        events,
    )
    controller = TTTDRunController(
        harness,
        reader,
        TTTDRunStateStore(tmp_path / "state.json", _identity()),
        poll_interval_s=0.01,
        sleep=lambda _seconds: events.append(("sleep", None)),
    )

    outcome = controller.run(1)

    assert outcome.runtime_load_id == "v1"
    assert ("sleep", None) in events


@pytest.mark.unit
def test_resume_rebinds_session_scoped_runtime_load_id(tmp_path) -> None:
    store = TTTDRunStateStore(tmp_path / "state.json", _identity())
    store.save_committed(next_step=1, runtime_load_id="v1", archive={"steps": [0]})
    events = []
    controller = TTTDRunController(
        _Harness(events),
        _StatusReader([_status(1, "new-session-v1")]),
        store,
        poll_interval_s=0.01,
        sleep=lambda _seconds: None,
        emit=events.append,
    )

    outcome = controller.run(1)

    assert outcome.results == ()
    assert outcome.runtime_load_id == "new-session-v1"
    assert store.load()["runtime_load_id"] == "new-session-v1"
    assert events == [
        {
            "event": "tttd_runtime_load_id_rebound",
            "step": 1,
            "previous_runtime_load_id": "v1",
            "runtime_load_id": "new-session-v1",
        }
    ]


@pytest.mark.unit
def test_resume_fails_closed_when_reef_and_archive_steps_disagree(tmp_path) -> None:
    store = TTTDRunStateStore(tmp_path / "state.json", _identity())
    store.save_committed(next_step=1, runtime_load_id="v1", archive={"steps": [0]})
    controller = TTTDRunController(
        _Harness([]),
        _StatusReader([_status(2, "v2")]),
        store,
        poll_interval_s=0.01,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(TTTDRunStateError, match="beyond PUCT step"):
        controller.run(2)


@pytest.mark.unit
def test_status_client_reads_authenticated_public_training_status() -> None:
    observed = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "error": None,
                    "preload_errors": {},
                    "scenarios": {
                        "smoke": {
                            "scenario_step": 1,
                            "current_runtime_load_id": "v1",
                            "batch_ready": False,
                            "processor": {"failed_steps": []},
                        }
                    },
                }
            ).encode()

    def opener(request, *, timeout):
        observed.update(url=request.full_url, authorization=request.get_header("Authorization"), timeout=timeout)
        return _Response()

    status = ReefTrainingStatusClient(
        "http://reef:8900/",
        token="secret",
        timeout_s=7,
        opener=opener,
    ).scenario_status("smoke")

    assert status == _status(1, "v1")
    assert observed == {
        "url": "http://reef:8900/reef/status",
        "authorization": "Bearer secret",
        "timeout": 7.0,
    }


@pytest.mark.unit
def test_controller_fails_fast_when_reef_discards_a_mixed_artifact_step(tmp_path) -> None:
    failure = ScenarioTrainingFailure(
        step=0,
        reason="mixed_release_ids",
        release_ids=("v0", "v1"),
    )
    store = TTTDRunStateStore(tmp_path / "state.json", _identity())
    store.save_pending(next_step=1, previous_runtime_load_id="v0", archive={"steps": [0]})
    controller = TTTDRunController(
        _Harness([]),
        _StatusReader([ScenarioTrainingStatus(0, "v0", False, (failure,))]),
        store,
        poll_interval_s=0.01,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(TTTDRunStateError, match="mixed_release_ids"):
        controller.run(1)
