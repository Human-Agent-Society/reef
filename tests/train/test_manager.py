"""The training manager's values and control's manager worker, against a fake target."""

from __future__ import annotations

import pytest

from reef.dispatcher.manager_worker import ManagedTraining, TrainingManagerWorker
from reef.train.manager import Cancel, ManagedJob, ManagerView, Train, TrainFrom, TrainingManager


def view(scenario: str = "s", **fields) -> ManagerView:
    values = {
        "component": "weights",
        "training_start": "a0",
        "selection_serves": True,
        "untrained_samples": ("x", "y"),
        "job_samples": 2,
        "waiting_requests": 0,
    }
    return ManagerView(scenario=scenario, **{**values, **fields})


class RecordingTarget(ManagedTraining):
    """Gives ``views`` and records each action that it runs."""

    def __init__(self, views: tuple[ManagerView, ...]) -> None:
        self.views = views
        self.ran: list[tuple[str, object, int]] = []

    def manager_views(self) -> tuple[ManagerView, ...]:
        return self.views

    def run_manager_action(self, view: ManagerView, action, *, job_limit: int) -> None:
        self.ran.append((view.scenario, action, job_limit))


class TrainThenFail(TrainingManager):
    """Trains what waits in scenario "good"; fails for any other scenario."""

    @property
    def job_limit(self) -> int:
        return 3

    def actions(self, view: ManagerView):
        if view.scenario != "good":
            raise RuntimeError("the manager's model is away")
        return (Train(view.untrained_samples, "enough samples wait"), TrainFrom("a1"), Cancel("j1"))


@pytest.mark.unit
def test_the_manager_values_check_their_conditions():
    assert Train(("x",), "why").train_from is None
    for samples in ((), ("x", "x")):
        with pytest.raises(ValueError, match="distinct samples"):
            Train(samples, "why")
    with pytest.raises(ValueError):
        Train(("x",), "")
    with pytest.raises(ValueError, match="positive int"):
        view(job_samples=0)
    with pytest.raises(ValueError, match="not negative"):
        view(waiting_requests=-1)
    with pytest.raises(ValueError, match="queued or running"):
        ManagedJob("j", "finished", None, False)
    assert view(jobs=(ManagedJob("j", "running", "a0", True),)).jobs[0].managed


@pytest.mark.unit
def test_a_round_runs_each_views_actions_in_order_and_a_failed_view_does_not_stop_the_others():
    target = RecordingTarget((view("bad"), view("good")))
    TrainingManagerWorker(TrainThenFail(), target).manage()
    assert target.ran == [
        ("good", Train(("x", "y"), "enough samples wait"), 3),
        ("good", TrainFrom("a1"), 3),
        ("good", Cancel("j1"), 3),
    ]
