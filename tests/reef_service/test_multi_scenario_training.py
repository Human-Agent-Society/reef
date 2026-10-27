"""Training scenarios on one deployment: one per process, or several on a runtime that serves each its own weights."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import VERSION_FILE, StubInferenceRuntime, StubTrainingComponent

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.storage.sqlite import SQLiteScenarioStorage


class SharedRuntime(StubInferenceRuntime):
    """A stub runtime that several training scenarios can share."""

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True


def _dispatcher(
    tmp_path, runtime: StubInferenceRuntime, *, shares_scenarios: bool = False
) -> tuple[Dispatcher, StubTrainingComponent]:
    initial = tmp_path / "initial"
    initial.mkdir()
    component = StubTrainingComponent(tmp_path / "checkpoints", shares_scenarios=shares_scenarios)
    dispatcher = Dispatcher(
        TestPolicyRecipe(component, runtime=runtime),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    return dispatcher, component


def _train(dispatcher: Dispatcher, runtime: StubInferenceRuntime, scenario: str) -> None:
    """Feed one scored exchange of ``scenario``, served by the current version, and wait until its job ends."""
    dispatcher.get_or_create_scenario(scenario)
    inference = f"{scenario}-i1"
    dispatcher.accept_record(
        AgentRecord.create(
            scenario=scenario,
            request_type=RequestType.INFERENCE,
            payload={
                "tokens": [1, 2],
                "loss_mask": [0, 1],
                "rollout_log_probs": [-0.2],
                "runtime_load_id": runtime.current_runtime_load_id(),
            },
            agent_record_id=inference,
        )
    )
    dispatcher.accept_record(
        AgentRecord.create(
            scenario=scenario,
            request_type=RequestType.REPORT,
            payload={"score": 1.0, "references": [inference]},
            agent_record_id=f"{scenario}-r1",
            references=(inference,),
        )
    )
    deadline = time.monotonic() + 10
    while dispatcher.build_training_status()["scenarios"][scenario]["finished_jobs"] < 1:
        if time.monotonic() > deadline:
            raise AssertionError(f"the job of {scenario!r} did not end")
        time.sleep(0.05)


@pytest.mark.unit
def test_a_runtime_that_serves_one_model_binds_training_to_one_scenario(tmp_path) -> None:
    dispatcher, _ = _dispatcher(tmp_path, StubInferenceRuntime())
    try:
        dispatcher.get_or_create_scenario("math")
        with pytest.raises(ReefError, match="training is already bound to scenario 'math'"):
            dispatcher.get_or_create_scenario("code")
        assert dispatcher.get_or_create_scenario("math").training_component is not None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_shared_runtime_binds_one_scenario_when_the_training_component_does_not_share(tmp_path) -> None:
    dispatcher, _ = _dispatcher(tmp_path, SharedRuntime(), shares_scenarios=False)
    try:
        dispatcher.get_or_create_scenario("math")
        with pytest.raises(ReefError, match="training is already bound to scenario 'math'"):
            dispatcher.get_or_create_scenario("code")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_shared_runtime_trains_each_scenario_on_the_one_training_component(tmp_path) -> None:
    runtime = SharedRuntime()
    dispatcher, component = _dispatcher(tmp_path, runtime, shares_scenarios=True)
    try:
        # The stub serves one version for every scenario, so each exchange names the version served when it arrives.
        _train(dispatcher, runtime, "math")
        _train(dispatcher, runtime, "code")
        # Each scenario submitted its own job of its own records, and each one's checkpoint became its head.
        assert [source_record_id(job.data.items[0]) for job in component.submitted] == ["math-i1", "code-i1"]
        for scenario, version in (("math", "w1"), ("code", "w2")):
            head = dispatcher.current_artifact(scenario).materialize()
            assert (Path(head.local_path) / VERSION_FILE).read_text() == version
    finally:
        dispatcher.close()
