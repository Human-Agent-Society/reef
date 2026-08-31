"""One training thread, several scenarios, one adapter-serving runtime."""

from __future__ import annotations

import time

import pytest
from reef_service.test_commit_log import RecordingRuntime, build_training_dispatcher, wait_for_step

from reef.artifact import InMemoryRepositoryBackend
from reef.core.errors import ReefError
from reef.core.records_types import RequestType
from reef.records import AgentRecord
from reef.surface import adapter_name


class AdapterRuntime(RecordingRuntime):
    """A recording runtime that time-slices one adapter slot between scenarios."""

    def __init__(self) -> None:
        super().__init__(served_version="w0")
        self.scenarios: list[str] = []
        self.adapter_versions: dict[str, str] = {}

    @property
    def concurrent_training_scenarios(self) -> bool:
        return True

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        return self.adapter_versions.get(scenario)

    def train_candidate(self, payload):
        self.scenarios.append(payload["scenario"])
        return super().train_candidate(payload)

    def activate_candidate(self, candidate):
        activated = super().activate_candidate(candidate)
        self.adapter_versions[self.scenarios[-1]] = activated.runtime_load_id
        return activated


def _inference(scenario: str, agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=agent_record_id,
    )


def _report(scenario: str, agent_record_id: str, reference: str) -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference]},
        agent_record_id=agent_record_id,
        references=(reference,),
    )


def _feed(dispatcher, scenario: str, index: int) -> None:
    dispatcher.accept_record(_inference(scenario, f"{scenario}-i{index}"), recipe="test_policy")
    dispatcher.accept_record(_report(scenario, f"{scenario}-r{index}", f"{scenario}-i{index}"), recipe="test_policy")


@pytest.mark.unit
def test_two_scenarios_train_through_one_adapter_runtime(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = AdapterRuntime()
    dispatcher = build_training_dispatcher(
        runtime, tmp_path, InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    )
    try:
        _feed(dispatcher, "math", 1)
        _feed(dispatcher, "code", 1)
        wait_for_step(dispatcher, 1, scenario="math")
        wait_for_step(dispatcher, 1, scenario="code")
        _feed(dispatcher, "code", 2)
        wait_for_step(dispatcher, 2, scenario="code")

        assert dispatcher._registry.training_scenario_names == ("math", "code")
        assert runtime.scenarios == ["math", "code", "code"], "every job names the scenario whose adapter it trains"
        math = dispatcher.get_or_create_scenario("math")
        code = dispatcher.get_or_create_scenario("code")
        assert math.scenario_step == 1 and code.scenario_step == 2
        assert math.current_artifact_ref().runtime_load_id == "w1"
        assert code.current_artifact_ref().runtime_load_id == "w3"

        status = dispatcher.training_status
        assert set(status["scenarios"]) == {"math", "code"}
        assert status["scenarios"]["math"]["adapter_runtime_load_id"] == "w1"
        assert status["scenarios"]["code"]["adapter_runtime_load_id"] == "w3"

        # Each scenario's surface routes to its own adapter revision.
        payload = {"messages": []}
        math_hooks = math.surface.inference
        code_hooks = code.surface.inference
        assert math_hooks is not None and code_hooks is not None
        from reef.artifact import Artifact

        served_math = math_hooks.prepare_request(
            Artifact(math.current_artifact_ref(), math.repository), "/v1", payload
        )
        served_code = code_hooks.prepare_request(
            Artifact(code.current_artifact_ref(), code.repository), "/v1", payload
        )
        assert served_math["lora_path"] == adapter_name("math", "w1")
        assert served_code["lora_path"] == adapter_name("code", "w3")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_full_weight_runtime_still_trains_one_scenario_only(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = build_training_dispatcher(
        RecordingRuntime(), tmp_path, InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    )
    try:
        dispatcher.get_or_create_scenario("math", "test_policy")
        with pytest.raises(ReefError, match="already bound"):
            dispatcher.get_or_create_scenario("code", "test_policy")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_one_scenarios_failure_reloads_only_that_scenario(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()

    class FlakyRuntime(AdapterRuntime):
        def train_candidate(self, payload):
            if payload["scenario"] == "code" and not getattr(self, "_failed", False):
                self._failed = True
                self.scenarios.append("code")
                raise RuntimeError("code exploded")
            return super().train_candidate(payload)

    runtime = FlakyRuntime()
    dispatcher = build_training_dispatcher(
        runtime, tmp_path, InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    )
    try:
        _feed(dispatcher, "math", 1)
        _feed(dispatcher, "code", 1)
        wait_for_step(dispatcher, 1, scenario="math")
        for _ in range(1000):
            if "code" in runtime.scenarios:
                break
            time.sleep(0.001)
        assert runtime.scenarios.count("code") >= 1, "the failing scenario was attempted"
        assert dispatcher.get_or_create_scenario("math").scenario_step == 1, "math committed despite code's failure"
        # The failed turn reloaded code alone; its next accept retries the batch.
        _feed(dispatcher, "code", 2)
        wait_for_step(dispatcher, 1, scenario="code")
        assert dispatcher.get_or_create_scenario("math").scenario_step == 1
    finally:
        dispatcher.close()
