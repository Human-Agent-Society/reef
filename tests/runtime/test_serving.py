"""The serving runtimes: the control plane loads each published release into the serving engines."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.core.errors import ReefError
from reef.runtime.deployment import InferenceService
from reef.runtime.interfaces import InferenceBackend, InferenceHandler, RuntimeContractError
from reef.runtime.residency import AdapterCapacityExhausted, AdapterResidencyError
from reef.runtime.serving import ADAPTER_DIR, AdapterServingRuntime, FullWeightServingRuntime
from reef.surface.adapter import adapter_name, parse_adapter_name


class Engines(InferenceBackend):
    """Serving engines that record each control call.

    ``misreport`` names a version they report after a load; ``dead`` is the
    number of engines the next ``recover`` replaces.
    """

    def __init__(self) -> None:
        self.events: list = []
        self.versions = ["boot"]
        self.misreport: str | None = None
        self.dead = 0

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        self.events.append(("load", files.layout, adapter_name, runtime_load_id, files.path))
        self.versions = [runtime_load_id if self.misreport is None else self.misreport]

    def initialize_version(self, runtime_load_id):
        self.events.append(("initialize", runtime_load_id))
        self.versions = [runtime_load_id]

    def inference_url(self):
        return "http://component.example"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def recover(self):
        replaced, self.dead = self.dead, 0
        self.events.append(("recover", replaced))
        return replaced

    def abort(self):
        self.events.append("abort")
        self.dead = 2

    def unload_adapter(self, name):
        self.events.append(("unload", name))

    def loads(self) -> list:
        return [event for event in self.events if event[0] == "load"]


class Handler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        raise AssertionError("unused")


class Service(InferenceService):
    """The service that owns the engines; only ``close`` is reached."""

    connection_protocol = "test-engines-v1"

    def __init__(self) -> None:
        self.closed = 0

    def start(self, resources):
        raise AssertionError("unused")

    def backend(self, connection):
        raise AssertionError("unused")

    def check_health(self):
        raise AssertionError("unused")

    def poll(self):
        raise AssertionError("unused")

    def close(self):
        self.closed += 1


def saved_adapter(path: Path) -> Path:
    (path / ADAPTER_DIR).mkdir(parents=True)
    (path / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (path / "tinker-checkpoint.json").write_text("{}")
    return path


def saved_full_weights(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def base_release(path: Path) -> Path:
    """A release with only the repository's files: the base model."""
    path.mkdir(parents=True)
    (path / "reef-artifact.json").write_text("{}")
    return path


def release(path: Path, content_id: str) -> Artifact:
    return Artifact(ArtifactRef(content_id, f"release-{content_id}", None), None, local_path=path)


@pytest.fixture
def engines():
    return Engines()


@pytest.fixture
def serving(engines):
    return AdapterServingRuntime(engines, Handler(), adapter_capacity=2)


def test_the_engines_start_serving_the_base_model_under_a_fresh_version(serving, engines):
    [(_, version), resumed] = engines.events
    assert (engines.events[0][0], resumed) == ("initialize", "resume")
    assert serving.current_runtime_load_id() == version == serving.base_version
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.base_url == "http://component.example"
    assert serving.per_scenario_adapters and serving.shared_by_training_scenarios


def test_each_release_loads_under_a_new_version_and_the_same_content_loads_nothing(serving, engines, tmp_path):
    checkpoint = saved_adapter(tmp_path / "job-1")
    version = serving.activate_checkpoint(release(checkpoint, "c1"), scenario="math")
    assert engines.events[-4:] == [
        "pause",
        ("recover", 0),
        ("load", "adapter", adapter_name("math", version), version, checkpoint / ADAPTER_DIR),
        "resume",
    ]
    assert serving.serving_adapter_runtime_load_id("math") == version
    # The committer activates the staged release, then the published one with the same content.
    calls = len(engines.events)
    assert serving.activate_checkpoint(release(checkpoint, "c1"), scenario="math") == version
    assert len(engines.events) == calls
    newer = serving.activate_checkpoint(release(saved_adapter(tmp_path / "job-2"), "c2"), scenario="math")
    assert newer != version and serving.serving_adapter_runtime_load_id("math") == newer


def test_scenarios_serve_their_own_adapters_and_a_restart_loads_each_release_again(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=2)
    math = serving.activate_checkpoint(release(saved_adapter(tmp_path / "math"), "m"), scenario="math")
    code = serving.activate_checkpoint(release(saved_adapter(tmp_path / "code"), "c"), scenario="code")
    assert (serving.serving_adapter_runtime_load_id("math"), serving.serving_adapter_runtime_load_id("code")) == (
        math,
        code,
    )
    assert serving.adapter_residency_status()["counters"]["loads"] == 2
    restarted = AdapterServingRuntime(Engines(), Handler(), adapter_capacity=2)
    again = restarted.activate_checkpoint(release(tmp_path / "math", "m"), scenario="math")
    # A new incarnation never reuses a version for other weights.
    assert again.split(":")[0] != math.split(":")[0]


def test_a_rollback_loads_another_release_and_a_release_without_weights_serves_the_base(serving, tmp_path):
    first = serving.activate_checkpoint(release(saved_adapter(tmp_path / "job-1"), "j1"), scenario="math")
    older = serving.activate_checkpoint(release(saved_adapter(tmp_path / "older"), "o"), scenario="math")
    assert older != first and serving.serving_adapter_runtime_load_id("math") == older
    serving.activate_checkpoint(release(base_release(tmp_path / "empty"), "base"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") is None


def test_a_base_model_release_unloads_the_scenario_adapter_and_serves_the_base_version(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=1)
    base = serving.current_runtime_load_id()
    assert base == serving.base_version
    math = serving.activate_checkpoint(release(saved_adapter(tmp_path / "math"), "m"), scenario="math")
    assert serving.current_runtime_load_id() == math != base
    assert serving.activate_checkpoint(release(base_release(tmp_path / "empty"), "base"), scenario="math") == base
    # Generation pauses while the adapter unloads.
    assert engines.events[-3:] == ["pause", ("unload", adapter_name("math", math)), "resume"]
    assert serving.current_runtime_load_id() == base
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.adapter_residency_status()["resident"] == 0
    # The freed slot takes another scenario's adapter without an eviction.
    code = serving.activate_checkpoint(release(saved_adapter(tmp_path / "code"), "c"), scenario="code")
    assert serving.current_runtime_load_id() == code
    assert serving.adapter_residency_status()["counters"]["evictions"] == 0


def test_only_a_saved_adapter_of_a_named_scenario_is_served(serving, tmp_path):
    with pytest.raises(ReefError, match="named scenario"):
        serving.activate_checkpoint(release(saved_adapter(tmp_path / "a"), "a"))
    manifest_only = tmp_path / "manifest-only"
    manifest_only.mkdir()
    (manifest_only / "tinker-checkpoint.json").write_text("{}")
    with pytest.raises(ReefError, match="holds no PEFT adapter"):
        serving.activate_checkpoint(release(manifest_only, "m"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") is None


def test_a_scenario_replaces_its_own_adapter_but_never_evicts_another_scenarios(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=1)
    first = serving.activate_checkpoint(release(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    # Generation is paused during the load, so the scenario's current adapter makes room for its next one.
    second = serving.activate_checkpoint(release(saved_adapter(tmp_path / "m2"), "m2"), scenario="math")
    assert ("unload", adapter_name("math", first)) in engines.events
    with pytest.raises(AdapterCapacityExhausted, match="raise --max-loaded-loras"):
        serving.activate_checkpoint(release(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    assert engines.events[-1] == "resume"
    assert serving.serving_adapter_runtime_load_id("math") == second
    assert serving.serving_adapter_runtime_load_id("code") is None


def test_engines_that_report_another_version_after_an_adapter_load_fail_the_activation(serving, engines, tmp_path):
    engines.misreport = "stale:1"
    with pytest.raises(AdapterResidencyError, match="could not load") as raised:
        serving.activate_checkpoint(release(saved_adapter(tmp_path / "a"), "a"), scenario="math")
    assert isinstance(raised.value.__cause__, RuntimeContractError)
    assert "after loading" in str(raised.value.__cause__)
    # The engines resume, the failed load is dropped, and the scenario serves nothing.
    assert engines.events[-2:] == [("unload", adapter_name("math", engines.loads()[-1][3])), "resume"]
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.adapter_residency_status()["resident"] == 0


def test_a_failed_load_that_took_the_scenarios_own_slot_leaves_it_serving_no_adapter(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=1)
    first = serving.activate_checkpoint(release(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    engines.misreport = "stale:1"
    with pytest.raises(AdapterResidencyError, match="could not load"):
        serving.activate_checkpoint(release(saved_adapter(tmp_path / "m2"), "m2"), scenario="math")
    # The new adapter evicted the first one, so requests sample the base model rather than name a gone adapter.
    assert ("unload", adapter_name("math", first)) in engines.events
    assert serving.serving_adapter_runtime_load_id("math") is None
    # The first release is not served any more, so activating it again loads it under a new version.
    engines.misreport = None
    again = serving.activate_checkpoint(release(tmp_path / "m1", "m1"), scenario="math")
    assert again != first and serving.serving_adapter_runtime_load_id("math") == again


def test_shutdown_closes_admission_and_the_service_that_owns_the_engines(engines):
    service = Service()
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=1, service=service)
    serving.shutdown()
    assert service.closed == 1
    assert serving.inference_admission_status["open"] is False


def test_full_weights_load_under_a_new_version_and_the_same_content_loads_nothing(engines, tmp_path):
    serving = FullWeightServingRuntime(engines, Handler(), model_path="Qwen/Qwen3-8B")
    initial = serving.current_runtime_load_id()
    assert not serving.per_scenario_adapters and not serving.shared_by_training_scenarios
    # Before any load, a release without weights is the base model that the engines already serve.
    assert serving.activate_checkpoint(release(base_release(tmp_path / "base"), "base")) == initial
    weights = saved_full_weights(tmp_path / "step-1")
    version = serving.activate_checkpoint(release(weights, "w1"))
    assert version != initial
    assert engines.events[-4:] == ["pause", ("recover", 0), ("load", "full", None, version, weights), "resume"]
    assert serving.current_runtime_load_id() == version
    calls = len(engines.events)
    assert serving.activate_checkpoint(release(weights, "w1")) == version
    assert len(engines.events) == calls
    rolled_back = serving.activate_checkpoint(release(saved_full_weights(tmp_path / "step-0"), "w0"))
    assert rolled_back not in (initial, version)


def test_full_weights_cannot_return_to_the_base_model(engines, tmp_path):
    serving = FullWeightServingRuntime(engines, Handler())
    version = serving.activate_checkpoint(release(saved_full_weights(tmp_path / "step-1"), "w1"))
    calls = len(engines.events)
    with pytest.raises(ReefError, match="cannot return to the base model"):
        serving.activate_checkpoint(release(base_release(tmp_path / "base"), "base"))
    assert len(engines.events) == calls
    assert serving.current_runtime_load_id() == version


def test_engines_that_report_another_version_after_a_full_load_fail_it_and_keep_the_served_version(engines, tmp_path):
    serving = FullWeightServingRuntime(engines, Handler())
    initial = serving.current_runtime_load_id()
    engines.misreport = "stale:1"
    weights = saved_full_weights(tmp_path / "step-1")
    with pytest.raises(RuntimeContractError, match=r"report \['stale:1'\] after loading"):
        serving.activate_checkpoint(release(weights, "w1"))
    assert engines.events[-1] == "resume"
    assert serving.current_runtime_load_id() == initial
    # The release is not recorded as served: a retry loads it again under a new version.
    engines.misreport = None
    retried = serving.activate_checkpoint(release(weights, "w1"))
    assert len(engines.loads()) == 2 and retried == serving.current_runtime_load_id()


def test_each_full_weight_load_first_replaces_the_dead_engines_while_they_are_paused(engines, tmp_path):
    serving = FullWeightServingRuntime(engines, Handler())
    engines.dead = 1
    serving.activate_checkpoint(release(saved_full_weights(tmp_path / "w1"), "c1"))
    order = [event if isinstance(event, str) else event[0] for event in engines.events[2:]]
    assert order == ["pause", "recover", "load", "resume"]


def test_a_replaced_engine_restarts_every_engine_and_reloads_each_scenarios_adapter(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=3)
    math = serving.activate_checkpoint(release(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    engines.dead = 1
    code = serving.activate_checkpoint(release(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    loads = [(adapter, version) for _, _, adapter, version, _ in engines.loads()]
    # The restart loads math's current adapter again before code's new one.
    assert loads == [
        (adapter_name("math", math), math),
        (adapter_name("math", math), math),
        (adapter_name("code", code), code),
    ]
    assert "abort" in engines.events
    assert serving.adapter_residency_status()["resident"] == 2
    assert serving.serving_adapter_runtime_load_id("math") == math


def test_a_restart_that_cannot_load_one_scenarios_adapter_still_loads_the_others(engines, tmp_path):
    class FailingMath(Engines):
        def load_weight_files(self, files, *, adapter_name, runtime_load_id):
            if parse_adapter_name(adapter_name)[0] == "math" and "abort" in self.events:
                raise RuntimeError("the engine refused the adapter")
            super().load_weight_files(files, adapter_name=adapter_name, runtime_load_id=runtime_load_id)

    engines = FailingMath()
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=3)
    serving.activate_checkpoint(release(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    code = serving.activate_checkpoint(release(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    with pytest.raises(AdapterResidencyError, match=r"scenarios \['math'\]"):
        serving.restart_engines()
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.serving_adapter_runtime_load_id("code") == code


def test_a_load_without_dead_engines_restarts_nothing(serving, engines, tmp_path):
    serving.activate_checkpoint(release(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    assert ("recover", 0) in engines.events
    assert "abort" not in engines.events


class RestartingEngines(Engines):
    """Engines that a component restart can lose; ``recovered`` is set when they resume after a recovery."""

    def __init__(self) -> None:
        super().__init__()
        self.recovered = threading.Event()
        self.replaced = 0

    def recover(self):
        replaced = super().recover()
        self.replaced += replaced
        return replaced

    def resume(self):
        super().resume()
        if self.replaced:
            self.recovered.set()


class WatchedService(Service):
    """The engines' service: ``restart`` makes the next health check fail as after a component restart."""

    def __init__(self, engines: RestartingEngines, *, lost: int) -> None:
        super().__init__()
        self.engines = engines
        self.lost = lost
        self.restarted = threading.Event()

    def check_health(self):
        if self.restarted.is_set():
            self.restarted.clear()
            self.engines.dead, self.engines.versions = self.lost, ["restarted"]
            raise RuntimeError("the component restarted")

    def restart(self) -> None:
        self.restarted.set()


def test_a_failed_health_check_restarts_the_engines_and_loads_each_scenarios_adapter_again(tmp_path):
    engines = RestartingEngines()
    service = WatchedService(engines, lost=2)
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=2, service=service, health_interval_s=0.01)
    try:
        math = serving.activate_checkpoint(release(saved_adapter(tmp_path / "math"), "m"), scenario="math")
        code = serving.activate_checkpoint(release(saved_adapter(tmp_path / "code"), "c"), scenario="code")
        loaded = len(engines.loads())
        service.restart()
        assert engines.recovered.wait(timeout=10)
        reloaded = [(adapter, version) for _, _, adapter, version, _ in engines.loads()[loaded:]]
        assert reloaded == [(adapter_name("math", math), math), (adapter_name("code", code), code)]
        assert engines.events[-1] == "resume" and "abort" in engines.events
        assert serving.serving_adapter_runtime_load_id("math") == math
    finally:
        serving.shutdown()
    assert service.closed == 1


def test_a_failed_health_check_loads_the_served_full_weights_again_under_their_version(tmp_path):
    engines = RestartingEngines()
    service = WatchedService(engines, lost=1)
    serving = FullWeightServingRuntime(engines, Handler(), service=service, health_interval_s=0.01)
    try:
        weights = saved_full_weights(tmp_path / "step-1")
        version = serving.activate_checkpoint(release(weights, "w1"))
        calls = len(engines.events)
        service.restart()
        assert engines.recovered.wait(timeout=10)
        assert engines.events[calls:] == ["pause", ("recover", 1), ("load", "full", None, version, weights), "resume"]
        assert serving.current_runtime_load_id() == version and engines.runtime_load_ids() == [version]
    finally:
        serving.shutdown()


def test_recovery_that_replaces_no_engine_loads_nothing(engines, tmp_path):
    serving = FullWeightServingRuntime(engines, Handler())
    serving.activate_checkpoint(release(saved_full_weights(tmp_path / "step-1"), "w1"))
    calls = len(engines.events)
    serving.recover_engines()
    assert engines.events[calls:] == ["pause", ("recover", 0), "resume"]
