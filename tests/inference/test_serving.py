"""The serving runtimes: the control plane loads each published snapshot into the serving engines."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.support.runtime_stubs import RunningEngines, StubInferenceRuntime
from tests.support.serving import Engines

from reef.artifact import (
    BASE_FILE,
    REFERENCE_FILE,
    BaseRoot,
    DownloadedSnapshot,
    HuggingFaceRevision,
    HuggingFaceSource,
)
from reef.artifact.artifact import Artifact, SnapshotRef
from reef.core.errors import ReefError
from reef.core.served_models import ServedArtifact
from reef.core.weight_layout import ADAPTER_DIR
from reef.inference.checkpoint_slots import CheckpointSlotsRuntime
from reef.inference.deployment import InferenceService
from reef.inference.interfaces import InferenceHandler, RuntimeContractError, full_weight_name
from reef.inference.residency import (
    AdapterCapacityExhaustedError,
    AdapterResidencyError,
    adapter_name,
    parse_adapter_name,
)
from reef.inference.serving import AdapterServingRuntime, FullWeightServingRuntime, OpenedEngines, ServingEngines


class ServiceEngines(RunningEngines):
    """Running engines that ``service`` owns and watches."""

    def __init__(self, backend, service) -> None:
        super().__init__(backend)
        self.service = service

    def check_health(self) -> None:
        self.service.check_health()

    def close(self) -> None:
        self.service.close()


def opened(runtime):
    """``runtime`` after control opened it."""
    runtime.open()
    return runtime


class Handler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        raise AssertionError("unused")


class Service(InferenceService):
    """The service that owns the engines; only ``close`` is reached."""

    def __init__(self) -> None:
        self.closed = 0

    @classmethod
    def from_config(cls, config):
        raise AssertionError("unused")

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


def saved_base_root(path: Path, model: str = "Qwen/Qwen3-8B") -> Path:
    """A base root: the base file alone, which names the default start of ``model``."""
    path.mkdir(parents=True)
    (path / BASE_FILE).write_text(BaseRoot({"model": model, "lora_rank": "16"}).to_text())
    return path


def published_snapshot(path: Path, artifact_id: str) -> Artifact:
    return Artifact(SnapshotRef(artifact_id, None), None, artifact_id=artifact_id, local_path=path)


@pytest.fixture
def engines():
    return Engines()


@pytest.fixture
def serving(engines):
    return opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2))


def test_the_engines_start_serving_the_base_model_under_a_fresh_version(serving, engines):
    [(_, version), resumed] = engines.events
    assert (engines.events[0][0], resumed) == ("initialize", "resume")
    assert version == serving.base_version
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.base_url == "http://component.example"
    assert serving.per_scenario_adapters and serving.shared_by_training_scenarios
    assert serving.served_weight_kind == "model_adapter"


def test_each_artifact_loads_under_a_new_version_and_the_same_artifact_loads_nothing(serving, engines, tmp_path):
    checkpoint = saved_adapter(tmp_path / "job-1")
    # Before its activation, an artifact has no binding and so no route.
    assert serving.served_artifact("c1", "math") is None
    version = serving.activate_checkpoint(published_snapshot(checkpoint, "c1"), scenario="math")
    assert engines.events[-4:] == [
        "pause",
        ("recover", 0),
        ("load", "adapter", adapter_name("math", version), version, checkpoint / ADAPTER_DIR),
        "resume",
    ]
    assert serving.serving_adapter_runtime_load_id("math") == version
    # The route names the activated artifact's resident adapter; no other artifact or scenario has one.
    route = serving.served_artifact("c1", "math")
    assert route == ServedArtifact("c1", "http://component.example", adapter_name("math", version))
    assert serving.served_artifact("c2", "math") is None and serving.served_artifact("c1", "code") is None
    # Repeating the same artifact does not reload the adapter.
    calls = len(engines.events)
    assert serving.activate_checkpoint(published_snapshot(checkpoint, "c1"), scenario="math") == version
    assert len(engines.events) == calls
    newer = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "job-2"), "c2"), scenario="math")
    assert newer != version and serving.serving_adapter_runtime_load_id("math") == newer


def test_scenarios_serve_their_own_adapters_and_a_restart_loads_each_snapshot_again(engines, tmp_path):
    serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2))
    math = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "math"), "m"), scenario="math")
    code = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "code"), "c"), scenario="code")
    assert (serving.serving_adapter_runtime_load_id("math"), serving.serving_adapter_runtime_load_id("code")) == (
        math,
        code,
    )
    assert serving.adapter_residency_status()["counters"]["loads"] == 2
    restarted = opened(AdapterServingRuntime(RunningEngines(Engines()), Handler(), adapter_capacity=2))
    again = restarted.activate_checkpoint(published_snapshot(tmp_path / "math", "m"), scenario="math")
    # A new incarnation never reuses a version for other weights.
    assert again.split(":")[0] != math.split(":")[0]


def test_an_activation_loads_another_snapshot_and_the_default_start_loads_as_any_adapter(serving, tmp_path):
    first = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "job-1"), "j1"), scenario="math")
    older = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "older"), "o"), scenario="math")
    assert older != first and serving.serving_adapter_runtime_load_id("math") == older
    # A root snapshot of an older Reef holds the untrained adapter, which loads under a version of its own.
    base = serving.activate_checkpoint(
        published_snapshot(saved_adapter(tmp_path / "untrained"), "base"), scenario="math"
    )
    assert base not in (first, older, serving.base_version)
    assert serving.serving_adapter_runtime_load_id("math") == base


def test_a_base_root_serves_the_base_model_and_loads_no_adapter(engines, tmp_path):
    serving = opened(
        AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2, model_path="Qwen/Qwen3-8B")
    )
    base = published_snapshot(saved_base_root(tmp_path / "base"), "base")
    # The route of the base root names the base model's load, which the receiver admits; no other artifact has it.
    base_route = ServedArtifact(
        "base", "http://component.example", full_weight_name("Qwen/Qwen3-8B", serving.base_version)
    )
    calls = len(engines.events)
    # Before any adapter: the engines serve the base model already, and nothing changes in them.
    assert serving.activate_checkpoint(base, scenario="math") == serving.base_version
    assert engines.events[calls:] == []
    assert serving.served_artifact("base", "math") == base_route
    assert serving.served_artifact("base", "code") is None

    trained = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "job-1"), "j1"), scenario="math")
    assert serving.served_artifact("base", "math") is None
    other = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "code"), "c1"), scenario="code")
    route = serving.served_artifact("j1", "math")
    calls = len(engines.events)
    # After an adapter: the scenario serves no adapter again. The requests that still name the adapter finish on
    # it, so nothing unloads until its route stops, but no route names the adapter's artifact any more.
    assert serving.activate_checkpoint(base, scenario="math") == serving.base_version
    assert engines.events[calls:] == []
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.served_model_name("math") == base_route.model_name
    assert serving.served_artifact("j1", "math") is None and serving.served_artifact("base", "math") == base_route
    assert serving.serving_adapter_runtime_load_id("code") == other
    serving.stop_serving(route)
    assert engines.events[-1] == ("unload", adapter_name("math", trained))
    # A recovery that replaces the engines loads the other scenario's adapter again and keeps the base root bound.
    engines.dead = 1
    serving.recover_engines()
    assert engines.loads()[-1][2] == adapter_name("code", other)
    assert serving.served_artifact("base", "math") == base_route
    # The adapter loads again under a new version when the scenario selects it again.
    again = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "job-1b"), "j1"), scenario="math")
    assert again not in (trained, serving.base_version) and serving.served_model_name("math") == adapter_name(
        "math", again
    )

    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / BASE_FILE).write_text("{}")
    with pytest.raises(ReefError, match="no readable base file"):
        serving.activate_checkpoint(published_snapshot(tmp_path / "broken", "broken"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") == again


def test_only_a_saved_adapter_of_a_named_scenario_is_served(serving, tmp_path):
    with pytest.raises(ReefError, match="named scenario"):
        serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "a"), "a"))
    manifest_only = tmp_path / "manifest-only"
    manifest_only.mkdir()
    (manifest_only / "tinker-checkpoint.json").write_text("{}")
    with pytest.raises(ReefError, match="holds no PEFT adapter"):
        serving.activate_checkpoint(published_snapshot(manifest_only, "m"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") is None


def test_a_scenario_replaces_its_own_adapter_but_never_evicts_another_scenarios(engines, tmp_path):
    serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=1))
    first = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    # Generation is paused during the load, so the scenario's current adapter makes room for its next one.
    second = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m2"), "m2"), scenario="math")
    assert ("unload", adapter_name("math", first)) in engines.events
    with pytest.raises(AdapterCapacityExhaustedError, match="raise --max-loaded-loras"):
        serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    assert engines.events[-1] == "resume"
    assert serving.serving_adapter_runtime_load_id("math") == second
    assert serving.serving_adapter_runtime_load_id("code") is None


def test_engines_that_report_another_version_after_an_adapter_load_fail_the_activation(serving, engines, tmp_path):
    engines.misreport = "stale:1"
    with pytest.raises(AdapterResidencyError, match="could not load") as raised:
        serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "a"), "a"), scenario="math")
    assert isinstance(raised.value.__cause__, RuntimeContractError)
    assert "after loading" in str(raised.value.__cause__)
    # The failed load is dropped and the scenario serves nothing. Generation stays paused until recovery.
    assert engines.events[-1] == ("unload", adapter_name("math", engines.loads()[-1][3]))
    assert serving.recovery_pending
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.adapter_residency_status()["resident"] == 0


def test_a_failed_load_that_took_the_scenarios_own_slot_serves_nothing_until_recovery_restores_it(engines, tmp_path):
    serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=1))
    first = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    engines.misreport = "stale:1"
    with pytest.raises(AdapterResidencyError, match="could not load"):
        serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m2"), "m2"), scenario="math")
    # The new adapter evicted the first one, so neither artifact may publish a route to the base model.
    assert ("unload", adapter_name("math", first)) in engines.events
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.served_artifact("m1", "math") is None
    assert serving.served_artifact("m2", "math") is None
    assert engines.events[-1] != "resume" and serving.recovery_pending
    # The next activation first restores the adapter that served before the failure, under its version.
    engines.misreport = None
    again = serving.activate_checkpoint(published_snapshot(tmp_path / "m1", "m1"), scenario="math")
    assert again == first and serving.served_artifact("m1", "math") is not None
    assert engines.events[-1] == "resume"


def test_shutdown_closes_admission_and_the_service_that_owns_the_engines(engines):
    service = Service()
    serving = opened(AdapterServingRuntime(ServiceEngines(engines, service), Handler(), adapter_capacity=1))
    serving.shutdown()
    assert service.closed == 1
    assert serving.inference_admission_status["open"] is False


def test_full_weights_load_under_a_new_version_and_the_same_artifact_loads_nothing(engines, tmp_path):
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="Qwen/Qwen3-8B"))
    initial = serving.base_version
    assert not serving.per_scenario_adapters and not serving.shared_by_training_scenarios
    assert serving.served_weight_kind == "model"
    # It serves only the checkpoint that it has loaded.
    assert not serving.serves_any_artifact and serving.served_artifact_capacity == 1
    weights = saved_full_weights(tmp_path / "step-1")
    # The engines start with the base model, which binds no artifact: no route before the first activation.
    assert serving.served_artifact("w1", "math") is None
    version = serving.activate_checkpoint(published_snapshot(weights, "w1"))
    assert version != initial
    assert engines.events[-4:] == ["pause", ("recover", 0), ("load", "full", None, version, weights), "resume"]
    # The route names this load, so the receiver can refuse it once the engines serve another one.
    assert serving.served_artifact("w1", "math") == ServedArtifact(
        "w1", "http://component.example", full_weight_name("Qwen/Qwen3-8B", version)
    )
    calls = len(engines.events)
    assert serving.activate_checkpoint(published_snapshot(weights, "w1")) == version
    assert len(engines.events) == calls
    rolled_back = serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "step-0"), "w0"))
    assert rolled_back not in (initial, version)
    # Only the last activation has a route.
    assert serving.served_artifact("w1", "math") is None and serving.served_artifact("w0", "math") is not None


def test_a_base_root_loads_the_base_models_directory_as_full_weights(engines, tmp_path):
    model = saved_full_weights(tmp_path / "model")
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path=str(model)))
    trained = serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "step-1"), "w1"))
    base = published_snapshot(saved_base_root(tmp_path / "base", str(model)), "base")
    version = serving.activate_checkpoint(base)
    assert version not in (trained, serving.base_version)
    assert engines.events[-4:] == ["pause", ("recover", 0), ("load", "full", None, version, model), "resume"]
    calls = len(engines.events)
    assert serving.activate_checkpoint(base) == version
    assert len(engines.events) == calls
    # Recovery loads the base model again under the same version.
    engines.dead = 1
    serving.recover_engines()
    assert engines.loads()[-1] == ("load", "full", None, version, model)

    # A model path that is no directory on this host cannot be loaded again: the activation fails and changes nothing.
    hub = opened(FullWeightServingRuntime(RunningEngines(Engines()), Handler(), model_path="Qwen/Qwen3-8B"))
    with pytest.raises(ReefError, match="is no directory on this host"):
        hub.activate_checkpoint(base)
    assert hub.served_artifact_id is None


@pytest.mark.parametrize("adapter", [False, True])
def test_weight_load_reuse_follows_artifact_identity_across_storage_snapshots(engines, tmp_path, adapter):
    if adapter:
        serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2))
        path = saved_adapter(tmp_path / "weights")
    else:
        serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
        path = saved_full_weights(tmp_path / "weights")
    first = Artifact(SnapshotRef("snapshot:one", None), None, local_path=path, artifact_id="a")
    same = Artifact(SnapshotRef("snapshot:two", None), None, local_path=path, artifact_id="a")
    other = Artifact(SnapshotRef("snapshot:three", None), None, local_path=path, artifact_id="b")
    try:
        version = serving.activate_checkpoint(first, scenario="math")
        calls = len(engines.events)
        assert serving.activate_checkpoint(same, scenario="math") == version
        assert len(engines.events) == calls
        assert serving.activate_checkpoint(other, scenario="math") != version
        assert len(engines.events) > calls
    finally:
        serving.shutdown()


def test_a_snapshot_that_refers_to_a_revision_loads_the_revisions_snapshot(engines, tmp_path, monkeypatch):
    revision = HuggingFaceRevision("org/model", "0123456789abcdef0123456789abcdef01234567")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()))
    snapshot = saved_full_weights(tmp_path / "snapshots" / revision.commit)
    downloaded = []

    def download(source):
        downloaded.append(source)
        return DownloadedSnapshot(snapshot, revision.commit)

    monkeypatch.setattr("reef.inference.serving.download_huggingface_snapshot", download)
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    version = serving.activate_checkpoint(published_snapshot(reference, "hf"))
    assert engines.events[-2] == ("load", "full", None, version, snapshot)
    assert downloaded == [HuggingFaceSource(revision.repo, revision.commit)]
    (reference / REFERENCE_FILE).write_text("{}")
    with pytest.raises(ReefError, match="no readable reference"):
        serving.activate_checkpoint(published_snapshot(reference, "bad"))


def test_engines_that_report_another_version_after_a_full_load_fail_it_and_keep_the_served_version(engines, tmp_path):
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    initial = serving.base_version
    engines.misreport = "stale:1"
    weights = saved_full_weights(tmp_path / "step-1")
    with pytest.raises(RuntimeContractError, match=r"report \['stale:1'\] after loading"):
        serving.activate_checkpoint(published_snapshot(weights, "w1"))
    assert engines.events[-1][0] == "load"
    # The failed load stays paused. A retry loads it again under a new version before resuming.
    engines.misreport = None
    retried = serving.activate_checkpoint(published_snapshot(weights, "w1"))
    assert len(engines.loads()) == 2 and retried not in (initial, "stale:1")


class PartialFullWeightEngines(Engines):
    """Two replicas whose first weight replacement can succeed before the load raises.

    ``weights`` has one checkpoint path or None for each replica. ``paused`` prevents generation;
    ``fail_load`` makes a load replace only the first replica while retaining both previous load IDs.
    ``resumed`` is set after each resume, so a test can wait for background recovery.
    Tests generate the resident paths to distinguish their contents from the reported load IDs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weights: tuple[Path | None, Path | None] = (None, None)
        self.paused = False
        self.fail_load = False
        self.resumed = threading.Event()

    def initialize_version(self, runtime_load_id):
        super().initialize_version(runtime_load_id)
        self.versions *= 2

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        self.weights = (files.path, self.weights[1])
        if self.fail_load:
            raise RuntimeError("one replica changed weights before the load failed")
        self.weights = (files.path, files.path)
        super().load_weight_files(files, adapter_name=adapter_name, runtime_load_id=runtime_load_id)
        self.versions *= 2

    def pause(self):
        super().pause()
        self.paused = True

    def resume(self):
        super().resume()
        self.paused = False
        self.resumed.set()

    def generate(self):
        if self.paused:
            raise RuntimeError("generation is paused")
        return self.weights


@pytest.mark.parametrize("failure", ["partial_load", "wrong_load_id"])
@pytest.mark.parametrize("recovery", ["previous_artifact", "new_artifact", "explicit_recovery"])
def test_full_weight_failure_blocks_generation_until_a_verified_reload(tmp_path, failure, recovery):
    engines = PartialFullWeightEngines()
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="test-model"))
    previous = published_snapshot(saved_full_weights(tmp_path / "previous"), "previous")
    candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
    serving.activate_checkpoint(previous)
    assert engines.generate() == (previous.local_path, previous.local_path)
    assert serving.served_artifact("previous", "math") is not None
    assert serving.served_artifact("candidate", "math") is None

    engines.fail_load = failure == "partial_load"
    engines.misreport = "stale:1" if failure == "wrong_load_id" else None
    with pytest.raises((RuntimeError, RuntimeContractError), match=r"one replica changed|after loading"):
        serving.activate_checkpoint(candidate)
    with pytest.raises(RuntimeError, match="generation is paused"):
        engines.generate()
    assert serving.served_artifact("previous", "math") is None
    assert serving.served_artifact("candidate", "math") is None

    engines.fail_load = False
    engines.misreport = None
    if recovery == "previous_artifact":
        serving.activate_checkpoint(previous)
        expected = previous.local_path
    elif recovery == "new_artifact":
        serving.activate_checkpoint(candidate)
        expected = candidate.local_path
    else:
        # Both engines are alive and can report the old ID after a partial load.
        # A health retry must restore the contents even when recover() replaces none.
        serving.recover_engines()
        expected = previous.local_path
    assert engines.generate() == (expected, expected)
    expected_artifact = "candidate" if recovery == "new_artifact" else "previous"
    assert serving.served_artifact(expected_artifact, "math") is not None


def test_failed_full_weight_recovery_retries_even_when_all_engines_are_alive(tmp_path):
    engines = PartialFullWeightEngines()
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    previous = published_snapshot(saved_full_weights(tmp_path / "previous"), "previous")
    serving.activate_checkpoint(previous)
    engines.dead = 1
    engines.fail_load = True
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.recover_engines()
    assert engines.dead == 0
    with pytest.raises(RuntimeError, match="generation is paused"):
        engines.generate()
    # The replacement survived, but its restoration did not. Retry the load, not just liveness.
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.recover_engines()
    engines.fail_load = False
    serving.recover_engines()
    assert engines.generate() == (previous.local_path, previous.local_path)


def test_failed_first_full_weight_load_requires_an_explicit_activation(tmp_path):
    engines = PartialFullWeightEngines()
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
    engines.fail_load = True
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.activate_checkpoint(candidate)
    engines.fail_load = False
    with pytest.raises(ReefError, match="no verified full-weight artifact"):
        serving.recover_engines()
    with pytest.raises(RuntimeError, match="generation is paused"):
        engines.generate()
    serving.activate_checkpoint(candidate)
    assert engines.generate() == (candidate.local_path, candidate.local_path)


def test_a_released_checkpoint_leaves_paused_engines_that_load_another_checkpoint(tmp_path):
    engines = PartialFullWeightEngines()
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="test-model"))
    first = published_snapshot(saved_full_weights(tmp_path / "first"), "first")
    second = published_snapshot(saved_full_weights(tmp_path / "second"), "second")
    version = serving.activate_checkpoint(first)
    calls = len(engines.events)
    serving.release_checkpoint()
    # The release only pauses. The engines keep the weights that they hold, and no route names them.
    assert engines.events[calls:] == ["pause"]
    assert serving.served_artifact_id is None and serving.awaits_first_activation and serving.recovery_pending
    assert serving.served_artifact("first", "math") is None
    with pytest.raises(RuntimeError, match="generation is paused"):
        engines.generate()
    # Nothing is left to restore, and a second release keeps the engines paused.
    with pytest.raises(ReefError, match="no verified full-weight artifact"):
        serving.recover_engines()
    serving.release_checkpoint()
    assert len(engines.loads()) == 1 and "resume" not in engines.events[calls:] and engines.paused
    reused = serving.activate_checkpoint(second)
    assert reused not in (serving.base_version, version)
    load = ("load", "full", None, reused, second.local_path)
    assert engines.events[-4:] == ["pause", ("recover", 0), load, "resume"]
    assert engines.generate() == (second.local_path, second.local_path)
    assert serving.served_artifact("first", "math") is None
    assert serving.served_artifact("second", "math") == ServedArtifact(
        "second", "http://component.example", full_weight_name("test-model", reused)
    )


def test_a_release_whose_pause_fails_keeps_the_checkpoint_as_the_recovery_target(tmp_path, monkeypatch):
    engines = PartialFullWeightEngines()
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="test-model"))
    served = published_snapshot(saved_full_weights(tmp_path / "served"), "served")
    serving.activate_checkpoint(served)
    route = serving.served_artifact("served", "math")

    def refuse_pause():
        raise RuntimeError("the engines did not pause")

    with monkeypatch.context() as patched:
        patched.setattr(engines, "pause", refuse_pause)
        with pytest.raises(RuntimeError, match="did not pause"):
            serving.release_checkpoint()
    # The pause is unconfirmed, so the route is withheld until recovery loads the same checkpoint again.
    assert serving.served_artifact_id == "served" and not serving.awaits_first_activation
    assert serving.recovery_pending and serving.served_artifact("served", "math") is None
    serving.recover_engines()
    assert engines.generate() == (served.local_path, served.local_path)
    assert route is not None and serving.served_artifact("served", "math") == route


class SlotEngines(RunningEngines):
    """The engines of the checkpoint slot at ``url``, which a test observes and makes fail.

    ``backend`` is a ``PartialFullWeightEngines``. ``owner_sequence`` is the last serving-owner sequence that the
    engines received, or None. ``opens`` counts the calls that opened them, and ``closes`` each call that tried to
    close them. ``refuse_open`` and ``refuse_close`` make those calls raise.
    """

    def __init__(self, url: str) -> None:
        super().__init__(PartialFullWeightEngines())
        self.url = url
        self.owner_sequence: int | None = None
        self.opens = 0
        self.closes = 0
        self.refuse_open = False
        self.refuse_close = False

    @property
    def base_url(self) -> str:
        return self.url

    def set_owner_sequence(self, sequence: int) -> None:
        self.owner_sequence = sequence

    def open(self) -> OpenedEngines:
        if self.refuse_open:
            raise RuntimeError(f"the engines at {self.url} did not open")
        self.opens += 1
        return super().open()

    def close(self) -> None:
        self.closes += 1
        if self.refuse_close:
            raise RuntimeError(f"the engines at {self.url} did not close")


@pytest.fixture
def slot_engines():
    return SlotEngines("http://slot-a.example"), SlotEngines("http://slot-b.example")


def checkpoint_slots(*slot_engines: SlotEngines) -> CheckpointSlotsRuntime:
    """One checkpoint slot of ``test-model`` on each of ``slot_engines``, before control opens them."""
    return CheckpointSlotsRuntime(
        [FullWeightServingRuntime(engines, Handler(), model_path="test-model") for engines in slot_engines]
    )


def checkpoints(path: Path) -> list[Artifact]:
    """The saved full-weight releases ``a``, ``b`` and ``c``."""
    return [published_snapshot(saved_full_weights(path / name), name) for name in "abc"]


def test_checkpoint_slots_serve_two_checkpoints_and_switch_between_them_without_a_load(slot_engines, tmp_path):
    first, second = slot_engines
    serving = checkpoint_slots(first, second)
    a, b, c = checkpoints(tmp_path)
    # A process that opens nothing finds each slot's handler. Building the runtime touched no engine.
    handlers = [slot.inference_handler for slot in serving.slots]
    assert [serving.handler_for(engines.url) for engines in slot_engines] == handlers
    assert handlers[0] is not handlers[1] and serving.inference_handler is handlers[0]
    assert serving.handler_for("http://slot-c.example") is None
    assert serving.served_artifact_capacity == 2 and serving.serving_slots == 1 and not serving.serves_any_artifact
    with pytest.raises(ReefError, match="not open"):
        serving.activate_checkpoint(a)
    assert first.backend.events == second.backend.events == []

    serving.open()
    assert serving.served_artifact("a", "math") is None and serving.served_model_name("math") == "test-model"
    loaded_a = serving.activate_checkpoint(a)
    route_b = serving.serve_artifact(b, scenario="math")
    loaded_b = second.backend.loads()[-1][3]
    route_a = ServedArtifact("a", "http://slot-a.example", full_weight_name("test-model", loaded_a))
    assert serving.served_artifact("a", "math") == route_a
    assert route_b == ServedArtifact("b", "http://slot-b.example", full_weight_name("test-model", loaded_b))
    assert route_a.model_name != route_b.model_name and serving.served_model_name("math") == route_a.model_name
    assert first.backend.generate() == (a.local_path, a.local_path)
    assert second.backend.generate() == (b.local_path, b.local_path)

    # A promotion and a rollback between the two bound checkpoints load nothing and keep both routes.
    calls = [len(engines.backend.events) for engines in slot_engines]
    assert serving.activate_checkpoint(b) == loaded_b and serving.served_model_name("math") == route_b.model_name
    assert serving.activate_checkpoint(a) == loaded_a and serving.serve_artifact(b, scenario="math") == route_b
    # Both slots are bound, so a third checkpoint gets none, neither beside the active one nor as the active one.
    with pytest.raises(ReefError, match=r"the 2 checkpoint slots .* \(a, b\), so artifact 'c' has no slot"):
        serving.serve_artifact(c, scenario="math")
    with pytest.raises(ReefError, match="has no slot"):
        serving.activate_checkpoint(c)
    # Only the exact route of a checkpoint's last verified load releases its slot.
    serving.stop_serving(ServedArtifact("b", route_b.url, route_a.model_name))
    serving.stop_serving(ServedArtifact("b", route_a.url, route_b.model_name))
    serving.stop_serving(ServedArtifact("c", route_b.url, route_b.model_name))
    assert [len(engines.backend.events) for engines in slot_engines] == calls
    assert serving.served_artifact("a", "math") == route_a and serving.served_artifact("b", "math") == route_b
    assert serving.served_model_name("math") == route_a.model_name

    serving.stop_serving(route_b)
    assert second.backend.events[calls[1] :] == ["pause"] and serving.served_artifact("b", "math") is None
    with pytest.raises(RuntimeError, match="generation is paused"):
        second.backend.generate()
    # The freed slot loads the third checkpoint under a new load ID, and the stopped route releases nothing again.
    route_c = serving.serve_artifact(c, scenario="math")
    serving.stop_serving(route_b)
    loaded_c = second.backend.loads()[-1][3]
    assert loaded_c != loaded_b
    assert route_c == ServedArtifact("c", "http://slot-b.example", full_weight_name("test-model", loaded_c))
    assert serving.served_artifact("c", "math") == route_c and serving.served_artifact("a", "math") == route_a
    assert second.backend.generate() == (c.local_path, c.local_path)
    assert first.backend.generate() == (a.local_path, a.local_path)


def test_a_stop_releases_the_slot_of_a_checkpoint_that_the_slot_has_not_restored(slot_engines, tmp_path):
    first, second = slot_engines
    serving = opened(checkpoint_slots(first, second))
    a, b, c = checkpoints(tmp_path)
    loaded_a = serving.activate_checkpoint(a)
    route_a = serving.served_artifact("a", "math")
    serving.activate_checkpoint(b)
    route_b = serving.served_artifact("b", "math")
    assert route_a is not None and route_b is not None
    first.backend.dead = 1
    first.backend.fail_load = True
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.slots[0].recover_engines()
    assert serving.served_artifact("a", "math") is None
    # A rollback that cannot load selects nothing: b stays active, and a keeps its slot for the restoration.
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.activate_checkpoint(a)
    assert serving.served_model_name("math") == route_b.model_name and serving.served_artifact("b", "math") == route_b
    with pytest.raises(ReefError, match="has no slot"):
        serving.serve_artifact(c, scenario="math")
    # The slot gives no route while it must restore a, so the last verified route is what releases it.
    serving.stop_serving(route_a)
    first.backend.fail_load = False
    with pytest.raises(ReefError, match="no verified full-weight artifact"):
        serving.slots[0].recover_engines()
    route_c = serving.serve_artifact(c, scenario="math")
    loaded_c = first.backend.loads()[-1][3]
    assert loaded_c != loaded_a
    assert route_c == ServedArtifact("c", "http://slot-a.example", full_weight_name("test-model", loaded_c))
    assert first.backend.generate() == (c.local_path, c.local_path)
    assert serving.served_artifact("a", "math") is None and serving.served_artifact("b", "math") == route_b


def test_a_stop_of_the_active_checkpoint_frees_its_slot_and_leaves_none_active(slot_engines, tmp_path):
    first, second = slot_engines
    serving = opened(checkpoint_slots(first, second))
    a, b, c = checkpoints(tmp_path)
    serving.activate_checkpoint(a)
    route_a = serving.served_artifact("a", "math")
    route_b = serving.serve_artifact(b, scenario="math")
    assert route_a is not None and route_b is not None
    # No route uses the active checkpoint any more, as after its scenario was removed: its stop frees the slot.
    calls = len(first.backend.events)
    serving.stop_serving(route_a)
    assert first.backend.events[calls:] == ["pause"] and serving.served_artifact("a", "math") is None
    assert serving.served_model_name("math") == "test-model" and serving.served_artifact("b", "math") == route_b
    # The freed slot takes the next active checkpoint. A route that is not a binding then releases nothing:
    # neither the stopped route of a, nor a route that names c or b with the URL or the name of another load.
    loaded_c = serving.activate_checkpoint(c)
    route_c = ServedArtifact("c", "http://slot-a.example", full_weight_name("test-model", loaded_c))
    calls = len(first.backend.events) + len(second.backend.events)
    serving.stop_serving(route_a)
    serving.stop_serving(ServedArtifact("c", route_a.url, route_a.model_name))
    serving.stop_serving(ServedArtifact("b", route_c.url, route_b.model_name))
    assert len(first.backend.events) + len(second.backend.events) == calls
    assert serving.served_artifact("c", "math") == route_c and serving.served_artifact("b", "math") == route_b
    assert serving.served_model_name("math") == route_c.model_name
    # The last active checkpoint retires without a successor, and b keeps its slot beside the free one.
    serving.stop_serving(route_c)
    assert serving.served_artifact("c", "math") is None and serving.served_model_name("math") == "test-model"
    assert serving.served_artifact("b", "math") == route_b
    assert serving.serve_artifact(a, scenario="math") is not None
    assert first.backend.generate() == (a.local_path, a.local_path)


def test_checkpoint_slots_release_only_the_bindings_that_no_retained_route_uses(tmp_path, monkeypatch):
    engines = [SlotEngines(f"http://slot-{name}.example") for name in "abc"]
    serving = opened(checkpoint_slots(*engines))
    a, b, c = checkpoints(tmp_path)
    d = published_snapshot(saved_full_weights(tmp_path / "d"), "d")
    serving.activate_checkpoint(a)
    route_a = serving.served_artifact("a", "math")
    route_b = serving.serve_artifact(b, scenario="math")
    assert route_a is not None and route_b is not None and serving.serve_artifact(c, scenario="math") is not None
    # A route uses a binding through its URL and model name, whatever artifact it names. b waits for its stop.
    retained = [ServedArtifact("alias", route_a.url, route_a.model_name), route_b]

    def refuse_pause():
        raise RuntimeError("the engines did not pause")

    # No route uses c, which a work loaded and did not publish. A release that fails keeps c in its slot.
    with monkeypatch.context() as patched:
        patched.setattr(engines[2].backend, "pause", refuse_pause)
        with pytest.raises(RuntimeError, match="did not pause"):
            serving.release_unreferenced_artifacts(retained)
    with pytest.raises(ReefError, match="has no slot"):
        serving.serve_artifact(d, scenario="math")
    calls = [len(slot.backend.events) for slot in engines]
    serving.release_unreferenced_artifacts(retained)
    assert [len(slot.backend.events) for slot in engines] == [calls[0], calls[1], calls[2] + 1]
    assert engines[2].backend.events[-1] == "pause"
    assert serving.served_artifact("a", "math") == route_a and serving.served_artifact("b", "math") == route_b
    assert serving.served_model_name("math") == route_a.model_name
    # The slot that c left serves another checkpoint.
    route_d = serving.serve_artifact(d, scenario="math")
    assert route_d is not None and route_d.url == "http://slot-c.example"
    # Without a route, every binding goes, the active one too.
    serving.release_unreferenced_artifacts([])
    assert all(serving.served_artifact(name, "math") is None for name in "abd")
    assert serving.served_model_name("math") == "test-model"
    assert all(slot.backend.events[-1] == "pause" for slot in engines)


def test_a_stop_whose_pause_fails_keeps_the_checkpoint_in_its_slot(slot_engines, tmp_path, monkeypatch):
    first, second = slot_engines
    serving = opened(checkpoint_slots(first, second))
    a, b, c = checkpoints(tmp_path)
    serving.activate_checkpoint(a)
    route_b = serving.serve_artifact(b, scenario="math")
    assert route_b is not None

    def refuse_pause():
        raise RuntimeError("the engines did not pause")

    with monkeypatch.context() as patched:
        patched.setattr(second.backend, "pause", refuse_pause)
        with pytest.raises(RuntimeError, match="did not pause"):
            serving.stop_serving(route_b)
    # The slot keeps b and restores it under the same route, so c gets no slot until a later stop releases b.
    assert serving.served_artifact("b", "math") is None
    serving.slots[1].recover_engines()
    assert serving.served_artifact("b", "math") == route_b
    with pytest.raises(ReefError, match="has no slot"):
        serving.serve_artifact(c, scenario="math")
    serving.stop_serving(route_b)
    assert serving.served_artifact("b", "math") is None
    route_c = serving.serve_artifact(c, scenario="math")
    assert route_c is not None and route_c.url == "http://slot-b.example"
    assert second.backend.generate() == (c.local_path, c.local_path)


def test_a_load_that_fails_in_a_free_slot_binds_nothing_and_leaves_the_slot_free(slot_engines, tmp_path):
    first, second = slot_engines
    serving = opened(checkpoint_slots(first, second))
    a, b, c = checkpoints(tmp_path)
    serving.activate_checkpoint(a)
    route_a = serving.served_artifact("a", "math")
    assert route_a is not None
    second.backend.fail_load = True
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.serve_artifact(b, scenario="math")
    with pytest.raises(RuntimeError, match="one replica changed"):
        serving.activate_checkpoint(b)
    # Neither failure bound b or selected it: a stays active under its route, and the other slot stays paused.
    assert serving.served_artifact("b", "math") is None and serving.served_artifact("a", "math") == route_a
    assert serving.served_model_name("math") == route_a.model_name
    with pytest.raises(RuntimeError, match="generation is paused"):
        second.backend.generate()
    # The slot is still free, so another checkpoint loads into it and then holds it.
    second.backend.fail_load = False
    route_c = serving.serve_artifact(c, scenario="math")
    assert route_c is not None and route_c.url == "http://slot-b.example"
    assert second.backend.generate() == (c.local_path, c.local_path)
    with pytest.raises(ReefError, match="has no slot"):
        serving.serve_artifact(b, scenario="math")


def test_checkpoint_slots_open_after_a_slot_did_not_open_and_shut_every_slot_down(slot_engines, tmp_path, caplog):
    first, second = slot_engines
    serving = checkpoint_slots(first, second)
    a, b = checkpoints(tmp_path)[:2]
    serving.set_owner_sequence(7)
    assert (first.owner_sequence, second.owner_sequence) == (7, 7)
    second.refuse_open = True
    with pytest.raises(RuntimeError, match=r"slot-b\.example did not open"):
        serving.open()
    # The first slot is open and paused. No checkpoint binds before the second slot is open too.
    with pytest.raises(ReefError, match="not open"):
        serving.activate_checkpoint(a)
    assert (first.opens, second.opens) == (1, 0) and first.backend.loads() == []
    second.refuse_open = False
    serving.open()
    serving.open()
    assert (first.opens, second.opens) == (1, 1)
    serving.activate_checkpoint(a)
    assert first.backend.generate() == (a.local_path, a.local_path)

    first.refuse_close = second.refuse_close = True
    with pytest.raises(RuntimeError, match=r"slot-a\.example did not close"):
        serving.shutdown()
    # Each slot's shutdown ran. The first failure is raised, and the second one is logged.
    assert (first.closes, second.closes) == (1, 1)
    assert "http://slot-b.example did not shut down either" in caplog.text
    assert not serving.inference_admission_status["open"]
    assert not any(slot.inference_admission_status["open"] for slot in serving.slots)
    # The shutdown is final: no checkpoint loads afterwards, and a second shutdown reaches no slot again.
    with pytest.raises(ReefError, match="shut down"):
        serving.serve_artifact(b, scenario="math")
    serving.shutdown()
    assert second.backend.loads() == [] and (first.closes, second.closes) == (1, 1)


def test_a_shutdown_during_an_open_waits_and_closes_the_slot_that_was_opening(slot_engines, monkeypatch):
    first, second = slot_engines
    serving = checkpoint_slots(first, second)
    opening, may_open, closing = threading.Event(), threading.Event(), threading.Event()
    open_second, close_admission = second.open, serving.close_admission

    def open_when_allowed():
        opening.set()
        assert may_open.wait(10)
        return open_second()

    def close_admission_and_report():
        close_admission()
        closing.set()

    monkeypatch.setattr(second, "open", open_when_allowed)
    monkeypatch.setattr(serving, "close_admission", close_admission_and_report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        opened_slots = pool.submit(serving.open)
        assert opening.wait(10)
        shut_down = pool.submit(serving.shutdown)
        # The shutdown closed admission and now waits for the open, which still holds the second slot.
        assert closing.wait(10)
        may_open.set()
        opened_slots.result(10)
        shut_down.result(10)
    # The open ended before any slot shut down, so the shutdown closed both slots and left no engines open.
    assert (first.opens, second.opens) == (1, 1) and (first.closes, second.closes) == (1, 1)
    with pytest.raises(ReefError, match="shut down"):
        serving.open()


def test_checkpoint_slots_that_opened_in_part_shut_down_the_slot_that_opened(slot_engines):
    first, second = slot_engines
    serving = checkpoint_slots(first, second)
    second.refuse_open = True
    with pytest.raises(RuntimeError, match="did not open"):
        serving.open()
    serving.shutdown()
    assert (first.closes, second.closes) == (1, 0) and not serving.inference_admission_status["open"]
    # A runtime that shut down opens nothing, so the slot that never opened takes no engines afterwards.
    second.refuse_open = False
    with pytest.raises(ReefError, match="shut down"):
        serving.open()
    assert (first.opens, second.opens) == (1, 0)


def test_checkpoint_slots_need_two_unbound_runtimes_of_one_model_at_distinct_http_urls(engines, tmp_path):
    def slot(url, model_path="test-model", inference_timeout_s=300.0):
        return FullWeightServingRuntime(
            SlotEngines(url), Handler(), model_path=model_path, inference_timeout_s=inference_timeout_s
        )

    first = slot("http://slot-a.example")
    with pytest.raises(ValueError, match="at least two"):
        CheckpointSlotsRuntime([first])
    with pytest.raises(ValueError, match="distinct base URLs"):
        CheckpointSlotsRuntime([first, slot("http://slot-a.example/")])
    with pytest.raises(ValueError, match="one model with one request timeout"):
        CheckpointSlotsRuntime([first, slot("http://slot-b.example", model_path="other-model")])
    with pytest.raises(ValueError, match="one model with one request timeout"):
        CheckpointSlotsRuntime([first, slot("http://slot-b.example", inference_timeout_s=30.0)])
    with pytest.raises(ValueError, match=r"unix:///run/engines\.sock' can have no route"):
        CheckpointSlotsRuntime([first, slot("unix:///run/engines.sock")])
    with pytest.raises(ValueError, match="can have no route: a full-weight route needs a model path"):
        CheckpointSlotsRuntime([slot("http://slot-a.example", model_path=""), slot("http://slot-b.example", "")])
    bound = opened(slot("http://slot-b.example"))
    bound.activate_checkpoint(checkpoints(tmp_path)[0])
    with pytest.raises(ValueError, match=r"slot-b\.example' already binds a checkpoint"):
        CheckpointSlotsRuntime([first, bound])
    with pytest.raises(TypeError, match="FullWeightServingRuntime"):
        CheckpointSlotsRuntime([first, AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=1)])


def test_only_a_runtime_that_answers_from_any_artifact_has_unlimited_artifact_capacity():
    runtime = StubInferenceRuntime()
    assert runtime.served_artifact_capacity == 1
    runtime.answers_each_artifact = True
    assert runtime.served_artifact_capacity is None


def test_a_new_full_weight_owner_does_not_resume_an_unverified_attached_group(tmp_path):
    class AttachedEngines(RunningEngines):
        def open(self):
            return OpenedEngines(self.backend, attached=True)

    engines = PartialFullWeightEngines()
    previous = published_snapshot(saved_full_weights(tmp_path / "previous"), "previous")
    candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
    first = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="test-model"))
    first.activate_checkpoint(previous)
    served = first.served_artifact("previous", "math")
    assert served is not None
    old_ids = engines.runtime_load_ids()
    engines.fail_load = True
    with pytest.raises(RuntimeError, match="one replica changed"):
        first.activate_checkpoint(candidate)
    assert engines.runtime_load_ids() == old_ids
    first.shutdown()

    restarted = opened(FullWeightServingRuntime(AttachedEngines(engines), Handler(), model_path="test-model"))
    assert not restarted.adopt(served)
    with pytest.raises(RuntimeError, match="generation is paused"):
        engines.generate()
    assert restarted.served_artifact("previous", "math") is None
    engines.fail_load = False
    restarted.activate_checkpoint(previous)
    assert engines.generate() == (previous.local_path, previous.local_path)
    assert restarted.served_artifact("previous", "math") is not None


def test_each_full_weight_load_first_replaces_the_dead_engines_while_they_are_paused(engines, tmp_path):
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    engines.dead = 1
    serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "w1"), "c1"))
    order = [event if isinstance(event, str) else event[0] for event in engines.events[2:]]
    assert order == ["pause", "recover", "load", "resume"]


def test_replaced_engines_reload_each_scenarios_adapter_without_a_second_restart(engines, tmp_path):
    serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=3))
    math = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    engines.dead = 1
    code = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    loads = [(adapter, version) for _, _, adapter, version, _ in engines.loads()]
    # Recovery replaces every engine or none, so the runtime loads math's current adapter again before code's new one.
    assert loads == [
        (adapter_name("math", math), math),
        (adapter_name("math", math), math),
        (adapter_name("code", code), code),
    ]
    assert "abort" not in engines.events
    assert serving.adapter_residency_status()["resident"] == 2
    assert serving.serving_adapter_runtime_load_id("math") == math


def test_a_restart_that_cannot_load_one_scenarios_adapter_still_loads_the_others(engines, tmp_path):
    class FailingMath(Engines):
        def load_weight_files(self, files, *, adapter_name, runtime_load_id):
            if parse_adapter_name(adapter_name)[0] == "math" and "abort" in self.events:
                raise RuntimeError("the engine refused the adapter")
            super().load_weight_files(files, adapter_name=adapter_name, runtime_load_id=runtime_load_id)

    engines = FailingMath()
    serving = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=3))
    serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
    code = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "c1"), "c1"), scenario="code")
    with pytest.raises(AdapterResidencyError, match=r"scenarios \['math'\]"):
        serving.restart_engines()
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.serving_adapter_runtime_load_id("code") == code


def test_a_load_without_dead_engines_restarts_nothing(serving, engines, tmp_path):
    serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "m1"), "m1"), scenario="math")
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

    def __init__(self, engines: Engines, *, lost: int) -> None:
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


def test_health_watcher_restores_an_unfinished_full_load_even_when_the_service_is_healthy(tmp_path):
    engines = PartialFullWeightEngines()
    service = WatchedService(engines, lost=0)
    serving = opened(FullWeightServingRuntime(ServiceEngines(engines, service), Handler(), health_interval_s=0.01))
    try:
        previous = published_snapshot(saved_full_weights(tmp_path / "previous"), "previous")
        candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
        serving.activate_checkpoint(previous)
        engines.resumed.clear()
        engines.fail_load = True
        with pytest.raises(RuntimeError, match="one replica changed"):
            serving.activate_checkpoint(candidate)
        assert not engines.resumed.is_set()
        engines.fail_load = False
        assert engines.resumed.wait(timeout=5)
        assert engines.generate() == (previous.local_path, previous.local_path)
    finally:
        serving.shutdown()


class CountedService(WatchedService):
    """A watched service that counts its health checks, so that a test can wait for the watcher without a sleep.

    ``checks`` is the number of health checks that have started. It starts at zero and never decreases.
    ``checked`` is the condition that protects ``checks`` and notifies its waiters at each check.
    The count lets a test wait until the watcher has finished a whole cycle: one check and the recovery after it.
    ``restart`` and the failing check keep the ``WatchedService`` behavior.
    """

    def __init__(self, engines: Engines, *, lost: int) -> None:
        super().__init__(engines, lost=lost)
        self.checks = 0
        self.checked = threading.Condition()

    def check_health(self) -> None:
        with self.checked:
            self.checks += 1
            self.checked.notify_all()
        super().check_health()

    def wait_for_a_whole_check(self) -> None:
        """Return once the watcher has finished a check that began after this call, and the recovery that follows it.

        The watcher begins a check only after it finished the one before, so two later checks show it.
        """
        with self.checked:
            target = self.checks + 2
            assert self.checked.wait_for(lambda: self.checks >= target, timeout=5)


@pytest.mark.parametrize("state", ["new_engines", "failed_first_activation", "released_checkpoint", "lost_engine"])
def test_the_health_watcher_restores_nothing_before_the_first_verified_full_weight_activation(tmp_path, caplog, state):
    engines = PartialFullWeightEngines()
    service = CountedService(engines, lost=1)
    watched = ServiceEngines(engines, service)
    serving = opened(FullWeightServingRuntime(watched, Handler(), model_path="test-model", health_interval_s=0.01))
    candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
    try:
        if state == "failed_first_activation":
            engines.fail_load = True
            with pytest.raises(RuntimeError, match="one replica changed"):
                serving.activate_checkpoint(candidate)
            engines.fail_load = False
        elif state == "released_checkpoint":
            # A released checkpoint is no recovery target: the runtime is as before its first activation.
            serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "released"), "released"))
            serving.release_checkpoint()
        calls = len(engines.events)
        if state == "lost_engine":
            service.restart()
        service.wait_for_a_whole_check()
        # No verified artifact exists. Healthy engines get no call; a lost engine is replaced, and nothing loads.
        assert engines.events[calls:] == (["pause", ("recover", 1)] if state == "lost_engine" else [])
        # The log reports the failed health check only, and no failed recovery.
        logged = [record.levelno for record in caplog.records if record.name == "reef.inference.serving"]
        assert logged == ([logging.WARNING] if state == "lost_engine" else [])
        # Admission stays closed until a later explicit activation, which serves as usual.
        assert serving.recovery_pending and serving.served_artifact("candidate", "math") is None
        with pytest.raises(RuntimeError, match="generation is paused"):
            engines.generate()
        serving.activate_checkpoint(candidate)
        assert engines.generate() == (candidate.local_path, candidate.local_path)
        assert serving.served_artifact("candidate", "math") is not None
    finally:
        serving.shutdown()


@pytest.mark.parametrize("shutdown_during_load", [False, True])
def test_waiting_health_tick_does_not_run_after_activation_or_shutdown(tmp_path, shutdown_during_load):
    class SlowLoadEngines(Engines):
        """A load of ``blocked_path`` waits for ``release_load`` after setting ``entered_load``.

        The shutdown case fails that load after mutation; other loads and operations record their calls normally.
        """

        def __init__(self, blocked_path):
            super().__init__()
            self.blocked_path = blocked_path
            self.entered_load = threading.Event()
            self.release_load = threading.Event()

        def load_weight_files(self, files, *, adapter_name, runtime_load_id):
            if files.path == self.blocked_path:
                self.entered_load.set()
                if not self.release_load.wait(5):
                    raise TimeoutError("the test did not release the load")
            super().load_weight_files(files, adapter_name=adapter_name, runtime_load_id=runtime_load_id)
            if files.path == self.blocked_path and shutdown_during_load:
                raise RuntimeError("load failed during shutdown")

    class ObservedService(Service):
        """Healthy checks wait for the blocked load, then signal ``during_load`` and ``next_tick`` in order."""

        def __init__(self, engines):
            super().__init__()
            self.engines = engines
            self.during_load = threading.Event()
            self.next_tick = threading.Event()

        def check_health(self):
            self.engines.entered_load.wait(5)
            if self.engines.entered_load.is_set():
                if self.during_load.is_set():
                    self.next_tick.set()
                else:
                    self.during_load.set()

    engines = SlowLoadEngines(tmp_path / "candidate")
    service = ObservedService(engines)
    serving = opened(FullWeightServingRuntime(ServiceEngines(engines, service), Handler(), health_interval_s=0.01))
    try:
        serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "previous"), "previous"))
        engines.events.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            candidate = published_snapshot(saved_full_weights(tmp_path / "candidate"), "candidate")
            loaded = executor.submit(serving.activate_checkpoint, candidate)
            stopped = None
            try:
                assert engines.entered_load.wait(2)
                assert service.during_load.wait(2)
                if shutdown_during_load:
                    stopped = executor.submit(serving.shutdown)
                    assert serving._stopping.wait(2)
            finally:
                engines.release_load.set()
            if shutdown_during_load:
                with pytest.raises(RuntimeError, match="load failed during shutdown"):
                    loaded.result(timeout=5)
                stopped.result(timeout=5)
            else:
                loaded.result(timeout=5)
                assert service.next_tick.wait(2)
        assert engines.events.count("pause") == 1
        assert len(engines.loads()) == 1
    finally:
        serving.shutdown()


def test_a_failed_health_check_restarts_the_engines_and_loads_each_scenarios_adapter_again(tmp_path):
    engines = RestartingEngines()
    service = WatchedService(engines, lost=2)
    serving = opened(
        AdapterServingRuntime(ServiceEngines(engines, service), Handler(), adapter_capacity=2, health_interval_s=0.01)
    )
    try:
        math = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "math"), "m"), scenario="math")
        code = serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "code"), "c"), scenario="code")
        loaded = len(engines.loads())
        service.restart()
        assert engines.recovered.wait(timeout=10)
        reloaded = [(adapter, version) for _, _, adapter, version, _ in engines.loads()[loaded:]]
        assert reloaded == [(adapter_name("math", math), math), (adapter_name("code", code), code)]
        assert engines.events[-1] == "resume" and "abort" not in engines.events
        assert serving.serving_adapter_runtime_load_id("math") == math
    finally:
        serving.shutdown()
    assert service.closed == 1


def test_a_failed_health_check_loads_the_served_full_weights_again_under_their_version(tmp_path):
    engines = RestartingEngines()
    service = WatchedService(engines, lost=1)
    serving = opened(FullWeightServingRuntime(ServiceEngines(engines, service), Handler(), health_interval_s=0.01))
    try:
        weights = saved_full_weights(tmp_path / "step-1")
        version = serving.activate_checkpoint(published_snapshot(weights, "w1"))
        calls = len(engines.events)
        service.restart()
        assert engines.recovered.wait(timeout=10)
        assert engines.events[calls:] == ["pause", ("recover", 1), ("load", "full", None, version, weights), "resume"]
        assert engines.runtime_load_ids() == [version]
    finally:
        serving.shutdown()


def test_recovery_that_replaces_no_engine_loads_nothing(engines, tmp_path):
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler()))
    serving.activate_checkpoint(published_snapshot(saved_full_weights(tmp_path / "step-1"), "w1"))
    calls = len(engines.events)
    serving.recover_engines()
    assert engines.events[calls:] == ["pause", ("recover", 0), "resume"]


def test_new_full_weight_engines_stay_paused_until_the_first_activation(engines):
    serving = opened(FullWeightServingRuntime(RunningEngines(engines), Handler(), model_path="Qwen/Qwen3-8B"))
    # The base model is no published artifact, so recovery has nothing to restore and resumes nothing.
    engines.dead = 1
    with pytest.raises(ReefError, match="no verified full-weight artifact"):
        serving.recover_engines()
    assert "resume" not in engines.events and serving.recovery_pending
    assert serving.served_artifact("w1", "math") is None


def test_a_new_adapter_runtime_takes_over_no_adapter_and_stops_a_replaced_one(engines, tmp_path):
    first = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2))
    artifact = published_snapshot(saved_adapter(tmp_path / "math"), "artifact-m")
    first.activate_checkpoint(artifact, scenario="math")
    served = first.served_artifact("artifact-m", "math")
    assert served is not None

    # The engines' load IDs do not prove which adapters are resident, so a new runtime loads the artifact again.
    restarted = opened(AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=2))
    assert not restarted.adopt(served)
    loads = len(engines.loads())
    version = restarted.activate_checkpoint(artifact, scenario="math")
    assert len(engines.loads()) == loads + 1
    current = restarted.served_artifact("artifact-m", "math")
    assert current is not None and current.model_name == adapter_name("math", version) != served.model_name

    # A newer snapshot replaces it; when its stop is due, the engines unload the old adapter.
    newer = published_snapshot(saved_adapter(tmp_path / "newer"), "n")
    restarted.activate_checkpoint(newer, scenario="math")
    restarted.stop_serving(current)
    assert ("unload", current.model_name) in engines.events
    # The current adapter never stops.
    assert restarted.served_artifact("wrong-artifact", "math") is None
    restarted.stop_serving(restarted.served_artifact(newer.artifact_id, "math"))
    assert ("unload", restarted.served_model_name("math")) not in engines.events


@pytest.mark.parametrize(
    "build",
    [
        lambda engines: AdapterServingRuntime(engines, Handler(), adapter_capacity=1),
        lambda engines: FullWeightServingRuntime(engines, Handler()),
    ],
    ids=["adapter", "full"],
)
def test_saved_weight_serving_refuses_engines_that_declare_no_request_protection(build):
    class PlainEngines(ServingEngines):
        """Engines of an integration that leaves ``request_pinning`` at its default."""

        base_url = "http://component.example"

        def open(self):
            raise AssertionError("unused")

        def check_health(self):
            raise AssertionError("unused")

    with pytest.raises(RuntimeContractError, match="request_pinning"):
        build(PlainEngines())


def test_a_runtime_that_control_did_not_open_changes_no_engine(engines, tmp_path):
    # A process that runs no control builds the same runtime to send requests, and must leave the engines alone.
    serving = AdapterServingRuntime(RunningEngines(engines), Handler(), adapter_capacity=1)
    assert engines.events == []
    with pytest.raises(ReefError, match="not open"):
        serving.activate_checkpoint(published_snapshot(saved_adapter(tmp_path / "math"), "m"), scenario="math")
    serving.shutdown()
    assert engines.events == []
