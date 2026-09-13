"""Backend-neutral startup, connection, readiness and failure ownership."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from reef.runtime.deployment import InferenceConnection, ModelDeploymentPlan
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.service import training_driver
from reef.service.training_driver import ModelDeployment


class Resources:
    def __init__(self, events, failures):
        self.events = events
        self.failures = failures

    def event(self, name):
        self.events.append(name)
        if name in self.failures:
            raise RuntimeError(name)

    def start(self):
        self.event("allocate")

    def close(self):
        self.event("release")


class Engine:
    def __init__(self):
        self.weight = 0

    def update(self, weight):
        self.weight = weight


class Inference:
    connection_protocol = "test-weights-v1"

    def __init__(self, resources):
        self.resources = resources
        self.engine = Engine()
        self.executor = UniProcExecutor.from_workers([self.engine])
        self.probes = 0

    def start(self, resources):
        assert resources is self.resources
        resources.event("inference-start")
        return InferenceConnection(self.connection_protocol, self.executor)

    def check_health(self):
        self.probes += 1
        self.resources.event(f"inference-health-{self.probes}")

    def close(self):
        self.resources.event("inference-close")
        self.executor.shutdown()


class Training:
    inference_protocol = "test-weights-v1"

    def __init__(self, resources):
        self.resources = resources
        self.inference = None

    def start(self, resources, inference):
        assert resources is self.resources
        self.inference = inference
        resources.event("training-start")
        if inference is not None:
            inference.control.rpc(0, "update", args=(7,))

    def check_health(self):
        self.resources.event("training-health")

    def close(self):
        self.resources.event("training-close")


class OtherTraining(Training):
    """Second implementation consumes the same connection through another RPC API."""

    def start(self, resources, inference):
        assert inference is not None
        self.inference = inference
        resources.event("training-start")
        inference.control.collective_rpc("update", args=(9,))


def plan_for(*failures, training_type=Training):
    events = []
    resources = Resources(events, failures)
    return ModelDeploymentPlan(resources, Inference(resources), training_type(resources)), events


@pytest.mark.parametrize("training_type,weight", [(Training, 7), (OtherTraining, 9)])
def test_owner_composes_backends_and_shuts_down_in_dependency_order(training_type, weight):
    plan, events = plan_for(training_type=training_type)
    owner = ModelDeployment(plan)
    owner.start()
    assert plan.inference.engine.weight == weight
    assert plan.training.inference.control is plan.inference.executor
    owner.close()
    owner.close()
    assert events == [
        "allocate",
        "inference-start",
        "inference-health-1",
        "training-start",
        "training-health",
        "inference-health-2",
        "training-close",
        "inference-close",
        "release",
    ]
    with pytest.raises(RuntimeError, match="once"):
        owner.start()


@pytest.mark.parametrize(
    "failure,cleanup",
    [
        ("allocate", ["release"]),
        ("inference-start", ["inference-close", "release"]),
        ("inference-health-1", ["inference-close", "release"]),
        ("training-start", ["training-close", "inference-close", "release"]),
        ("training-health", ["training-close", "inference-close", "release"]),
        ("inference-health-2", ["training-close", "inference-close", "release"]),
    ],
)
def test_partial_start_closes_only_attempted_components(failure, cleanup):
    plan, events = plan_for(failure)
    owner = ModelDeployment(plan)
    with pytest.raises(RuntimeError, match=failure):
        owner.start()
    owner.close()
    assert events[events.index(failure) + 1 :] == cleanup


@pytest.mark.parametrize("failure", ["training-close", "inference-close", "release"])
def test_cleanup_failure_does_not_skip_other_components(failure):
    plan, events = plan_for(failure)
    owner = ModelDeployment(plan)
    owner.start()
    with pytest.raises(RuntimeError, match=failure):
        owner.close()
    owner.close()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_startup_error_survives_cleanup_error():
    plan, events = plan_for("training-start", "inference-close", "release")
    with pytest.raises(RuntimeError, match="training-start"):
        ModelDeployment(plan).start()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_incompatible_control_protocol_fails_before_allocation():
    plan, events = plan_for()
    plan.training.inference_protocol = "another-protocol"
    with pytest.raises(ValueError, match="incompatible inference control protocol"):
        ModelDeployment(plan).start()
    assert events == []


def test_combined_compatibility_is_explicit_not_a_startup_fallback():
    plan, events = plan_for()
    plan.training.inference_protocol = None
    owner = ModelDeployment(ModelDeploymentPlan(plan.resources, None, plan.training))
    owner.start()
    owner.close()
    assert events == ["allocate", "training-start", "training-health", "training-close", "release"]


def test_driver_publishes_readiness_only_after_both_components_and_removes_it(tmp_path, monkeypatch):
    plan, events = plan_for()
    ready = tmp_path / "ready"
    ready.write_text("stale")

    class Stopping:
        def set(self):
            pass

        def is_set(self):
            return False

        def wait(self):
            assert ready.read_text().strip() == training_driver.READY_MARKER
            assert events[-1] == "inference-health-2"

    monkeypatch.setattr(training_driver, "threading", SimpleNamespace(Event=Stopping))
    assert training_driver.run_deployment(plan, ready) == 0
    assert not ready.exists()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_driver_selects_backend_plan_and_clears_stale_readiness_on_preflight_error(tmp_path, monkeypatch):
    config = {"reef": {"training_backend": "test", "recipe": "recipes.sao.recipe:SAORecipe"}}
    ready = tmp_path / "ready"
    ready.write_text("stale")
    monkeypatch.setenv("REEF_CONFIG", "unused")
    monkeypatch.setattr(training_driver, "load_config", lambda path: config)

    class Definition:
        def create_model_plan(self, received, *, loss_family):
            assert loss_family == "sao"
            assert received is config
            assert not ready.exists()
            raise ValueError("bad combination")

    monkeypatch.setattr(
        training_driver, "training_deployment_for", lambda name: Definition() if name == "test" else None
    )
    with pytest.raises(ValueError, match="bad combination"):
        training_driver.main(["--ready-file", str(ready)])
    assert not ready.exists()


def test_shared_driver_imports_without_optional_model_frameworks():
    script = """
import sys
class NoFrameworks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'slime', 'ray', 'sglang', 'megatron'}:
            raise RuntimeError('unexpected framework import: ' + fullname)
sys.meta_path.insert(0, NoFrameworks())
import reef.service.training_driver
import reef.runtime.deployment
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_returned_connection_must_match_the_declared_protocol(monkeypatch):
    plan, events = plan_for()

    def start(resources):
        resources.event("inference-start")
        return InferenceConnection("wrong-protocol", plan.inference.executor)

    monkeypatch.setattr(plan.inference, "start", start)
    with pytest.raises(ValueError, match="incompatible protocol"):
        ModelDeployment(plan).start()
    assert events == ["allocate", "inference-start", "inference-close", "release"]


def test_readiness_write_error_survives_cleanup_failure(tmp_path, monkeypatch):
    plan, events = plan_for("training-close")
    ready = tmp_path / "ready"

    def write(*args):
        raise OSError("readiness write failed")

    monkeypatch.setattr(training_driver, "_write_ready_file", write)
    with pytest.raises(OSError, match="readiness write failed"):
        training_driver.run_deployment(plan, ready)
    assert not ready.exists()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_stop_requested_during_startup_does_not_publish_readiness(tmp_path, monkeypatch):
    import threading

    plan, events = plan_for()
    stopping = threading.Event()
    stopping.set()
    monkeypatch.setattr(training_driver, "threading", SimpleNamespace(Event=lambda: stopping))
    ready = tmp_path / "ready"

    def unexpected(*args):
        pytest.fail("a stopped deployment must not advertise readiness")

    monkeypatch.setattr(training_driver, "_write_ready_file", unexpected)
    assert training_driver.run_deployment(plan, ready) == 0
    assert not ready.exists()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_legacy_cli_delegates_lifecycle_and_preserves_its_native_arguments(tmp_path, monkeypatch):
    from reef.service import slime_driver
    from reef.train.slime_backend import driver

    config = {"reef": {"recipe": "recipes.sao.recipe:SAORecipe"}}
    plan, _ = plan_for()
    captured = {}

    def create(received, arguments, *, loss_family):
        assert received is config
        assert loss_family == "sao"
        captured["arguments"] = list(arguments)
        return plan

    def run(received, ready_file, *, marker):
        assert received is plan
        assert ready_file == tmp_path / "ready"
        assert marker == "reef-slime-bridge-ready"
        return 0

    monkeypatch.setenv("REEF_CONFIG", "unused")
    monkeypatch.setattr(slime_driver, "load_config", lambda path: config)
    monkeypatch.setattr(driver, "create_model_plan", create)
    monkeypatch.setattr(slime_driver, "run_deployment", run)
    assert slime_driver.main(["serve", "--ready-file", str(tmp_path / "ready"), "--lr=1e-6"]) == 0
    assert captured["arguments"] == ["--lr=1e-6"]
