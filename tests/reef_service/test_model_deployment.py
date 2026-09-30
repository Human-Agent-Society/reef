"""Backend-neutral startup, connection, readiness and failure ownership."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from reef.runtime.deployment import (
    CoordinatorConfig,
    DeploymentResources,
    InferenceConnection,
    InferenceService,
    ModelDeploymentPlan,
    TrainingService,
)
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.interfaces import InferenceBackend, TrainingBackend
from reef.service import training_driver
from reef.service.training_driver import ModelDeployment
from reef.train.deployment import TrainingDeploymentPlan


class Resources(DeploymentResources):
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


class Inference(InferenceService, InferenceBackend):
    connection_protocol = "test-weights-v1"

    def __init__(self, resources):
        self.resources = resources
        self.engine = Engine()
        self.executor = UniProcExecutor.from_workers([self.engine])
        self.probes = 0
        self.version = "engine:0"

    def start(self, resources):
        assert resources is self.resources
        resources.event("inference-start")
        return InferenceConnection(self.connection_protocol, self.executor)

    def prepare_weight_transfer(self, connection):
        self.resources.event("transfer-prepare")

    def backend(self, connection):
        self.resources.event("inference-operations")
        return self

    def inference_url(self):
        return "http://inference"

    def runtime_load_ids(self):
        return [self.version]

    def initialize_version(self, runtime_load_id):
        self.version = runtime_load_id

    def pause(self):
        self.resources.event("inference-pause")

    def resume(self):
        self.resources.event("inference-resume")

    def abort(self):
        self.resources.event("inference-abort")

    def check_health(self):
        self.probes += 1
        self.resources.event(f"inference-health-{self.probes}")

    def close(self):
        self.resources.event("inference-close")
        self.executor.shutdown()

    def poll(self):
        self.resources.event("inference-poll")

    def recover(self):
        raise AssertionError("unexpected recover in this fixture")

    def offload(self, tags):
        raise AssertionError("unexpected offload in this fixture")

    def onload_weights(self):
        raise AssertionError("unexpected onload_weights in this fixture")

    def onload_kv(self):
        raise AssertionError("unexpected onload_kv in this fixture")

    def unload_adapter(self, name):
        raise AssertionError("unexpected unload_adapter in this fixture")


class Training(TrainingService):
    weight_transfer_protocol = "test-weights-v1"

    def __init__(self, resources):
        self.resources = resources
        self.inference = None

    def start(self, resources):
        assert resources is self.resources
        resources.event("training-start")

    def attach_weight_transport(self, session):
        self.inference = session
        self.resources.event("training-attach")
        session.receiver.rpc(0, "update", args=(7,))

    def backend(self):
        self.resources.event("training-operations")
        return self

    def check_health(self):
        self.resources.event("training-health")

    def close(self):
        self.resources.event("training-close")

    def poll(self):
        self.resources.event("training-poll")


class OtherTraining(Training):
    """Second implementation consumes the same connection through another RPC API."""

    def attach_weight_transport(self, session):
        self.inference = session
        self.resources.event("training-attach")
        session.receiver.collective_rpc("update", args=(9,))


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
    assert plan.training.inference.receiver is plan.inference.executor
    owner.close()
    owner.close()
    assert events == [
        "allocate",
        "inference-start",
        "inference-health-1",
        "transfer-prepare",
        "training-start",
        "training-attach",
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
        ("transfer-prepare", ["inference-close", "release"]),
        ("training-start", ["training-close", "inference-close", "release"]),
        ("training-attach", ["training-close", "inference-close", "release"]),
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
    plan.training.weight_transfer_protocol = "another-protocol"
    with pytest.raises(ValueError, match="incompatible inference control protocol"):
        ModelDeployment(plan).start()
    assert events == []


def test_combined_compatibility_is_explicit_not_a_startup_fallback():
    plan, events = plan_for()
    plan.training.weight_transfer_protocol = None
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
        def create_training_plan(self, received, *, loss_family):
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
    monkeypatch.setattr(driver, "create_training_plan", create)
    monkeypatch.setattr(slime_driver, "assemble_model_plan", lambda config, training: training)
    monkeypatch.setattr(slime_driver, "run_deployment", run)
    assert slime_driver.main(["serve", "--ready-file", str(tmp_path / "ready"), "--lr=1e-6"]) == 0
    assert captured["arguments"] == ["--lr=1e-6"]


def test_rebuild_assigns_a_new_weight_transfer_session():
    sessions = []
    for _ in range(2):
        plan, _ = plan_for()
        deployment = ModelDeployment(plan)
        deployment.start()
        sessions.append(deployment.weight_transfer_session.session_id)
        deployment.close()
    assert sessions[0] != sessions[1]


def test_reef_coordinator_closes_backend_operations_before_local_cleanup():
    from dataclasses import replace

    from reef.runtime.interfaces import TrainingContext, TrainingCoordinationConfig

    plan, events = plan_for()
    plan.training.config = TrainingCoordinationConfig(save_hf_template=None)
    plan.training.context = TrainingContext()

    # The native operations may recreate workers inside the coordinator.
    # Its shutdown closes that copy; the service still handles partial starts.
    class Operations(TrainingBackend):
        config = plan.training.config
        context = plan.training.context

        def start(self):
            pass

        def check_health(self):
            pass

        def initialize_version(self, runtime_load_id):
            self.context.runtime_load_id = runtime_load_id

        def close(self):
            events.append("operations-close")

        def prepare_training_step(self, batch, objective, algorithm_state, scheduling):
            raise AssertionError("unexpected prepare_training_step in this fixture")

        def prepare(self, payload, *, job_id, scenario_step, prior_marker):
            raise AssertionError("unexpected prepare in this fixture")

        def prepare_weights(self, runtime_load_id, *, force_full):
            raise AssertionError("unexpected prepare_weights in this fixture")

        def send_weights(self, runtime_load_id, *, force_full):
            raise AssertionError("unexpected send_weights in this fixture")

        def activate_scenario(self, scenario):
            raise AssertionError("unexpected activate_scenario in this fixture")

        def send_adapter(self, scenario, name):
            raise AssertionError("unexpected send_adapter in this fixture")

    plan.training.backend = lambda: Operations()
    plan = replace(plan, coordinator=CoordinatorConfig(backend="uni"))
    deployment = ModelDeployment(plan)
    deployment.start()
    assert events.index("transfer-prepare") < events.index("training-start")
    deployment.poll()
    deployment.close()
    assert events.count("training-close") == 1
    assert events.index("operations-close") < events.index("training-close")
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_coordinator_construction_failure_releases_started_backends(monkeypatch):
    from dataclasses import replace

    from reef.runtime.executor import Executor

    plan, events = plan_for()
    plan = replace(plan, coordinator=CoordinatorConfig(backend="uni"))

    def fail(config):
        assert config.workers[0].worker_cls.__module__ == "reef.runtime.scheduler"
        assert events[-2:] == ["training-operations", "inference-operations"]
        raise RuntimeError("coordinator recovery failed")

    monkeypatch.setattr(Executor, "create", fail)
    with pytest.raises(RuntimeError, match="coordinator recovery failed"):
        ModelDeployment(plan).start()
    assert events[-3:] == ["training-close", "inference-close", "release"]


def test_reef_assembles_independently_selected_backend_definitions(monkeypatch):
    existing, events = plan_for()
    config = {"reef": {"training_backend": "another-trainer", "inference_backend": "another-receiver"}}
    native_options = {"model": "tiny", "parallel": 2}

    class Definition:
        def create_training_plan(self, received, *, loss_family):
            assert received is config
            assert loss_family == "custom-loss"
            return TrainingDeploymentPlan(existing.resources, existing.training, native_options)

    def training_definition(name):
        assert name == "another-trainer"
        return Definition()

    def inference_definition(name, options):
        assert name == "another-receiver"
        assert options is native_options
        return existing.inference

    monkeypatch.setattr(training_driver, "training_deployment_for", training_definition)
    monkeypatch.setattr(training_driver, "inference_service_for", inference_definition)
    plan = training_driver.ConfiguredModelPlanSource(config, "custom-loss").create()
    assert events == []
    assert plan.training is existing.training
    assert plan.inference is existing.inference
    plan.health.poll()
    assert events == ["inference-poll", "training-poll"]


def test_assembler_rejects_incompatible_backend_pair_before_allocation(monkeypatch):
    existing, events = plan_for()
    existing.training.weight_transfer_protocol = "unsupported-transport"
    monkeypatch.setattr(training_driver, "inference_service_for", lambda name, config: existing.inference)
    with pytest.raises(ValueError, match="incompatible inference control protocol"):
        training_driver.assemble_model_plan({}, TrainingDeploymentPlan(existing.resources, existing.training))
    assert events == []


def test_pending_coordinator_probe_does_not_trigger_restart_or_queue_more_work():
    class Probe:
        result_value = None

        def result(self, timeout):
            if self.result_value is None:
                raise TimeoutError("still training")
            return self.result_value

    class Coordinator:
        failure = None

        def __init__(self):
            self.probe = Probe()
            self.submitted = 0

        def rpc(self, rank, method, *, non_block):
            self.submitted += 1
            return self.probe

    plan, _ = plan_for()
    deployment = ModelDeployment(plan)
    coordinator = Coordinator()
    deployment._coordinator = coordinator
    for _ in range(5):
        deployment.poll()
    assert coordinator.submitted == 1
    coordinator.probe.result_value = {"ok": False, "recoverable": True}
    deployment.poll()
    assert coordinator.submitted == 1
    coordinator.probe.result_value = {"ok": False}
    with pytest.raises(RuntimeError, match="health check"):
        deployment.poll()
    assert coordinator.submitted == 2


def test_explicitly_disabled_supervision_stays_disabled_with_an_owned_coordinator(tmp_path, monkeypatch):
    from dataclasses import replace

    plan, _ = plan_for()
    plan = replace(plan, coordinator=CoordinatorConfig(backend="uni"), health=None)
    # This test isolates the supervisor selection from coordinator execution,
    # which the separate real-Ray deployment contract exercises.
    monkeypatch.setattr(ModelDeployment, "_start_coordinator", lambda *args: None)
    monkeypatch.setattr(
        training_driver, "supervise_deployment", lambda *args, **kwargs: pytest.fail("supervision enabled")
    )
    ready = tmp_path / "ready"

    class Stopping:
        def set(self):
            pass

        def is_set(self):
            return False

        def wait(self):
            assert ready.exists()

    monkeypatch.setattr(training_driver, "threading", SimpleNamespace(Event=Stopping))
    assert training_driver.run_deployment(plan, ready, source=SimpleNamespace()) == 0
    assert not ready.exists()


def test_coordinator_is_placed_on_the_trainers_node(monkeypatch):
    from dataclasses import replace

    from reef.runtime.executor import Executor

    class PlacedResources(Resources):
        @property
        def training_node_id(self):
            return "node-1"

    configs = []

    def capture(config):
        configs.append(config)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(Executor, "create", capture)
    placed = PlacedResources([], ())
    plan = ModelDeploymentPlan(
        placed, Inference(placed), Training(placed), coordinator=CoordinatorConfig(backend="uni")
    )
    with pytest.raises(RuntimeError, match="stop after capture"):
        ModelDeployment(plan).start()
    assert configs[0].node_id == "node-1"

    # An allocation that names no trainer node leaves the coordinator unplaced.
    plan, _ = plan_for()
    plan = replace(plan, coordinator=CoordinatorConfig(backend="uni"))
    with pytest.raises(RuntimeError, match="stop after capture"):
        ModelDeployment(plan).start()
    assert configs[1].node_id is None
