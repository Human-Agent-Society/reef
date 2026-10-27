"""SGLang receiver results and failures crossing the runtime control boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.config import SGLangConfig, SGLangGroupConfig
from reef.inference.sglang.service import INFERENCE_PROTOCOL, SGLangInferenceService
from reef.runtime.deployment import InferenceConnection
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor


class Receiver:
    def __init__(self, versions=("weights:1", "weights:1")):
        self.versions = versions
        self.paused = False
        self.closed = False

    def inference_url(self):
        return "http://inference:8000"

    def get_runtime_load_ids(self):
        return self.versions

    def pause_generation_for_update(self):
        self.paused = True

    def continue_generation_after_update(self):
        self.paused = False

    def shutdown(self):
        self.closed = True


def test_service_adapts_only_compatible_connections_without_owning_workers():
    receiver = Receiver()
    executor = UniProcExecutor.from_workers([receiver])
    service = SGLangInferenceService(SGLangConfig("model", 1, 1, 1))
    operations = service.backend(InferenceConnection(INFERENCE_PROTOCOL, executor))

    assert operations.inference_url() == "http://inference:8000"
    assert operations.runtime_load_ids() == ("weights:1", "weights:1")
    operations.pause()
    assert receiver.paused
    operations.resume()
    assert not receiver.paused
    service.close()
    assert not receiver.closed
    assert operations.runtime_load_ids() == ("weights:1", "weights:1")

    with pytest.raises(ValueError, match="incompatible SGLang"):
        service.backend(InferenceConnection("other-protocol", executor))


@pytest.mark.parametrize("versions", ["weights:1", [None], [1], [""], {"engine": "weights:1"}])
def test_receiver_rejects_malformed_versions_before_publication(versions):
    operations = SGLangInferenceBackend(UniProcExecutor.from_workers([Receiver(versions)]))
    with pytest.raises(RuntimeError, match="invalid runtime load IDs"):
        operations.runtime_load_ids()


def test_receiver_preserves_mixed_versions_for_coordinator_consistency_check():
    operations = SGLangInferenceBackend(UniProcExecutor.from_workers([Receiver(["weights:1", "weights:2"])]))
    assert operations.runtime_load_ids() == ("weights:1", "weights:2")


def test_receiver_preserves_pause_failure_without_attempting_resume():
    class UncertainReceiver(Receiver):
        def pause_generation_for_update(self):
            self.paused = True
            raise TimeoutError("pause acknowledgement lost")

    receiver = UncertainReceiver()
    operations = SGLangInferenceBackend(UniProcExecutor.from_workers([receiver]))
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        operations.pause()
    assert receiver.paused


@pytest.mark.parametrize("replaced", [0, 2])
def test_receiver_recovery_returns_the_refilled_engine_count(replaced):
    receiver = Receiver()
    receiver.recover_updatable_engines = lambda: replaced
    assert SGLangInferenceBackend(UniProcExecutor.from_workers([receiver])).recover() == replaced


@pytest.mark.parametrize("replaced", [-1, True, None, [object()]])
def test_receiver_recovery_rejects_a_malformed_engine_count(replaced):
    receiver = Receiver()
    receiver.recover_updatable_engines = lambda: replaced
    with pytest.raises(RuntimeError, match="invalid replaced-engine count"):
        SGLangInferenceBackend(UniProcExecutor.from_workers([receiver])).recover()


def test_worker_recovery_counts_the_dead_engines_it_refilled():
    from reef.inference.sglang import worker as worker_module

    group = SimpleNamespace(all_engines=[object(), None, None])
    restarts = []

    def restart_dead_engines():
        restarts.append(len(group.all_engines))
        group.all_engines = [engine or object() for engine in group.all_engines]

    worker = worker_module.SGLangWorker.__new__(worker_module.SGLangWorker)
    worker.servers = {
        "actor": SimpleNamespace(update_weights=True, server_groups=[group], recover=restart_dead_engines)
    }
    engines = worker_module._SGLangEngineGroup(worker)

    assert engines.recover() == 2
    assert engines.recover() == 0, "no dead engine: nothing to restart"
    assert restarts == [3]


@pytest.mark.parametrize("fail_second", [False, True])
def test_initial_version_stamping_keeps_receiver_paused_on_partial_failure(monkeypatch, fail_second):
    versions = ["native:0", "native:0"]

    class EngineGroup:
        def collective_rpc(self, method, *, args, timeout):
            assert method == "set_runtime_load_id"
            versions[0] = args[0]
            if fail_second:
                raise TimeoutError("second engine acknowledgement lost")
            versions[1] = args[0]

    receiver = Receiver(versions)
    receiver.updatable_engines = lambda: [object(), object()]
    monkeypatch.setattr(RayExecutor, "from_workers", lambda workers: EngineGroup())
    operations = SGLangInferenceBackend(UniProcExecutor.from_workers([receiver]))
    operations.pause()
    if fail_second:
        with pytest.raises(TimeoutError, match="acknowledgement lost"):
            operations.initialize_version("reef:initial")
        assert operations.runtime_load_ids() == ("reef:initial", "native:0")
    else:
        operations.initialize_version("reef:initial")
        assert operations.runtime_load_ids() == ("reef:initial", "reef:initial")
    assert receiver.paused


@pytest.mark.parametrize("last_result", [{"success": False}, {"success": True}, None])
def test_adapter_eviction_requires_every_engine_acknowledgement(monkeypatch, last_result):
    unloaded = []

    class Engine:
        def __init__(self, result):
            self.result = result

        def unload_lora_adapter(self, *, lora_name):
            unloaded.append(lora_name)
            return self.result

    engines = [Engine({"success": True}), Engine(last_result)]
    receiver = SimpleNamespace(updatable_engines=lambda: engines)

    class EngineGroup:
        def collective_rpc(self, method, *, kwargs, timeout):
            assert method == "unload_lora_adapter"
            return [engine.unload_lora_adapter(**kwargs) for engine in engines]

    monkeypatch.setattr(RayExecutor, "from_workers", lambda workers: EngineGroup())
    operations = SGLangInferenceBackend(UniProcExecutor.from_workers([receiver]))
    if last_result == {"success": False}:
        with pytest.raises(RuntimeError, match="engine kept adapter"):
            operations.unload_adapter("scenario/weights:2")
    else:
        operations.unload_adapter("scenario/weights:2")
    assert unloaded == ["scenario/weights:2", "scenario/weights:2"]


def test_worker_loads_an_adapter_directory_into_every_engine_then_stamps_the_version(monkeypatch):
    from reef.inference.sglang import worker as worker_module

    calls = []

    class Remote:
        def __init__(self, name, result=None):
            self.name, self.result = name, result

        def remote(self, *args, **kwargs):
            calls.append((self.name, args, kwargs))
            return self.result

    def engine(result=None):
        return SimpleNamespace(
            load_lora_adapter_from_disk=Remote("load", result), set_runtime_load_id=Remote("version")
        )

    monkeypatch.setattr(worker_module.ray, "get", lambda refs: list(refs))
    worker = worker_module.SGLangWorker.__new__(worker_module.SGLangWorker)
    worker.servers = {"default": SimpleNamespace(update_weights=True, engines=[engine(), engine()])}
    worker.load_adapter_from_disk("reef-adapter-math.v1", "/ckpt/adapter", "inc:1")
    assert [name for name, _, _ in calls] == ["load", "load", "version", "version"]
    assert calls[0][2] == {"lora_name": "reef-adapter-math.v1", "lora_path": "/ckpt/adapter"}
    assert calls[2][1] == ("inc:1",)
    calls.clear()
    worker.load_adapter_from_disk("reef-adapter-math.v1", "/ckpt/adapter")
    assert [name for name, _, _ in calls] == ["load", "load"]
    worker.servers = {
        "default": SimpleNamespace(update_weights=True, engines=[engine({"success": False, "message": "rank"})])
    }
    with pytest.raises(RuntimeError, match="refused adapter"):
        worker.load_adapter_from_disk("x", "/p", "inc:2")
    worker.servers = {}
    with pytest.raises(RuntimeError, match="no updatable"):
        worker.load_adapter_from_disk("x", "/p", "inc:2")


def test_worker_reloads_full_weights_into_every_updatable_engine(monkeypatch):
    from reef.inference.sglang import worker as worker_module

    calls = []

    class Remote:
        def __init__(self, result):
            self.result = result

        def remote(self, *args, **kwargs):
            calls.append(kwargs)
            return self.result

    def engine(result=None):
        return SimpleNamespace(update_weights_from_disk=Remote(result or {"success": True}))

    monkeypatch.setattr(worker_module.ray, "get", lambda refs: list(refs))
    worker = worker_module.SGLangWorker.__new__(worker_module.SGLangWorker)
    worker.servers = {"default": SimpleNamespace(update_weights=True, engines=[engine(), engine()])}
    worker.load_weights_from_disk("/ckpt/hf", "inc:3")
    assert calls == [{"model_path": "/ckpt/hf", "runtime_load_id": "inc:3"}] * 2
    worker.servers = {"default": SimpleNamespace(update_weights=True, engines=[engine({"success": False})])}
    with pytest.raises(RuntimeError, match="refused weights"):
        worker.load_weights_from_disk("/ckpt/hf", "inc:3")


def test_engine_control_and_weight_loads_skip_encoder_groups(monkeypatch):
    from reef.inference.sglang import worker as worker_module
    from reef.inference.sglang.launch import SGLangEngineGroup, SGLangModel
    from reef.runtime.recovery import InferenceControl

    calls = []

    class Remote:
        def __init__(self, engine_name, method, result=None):
            self.engine_name, self.method, self.result = engine_name, method, result

        def remote(self, *args, **kwargs):
            calls.append((self.engine_name, self.method))
            return self.result

    def engine(name):
        methods = ("pause_generation", "flush_cache", "continue_generation")
        handle = SimpleNamespace(**{method: Remote(name, method) for method in methods})
        handle.get_runtime_load_id = Remote(name, "get_runtime_load_id", "weights:1")
        return handle

    config = SGLangConfig("model", 2, 1, 2, pause_mode="retract")
    encoder = SGLangEngineGroup(config, SGLangGroupConfig("encoder", 1, 1), None, 0, ("router", 3000))
    regular = SGLangEngineGroup(config, SGLangGroupConfig("regular", 1, 1), None, 1, ("router", 3000))
    encoder.all_engines = [engine("encoder")]
    regular.all_engines = [engine("regular")]
    encoder.num_new_engines = 1
    regular.num_new_engines = 1
    monkeypatch.setattr(worker_module.ray, "get", lambda refs, **_: list(refs))
    monkeypatch.setattr(worker_module, "retire_engines", lambda engines: None)
    worker = worker_module.SGLangWorker.__new__(worker_module.SGLangWorker)
    worker.config = config
    worker._health_monitors = []
    worker.servers = {"actor": SGLangModel([encoder, regular])}
    worker._control = InferenceControl(
        worker_module._SGLangEngineGroup(worker), worker_module._SGLangInferenceMonitor(worker)
    )

    assert worker.get_runtime_load_ids() == ["weights:1"]
    worker.pause_generation_for_update()
    worker.continue_generation_after_update()
    assert worker.updatable_engines() == regular.engines
    assert worker.terminate_updatable_engines() == 1
    assert encoder.all_engines[0] is not None and regular.all_engines == [None]
    assert calls == [
        ("regular", "get_runtime_load_id"),
        ("regular", "pause_generation"),
        ("regular", "flush_cache"),
        ("regular", "continue_generation"),
    ]
