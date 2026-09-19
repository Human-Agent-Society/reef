"""SGLang receiver results and failures crossing the runtime control boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.config import SGLangConfig
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

    def prepare_training_connection(self):
        self.paused = True

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


def test_preparing_weight_transfer_fences_only_a_compatible_receiver():
    receiver = Receiver()
    executor = UniProcExecutor.from_workers([receiver])
    service = SGLangInferenceService(SGLangConfig("model", 1, 1, 1))
    with pytest.raises(ValueError, match="incompatible SGLang"):
        service.prepare_weight_transfer(InferenceConnection("other-protocol", executor))
    assert not receiver.paused

    connection = InferenceConnection(INFERENCE_PROTOCOL, executor)
    service.prepare_weight_transfer(connection)
    assert receiver.paused
    assert not receiver.closed


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
    receiver.get_updatable_engines_and_lock = lambda: ([object(), object()], None, 0, [], [], [])
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
    receiver = SimpleNamespace(get_updatable_engines_and_lock=lambda: (engines, None, 0, [], [], []))

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
