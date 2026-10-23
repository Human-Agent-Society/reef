"""``inference.backend: none``: the receiver contract without an engine, and refused inference requests."""

from __future__ import annotations

import asyncio
import pickle

import pytest

from reef.artifact import Artifact
from reef.core.errors import InferenceUnavailable, ReefError
from reef.inference.deployment import inference_service_for
from reef.inference.null import (
    NULL_INFERENCE_URL,
    UNSTAMPED_RUNTIME_LOAD_ID,
    NullInferenceBackend,
    NullInferenceHandler,
    NullInferenceService,
    ServedVersionRecord,
)
from reef.runtime.deployment import NO_WEIGHT_TRANSFER_PROTOCOL, InferenceConnection
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.interfaces import WeightFiles


def test_backend_reports_the_version_it_or_the_trainer_recorded(tmp_path):
    record = UniProcExecutor.from_workers([ServedVersionRecord()])
    backend = NullInferenceBackend(record)
    try:
        # A new record reports what a freshly booted engine reports, so startup recovery reads one version.
        assert backend.runtime_load_ids() == (UNSTAMPED_RUNTIME_LOAD_ID,)
        backend.initialize_version("inc:1")
        assert backend.runtime_load_ids() == ("inc:1",)
        # The trainer records a publication through the session's receiver, in place of a send.
        record.rpc(0, "set_runtime_load_id", args=("inc:2",))
        for operation in (backend.pause, backend.resume, backend.recover, backend.abort):
            operation()
        backend.unload_adapter("scenario@inc:1")
        assert backend.runtime_load_ids() == ("inc:2",)
        assert backend.inference_url() == NULL_INFERENCE_URL
        with pytest.raises(ReefError, match="does not load weight files"):
            backend.load_weight_files(WeightFiles(tmp_path, "full"), adapter_name=None, runtime_load_id="inc:3")
        with pytest.raises(ValueError, match="non-empty"):
            record.rpc(0, "set_runtime_load_id", args=("",))
    finally:
        record.shutdown()


def test_backend_selection_builds_the_service_and_refuses_engine_settings():
    service = inference_service_for("none", {})
    assert isinstance(service, NullInferenceService)
    assert service.connection_protocol == NO_WEIGHT_TRANSFER_PROTOCOL
    assert service.supported_transfer_protocols == (NO_WEIGHT_TRANSFER_PROTOCOL,)
    with pytest.raises(ValueError, match="takes no engine settings; got num_gpus"):
        inference_service_for("none", {"num_gpus": 1})


def test_service_refuses_a_connection_of_another_protocol():
    service = NullInferenceService()
    with pytest.raises(ValueError, match="incompatible inference connection"):
        service.backend(InferenceConnection("slime-sglang-control-v2", UniProcExecutor.from_workers([object()])))
    with pytest.raises(RuntimeError, match="not running"):
        service.check_health()


def test_service_record_is_shared_by_the_trainer_and_the_coordinator_processes():
    ray = pytest.importorskip("ray", reason="requires the optional Ray runtime")
    ray.init(address="local", num_cpus=1, include_dashboard=False, log_to_driver=False)
    service = NullInferenceService()
    try:
        connection = service.start(object())
        service.check_health()
        service.prepare_weight_transfer(connection)
        backend = service.backend(connection)
        # The trainer and the coordinator each unpickle their own handle on the one record.
        trainer_receiver = pickle.loads(pickle.dumps(connection.control))
        coordinator_backend = pickle.loads(pickle.dumps(backend))
        trainer_receiver.rpc(0, "set_runtime_load_id", args=("inc:5",), timeout=30)
        assert coordinator_backend.runtime_load_ids() == ("inc:5",)
        service.poll()
        with pytest.raises(RuntimeError, match="once"):
            service.start(object())
    finally:
        service.close()
        service.close()
        ray.shutdown()


def test_handler_refuses_every_request(tmp_path):
    # The HTTP contract (503 on every inference route) is in tests/reef_service/test_runtime_registry.py.
    handler = NullInferenceHandler.from_config(NULL_INFERENCE_URL, model_path="/models/qwen", timeout_s=30)
    with pytest.raises(InferenceUnavailable, match=r"inference\.backend: none"):
        asyncio.run(handler.inference(Artifact.local(tmp_path), "/v1/chat/completions", {}))
