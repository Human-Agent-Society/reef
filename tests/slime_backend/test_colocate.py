"""Shared GPU serving drains requests and preserves artifact publication semantics."""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.sglang.colocate import ColocatedServingRuntime, build_colocated_serving
from reef.inference.sglang.executor import SGLangExecutor
from reef.inference.sglang.service import SGLangInferenceService
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.train.slime_backend.component import SlimeTrainingComponent
from reef.train.slime_backend.launch import SlimeDeployment


@pytest.fixture
def serving():
    backend = MagicMock(spec=SGLangInferenceBackend)
    backend.inference_url.return_value = "http://127.0.0.1:12345"
    runtime = ColocatedServingRuntime(
        backend,
        MagicMock(spec=SGLangInferenceHandler),
        service=MagicMock(spec=SGLangInferenceService),
        model_path="model",
        inference_timeout_s=1,
    )
    backend.reset_mock()
    yield runtime, backend
    runtime.shutdown()


def test_training_handoff_restores_incumbent_without_publishing_candidate(serving):
    runtime, backend = serving
    version = runtime.current_runtime_load_id()
    with runtime.training_phase():
        assert runtime.phase == "training"
        assert runtime.inference_admission_status == {"open": False, "active": 0}
        backend.release_memory.assert_called_once()
        backend.resume_memory.assert_not_called()
    assert [call[0] for call in backend.mock_calls] == ["pause", "release_memory", "resume_memory", "resume"]
    assert runtime.current_runtime_load_id() == version
    assert runtime.inference_admission_status["open"]
    backend.load_weight_files.assert_not_called()
    with runtime.training_phase():
        pass
    assert backend.release_memory.call_count == 2


@pytest.mark.parametrize("failure", ["release", "train", "restore"])
def test_failed_handoff_never_reopens_inference(serving, failure, tmp_path):
    runtime, backend = serving
    if failure == "release":
        backend.release_memory.side_effect = TimeoutError("worker still running")
    elif failure == "restore":
        backend.resume_memory.side_effect = TimeoutError("worker still running")
    with pytest.raises(TimeoutError), runtime.training_phase():
        if failure == "train":
            raise TimeoutError("worker still running")
    assert runtime.phase == "failed"
    backend.resume.assert_not_called()
    with pytest.raises(InferenceUnavailable):
        asyncio.run(runtime.acquire_inference())
    with pytest.raises(RuntimeError, match="cannot recover"):
        runtime.recover_engines()
    with pytest.raises(RuntimeError, match="cannot publish"):
        runtime.activate_checkpoint(Artifact.local(tmp_path))


def test_training_waits_for_active_request_and_blocks_new_requests(serving):
    runtime, backend = serving

    async def exercise():
        admitted = await runtime.acquire_inference()
        drained = threading.Event()
        finish = threading.Event()

        def train():
            with runtime.training_phase():
                drained.set()
                assert finish.wait(2)

        task = asyncio.create_task(asyncio.to_thread(train))
        for _ in range(100):
            if not runtime.inference_admission_status["open"]:
                break
            await asyncio.sleep(0.01)
        assert not runtime.inference_admission_status["open"]
        backend.release_memory.assert_not_called()
        next_request = asyncio.create_task(runtime.acquire_inference())
        admitted.release()
        assert await asyncio.to_thread(drained.wait, 2)
        assert not next_request.done()
        finish.set()
        await task
        handle = await asyncio.wait_for(next_request, 2)
        handle.release()

    asyncio.run(exercise())


@pytest.mark.parametrize("actor_width", [2, "${training.config.gpus}"])
def test_shared_config_starts_no_separate_inference_component(tmp_path, actor_width):
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {"model-path": "/models/demo", "num-gpus": 2, "tensor-parallel-size": 2},
        "training": {
            "backend": "slime",
            "colocate": True,
            "ray-address": "ray://head:10001",
            "config": {"gpus": 2},
            "options": {"global-batch-size": 4, "actor-num-nodes": 1, "actor-num-gpus-per-node": actor_width},
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert config["reef"]["training_backend_options"]["offload-train"] is True
    assert not config["reef"].get("inference_remote")


def test_shared_serving_binds_before_recovered_jobs_start(monkeypatch, serving):
    runtime, _ = serving
    component = MagicMock(spec=SlimeTrainingComponent)
    component.colocate = True
    component.resources = object()
    from reef.runtime.jobs import ReleaseFormat

    component.release_format = ReleaseFormat("full")
    events = []

    def build(resources, release_format, config, *, model_path, environ):
        assert resources is component.resources
        assert release_format == component.release_format
        events.append("serving-started")
        return runtime

    def start_jobs():
        assert component.serving is runtime
        events.append("jobs-started")

    monkeypatch.setattr("reef.inference.sglang.colocate.build_colocated_serving", build)
    component.start_jobs.side_effect = start_jobs
    config = {
        "type": "reef.inference.sglang.colocate:ColocatedServingFactory",
        "release_format": component.release_format.to_dict(),
    }
    assert SlimeDeployment().build_serving(component, config, model_path="model") is runtime
    assert events == ["serving-started", "jobs-started"]


def test_shared_serving_uses_the_native_sglang_executor(monkeypatch):
    from reef.runtime.deployment import InferenceResources
    from reef.runtime.jobs import ReleaseFormat

    resources = MagicMock(spec=InferenceResources)
    backend = MagicMock(spec=SGLangInferenceBackend)
    backend.inference_url.return_value = "http://127.0.0.1:12345"

    def start(service, borrowed):
        assert borrowed is resources
        assert service.config.executor is SGLangExecutor
        assert service.config.options["enable_memory_saver"] is True
        assert service.config.options["enable_weights_cpu_backup"] is True
        assert service.config.pause_mode == "in_place"

    monkeypatch.setattr(SGLangInferenceService, "start", start)
    monkeypatch.setattr(SGLangInferenceService, "backend", lambda self, connection: backend)
    monkeypatch.setattr(SGLangInferenceHandler, "from_config", lambda *args, **kwargs: MagicMock())
    runtime = build_colocated_serving(
        resources,
        ReleaseFormat("full"),
        {
            "options": {},
            "num_gpus": 1,
            "tensor_parallel_size": 1,
            "executor": "ray",
            "executor_options": {},
            "inference_timeout_s": 1,
            "inference_handler_config": {},
        },
        model_path="model",
        environ={},
    )
    runtime.shutdown()
