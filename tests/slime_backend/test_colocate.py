"""Shared GPU serving drains requests and preserves artifact publication semantics."""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.sglang.colocate import (
    ColocatedAdapterServingRuntime,
    ColocatedServingRuntime,
    build_colocated_serving,
)
from reef.inference.sglang.executor import SGLangExecutor
from reef.inference.sglang.service import SGLangInferenceService
from reef.runtime.deployment import BorrowedGpuResources
from reef.runtime.executor.placement import GpuBundles
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.serving import serving_runtime_config
from reef.train.slime_backend.component import SlimeTrainingComponent
from reef.train.slime_backend.launch import SlimeDeployment


@pytest.fixture(params=["full", "adapter"])
def serving(request):
    backend = MagicMock(spec=SGLangInferenceBackend)
    backend.inference_url.return_value = "http://127.0.0.1:12345"
    runtime_type = ColocatedAdapterServingRuntime if request.param == "adapter" else ColocatedServingRuntime
    options = {"adapter_capacity": 2} if request.param == "adapter" else {}
    backend.recover.return_value = 0
    runtime = runtime_type(
        backend,
        MagicMock(spec=SGLangInferenceHandler),
        service=MagicMock(spec=SGLangInferenceService),
        model_path="model",
        inference_timeout_s=1,
        **options,
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


@pytest.mark.parametrize("inference_tp", [1, 2])
@pytest.mark.parametrize("lora_rank", [0, 8])
@pytest.mark.parametrize("actor_width", [2, 4, "${training.config.gpus}"])
@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("remote", [False, True])
def test_shared_config_starts_no_separate_inference_component(
    tmp_path, actor_width, inference_tp, lora_rank, nodes, remote
):
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {
            "model-path": "/models/demo",
            "num-gpus": 2,
            "tensor-parallel-size": inference_tp,
            "prefix-sharing": True,
            "handler-factory": "reef.inference.sglang.chat.SGLangInferenceHandler",
            "handler-config": {"tool_call_parser": "qwen25", "rollout_continuation": True},
        },
        "training": {
            "backend": "slime",
            "colocate": True,
            "ray-address": "ray://head:10001",
            "config": {"gpus": 4},
            "options": {
                "global-batch-size": 4,
                "actor-num-nodes": nodes,
                "actor-num-gpus-per-node": actor_width,
                "megatron-lora-rank": lora_rank,
            },
        },
    }
    if remote:
        raw["inference"]["remote"] = {"url": "http://component:8000/"}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert config["reef"]["training_backend_options"]["offload-train"] is True
    if remote:
        assert config["reef"]["inference_remote"]["url"] == "http://component:8000"
    else:
        assert not config["reef"].get("inference_remote")
    from reef.runtime.jobs import ReleaseFormat

    serving = serving_runtime_config(service_config_from_mapping(config), SlimeDeployment(), ReleaseFormat("full"))
    assert serving["inference_handler_factory"] is SGLangInferenceHandler
    assert serving["prefix_sharing"] is True
    assert serving["inference_handler_config"] == raw["inference"]["handler-config"]


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


def test_custom_serving_without_handoff_contract_is_closed_before_jobs_start(monkeypatch):
    from reef.runtime.deployment import RuntimeConfigError, RuntimeRegistry
    from reef.runtime.interfaces import InferenceRuntime

    component = MagicMock(spec=SlimeTrainingComponent)
    component.colocate = True
    component.resources = object()
    unsupported = MagicMock(spec=InferenceRuntime)
    build = MagicMock(return_value=unsupported)
    monkeypatch.setattr(RuntimeRegistry, "build", build)
    with pytest.raises(RuntimeConfigError, match="shared GPU serving runtime"):
        SlimeDeployment().build_serving(component, {"type": "custom"}, model_path="model")
    assert build.call_args.kwargs["resources"] is component.resources
    unsupported.shutdown.assert_called_once()
    component.start_jobs.assert_not_called()


def test_custom_shared_serving_factory_preserves_its_config(tmp_path):
    from reef.runtime.jobs import ReleaseFormat

    factory = "reef.inference.sglang.colocate:ColocatedServingFactory"
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {"model-path": "/model", "serving": factory, "serving-config": {"options": {"custom": 1}}},
        "training": {"colocate": True, "options": {"global-batch-size": 2}},
    }
    resolved, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    config = serving_runtime_config(service_config_from_mapping(resolved), SlimeDeployment(), ReleaseFormat("full"))
    assert config == {"type": factory, "options": {"custom": 1}, "release_format": ReleaseFormat("full").to_dict()}
    assert [service["name"] for service in resolved["services"]] == ["reef"]


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("prefix_sharing", [False, True])
def test_shared_serving_uses_the_native_sglang_executor(monkeypatch, adapter, prefix_sharing):
    from reef.runtime.deployment import InferenceResources
    from reef.runtime.jobs import ReleaseFormat

    resources = MagicMock(spec=InferenceResources)
    resources.inference_placement = GpuBundles(object(), [2, 3], [0, 1])
    backend = MagicMock(spec=SGLangInferenceBackend)
    backend.inference_url.return_value = "http://127.0.0.1:12345"

    def start(service, borrowed):
        assert borrowed.inference_placement == resources.inference_placement.slice(0, 1)
        assert service.config.executor is SGLangExecutor
        assert service.config.options["enable_memory_saver"] is True
        assert service.config.options["enable_weights_cpu_backup"] is True
        assert service.config.pause_mode == ("retract" if prefix_sharing else "in_place")
        assert service.config.options["disable_radix_cache"] is not prefix_sharing
        if adapter:
            assert service.config.options["enable_lora"] is True
            assert service.config.options["max_lora_rank"] == 8
            assert service.config.options["max_loaded_loras"] == 2

    monkeypatch.setattr("reef.inference.sglang.deployment.require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(BorrowedGpuResources, "node_width", lambda self, tp: 1)
    monkeypatch.setattr("reef.inference.sglang.config.adapter_scoped_prefix_cache_supported", lambda: True)
    monkeypatch.setattr("reef.inference.sglang.deployment.require_lora_distributed_request_schema", lambda: None)
    monkeypatch.setattr(SGLangInferenceService, "start", start)
    monkeypatch.setattr(SGLangInferenceService, "backend", lambda self, connection: backend)
    custom_handler = MagicMock()
    runtime = build_colocated_serving(
        resources,
        ReleaseFormat("adapter", 8, ("q_proj", "v_proj")) if adapter else ReleaseFormat("full"),
        {
            "options": {},
            "num_gpus": 1,
            "tensor_parallel_size": 1,
            "executor": "ray",
            "executor_options": {},
            "inference_timeout_s": 1,
            "inference_handler_config": {},
            "inference_handler_factory": custom_handler,
            "adapter_capacity": 2,
            "prefix_sharing": prefix_sharing,
        },
        model_path="model",
        environ={},
    )
    custom_handler.from_config.assert_called_once_with("http://127.0.0.1:12345", model_path="model", timeout_s=1)
    assert runtime.inference_handler is custom_handler.from_config.return_value
    runtime.shutdown()


def test_adapter_publication_and_residency_survive_handoff(serving, tmp_path):
    runtime, backend = serving
    if not isinstance(runtime, ColocatedAdapterServingRuntime):
        return

    def loaded(files, *, adapter_name, runtime_load_id):
        backend.runtime_load_ids.return_value = [runtime_load_id, runtime_load_id]

    backend.load_weight_files.side_effect = loaded
    for step in range(2):
        release = tmp_path / str(step)
        adapter = release / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}))
        published = runtime.activate_checkpoint(Artifact.local(release), scenario="math")
        assert runtime.serving_adapter_runtime_load_id("math") == published
        residency = runtime.adapter_residency_status()
        backend.reset_mock()
        with runtime.training_phase():
            pass
        assert runtime.serving_adapter_runtime_load_id("math") == published
        assert runtime.adapter_residency_status() == residency
        backend.load_weight_files.assert_not_called()
    backend.recover.return_value = 0
    runtime.recover_engines()
    assert runtime.serving_adapter_runtime_load_id("math") == published
