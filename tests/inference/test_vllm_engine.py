"""The vLLM engine actor maps Reef's control vocabulary onto vLLM's HTTP routes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reef.inference.vllm import engine as engine_module
from reef.inference.vllm.config import REEF_CONNECTOR_CONFIG, VLLMConfig
from reef.inference.vllm.engine import ReefVLLMEngine


class FakeHTTP:
    """Record every request and answer from a queue of prepared responses."""

    def __init__(self):
        self.calls = []
        self.responses = []

    def answer(self, body=None, *, status=200, text=""):
        self.responses.append((status, body, text))

    def _respond(self, method, url, **kwargs):
        self.calls.append((method, url.split("/", 3)[3], kwargs.get("params"), kwargs.get("json")))
        status, body, text = self.responses.pop(0) if self.responses else (200, {}, "")
        content = json.dumps(body).encode() if body is not None else text.encode()
        response = SimpleNamespace(status_code=status, ok=status < 400, content=content, text=content.decode())
        response.json = lambda: json.loads(content)

        def raise_for_status():
            if status >= 400:
                raise engine_module.requests.HTTPError(f"HTTP {status}", request=None, response=response)

        response.raise_for_status = raise_for_status
        return response

    def get(self, url, **kwargs):
        return self._respond("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, **kwargs)


@pytest.fixture
def engine(monkeypatch):
    http = FakeHTTP()
    monkeypatch.setattr(engine_module.requests, "get", http.get)
    monkeypatch.setattr(engine_module.requests, "post", http.post)
    config = VLLMConfig(
        "model", 2, 2, 8, options={"max-model-len": 4096, "enable_lora": True}, offload=True, shared_gpus=2
    )
    engine = ReefVLLMEngine(config, rank=0, gpu_ids=(6, 7))
    engine.server_host, engine.server_port = "10.0.0.5", 18900
    return engine, http


def test_server_command_line_carries_placement_reef_requirements_then_options(engine):
    engine, _ = engine
    arguments = engine.server_arguments("[10.0.0.5]", 18900)
    assert arguments[:6] == ["--host", "10.0.0.5", "--port", "18900", "--tensor-parallel-size", "2"]
    assert "--distributed-executor-backend" in arguments and "--enable-sleep-mode" in arguments
    assert arguments[arguments.index("--max-model-len") + 1] == "4096"
    assert "--enable-lora" in arguments
    assert "--no-enable-prefix-caching" in arguments
    assert json.loads(arguments[arguments.index("--kv-transfer-config") + 1]) == REEF_CONNECTOR_CONFIG
    environment = engine.server_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"
    with pytest.raises(ValueError, match="needs 2 GPUs"):
        ReefVLLMEngine(engine.config, rank=1, gpu_ids=(3,))


def test_version_round_trips_through_vllm_weight_routes(engine):
    engine, http = engine
    http.answer({"weight_version": "default"})
    assert engine.get_runtime_load_id() == "default"
    http.answer({"success": True, "new_version": "reef:1"})
    assert engine.set_runtime_load_id("reef:1")["new_version"] == "reef:1"
    assert http.calls == [
        ("GET", "weight_info", None, None),
        ("POST", "update_weight_version", None, {"new_version": "reef:1"}),
    ]
    http.answer({})
    with pytest.raises(RuntimeError, match="no weight_version"):
        engine.get_runtime_load_id()


def test_pause_keeps_requests_and_retract_also_frees_their_kv(engine):
    engine, http = engine
    http.answer({"status": "paused"})
    assert engine.pause_generation("in_place") == {"status": "paused"}
    http.answer({"status": "paused"})
    http.answer({"success": True})
    engine.pause_generation("retract")
    http.answer({"status": "resumed"})
    engine.continue_generation()
    assert http.calls == [
        ("POST", "pause", {"mode": "keep", "clear_cache": "false"}, None),
        ("POST", "pause", {"mode": "keep", "clear_cache": "false"}, None),
        ("POST", "reset_prefix_cache", {"reset_running_requests": "true"}, None),
        ("POST", "resume", None, None),
    ]
    http.answer({"status": "paused"})
    http.answer({"success": False})
    with pytest.raises(RuntimeError, match="refused the cache reset"):
        engine.pause_generation("retract")
    with pytest.raises(ValueError, match="unknown vLLM pause mode"):
        engine.pause_generation("drain")


def test_memory_release_sleeps_at_level_two_and_resume_wakes_named_regions(engine):
    engine, http = engine
    engine.release_memory_occupation()
    engine.release_memory_occupation()  # already released: no second sleep
    engine.resume_memory_occupation(["weights"])
    engine.resume_memory_occupation(["kv_cache", "cuda_graph"])
    engine.resume_memory_occupation(["kv_cache"])  # already resident: nothing to wake
    assert http.calls == [
        ("POST", "sleep", {"level": "2", "mode": "keep"}, None),
        ("POST", "wake_up", {"tags": ["weights"]}, None),
        ("POST", "wake_up", {"tags": ["kv_cache"]}, None),
    ]
    with pytest.raises(ValueError, match="only together with the weights"):
        engine.release_memory_occupation(["kv_cache", "cuda_graph"])
    with pytest.raises(ValueError, match="unknown inference memory regions"):
        engine.resume_memory_occupation(["weights", "optimizer"])


def test_weights_reload_and_adapters_report_vllm_answers(engine):
    engine, http = engine
    http.answer({"results": [None]})
    http.answer({"success": True, "new_version": "reef:2"})
    engine.update_weights_from_disk("/ckpt/step-2", runtime_load_id="reef:2")
    http.answer(text="Success: LoRA adapter 'a' added successfully.")
    assert engine.load_lora_adapter_from_disk("a", "/ckpt/adapter") == {
        "success": True,
        "message": "Success: LoRA adapter 'a' added successfully.",
    }
    http.answer({"error": {"message": "no such adapter"}}, status=404)
    assert engine.unload_lora_adapter("b")["success"] is False
    assert http.calls[:2] == [
        ("POST", "collective_rpc", None, {"method": "reload_weights", "kwargs": {"weights_path": "/ckpt/step-2"}}),
        ("POST", "update_weight_version", None, {"new_version": "reef:2"}),
    ]
    assert http.calls[2][1:] == ("v1/load_lora_adapter", None, {"lora_name": "a", "lora_path": "/ckpt/adapter"})
    http.answer({"detail": "engine busy"}, status=503)
    with pytest.raises(engine_module.requests.HTTPError, match="engine busy"):
        engine.flush_cache()


def test_init_launches_the_server_and_waits_for_its_health_route(monkeypatch):
    launched = []
    process = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(
        engine_module,
        "launch_server",
        lambda model_path, arguments, env: launched.append((model_path, arguments, env)) or process,
    )
    waited = []
    monkeypatch.setattr(
        engine_module, "wait_ready", lambda url, proc, timeout, path: waited.append((url, proc, timeout, path))
    )
    engine = ReefVLLMEngine(VLLMConfig("model", 1, 1, 1, startup_timeout=7), rank=0, gpu_ids=(0,))
    engine.init("10.0.0.5", 18900)
    assert engine.get_url() == "http://10.0.0.5:18900"
    assert launched[0][0] == "model" and launched[0][2]["CUDA_VISIBLE_DEVICES"] == "0"
    assert waited == [("http://10.0.0.5:18900", process, 7, "/health")]
    assert engine.process is process
