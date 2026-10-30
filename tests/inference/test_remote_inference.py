"""The remote inference component and its control-plane client, over real HTTP with fake engines."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from reef.artifact import Artifact
from reef.core.errors import ReefError
from reef.core.weight_layout import ADAPTER_DIR
from reef.inference.remote import server as server_module
from reef.inference.remote.config import CONTROL_PROTOCOL, SHARED_GPU_CONTROL_CAPABILITY, RemoteInferenceSettings
from reef.inference.remote.runtime import RemoteServingRuntimeFactory
from reef.inference.remote.server import RemoteInferenceServer
from reef.inference.remote.service import RemoteControl, RemoteInferenceBackend, RemoteInferenceService
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.runtime.deployment import (
    BorrowedGpuResources,
    InferenceConnection,
    InferenceService,
    RuntimeConfigError,
    RuntimeRegistry,
    runtime_factory_for,
)
from reef.runtime.executor.placement import GpuBundles
from reef.runtime.interfaces import MemoryManagedInferenceBackend, WeightFiles
from reef.runtime.jobs import ReleaseFormat
from reef.runtime.serving import AdapterServingRuntime, FullWeightServingRuntime
from reef.surface.adapter import adapter_name

CONTROL_TOKEN = "control-secret"
ADAPTER = ReleaseFormat("adapter", 16, ("all",))
INFERENCE_TOKEN = "inference-secret"


@contextmanager
def serving(app: web.Application):
    """Run ``app`` on a free loopback port in its own event loop thread and yield its base URL."""
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = runner.addresses[0][1]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=30)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=30)
        loop.close()


def engine_app() -> web.Application:
    """An engine endpoint: echoes generate requests and streams three chunks."""

    async def generate(request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {"payload": await request.json(), "authorization": request.headers.get("Authorization")}
        )

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index in range(3):
            await response.write(f"data: {index}\n\n".encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.router.add_get("/stream", stream)
    return app


class FakeEngines(MemoryManagedInferenceBackend):
    def __init__(self, engine_url: str, events: list):
        self.engine_url = engine_url
        self.events = events
        self.versions = ["base:0"]
        self.release_load = threading.Event()
        self.release_load.set()

    def initialize_version(self, runtime_load_id):
        self.versions = [runtime_load_id]

    def inference_url(self):
        return self.engine_url

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def release_memory(self):
        self.events.append("release-memory")

    def resume_memory(self):
        self.events.append("resume-memory")

    def recover(self):
        self.events.append("recover")
        return 1

    def abort(self):
        self.events.append("abort")

    def unload_adapter(self, name):
        self.events.append(("unload", name))

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        assert self.release_load.wait(timeout=30)
        contents = {path.name: path.read_text() for path in files.path.iterdir()}
        self.events.append(("load", files.layout, adapter_name, runtime_load_id, contents, files.path))
        if runtime_load_id is not None:
            self.versions = [runtime_load_id]


class FakeService(InferenceService):
    connection_protocol = "fake-engine-v1"

    def __init__(self, config, events, engine_url):
        self.config = config
        self.events = events
        self.engines = FakeEngines(engine_url, events)

    def start(self, resources):
        self.events.append(("start", dict(self.config), resources.inference_placement))
        return InferenceConnection(self.connection_protocol, None)

    def backend(self, connection):
        return self.engines

    def check_health(self):
        self.events.append("health")

    def poll(self):
        self.events.append("poll")

    def close(self):
        self.events.append("close")


class FakeReservation:
    def __init__(self, layout, events):
        self.inference = f"placement-{layout.inference_gpus}"
        self.events = events

    def release(self):
        self.events.append("release")


@pytest.fixture
def server(tmp_path):
    """The component's server, which a test can restart (``restart``)."""
    return RemoteInferenceServer(
        engine="sglang", control_token=CONTROL_TOKEN, inference_token=INFERENCE_TOKEN, work_dir=tmp_path / "work"
    )


def restart(server: RemoteInferenceServer) -> None:
    """Make ``server`` act as a new component process at the same URL: another instance, which runs no engines."""
    server.stop_stack()
    server.instance_id = "restarted-process"


@pytest.fixture
def component(server, tmp_path, monkeypatch):
    """A running component whose engines are fakes, and the events they record."""
    events: list = []
    with serving(engine_app()) as engine_url:
        services = []

        def create_service(name, config):
            services.append(FakeService({"backend": name, **config}, events, engine_url))
            return services[-1]

        monkeypatch.setattr(server_module, "inference_service_for", create_service)
        monkeypatch.setattr(server_module, "reserve_model_gpus", lambda layout: FakeReservation(layout, events))
        with serving(server.application()) as url:
            yield url, events, tmp_path / "work", services


def started_service(url: str) -> RemoteInferenceService:
    service = RemoteInferenceService(
        RemoteInferenceSettings(url=url, startup_timeout_s=30),
        engine="sglang",
        engine_config={"model_path": "Qwen/Qwen3-8B"},
        num_gpus=2,
        control_token=CONTROL_TOKEN,
    )
    connection = service.start(None)
    assert connection == InferenceConnection(CONTROL_PROTOCOL, None)
    return service


@pytest.fixture
def shared_resources(monkeypatch):
    group = SimpleNamespace(id=SimpleNamespace(hex=lambda: "shared-group"))
    resources = BorrowedGpuResources(GpuBundles(group, [2, 3], [0, 1]))

    def attach(cls, config):
        assert config == resources.to_config()
        return resources

    monkeypatch.setattr(BorrowedGpuResources, "from_config", classmethod(attach))
    monkeypatch.setattr(BorrowedGpuResources, "node_width", lambda self, tp: 2)
    return resources


def test_shared_client_refuses_an_old_component_before_allocating_engines(shared_resources, monkeypatch):
    calls = []

    def legacy(self, method, route, **kwargs):
        calls.append((method, route))
        return {"ok": True, "engines": False, "instance": "legacy"}

    monkeypatch.setattr(RemoteControl, "call", legacy)
    service = RemoteInferenceService(
        RemoteInferenceSettings(url="http://component"),
        engine="sglang",
        num_gpus=2,
        engine_config={},
        control_token=CONTROL_TOKEN,
    )
    with pytest.raises(RuntimeError, match="does not support shared GPU memory"):
        service.start(shared_resources)
    service.close()
    assert calls == [("GET", "health")]


@pytest.mark.parametrize("failure", [None, "release-memory", "resume-memory"])
def test_remote_handoff_preserves_ownership_and_blocks_requests_until_restored(
    component, server, shared_resources, monkeypatch, failure
):
    url, events, _, services = component
    service = RemoteInferenceService(
        RemoteInferenceSettings(url=url),
        engine="sglang",
        num_gpus=2,
        control_token=CONTROL_TOKEN,
        engine_config={
            "model_path": "model",
            "gpus_per_engine": 2,
            "gpus_per_node": 2,
            "options": {"enable_memory_saver": True, "enable_weights_cpu_backup": True},
        },
    )
    backend = service.backend(service.start(shared_resources))
    assert server.resources is shared_resources
    assert service.control.engine_instance == server.engine_instance
    backend.resume()
    payload = json.dumps({"input_ids": [1]}).encode()

    def generate():
        request = urllib.request.Request(
            url + "/generate",
            data=payload,
            headers={"Authorization": f"Bearer {INFERENCE_TOKEN}", "Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=10)

    with generate() as response:
        assert response.status == 200
    backend.pause()
    with pytest.raises(urllib.error.HTTPError) as paused:
        generate()
    assert paused.value.code == 503
    if failure:

        def fail():
            raise TimeoutError("one engine did not acknowledge")

        monkeypatch.setattr(services[-1].engines, failure.replace("-", "_"), fail)
    if failure == "release-memory":
        with pytest.raises(RuntimeError, match="acknowledge"):
            backend.release_memory()
    else:
        backend.release_memory()
        assert server.memory_state == "released"
        with pytest.raises(RuntimeError, match="released"):
            backend.resume()
        if failure == "resume-memory":
            with pytest.raises(RuntimeError, match="acknowledge"):
                backend.resume_memory()
        else:
            backend.resume_memory()
            backend.resume()
            with generate() as response:
                assert response.status == 200
    if failure:
        assert server.memory_state == "failed"
        with pytest.raises(RuntimeError, match="failed"):
            backend.resume()
        with pytest.raises(urllib.error.HTTPError):
            generate()
    expected_instance = server.engine_instance
    server.engine_instance = "replacement"
    with pytest.raises(RuntimeError, match="changed"):
        backend.runtime_load_ids()
    with pytest.raises(RuntimeError, match="changed"):
        service.control.call("DELETE", "engines")
    assert server.resources is shared_resources
    server.engine_instance = expected_instance
    service.close()
    assert "release" not in events


def test_shared_pause_waits_for_the_entire_forwarded_response(component, server, shared_resources, monkeypatch):
    url, events, _, _ = component
    service = RemoteInferenceService(
        RemoteInferenceSettings(url=url),
        engine="sglang",
        num_gpus=2,
        control_token=CONTROL_TOKEN,
        engine_config={
            "model_path": "model",
            "gpus_per_engine": 2,
            "gpus_per_node": 2,
            "options": {"enable_memory_saver": True, "enable_weights_cpu_backup": True},
        },
    )
    backend = service.backend(service.start(shared_resources))
    backend.resume()
    entered, release, draining = threading.Event(), threading.Event(), threading.Event()
    original_wait = server.drained.wait

    async def wait_for_stream():
        draining.set()
        await original_wait()

    async def stream(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        entered.set()
        if not await asyncio.to_thread(release.wait, 10):
            raise TimeoutError("test did not release the response stream")
        await response.write(b"last")
        await response.write_eof()
        return response

    def read_stream():
        request = urllib.request.Request(url + "/stream", headers={"Authorization": f"Bearer {INFERENCE_TOKEN}"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read()

    monkeypatch.setattr(server, "forward_request", stream)
    monkeypatch.setattr(server.drained, "wait", wait_for_stream)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(read_stream)
        try:
            assert entered.wait(10)
            pausing = pool.submit(backend.pause)
            assert draining.wait(10)
            assert not pausing.done() and server.memory_state == "draining"
            with pytest.raises(urllib.error.HTTPError) as rejected:
                read_stream()
            assert rejected.value.code == 503
            assert [event for event in events if event in ("pause", "resume")][-1] == "resume"
        finally:
            release.set()
        assert reading.result(10) == b"firstlast"
        pausing.result(10)
    assert server.active_requests == 0 and server.memory_state == "paused"
    backend.release_memory()
    assert events[-2:] == ["pause", "release-memory"]
    service.close()


def test_service_starts_engines_and_publishes_weight_files_through_the_component(component, tmp_path):
    url, events, work_dir, _ = component
    service = started_service(url)
    # The engines start paused; the control plane decides when they serve.
    assert events == [
        ("start", {"backend": "sglang", "model_path": "Qwen/Qwen3-8B"}, "placement-2"),
        "health",
        "pause",
    ]
    connection = InferenceConnection(CONTROL_PROTOCOL, None)
    backend = service.backend(connection)
    assert isinstance(backend, RemoteInferenceBackend)
    assert backend.inference_url() == url
    backend.initialize_version("inc:1")
    assert backend.runtime_load_ids() == ["inc:1"]
    for operation in (backend.pause, backend.resume, backend.abort):
        operation()
    assert events[-3:] == ["pause", "resume", "abort"]
    # The recover route returns the engine slots the remote backend refilled.
    assert backend.recover() == 1

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    backend.load_weight_files(WeightFiles(adapter, "adapter"), adapter_name="math@inc:2", runtime_load_id="inc:2")
    _, layout, name, version, contents, served = events[-1]
    assert (layout, name, version, contents) == (
        "adapter",
        "math@inc:2",
        "inc:2",
        {"adapter_config.json": '{"r": 16}'},
    )
    assert served.parent == work_dir and served.is_dir()
    assert backend.runtime_load_ids() == ["inc:2"]
    backend.unload_adapter("math@inc:2")
    assert events[-1] == ("unload", "math@inc:2") and not served.exists()

    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    backend.load_weight_files(WeightFiles(weights, "full"), adapter_name=None, runtime_load_id="inc:3")
    _, layout, name, version, contents, served = events[-1]
    assert (layout, name, version, contents) == ("full", None, "inc:3", {"config.json": "{}"})
    assert not served.exists()  # engines read full weights once
    with pytest.raises(ValueError, match="adapter name is given exactly"):
        backend.load_weight_files(WeightFiles(weights, "full"), adapter_name="x", runtime_load_id="inc:4")

    service.check_health()
    assert events[-1] == "poll"
    service.close()
    service.close()
    assert events[-2:] == ["close", "release"]
    with pytest.raises(RuntimeError, match="409"):
        backend.runtime_load_ids()


def test_component_requires_its_tokens_and_running_engines(component):
    url, events, _, _ = component
    with pytest.raises(RuntimeError, match="401"):
        RemoteControl(url, "wrong").call("GET", "health")
    with pytest.raises(RuntimeError, match="409: no engines are running"):
        RemoteControl(url, CONTROL_TOKEN).call("GET", "runtime-load-ids")
    with pytest.raises(RuntimeError, match="400"):
        RemoteControl(url, CONTROL_TOKEN).call("PUT", "engines", payload={"backend": "sglang", "config": {}})
    started_service(url)
    started = list(events)
    for token in (None, CONTROL_TOKEN):
        request = urllib.request.Request(f"{url}/generate", data=b"{}", method="POST")
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        with pytest.raises(urllib.error.HTTPError, match="401"):
            urllib.request.urlopen(request, timeout=30)
    assert events == started


def test_component_starts_only_the_engine_its_operator_chose(component):
    url, events, _, services = component
    with pytest.raises(RuntimeError, match="400: this component starts only 'sglang' engines, not 'subprocess:run'"):
        RemoteControl(url, CONTROL_TOKEN).call(
            "PUT", "engines", payload={"backend": "subprocess:run", "config": {"echo": 1}, "num_gpus": 1}
        )
    assert services == [] and events == []


def test_health_reports_busy_engines_as_healthy_while_they_load_weights(component, tmp_path):
    url, events, _, services = component
    service = started_service(url)
    backend = service.backend(InferenceConnection(CONTROL_PROTOCOL, None))
    engines = services[0].engines
    engines.release_load.clear()
    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    loading = threading.Thread(
        target=backend.load_weight_files,
        args=(WeightFiles(weights, "full"),),
        kwargs={"adapter_name": None, "runtime_load_id": "inc:5"},
    )
    loading.start()
    try:
        service.check_health()  # answers while the load holds the control lock
        assert events[-1] == "poll" and not [event for event in events if event[0] == "load"]
    finally:
        engines.release_load.set()
        loading.join(timeout=30)
    assert events[-1][:4] == ("load", "full", None, "inc:5")


def test_health_reports_the_component_process_and_whether_engines_run(component, server):
    url, _, _, _ = component
    control = RemoteControl(url, CONTROL_TOKEN)
    assert control.call("GET", "health") == {
        "ok": True,
        "instance": server.instance_id,
        "engines": False,
        "engine_instance": None,
        "memory_state": "stopped",
        "capabilities": [SHARED_GPU_CONTROL_CAPABILITY],
    }
    service = started_service(url)
    assert service.instance == server.instance_id
    assert control.call("GET", "health") == {
        "ok": True,
        "instance": server.instance_id,
        "engines": True,
        "engine_instance": server.engine_instance,
        "memory_state": "resident",
        "capabilities": [SHARED_GPU_CONTROL_CAPABILITY],
    }
    assert not service.engines_lost()
    control.call("DELETE", "engines")
    # The component stopped the engines: the service has lost them although the process is the same.
    assert service.engines_lost()


def test_recovery_after_a_component_restart_starts_the_engines_again_at_the_initial_version(component, server):
    url, _, _, services = component
    service = started_service(url)
    backend = service.backend(InferenceConnection(CONTROL_PROTOCOL, None))
    backend.initialize_version("inc:1")
    restart(server)
    assert service.engines_lost()
    with pytest.raises(RuntimeError, match="restarted"):
        service.check_health()
    assert backend.recover() == 1
    assert len(services) == 2 and services[1].engines.versions == ["inc:1"]
    assert service.instance == "restarted-process" and not service.engines_lost()
    service.check_health()
    # Engines that the component still runs are recovered by the component itself.
    assert backend.recover() == 1 and len(services) == 2


def test_component_forwards_requests_to_the_engines_without_its_token(component):
    url, _, _, _ = component
    started_service(url)
    request = urllib.request.Request(
        f"{url}/generate",
        data=json.dumps({"input_ids": [1, 2]}).encode(),
        headers={"Authorization": f"Bearer {INFERENCE_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert json.loads(response.read()) == {"payload": {"input_ids": [1, 2]}, "authorization": None}
    request = urllib.request.Request(f"{url}/stream", headers={"Authorization": f"Bearer {INFERENCE_TOKEN}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert response.read() == b"data: 0\n\ndata: 1\n\ndata: 2\n\n"


def test_component_refuses_an_archive_that_leaves_its_directory(component, tmp_path):
    url, events, work_dir, _ = component
    started_service(url)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        body = b"escaped"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    size = archive.tell()
    archive.seek(0)
    with pytest.raises(RuntimeError, match="500"):
        RemoteControl(url, CONTROL_TOKEN).call(
            "PUT", "weights", upload=archive, upload_bytes=size, query={"layout": "full"}
        )
    assert not (work_dir / "escaped.txt").exists() and not (tmp_path / "escaped.txt").exists()
    assert not [event for event in events if isinstance(event, tuple) and event[0] == "load"]
    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ({}, "inference.remote requires a url"),
        ({"url": "ftp://host"}, "HTTP"),
        ({"url": "https://user:pw@host"}, "without credentials"),
        ({"url": "https://host?x=1"}, "without credentials, query"),
        ({"url": "https://host", "control-token-env": "SAME", "inference-token-env": "SAME"}, "different"),
        ({"url": "https://host", "startup-timeout-s": 0}, "positive"),
        ({"url": "https://host", "model-path": " "}, "model-path must be a non-empty string"),
        ({"url": "https://host", "control-token-env": ""}, "control-token-env must be"),
        ({"url": "https://host", "region": "us"}, "unknown inference.remote fields: region"),
    ],
)
def test_settings_refuse_invalid_sections(section, match):
    with pytest.raises(ValueError, match=match):
        RemoteInferenceSettings.from_mapping(section)


def test_settings_read_dashed_keys_and_normalize_the_url():
    settings = RemoteInferenceSettings.from_mapping(
        {"url": "https://host.example/", "control-token-env": "A", "inference-token-env": "B"}
    )
    assert (settings.url, settings.control_token_env, settings.inference_token_env) == (
        "https://host.example",
        "A",
        "B",
    )


def serving_config(url: str = "https://component.example", **values) -> dict:
    return {
        "type": "remote_serving",
        "remote": {"url": url, "startup-timeout-s": 30},
        "release_format": ADAPTER.to_dict(),
        "engine": "sglang",
        "num_gpus": 2,
        **values,
    }


def started_engines(event) -> tuple:
    """The backend, model, engine width, adapter table and GPUs of a recorded engine start."""
    _, config, placement = event
    return (
        config["backend"],
        config["model_path"],
        config["gpus_per_engine"],
        config["options"].get("max_loaded_loras"),
        placement,
    )


TOKENS = {"REEF_REMOTE_CONTROL_TOKEN": CONTROL_TOKEN, "REEF_REMOTE_INFERENCE_TOKEN": INFERENCE_TOKEN}


def saved_adapter(path):
    (path / ADAPTER_DIR).mkdir(parents=True)
    (path / ADAPTER_DIR / "adapter_config.json").write_text('{"r": 16}')
    return path


@pytest.mark.parametrize(
    ("values", "environ", "match"),
    [
        ({"engine": "my.engines:create"}, TOKENS, "explicit inference handler factory"),
        ({}, {"REEF_REMOTE_CONTROL_TOKEN": CONTROL_TOKEN}, "REEF_REMOTE_INFERENCE_TOKEN"),
        ({}, {"REEF_REMOTE_INFERENCE_TOKEN": INFERENCE_TOKEN}, "REEF_REMOTE_CONTROL_TOKEN"),
        ({"engine": " "}, TOKENS, "runtime.engine must be a non-empty string"),
        ({"release_format": {"layout": "shards"}}, TOKENS, "layout must be 'full' or 'adapter'"),
        ({"num_gpus": 0}, TOKENS, "runtime.num_gpus must be a positive integer"),
        ({"tensor_parallel_size": 3}, TOKENS, "multiple of runtime.tensor_parallel_size"),
        ({"engine": "vllm"}, TOKENS, "vllm serves full weights only"),
        ({"adapter_capacity": 0}, TOKENS, "runtime.adapter_capacity must be a positive integer"),
        ({"inference_timeout_s": 0}, TOKENS, "runtime.inference_timeout_s must be positive"),
        ({"remote": {"url": "ftp://host"}}, TOKENS, "HTTP"),
    ],
)
def test_runtime_refuses_invalid_settings_before_it_starts_engines(component, values, environ, match):
    url, events, _, _ = component
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeRegistry().build(serving_config(url, **values), model_path="Qwen/Qwen3-8B", environ=environ)
    assert events == []


def test_factory_is_registered_for_the_remote_serving_kind():
    assert isinstance(runtime_factory_for("remote_serving"), RemoteServingRuntimeFactory)


def test_adapter_runtime_starts_the_engines_and_loads_each_published_adapter_through_the_component(
    component, tmp_path
):
    url, events, work_dir, _ = component
    runtime = RuntimeRegistry().build(
        serving_config(url, adapter_capacity=2), model_path="Qwen/Qwen3-8B", environ=TOKENS
    )
    assert isinstance(runtime, AdapterServingRuntime)
    assert runtime.per_scenario_adapters and runtime.shared_by_training_scenarios
    # The engines start paused, then serve the base model under the runtime's first version.
    assert started_engines(events[0]) == ("sglang", "Qwen/Qwen3-8B", 1, 2, "placement-2")
    assert events[1:4] == ["health", "pause", "resume"]
    assert runtime.current_runtime_load_id() == runtime.base_version
    version = runtime.activate_checkpoint(Artifact.local(saved_adapter(tmp_path / "job-1")), scenario="math")
    _, layout, name, loaded, contents, served = events[-2]
    assert (layout, name, loaded, contents) == (
        "adapter",
        adapter_name("math", version),
        version,
        {"adapter_config.json": '{"r": 16}'},
    )
    assert served.parent == work_dir and events[-1] == "resume"
    assert runtime.current_runtime_load_id() == version
    assert runtime.serving_adapter_runtime_load_id("math") == version
    arguments = runtime.inference_handler._post_arguments(Artifact.local(tmp_path), "/generate", {"input_ids": [1]})
    assert (arguments["url"], arguments["headers"]["Authorization"]) == (
        f"{url}/generate",
        f"Bearer {INFERENCE_TOKEN}",
    )
    assert isinstance(runtime.inference_handler, SGLangInferenceHandler)
    runtime.shutdown()
    assert events[-2:] == ["close", "release"]


def test_full_weight_runtime_loads_a_published_checkpoint_through_the_component(component, tmp_path):
    url, events, _, _ = component
    runtime = RuntimeRegistry().build(
        serving_config(url, release_format=ReleaseFormat("full").to_dict()), model_path="Qwen/Qwen3-8B", environ=TOKENS
    )
    assert started_engines(events[0]) == ("sglang", "Qwen/Qwen3-8B", 1, None, "placement-2")
    assert isinstance(runtime, FullWeightServingRuntime)
    assert not runtime.per_scenario_adapters and not runtime.shared_by_training_scenarios
    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    version = runtime.activate_checkpoint(Artifact.local(weights))
    _, layout, name, loaded, contents, _ = events[-2]
    assert (layout, name, loaded, contents) == ("full", None, version, {"config.json": "{}"})
    assert runtime.current_runtime_load_id() == version
    runtime.shutdown()
    assert events[-2:] == ["close", "release"]


def test_a_failed_runtime_start_stops_the_engines_it_started(component, tmp_path):
    url, events, _, _ = component

    class BrokenHandler(SGLangInferenceHandler):
        @classmethod
        def from_config(cls, upstream_url, **config):
            raise ReefError("handler refused its configuration")

    with pytest.raises(ReefError, match="handler refused"):
        RuntimeRegistry().build(
            serving_config(url, inference_handler_factory=BrokenHandler), model_path="Qwen/Qwen3-8B", environ=TOKENS
        )
    assert [event for event in events if isinstance(event, str)][-2:] == ["close", "release"]


def test_the_serving_runtime_reloads_its_release_into_the_engines_of_a_restarted_component(
    component, server, tmp_path
):
    url, events, _, services = component
    runtime = RuntimeRegistry().build(
        serving_config(url, release_format=ReleaseFormat("full").to_dict()), model_path="Qwen/Qwen3-8B", environ=TOKENS
    )
    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    version = runtime.activate_checkpoint(Artifact.local(weights))
    restart(server)
    runtime.recover_engines()
    assert len(services) == 2
    _, layout, _, loaded, contents, _ = [event for event in events if event[0] == "load"][-1]
    assert (layout, loaded, contents) == ("full", version, {"config.json": "{}"})
    assert services[1].engines.versions == [version] and events[-1] == "resume"
    runtime.shutdown()


def test_shared_runtime_recovers_a_restarted_remote_component_on_the_same_reservation(
    component, server, shared_resources, tmp_path
):
    from reef.inference.sglang.colocate import build_colocated_serving

    url, events, _, services = component
    runtime = build_colocated_serving(
        shared_resources,
        ReleaseFormat("full"),
        {
            "remote": {"url": url},
            "num_gpus": 2,
            "tensor_parallel_size": 2,
            "options": {},
            "adapter_capacity": 1,
            "executor": "ray",
            "executor_options": {},
            "inference_timeout_s": 30,
            "inference_handler_config": {},
        },
        model_path="model",
        environ=TOKENS,
    )
    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    published = runtime.activate_checkpoint(Artifact.local(weights))
    previous_instance = server.engine_instance
    restart(server)
    runtime.recover_engines()
    assert server.engine_instance != previous_instance
    assert server.resources is shared_resources
    assert len(services) == 2 and services[-1].engines.versions == [published]
    assert runtime.current_runtime_load_id() == published
    assert runtime.inference_admission_status["open"] and server.memory_state == "resident"
    with runtime.training_phase():
        assert server.memory_state == "released"
    assert services[-1].engines.versions == [published]
    runtime.shutdown()
    assert "release" not in events
