"""The remote inference component and its control-plane client, over real HTTP with fake engines."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from aiohttp import web

from reef.artifact import Artifact
from reef.inference.remote import server as server_module
from reef.inference.remote.config import CONTROL_PROTOCOL, RemoteInferenceSettings
from reef.inference.remote.runtime import RemoteRuntimeFactory
from reef.inference.remote.server import RemoteInferenceServer
from reef.inference.remote.service import RemoteControl, RemoteInferenceService
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.runtime.deployment import (
    WEIGHT_FILES_PROTOCOL,
    InferenceConnection,
    InferenceService,
    RuntimeConfigError,
    RuntimeRegistry,
    runtime_factory_for,
)
from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import InferenceBackend, WeightFiles

CONTROL_TOKEN = "control-secret"
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


class FakeEngines(InferenceBackend):
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

    def recover(self):
        self.events.append("recover")

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

    def prepare_weight_transfer(self, connection):
        self.events.append("fence")

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
def component(tmp_path, monkeypatch):
    """A running component whose engines are fakes, and the events they record."""
    events: list = []
    with serving(engine_app()) as engine_url:
        services = []

        def create_service(name, config):
            services.append(FakeService({"backend": name, **config}, events, engine_url))
            return services[-1]

        monkeypatch.setattr(server_module, "inference_service_for", create_service)
        monkeypatch.setattr(server_module, "reserve_model_gpus", lambda layout: FakeReservation(layout, events))
        server = RemoteInferenceServer(
            engine="sglang", control_token=CONTROL_TOKEN, inference_token=INFERENCE_TOKEN, work_dir=tmp_path / "work"
        )
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


def test_service_starts_engines_and_publishes_weight_files_through_the_component(component, tmp_path):
    url, events, work_dir, _ = component
    service = started_service(url)
    assert service.supported_transfer_protocols == (WEIGHT_FILES_PROTOCOL,)
    assert events == [
        ("start", {"backend": "sglang", "model_path": "Qwen/Qwen3-8B"}, "placement-2"),
        "health",
        "fence",
    ]
    connection = InferenceConnection(CONTROL_PROTOCOL, None)
    service.prepare_weight_transfer(connection)
    backend = service.backend(connection)
    assert backend.inference_url() == url
    backend.initialize_version("inc:1")
    assert backend.runtime_load_ids() == ["inc:1"]
    for operation in (backend.pause, backend.resume, backend.recover, backend.abort):
        operation()
    assert events[-4:] == ["pause", "resume", "recover", "abort"]

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
    for token in (None, CONTROL_TOKEN):
        request = urllib.request.Request(f"{url}/generate", data=b"{}", method="POST")
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        with pytest.raises(urllib.error.HTTPError, match="401"):
            urllib.request.urlopen(request, timeout=30)
    assert "pause" not in events


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


class Coordinator(CoordinatorClient):
    def health(self):
        return {"ok": True}

    def prepare_training_step(self, batch, method, algorithm_state, scheduling):
        raise AssertionError("unused")

    def execute_training_job(self, payload):
        raise AssertionError("unused")

    def update_serving_weights(self, training_job_id):
        raise AssertionError("unused")

    def acknowledge_training_commit(self, training_job_id):
        raise AssertionError("unused")

    def reject_training_candidate(self, training_job_id):
        raise AssertionError("unused")


def test_runtime_sends_requests_to_the_component_with_the_inference_token(tmp_path):
    runtime = RuntimeRegistry().build(
        {
            "type": "remote",
            "control": Coordinator(),
            "inference_url": "https://component.example",
            "inference_token_env": "TOKEN",
        },
        model_path="Qwen/Qwen3-8B",
        environ={"TOKEN": INFERENCE_TOKEN},
    )
    handler = runtime.inference_handler
    assert isinstance(handler, SGLangInferenceHandler)
    arguments = handler._post_arguments(Artifact.local(tmp_path), "/generate", {"input_ids": [1]})
    assert arguments["url"] == "https://component.example/generate"
    assert arguments["headers"]["Authorization"] == f"Bearer {INFERENCE_TOKEN}"


@pytest.mark.parametrize(
    ("config", "environ", "match"),
    [
        ({"engine": "my.engines:create"}, {"TOKEN": "x"}, "explicit inference handler factory"),
        ({}, {}, "environment variable TOKEN"),
        ({"engine": " "}, {"TOKEN": "x"}, "runtime.engine must be a non-empty string"),
    ],
)
def test_runtime_refuses_an_unknown_dialect_or_a_missing_token(config, environ, match):
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeRegistry().build(
            {
                "type": "remote",
                "control": Coordinator(),
                "inference_url": "https://component.example",
                "inference_token_env": "TOKEN",
                **config,
            },
            model_path="Qwen/Qwen3-8B",
            environ=environ,
        )


def test_factory_is_registered_for_the_remote_kind():
    assert isinstance(runtime_factory_for("remote"), RemoteRuntimeFactory)
