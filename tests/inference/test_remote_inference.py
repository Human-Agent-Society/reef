"""The remote inference component and its control-plane client, over real HTTP with fake engines."""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import yaml
from aiohttp import ClientSession, web
from tests.support.inference_qualification import InferenceQualification, OutputOracle, qualify_inference

from reef.artifact import BASE_FILE, Artifact, BaseRoot
from reef.core.errors import ReefError, RuntimeConfigError
from reef.core.weight_layout import ADAPTER_DIR, ArtifactFormat, WeightFiles
from reef.inference import deployment as deployment_module
from reef.inference.checkpoint_slots import CheckpointSlotsRuntime
from reef.inference.deployment import InferenceConnection, InferenceService
from reef.inference.http import ProviderRequestHeaders
from reef.inference.interfaces import (
    FixedEngineClients,
    InferenceArtifact,
    InferenceBackend,
    RuntimeContractError,
    full_weight_name,
)
from reef.inference.registry import RuntimeRegistry, runtime_factory_for
from reef.inference.remote import server as server_module
from reef.inference.remote.config import RemoteInferenceSettings
from reef.inference.remote.runtime import ComponentEngines, RemoteServingRuntimeFactory
from reef.inference.remote.server import RemoteInferenceServer
from reef.inference.remote.service import RemoteControl, RemoteInferenceBackend, RemoteInferenceService
from reef.inference.residency import AdapterCapacityExhaustedError, adapter_name
from reef.inference.serving import AdapterServingRuntime, FullWeightServingRuntime
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.vllm.chat import VLLMInferenceHandler

CONTROL_TOKEN = "control-secret"
ADAPTER = ArtifactFormat("adapter", 16, ("all",))
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


def engine_app(services=None) -> web.Application:
    """An engine endpoint: echoes generate requests and streams three chunks."""

    async def generate(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if payload.get("refuse") == "http":
            return web.json_response({"error": {"message": "invalid sampling value"}}, status=400)
        if payload.get("refuse") == "server":
            return web.json_response({"error": {"message": "generation failed"}}, status=500)
        if payload.get("refuse") == "sse":
            return web.Response(
                body=b'data: {"error":{"message":"invalid sampling value"}}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )
        if services and (payload.get("contract") or "sampling_params" in payload):
            engine = services[-1].engines
            marker = engine.marker_for(payload.get("lora_path"))
            cached = engine.use_cache()
            engine.generation_started.set()
            if payload.get("hold") and not await asyncio.to_thread(engine.release_generation.wait, 10):
                raise RuntimeError("toy generation was not released")
            # Three tokens: one before the hold and two after it, so a weight change between them shows.
            tokens = [marker]
            while len(tokens) < 3:
                tokens.append(engine.marker_for(payload.get("lora_path")))
            text = "".join(tokens)
            engine.generation_done.set()
            token_ids = [ord(character) for character in text]
            return web.json_response(
                {
                    "text": text,
                    "meta_info": {
                        "finish_reason": "stop",
                        "cached_tokens": cached,
                        "completion_tokens": len(token_ids),
                        "output_token_logprobs": [[-0.1, token] for token in token_ids],
                    },
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "text": text,
                            "token_ids": token_ids,
                            "logprobs": {
                                "content": [{"token": f"token_id:{token}", "logprob": -0.1} for token in token_ids]
                            },
                        }
                    ],
                }
            )
        return web.json_response({"payload": payload, "authorization": request.headers.get("Authorization")})

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index in range(3):
            await response.write(f"data: {index}\n\n".encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.router.add_post("/inference/v1/generate", generate)
    app.router.add_get("/stream", stream)
    return app


class FakeEngines(InferenceBackend):
    """Toy engines that ``engine_app`` answers for; each control operation appends to ``events``.

    Values: ``versions`` holds the one runtime load ID that every engine reports. ``marker`` is the text of the
    full weights' ``marker.txt``, "base" before a load, and ``adapter_markers`` maps each loaded adapter name to
    its marker. A toy generation of three tokens repeats the marker of the weights that it read for each token.
    With the retracting pause the engines keep a prefix cache: ``cache`` counts the generations in it, and a pause
    empties it. ``paused`` is true from ``pause`` to ``resume``.

    Purpose: ``FakeService`` holds one, and a test controls it. A load sets ``entered_load`` and waits for
    ``release_load``. A generation sets ``generation_started``, waits for ``release_generation`` when it asks to
    ``hold``, and sets ``generation_done`` at its end. With ``fail_load`` each load raises after it changed the
    markers; with ``ignore_load`` a load keeps the markers and still reports its load ID, as engines that
    acknowledge weights that they do not serve.
    """

    def __init__(self, engine_url: str, events: list, *, retracting: bool = False):
        self.engine_url = engine_url
        self.events = events
        self.versions = ["base:0"]
        self.paused = False
        self.retracting = retracting
        self.cache = 0
        self.release_load = threading.Event()
        self.release_load.set()
        self.entered_load = threading.Event()
        self.marker = "base"
        self.adapter_markers = {}
        self.generation_started = threading.Event()
        self.generation_done = threading.Event()
        self.release_generation = threading.Event()
        self.fail_load = False
        self.ignore_load = False

    def marker_for(self, adapter):
        return self.marker if adapter is None else self.adapter_markers[adapter]

    def use_cache(self) -> int:
        """The cached generations that a new generation reuses; the cache then holds the new one too."""
        cached = self.cache
        if self.retracting:
            self.cache += 1
        return cached

    def initialize_version(self, runtime_load_id):
        self.versions = [runtime_load_id]

    def inference_url(self):
        return self.engine_url

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")
        self.paused = True
        self.cache = 0

    def resume(self):
        self.events.append("resume")
        self.paused = False

    def recover(self):
        self.events.append("recover")
        return 1

    def abort(self):
        self.events.append("abort")

    def unload_adapter(self, name):
        self.events.append(("unload", name))
        self.adapter_markers.pop(name, None)

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        self.entered_load.set()
        assert self.release_load.wait(timeout=30)
        contents = {path.name: path.read_text() for path in files.path.iterdir()}
        if not self.ignore_load:
            if adapter_name is None:
                self.marker = contents.get("marker.txt", "base")
            else:
                self.adapter_markers[adapter_name] = contents.get("marker.txt", "base")
        self.events.append(("load", files.layout, adapter_name, runtime_load_id, contents, files.path))
        if self.fail_load:
            raise RuntimeError("partial toy load")
        if runtime_load_id is not None:
            self.versions = [runtime_load_id]


class FakeService(InferenceService):
    """The engine service that the component's server starts in a test, with ``FakeEngines`` as its engines.

    Values: ``config`` is the start input with its ``backend``; each operation appends to ``events``. A
    ``pause_mode`` of "retract" gives engines with a prefix cache. Purpose: the ``component`` fixture makes one
    for each engine start and lists them in start order.
    """

    def __init__(self, config, events, engine_url):
        self.config = config
        self.events = events
        self.engines = FakeEngines(engine_url, events, retracting=config.get("pause_mode") == "retract")

    @classmethod
    def from_config(cls, config):
        raise AssertionError("unused")

    def start(self, resources):
        self.events.append(("start", dict(self.config), resources.inference_placement))
        return InferenceConnection(None)

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
def engine_name():
    """The built-in engine of the component's server; a test parametrizes ``engine_name`` to select the other one."""
    return "sglang"


@pytest.fixture
def server(tmp_path, engine_name):
    """The component's server, which a test can restart (``restart``)."""
    return RemoteInferenceServer(
        engine=engine_name, control_token=CONTROL_TOKEN, inference_token=INFERENCE_TOKEN, work_dir=tmp_path / "work"
    )


def restart(server: RemoteInferenceServer) -> None:
    """Make ``server`` act as a new component process at the same URL: another instance, which runs no engines."""
    server.stop_stack()
    server.instance_id = "restarted-process"


@pytest.fixture
def component(server, tmp_path, monkeypatch):
    """A running component whose engines are fakes, and the events they record."""
    events: list = []
    services = []
    with serving(engine_app(services)) as engine_url:

        def create_service(name, config):
            services.append(FakeService({"backend": name, **config}, events, engine_url))
            return services[-1]

        monkeypatch.setattr(server_module, "inference_service_for", create_service)
        monkeypatch.setattr(deployment_module, "reserve_model_gpus", lambda layout: FakeReservation(layout, events))
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
    assert connection == InferenceConnection(None)
    return service


def pinned_service(component, server, *, sequence=1, layout="full", timeout=2):
    """A CPU-only component with the production pin and owner-control protocol enabled."""
    service = RemoteInferenceService(
        RemoteInferenceSettings(url=component[0], startup_timeout_s=10),
        engine=server.engine,
        engine_config={"model_path": "toy"},
        num_gpus=1,
        control_token=CONTROL_TOKEN,
        request_pinning=True,
        weight_layout=layout,
        drain_timeout_s=timeout,
    )
    service.set_owner_sequence(sequence)
    service.start()
    return service, service.backend(InferenceConnection(None))


def load_marker(backend, tmp_path, marker, version, *, adapter=None):
    """Load distinguishable toy weights; emitted text comes from these files, not from the reported load ID."""
    directory = tmp_path / marker
    directory.mkdir(exist_ok=True)
    (directory / "marker.txt").write_text(marker)
    backend.load_weight_files(
        WeightFiles(directory, "full" if adapter is None else "adapter"), adapter_name=adapter, runtime_load_id=version
    )


def pinned_request(url, engine, name, *, adapter=None, hold=False, **options):
    path = "/generate" if engine == "sglang" else "/inference/v1/generate"
    payload = {"contract": True, "hold": hold, **options}
    if adapter is not None:
        payload["lora_path"] = adapter
    request = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {INFERENCE_TOKEN}",
            "Content-Type": "application/json",
            "x-reef-served-model": name,
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
def test_pinned_receiver_refuses_stale_routes_during_the_publication_gap(component, server, tmp_path):
    backend = pinned_service(component, server)[1]
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"))["text"] == "AAA"
    backend.pause()
    load_marker(backend, tmp_path, "B", "owner:7")
    backend.resume()
    # The old row is still cached and no new row has published yet: the receiver must refuse it.
    with pytest.raises(urllib.error.HTTPError) as refused:
        pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"))
    assert refused.value.code == 421
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:7"))["text"] == "BBB"
    with pytest.raises(RuntimeError, match="cannot be relabelled"):
        backend.initialize_version("owner:6")


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
def test_pinned_pause_waits_for_a_generation_before_mutating_weights(component, server, tmp_path):
    backend = pinned_service(component, server)[1]
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    engine = component[3][-1].engines
    with ThreadPoolExecutor(2) as pool:
        generation = pool.submit(
            pinned_request, component[0], server.engine, full_weight_name("toy", "owner:6"), hold=True
        )
        assert engine.generation_started.wait(5)
        pause = pool.submit(backend.pause)
        try:
            # Pause is blocked on the actual upstream generation, so a mutation cannot overtake it.
            with pytest.raises(TimeoutError):
                pause.result(timeout=0.1)
        finally:
            engine.release_generation.set()
        assert generation.result(timeout=5)["text"] == "AAA"
        pause.result(timeout=5)
    load_marker(backend, tmp_path, "B", "owner:7")
    backend.resume()
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:7"))["text"] == "BBB"


def test_adapter_pin_checks_residency_and_the_native_adapter_selection(component, server, tmp_path):
    backend = pinned_service(component, server, layout="adapter")[1]
    first, second = adapter_name("math", "owner:6"), adapter_name("code", "owner:7")
    load_marker(backend, tmp_path, "A", "owner:6", adapter=first)
    load_marker(backend, tmp_path, "B", "owner:7", adapter=second)
    backend.resume()
    # The engines now report load owner:7, but the resident immutable adapter A still produces A's output.
    assert pinned_request(component[0], server.engine, first, adapter=first)["text"] == "AAA"
    with pytest.raises(urllib.error.HTTPError) as refused:
        pinned_request(component[0], server.engine, first, adapter=second)
    assert refused.value.code == 421
    backend.unload_adapter(first)
    with pytest.raises(urllib.error.HTTPError):
        pinned_request(component[0], server.engine, first, adapter=first)
    assert pinned_request(component[0], server.engine, second, adapter=second)["text"] == "BBB"


def test_a_partial_load_invalidates_old_bindings_and_requires_stack_replacement(component, server, tmp_path):
    service, backend = pinned_service(component, server)
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    backend.pause()
    component[3][-1].engines.fail_load = True
    with pytest.raises(RuntimeError, match="partial toy load"):
        load_marker(backend, tmp_path, "B", "owner:7")
    with pytest.raises(RuntimeError, match="recovery is required"):
        backend.resume()
    with pytest.raises(urllib.error.HTTPError):
        pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"))
    assert backend.recover() == 1
    assert len(component[3]) == 2
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"))["text"] == "AAA"
    service.check_health()


def test_new_owner_fences_a_late_load_and_resume_before_replacing_the_stack(component, server, tmp_path):
    old, backend = pinned_service(component, server, timeout=0.05)
    engine = component[3][-1].engines
    engine.release_load.clear()
    with ThreadPoolExecutor(1) as pool:
        delayed = pool.submit(load_marker, backend, tmp_path, "A", "old:6")
        try:
            assert engine.entered_load.wait(5)
            newer = RemoteControl(component[0], CONTROL_TOKEN)
            newer.owner_sequence = 2
            assert newer.call("GET", "health")["binding_uncertain"] is True
            with pytest.raises(RuntimeError, match="earlier control operation is still running"):
                newer.call("POST", "resume")
            assert server.lock.locked() and len(component[3]) == 1
        finally:
            engine.release_load.set()
        with pytest.raises(RuntimeError, match="superseded"):
            delayed.result(timeout=5)
    with pytest.raises(RuntimeError, match="superseded"):
        backend.resume()
    with pytest.raises(RuntimeError, match="superseded"):
        old.close()
    current = pinned_service(component, server, sequence=2)[1]
    load_marker(current, tmp_path, "B", "new:7")
    current.resume()
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "new:7"))["text"] == "BBB"


def test_an_unpinned_start_cannot_disable_pinning_that_began_while_it_waited_for_the_lock(
    component, server, tmp_path, monkeypatch
):
    backend = started_service(component[0]).backend(InferenceConnection(None))
    engine = component[3][-1].engines
    engine.release_load.clear()
    queued = threading.Semaphore(0)
    control_lock = server.control_lock

    def announced(request):
        # The handler takes its place in the lock's queue in the event-loop step that makes this call.
        queued.release()
        return control_lock(request)

    late = RemoteControl(component[0], CONTROL_TOKEN)
    late.owner_sequence = 1
    unpinned_start = {"backend": server.engine, "config": {"model_path": "toy"}, "num_gpus": 1}
    with ThreadPoolExecutor(3) as pool:
        load = pool.submit(load_marker, backend, tmp_path, "A", "base:1")
        try:
            assert engine.entered_load.wait(5)
            monkeypatch.setattr(server, "control_lock", announced)
            pinned = pool.submit(pinned_service, component, server)
            assert queued.acquire(timeout=5)
            # Pinning is still off, so this start passes the check that precedes the lock.
            unpinned = pool.submit(late.call, "PUT", "engines", payload=unpinned_start)
            assert queued.acquire(timeout=5)
        finally:
            engine.release_load.set()
        load.result(timeout=5)
        pinned.result(timeout=5)
        with pytest.raises(RuntimeError, match="409: restart the component before disabling request pinning"):
            unpinned.result(timeout=5)
    assert server.pins.enabled and len(component[3]) == 2


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
def test_pinned_components_require_a_pin_and_forward_only_generation(component, server, tmp_path):
    backend = pinned_service(component, server)[1]
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    with pytest.raises(urllib.error.HTTPError) as refused:
        pinned_request(component[0], server.engine, "")
    assert refused.value.code == 421
    request = urllib.request.Request(
        component[0] + "/unsupported", headers={"Authorization": f"Bearer {INFERENCE_TOKEN}"}
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    assert refused.value.code == 404


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
@pytest.mark.parametrize(
    "selection", [{"model": "another-model"}, {"lora_path": "adapter"}, {"lora_id": 7}, {"lora_name": "adapter"}]
)
def test_a_full_weight_pin_refuses_conflicting_native_selections(component, server, tmp_path, selection):
    backend = pinned_service(component, server)[1]
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    with pytest.raises(urllib.error.HTTPError) as refused:
        pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"), **selection)
    assert refused.value.code == 421
    assert not component[3][-1].engines.generation_started.is_set()
    assert (
        pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"), model="toy")["text"] == "AAA"
    )


def test_disconnected_client_keeps_its_pin_until_native_generation_finishes(component, server, tmp_path):
    backend = pinned_service(component, server)[1]
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    engine = component[3][-1].engines

    async def run():
        try:
            async with ClientSession() as client:
                request = asyncio.create_task(
                    client.post(
                        component[0] + "/generate",
                        json={"contract": True, "hold": True},
                        headers={
                            "Authorization": f"Bearer {INFERENCE_TOKEN}",
                            "x-reef-served-model": full_weight_name("toy", "owner:6"),
                        },
                    )
                )
                assert await asyncio.to_thread(engine.generation_started.wait, 5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
            pause = asyncio.create_task(asyncio.to_thread(backend.pause))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(pause), 0.1)
            assert server.pins.active and not engine.generation_done.is_set()
            engine.release_generation.set()
            await asyncio.wait_for(pause, 5)
            assert not server.pins.active
        finally:
            engine.release_generation.set()

    asyncio.run(run())
    load_marker(backend, tmp_path, "B", "owner:7")
    backend.resume()
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:7"))["text"] == "BBB"


@pytest.mark.parametrize("refusal", ["http", "sse"])
def test_explicit_engine_refusals_do_not_invalidate_a_binding(component, server, tmp_path, refusal):
    service, backend = pinned_service(component, server)
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    request = urllib.request.Request(
        component[0] + "/generate",
        data=json.dumps({"refuse": refusal}).encode(),
        headers={
            "Authorization": f"Bearer {INFERENCE_TOKEN}",
            "Content-Type": "application/json",
            "x-reef-served-model": full_weight_name("toy", "owner:6"),
        },
    )
    if refusal == "http":
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=5)
        assert refused.value.code == 400
    else:
        with urllib.request.urlopen(request, timeout=5) as response:
            assert b"invalid sampling value" in response.read()
    service.check_health()
    assert server.pins.uncertain is False
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"))["text"] == "AAA"


def test_a_full_adapter_capacity_keeps_the_incumbent_scenario_available(component, server, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    runtime = RuntimeRegistry().build(
        serving_config(component[0], adapter_capacity=1), model_path="toy", environ=TOKENS
    )
    runtime.set_owner_sequence(1)
    runtime.open()
    path = saved_adapter(tmp_path / "math")
    (path / ADAPTER_DIR / "marker.txt").write_text("A")
    artifact = Artifact.local(path)
    try:
        runtime.activate_checkpoint(artifact, scenario="math")
        starts = len(component[3])
        with pytest.raises(AdapterCapacityExhaustedError):
            runtime.activate_checkpoint(Artifact.local(saved_adapter(tmp_path / "code")), scenario="code")
        assert len(component[3]) == starts and not runtime.recovery_pending
        served = runtime.served_artifact(artifact.artifact_id, "math")
        assert served is not None
        assert (
            pinned_request(component[0], server.engine, served.model_name, adapter=served.model_name)["text"] == "AAA"
        )
    finally:
        runtime.shutdown()


def test_a_server_error_cannot_prove_that_native_generation_stopped(component, server, tmp_path):
    service, backend = pinned_service(component, server)
    load_marker(backend, tmp_path, "A", "owner:6")
    backend.resume()
    with pytest.raises(urllib.error.HTTPError) as failed:
        pinned_request(component[0], server.engine, full_weight_name("toy", "owner:6"), refuse="server")
    assert failed.value.code == 500
    with pytest.raises(http.client.IncompleteRead):
        failed.value.read()
    with pytest.raises(RuntimeError, match="binding recovery"):
        service.check_health()


@pytest.mark.parametrize(
    "original", [FakeEngines.initialize_version, FakeEngines.load_weight_files], ids=lambda method: method.__name__
)
def test_adapter_restore_retries_every_scenario_after_a_partial_failure(
    component, server, tmp_path, monkeypatch, original
):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    runtime = RuntimeRegistry().build(
        serving_config(component[0], adapter_capacity=2), model_path="toy", environ=TOKENS
    )
    runtime.set_owner_sequence(1)
    runtime.open()
    artifacts = {}
    try:
        for scenario, marker in (("math", "A"), ("code", "B")):
            directory = tmp_path / scenario
            files = directory / "adapter"
            files.mkdir(parents=True)
            (files / "marker.txt").write_text(marker)
            (files / "adapter_config.json").write_text('{"r": 16}')
            artifacts[scenario] = Artifact.local(directory)
            runtime.activate_checkpoint(artifacts[scenario], scenario=scenario)
        failed = False

        def once(self, *args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("interrupted restore")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(FakeEngines, original.__name__, once)
        runtime.backend.abort()
        with pytest.raises((ReefError, RuntimeError)):
            runtime.recover_engines()
        assert runtime.recovery_pending
        assert all(
            runtime.served_artifact(artifact.artifact_id, scenario) is None for scenario, artifact in artifacts.items()
        )
        runtime.recover_engines()
        assert not runtime.recovery_pending
        starts = len(component[3])
        runtime.recover_engines()
        assert len(component[3]) == starts, "a stale health failure must not restart an already restored stack"
        for scenario, marker in (("math", "A"), ("code", "B")):
            served = runtime.served_artifact(artifacts[scenario].artifact_id, scenario)
            assert served is not None
            assert (
                pinned_request(component[0], server.engine, served.model_name, adapter=served.model_name)["text"]
                == marker * 3
            )
    finally:
        runtime.shutdown()


def test_owner_change_during_drain_prevents_the_old_adapter_unload(component, server, tmp_path):
    backend = pinned_service(component, server, layout="adapter")[1]
    name = adapter_name("math", "owner:6")
    load_marker(backend, tmp_path, "A", "owner:6", adapter=name)
    backend.resume()
    engine = component[3][-1].engines
    with ThreadPoolExecutor(2) as pool:
        generation = pool.submit(pinned_request, component[0], server.engine, name, adapter=name, hold=True)
        assert engine.generation_started.wait(5)
        unload = pool.submit(backend.unload_adapter, name)
        try:
            with pytest.raises(TimeoutError):
                unload.result(timeout=0.1)
            new_owner = RemoteControl(component[0], CONTROL_TOKEN)
            new_owner.owner_sequence = 2
            new_owner.call("GET", "health")
        finally:
            engine.release_generation.set()
        generation.result(timeout=5)
        with pytest.raises(RuntimeError, match="superseded"):
            unload.result(timeout=5)
    assert ("unload", name) not in component[1]


@pytest.mark.parametrize("layout", ["full", "adapter"])
@pytest.mark.parametrize("failure", ["pause", "health"])
def test_pinned_runtime_recovers_without_repeating_a_broken_pause(
    component, server, tmp_path, monkeypatch, layout, failure
):
    # With prefix sharing, the pause is also the cache clear that a load depends on.
    config = serving_config(
        component[0],
        artifact_format=(ArtifactFormat("full") if layout == "full" else ADAPTER).to_dict(),
        prefix_sharing=True,
    )
    runtime = RuntimeRegistry().build(config, model_path="toy", environ=TOKENS)
    runtime.set_owner_sequence(1)
    runtime.open()
    root = tmp_path / "artifact"
    files = root if layout == "full" else root / "adapter"
    files.mkdir(parents=True)
    (files / "marker.txt").write_text("A")
    if layout == "adapter":
        (files / "adapter_config.json").write_text('{"r": 16}')
    artifact = Artifact.local(root)

    def answer():
        served = runtime.served_artifact(artifact.artifact_id, "math")
        assert served is not None
        adapter = served.model_name if layout == "adapter" else None
        return pinned_request(component[0], server.engine, served.model_name, adapter=adapter)["text"]

    try:
        runtime.activate_checkpoint(artifact, scenario="math")
        previous = len(component[3])

        def broken():
            raise RuntimeError("dead toy engine")

        if failure == "pause":
            monkeypatch.setattr(component[3][-1].engines, "pause", broken)
            with pytest.raises(RuntimeError, match="dead toy engine"):
                runtime.backend.pause()
        else:
            monkeypatch.setattr(component[3][-1], "poll", broken)
            with pytest.raises(RuntimeError, match="binding recovery"):
                runtime.engines.check_health()
        # The route is still published, but admission stays closed until new engines replace the stack.
        with pytest.raises(urllib.error.HTTPError) as refused:
            answer()
        assert refused.value.code == 421 and server.pins.uncertain
        runtime.recover_engines()
        assert len(component[3]) == previous + 1
        assert answer() == "AAA"
        assert not server.pins.uncertain
    finally:
        runtime.shutdown()


@pytest.mark.parametrize(
    ("target", "original"),
    [(FakeService, FakeService.start), (FakeEngines, FakeEngines.initialize_version)],
    ids=["start", "initialize"],
)
def test_pinned_runtime_open_can_retry_a_partial_start(component, server, monkeypatch, target, original):
    config = serving_config(component[0], artifact_format=ArtifactFormat("full").to_dict())
    runtime = RuntimeRegistry().build(config, model_path="toy", environ=TOKENS)
    runtime.set_owner_sequence(1)
    attempts = 0

    def once(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary startup failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(target, original.__name__, once)
    with pytest.raises(RuntimeError, match="temporary startup failure"):
        runtime.open()
    try:
        runtime.open()
        assert attempts == 2
    finally:
        runtime.shutdown()


def test_cancelled_control_call_holds_the_lock_until_its_thread_finishes(server):
    entered, release, following = threading.Event(), threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)

    async def run():
        async def mutation(function):
            async with server.lock:
                await server.control_call(function)

        first = asyncio.create_task(mutation(blocked))
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        second = asyncio.create_task(mutation(following.set))
        try:
            await asyncio.sleep(0.05)
            first.cancel()
            await asyncio.sleep(0.05)
            assert not following.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert following.is_set()

    asyncio.run(run())


def test_upload_from_a_superseded_owner_cannot_mutate_the_new_stack(component, server, tmp_path):
    pinned_service(component, server)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        entry = tarfile.TarInfo("marker.txt")
        entry.size = 1
        tar.addfile(entry, io.BytesIO(b"A"))
    data = archive.getvalue()

    async def run():
        began, finish = asyncio.Event(), asyncio.Event()

        async def chunks():
            yield data[:512]
            began.set()
            await finish.wait()
            yield data[512:]

        async with ClientSession() as client:
            upload = asyncio.create_task(
                client.put(
                    component[0] + "/reef/control/v1/weights?layout=full&runtime_load_id=old:6",
                    data=chunks(),
                    headers={"Authorization": f"Bearer {CONTROL_TOKEN}", "x-reef-owner-sequence": "1"},
                )
            )
            await began.wait()
            try:
                current = (await asyncio.to_thread(pinned_service, component, server, sequence=2))[1]
            finally:
                finish.set()
            async with await upload as response:
                assert response.status == 412
            await asyncio.to_thread(load_marker, current, tmp_path, "B", "new:7")
            await asyncio.to_thread(current.resume)

    asyncio.run(run())
    assert pinned_request(component[0], server.engine, full_weight_name("toy", "new:7"))["text"] == "BBB"
    assert not any(isinstance(event, tuple) and event[0] == "load" and event[3] == "old:6" for event in component[1])


class MarkerTokenizer:
    """A CPU tokenizer whose generated token IDs encode the letters in the loaded toy weight files."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return [10, 11] if tokenize else "prompt"

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class MarkerOracle(OutputOracle):
    """Name the toy artifact of an answer from its text, not from the engine's reported version.

    Values: ``ids`` maps "AAA" to the ID of the first of two artifacts and "BBB" to the second's, because a toy
    generation repeats the marker of its weights three times. ``artifact_id`` raises ``KeyError`` for any other
    text, which neither artifact made.
    """

    def __init__(self, artifacts):
        self.ids = dict(zip(("AAA", "BBB"), (artifact.artifact_id for artifact in artifacts), strict=True))

    def artifact_id(self, response):
        return self.ids[response["choices"][0]["message"]["content"]]


def test_weight_serving_refuses_a_component_service_that_requests_no_request_protection():
    # The low-level service without the protection is for a trainer's rollout engines only.
    service = RemoteInferenceService(
        RemoteInferenceSettings(url="http://127.0.0.1:9"),
        engine="sglang",
        engine_config={"model_path": "toy"},
        num_gpus=1,
        control_token=CONTROL_TOKEN,
    )
    handler = SGLangInferenceHandler("http://127.0.0.1:9", model_path="toy", tokenizer=MarkerTokenizer())
    with pytest.raises(RuntimeContractError, match="request_pinning"):
        FullWeightServingRuntime(ComponentEngines(service), handler, model_path="toy")
    with pytest.raises(RuntimeContractError, match="request_pinning"):
        AdapterServingRuntime(ComponentEngines(service), handler, adapter_capacity=1, model_path="toy")


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"request_pinning": "yes"}, "request_pinning must be a bool"),
        ({"weight_layout": "shards"}, "weight_layout must be one of"),
        ({"drain_timeout_s": 0}, "drain_timeout_s must be a positive, finite"),
        ({"drain_timeout_s": float("inf")}, "drain_timeout_s must be a positive, finite"),
    ],
)
def test_a_component_service_refuses_an_invalid_pinning_configuration(values, match):
    with pytest.raises(ValueError, match=match):
        RemoteInferenceService(
            RemoteInferenceSettings(url="http://127.0.0.1:9"),
            engine="sglang",
            engine_config={"model_path": "toy"},
            num_gpus=1,
            control_token=CONTROL_TOKEN,
            **values,
        )


@pytest.mark.parametrize("engine_name", ["sglang", "vllm"])
@pytest.mark.parametrize("ignore_load", [False, True], ids=["protected", "false-load-ack"])
def test_shared_qualification_checks_real_output_headers_and_records(
    component, server, tmp_path, monkeypatch, ignore_load
):
    # This case promotes on a healthy stack; recovery must preserve the previously resident toy weights.
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    if ignore_load:
        original_load = FakeEngines.load_weight_files

        def load(engines, files, **kwargs):
            engines.ignore_load = (files.path / "marker.txt").read_text() == "B"
            original_load(engines, files, **kwargs)

        monkeypatch.setattr(FakeEngines, "load_weight_files", load)
    service = RemoteInferenceService(
        RemoteInferenceSettings(url=component[0]),
        engine=server.engine,
        engine_config={"model_path": "toy"},
        num_gpus=1,
        control_token=CONTROL_TOKEN,
        request_pinning=True,
    )
    handler_type = SGLangInferenceHandler if server.engine == "sglang" else VLLMInferenceHandler
    handler = handler_type(
        component[0],
        model_path="toy",
        tokenizer=MarkerTokenizer(),
        request_headers=ProviderRequestHeaders(INFERENCE_TOKEN),
    )
    runtime = FullWeightServingRuntime(ComponentEngines(service), handler, model_path="toy", health_interval_s=1000)
    artifacts = []
    for marker in ("A", "B"):
        directory = tmp_path / marker
        directory.mkdir()
        (directory / "marker.txt").write_text(marker)
        artifacts.append(Artifact.local(directory))
    case = InferenceQualification(
        runtime,
        FixedEngineClients(handler),
        tuple(artifacts),
        MarkerOracle(artifacts),
        {"messages": [{"role": "user", "content": "identify your weights"}], "max_tokens": 3},
        f"toy-{server.engine}",
        ("full-weights", "buffered"),
    )
    if not ignore_load:
        report = asyncio.run(qualify_inference(case, tmp_path, samples_per_artifact=2))
        assert report["completed_requests"] >= 4
        assert report["artifacts"] == [artifact.artifact_id for artifact in artifacts]
    else:
        # Detect a load that acknowledged B's ID without changing the weights.
        with pytest.raises(RuntimeContractError, match="generated output"):
            asyncio.run(qualify_inference(case, tmp_path, samples_per_artifact=2))


def test_service_starts_engines_and_publishes_weight_files_through_the_component(component, tmp_path):
    url, events, work_dir, _ = component
    service = started_service(url)
    # The engines start paused; the control plane decides when they serve.
    assert events == [
        ("start", {"backend": "sglang", "model_path": "Qwen/Qwen3-8B"}, "placement-2"),
        "health",
        "pause",
    ]
    connection = InferenceConnection(None)
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

    # A Hugging Face snapshot directory links its files into the cache; the component gets the files themselves.
    weights = tmp_path / "hf"
    weights.mkdir()
    (tmp_path / "blob").write_text("{}")
    (weights / "config.json").symlink_to(tmp_path / "blob")
    backend.load_weight_files(WeightFiles(weights, "full"), adapter_name=None, runtime_load_id="inc:3")
    _, layout, name, version, contents, served = events[-1]
    assert (layout, name, version, contents) == ("full", None, "inc:3", {"config.json": "{}"})
    assert not served.exists()  # engines read full weights once
    (weights / "config.json").write_text('{"restored": true}')
    backend.load_weight_files(WeightFiles(weights, "full"), adapter_name=None, runtime_load_id="inc:3")
    assert events[-1][1:5] == ("full", None, "inc:3", {"config.json": '{"restored": true}'})
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


@pytest.mark.parametrize(
    ("engine_name", "config", "option"),
    [
        ("sglang", {"options": {"hicache-storage-backend": "file"}}, "hicache_storage_backend"),
        ("sglang", {"options": {"enable_lmcache": True}}, "enable_lmcache"),
        ("vllm", {"options": {"kv_transfer_config": {"kv_connector": "OffloadingConnector"}}}, "kv_transfer_config"),
        ("vllm", {"options": {"kv-offloading-size": 4}}, "kv_offloading_size"),
    ],
)
def test_component_refuses_protected_engines_that_keep_kv_outside_them(component, server, config, option):
    url, events = component[:2]
    control = RemoteControl(url, CONTROL_TOKEN)
    control.owner_sequence = 1
    payload = {"backend": server.engine, "config": {"model_path": "toy", **config}, "num_gpus": 1}
    # No pause clears that KV before a change of weights, so the component starts no engine.
    with pytest.raises(RuntimeError, match=f"400: .*cannot use these options: {option}$"):
        control.call("PUT", "engines", payload={**payload, "request_pinning": True})
    assert events == [] and not server.pins.enabled
    # Rollout engines and plain endpoints claim no binding to published weights: the same input starts them.
    control.call("PUT", "engines", payload=payload)
    assert events[0] == ("start", {"backend": server.engine, "model_path": "toy", **config}, "placement-1")


def test_health_reports_busy_engines_as_healthy_while_they_load_weights(component, tmp_path):
    url, events, _, services = component
    service = started_service(url)
    backend = service.backend(InferenceConnection(None))
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
    assert control.call("GET", "health") == {"ok": True, "instance": server.instance_id, "engines": False}
    service = started_service(url)
    assert service.instance == server.instance_id
    assert control.call("GET", "health") == {"ok": True, "instance": server.instance_id, "engines": True}
    assert not service.engines_lost()
    control.call("DELETE", "engines")
    # The component stopped the engines: the service has lost them although the process is the same.
    assert service.engines_lost()


def test_recovery_after_a_component_restart_starts_the_engines_again_at_the_initial_version(component, server):
    url, _, _, services = component
    service = started_service(url)
    backend = service.backend(InferenceConnection(None))
    backend.initialize_version("inc:1")
    backend.pause()
    backend.pause()
    restart(server)
    assert service.engines_lost()
    with pytest.raises(RuntimeError, match="restarted"):
        service.check_health()
    assert backend.recover() == 1
    assert len(services) == 2 and services[1].engines.versions == ["inc:1"]
    assert services[1].engines.paused
    backend.resume()
    assert not services[1].engines.paused
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
        # A string is no list of URLs, and each member is a URL as inference.remote.url is.
        ({"url": "https://host", "slot-urls": "https://other"}, "slot-urls must be a list of URLs"),
        ({"url": "https://host", "slot-urls": ["https://other", 7]}, r"slot-urls\[1\] must be a string"),
        ({"url": "https://host", "slot-urls": ["ftp://other"]}, r"slot-urls\[0\] must be an HTTP"),
        ({"url": "https://host", "slot-urls": ["https://user:pw@other"]}, r"slot-urls\[0\] .* without credentials"),
        # Each URL names its own component: none repeats the first URL or another one, also with a trailing slash.
        ({"url": "https://host/", "slot-urls": ["https://other", "https://host"]}, "it repeats https://host$"),
        ({"url": "https://host", "slot_urls": ["https://other/", "https://other"]}, "it repeats https://other$"),
    ],
)
def test_settings_refuse_invalid_sections(section, match):
    with pytest.raises(ValueError, match=match):
        RemoteInferenceSettings.from_mapping(section)


def test_settings_read_dashed_keys_and_normalize_the_url():
    settings = RemoteInferenceSettings.from_mapping(
        {"url": "https://host.example/", "control-token-env": "A", "inference-token-env": "B"}
    )
    assert (settings.url, settings.control_token_env, settings.inference_token_env, settings.slot_urls) == (
        "https://host.example",
        "A",
        "B",
        (),
    )
    slots = RemoteInferenceSettings.from_mapping(
        {"url": "https://host.example", "slot-urls": ["https://b.example/", "http://c.example:8000"]}
    )
    assert slots.slot_urls == ("https://b.example", "http://c.example:8000")
    assert RemoteInferenceSettings(url="https://host.example", slot_urls=("https://b.example/",)).slot_urls == (
        "https://b.example",
    )
    # Resolution stores the fields under their own names as JSON values: the tuple of URLs is a list.
    full = RemoteInferenceSettings.from_mapping(
        {
            "url": "https://host.example/",
            "slot-urls": ["https://b.example/", "http://c.example:8000"],
            "control-token-env": "A",
            "inference-token-env": "B",
            "startup-timeout-s": 90.5,
            "model-path": "org/model",
        }
    )
    assert full.to_dict() == {
        "url": "https://host.example",
        "control_token_env": "A",
        "inference_token_env": "B",
        "startup_timeout_s": 90.5,
        "model_path": "org/model",
        "slot_urls": ["https://b.example", "http://c.example:8000"],
    }
    assert settings.to_dict()["slot_urls"] == []
    # A deployment file holds the mapping unchanged, and the mapping reads back as the same settings.
    for value in (settings, slots, full):
        wire = value.to_dict()
        assert RemoteInferenceSettings.from_mapping(wire) == value
        assert json.loads(json.dumps(wire)) == wire
        assert yaml.safe_load(yaml.safe_dump(wire)) == wire
        assert RemoteInferenceSettings.from_mapping(yaml.safe_load(yaml.safe_dump(wire))) == value


def test_rollout_engines_refuse_slot_urls_before_they_read_a_token(monkeypatch):
    monkeypatch.delenv("REEF_REMOTE_CONTROL_TOKEN", raising=False)
    config = {
        "remote": {"url": "https://a.example", "slot-urls": ["https://b.example"]},
        "engine": "sglang",
        "engine_config": {"model_path": "toy"},
        "num_gpus": 1,
    }
    with pytest.raises(ValueError, match="rollout engines run on the one component"):
        RemoteInferenceService.from_config(config)
    # The same input without slot-urls builds the service of the one component.
    monkeypatch.setenv("REEF_REMOTE_CONTROL_TOKEN", CONTROL_TOKEN)
    service = RemoteInferenceService.from_config({**config, "remote": {"url": "https://a.example"}})
    assert service.settings.url == "https://a.example" and not service.request_pinning


def serving_config(url: str = "https://component.example", **values) -> dict:
    return {
        "type": "remote_serving",
        "remote": {"url": url, "startup-timeout-s": 30},
        "artifact_format": ADAPTER.to_dict(),
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
#: An ``inference.remote`` section with a second component.
SLOTS = {"url": "https://a.example", "slot-urls": ["https://b.example"]}


def saved_adapter(path):
    (path / ADAPTER_DIR).mkdir(parents=True)
    (path / ADAPTER_DIR / "adapter_config.json").write_text('{"r": 16}')
    return path


class CustomHandler(SGLangInferenceHandler):
    """A deployment's own handler factory: it changes nothing of the built-in handler, but it is another class.

    Purpose: the runtime factory refuses it, because only a built-in handler is known to name the selected weights
    on every request.
    """


@pytest.mark.parametrize(
    ("values", "environ", "match"),
    [
        # Only the built-in engines with their own handlers keep each request on its weights; nothing turns that off.
        ({"engine": "my.engines:create"}, TOKENS, "cannot serve published weights"),
        ({"inference_handler_factory": CustomHandler}, TOKENS, "built-in request handlers"),
        ({"prefix_sharing": "sometimes"}, TOKENS, "runtime.prefix_sharing must be a valid bool"),
        ({"request_pinning": False}, TOKENS, "unknown config fields: request_pinning"),
        ({}, {"REEF_REMOTE_CONTROL_TOKEN": CONTROL_TOKEN}, "REEF_REMOTE_INFERENCE_TOKEN"),
        ({}, {"REEF_REMOTE_INFERENCE_TOKEN": INFERENCE_TOKEN}, "REEF_REMOTE_CONTROL_TOKEN"),
        ({"engine": " "}, TOKENS, "runtime.engine must be a non-empty string"),
        ({"artifact_format": {"layout": "shards"}}, TOKENS, "layout must be 'full' or 'adapter'"),
        ({"num_gpus": 0}, TOKENS, "runtime.num_gpus must be a positive integer"),
        ({"tensor_parallel_size": 3}, TOKENS, "multiple of runtime.tensor_parallel_size"),
        ({"engine": "vllm"}, TOKENS, "vllm serves full weights only"),
        ({"adapter_capacity": 0}, TOKENS, "runtime.adapter_capacity must be a positive integer"),
        ({"inference_timeout_s": 0}, TOKENS, "runtime.inference_timeout_s must be positive"),
        ({"remote": {"url": "ftp://host"}}, TOKENS, "HTTP"),
        # No pause clears KV that an engine keeps outside itself.
        ({"options": {"enable-lmcache": True}}, TOKENS, "cannot use these options: enable_lmcache$"),
        (
            {
                "engine": "vllm",
                "num_gpus": 1,
                "artifact_format": ArtifactFormat("full").to_dict(),
                "options": {"kv-transfer-config": '{"kv_connector": "OffloadingConnector"}'},
            },
            TOKENS,
            "cannot use these options: kv_transfer_config$",
        ),
        # Further components serve full-weight checkpoints only, and the engine input of each one is checked.
        ({"remote": SLOTS}, TOKENS, "slot-urls adds components that each serve one full-weight checkpoint"),
        ({"remote": {**SLOTS, "slot-urls": "https://b.example"}}, TOKENS, "slot-urls must be a list of URLs"),
        (
            {"remote": SLOTS, "artifact_format": ArtifactFormat("full").to_dict(), "options": {"enable-lmcache": 1}},
            TOKENS,
            "cannot use these options: enable_lmcache$",
        ),
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
    component, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, work_dir, _ = component
    runtime = RuntimeRegistry().build(
        serving_config(url, adapter_capacity=2), model_path="Qwen/Qwen3-8B", environ=TOKENS
    )
    assert isinstance(runtime, AdapterServingRuntime)
    assert runtime.per_scenario_adapters and runtime.shared_by_training_scenarios
    # Building the runtime touches no engine; control opens it. The engines start paused, then serve the base
    # model under the runtime's first version.
    assert events == []
    runtime.set_owner_sequence(1)
    runtime.open()
    # No setting asked for it: the factory always has the component keep each request on its weights.
    assert server.pins.enabled
    assert started_engines(events[0]) == ("sglang", "Qwen/Qwen3-8B", 1, 2, "placement-2")
    assert events[1:4] == ["health", "pause", "resume"]
    version = runtime.activate_checkpoint(Artifact.local(saved_adapter(tmp_path / "job-1")), scenario="math")
    _, layout, name, loaded, contents, served = events[-2]
    assert (layout, name, loaded, contents) == (
        "adapter",
        adapter_name("math", version),
        version,
        {"adapter_config.json": '{"r": 16}'},
    )
    assert served.parent == work_dir and events[-1] == "resume"
    assert runtime.serving_adapter_runtime_load_id("math") == version
    arguments = runtime.inference_handler.post_arguments(
        InferenceArtifact("artifact"), "/generate", {"input_ids": [1]}
    )
    assert (arguments["url"], arguments["headers"]["Authorization"]) == (
        f"{url}/generate",
        f"Bearer {INFERENCE_TOKEN}",
    )
    assert isinstance(runtime.inference_handler, SGLangInferenceHandler)
    # The engines are the inference component's: a control that stops leaves them serving.
    runtime.shutdown()
    assert "close" not in events


def test_full_weight_runtime_loads_a_published_checkpoint_through_the_component(
    component, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, _, _ = component
    runtime = RuntimeRegistry().build(
        serving_config(url, artifact_format=ArtifactFormat("full").to_dict()),
        model_path="Qwen/Qwen3-8B",
        environ=TOKENS,
    )
    runtime.set_owner_sequence(1)
    runtime.open()
    assert server.pins.enabled
    assert started_engines(events[0]) == ("sglang", "Qwen/Qwen3-8B", 1, None, "placement-2")
    # Without prefix sharing the engines pause in place, which keeps no prefix cache.
    assert events[0][1]["pause_mode"] == "in_place"
    assert isinstance(runtime, FullWeightServingRuntime)
    assert not runtime.per_scenario_adapters and not runtime.shared_by_training_scenarios
    weights = tmp_path / "hf"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    artifact = Artifact.local(weights)
    version = runtime.activate_checkpoint(artifact)
    _, layout, name, loaded, contents, _ = events[-2]
    assert (layout, name, loaded, contents) == ("full", None, version, {"config.json": "{}"})
    # The route names this load, and the component admits requests under that name only.
    served = runtime.served_artifact(artifact.artifact_id, "math")
    assert served is not None and served.model_name == full_weight_name("Qwen/Qwen3-8B", version)
    assert server.pins.full_load_id == version
    runtime.shutdown()
    assert "close" not in events


def test_a_base_root_reaches_the_component_as_the_base_models_files_or_as_no_load(
    component, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, _, _ = component
    # The base model as the deployment resolves it: a Hugging Face snapshot directory, whose files are links.
    model = tmp_path / "snapshots" / "commit"
    model.mkdir(parents=True)
    (tmp_path / "blob").write_text("{}")
    (model / "config.json").symlink_to(tmp_path / "blob")
    root = tmp_path / "root"
    root.mkdir()
    (root / BASE_FILE).write_text(BaseRoot({"model": str(model)}).to_text())
    base = Artifact.local(root)
    full = RuntimeRegistry().build(
        serving_config(url, artifact_format=ArtifactFormat("full").to_dict()), model_path=str(model), environ=TOKENS
    )
    full.set_owner_sequence(1)
    full.open()
    version = full.activate_checkpoint(base)
    _, layout, name, loaded, contents, _ = events[-2]
    assert (layout, name, loaded, contents) == ("full", None, version, {"config.json": "{}"})
    # The base root binds the load of the base model's files, which the component admits.
    served = full.served_artifact(base.artifact_id, "math")
    assert served is not None and served.model_name == full_weight_name(str(model), version)
    assert server.pins.full_load_id == version
    full.shutdown()

    adapters = RuntimeRegistry().build(serving_config(url, adapter_capacity=2), model_path=str(model), environ=TOKENS)
    adapters.set_owner_sequence(2)
    adapters.open()
    calls = len(events)
    assert adapters.activate_checkpoint(base, scenario="math") == adapters.base_version
    assert events[calls:] == []
    # The adapter engines serve the base model under the load ID that this owner gave the new engines, and the
    # component admits the base root's route under that ID only.
    served = adapters.served_artifact(base.artifact_id, "math")
    assert served is not None and served.model_name == full_weight_name(str(model), adapters.base_version)
    assert adapters.served_model_name("math") == served.model_name
    assert pinned_request(url, server.engine, served.model_name)["text"] == "basebasebase"
    with pytest.raises(urllib.error.HTTPError) as refused:
        pinned_request(url, server.engine, full_weight_name(str(model), version))
    assert refused.value.code == 421
    adapters.shutdown()


def test_slot_urls_serve_one_full_weight_checkpoint_on_each_component(component, server, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, services = component[0], component[1], component[3]
    other = RemoteInferenceServer(
        engine="sglang", control_token=CONTROL_TOKEN, inference_token=INFERENCE_TOKEN, work_dir=tmp_path / "other"
    )
    with serving(other.application()) as other_url:
        runtime = RuntimeRegistry().build(
            serving_config(
                remote={"url": url, "slot-urls": [f"{other_url}/"], "startup-timeout-s": 30},
                artifact_format=ArtifactFormat("full").to_dict(),
            ),
            model_path="Qwen/Qwen3-8B",
            environ=TOKENS,
        )
        # Building the runtime touches no engine. Each URL has a slot of the same model, with its own handler.
        assert isinstance(runtime, CheckpointSlotsRuntime) and runtime.served_artifact_capacity == 2
        assert [(slot.base_url, slot.model_path) for slot in runtime.slots] == [
            (url, "Qwen/Qwen3-8B"),
            (other_url, "Qwen/Qwen3-8B"),
        ]
        for slot_url in (url, other_url):
            handler = runtime.handler_for(slot_url)
            assert isinstance(handler, SGLangInferenceHandler)
            arguments = handler.post_arguments(InferenceArtifact("artifact"), "/generate", {"input_ids": [1]})
            assert (arguments["url"], arguments["headers"]["Authorization"]) == (
                f"{slot_url}/generate",
                f"Bearer {INFERENCE_TOKEN}",
            )
        assert events == [] and services == []
        runtime.set_owner_sequence(1)
        runtime.open()
        # Each component starts its own engines from the same input, and keeps each request on its weights.
        starts = [event for event in events if isinstance(event, tuple) and event[0] == "start"]
        assert len(starts) == 2 and starts[0] == starts[1]
        assert started_engines(starts[0]) == ("sglang", "Qwen/Qwen3-8B", 1, None, "placement-2")
        assert server.pins.enabled and other.pins.enabled
        # Two checkpoints serve at once: the first URL's component loads one and the second URL's the other.
        artifacts = []
        for name in ("first", "second"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "config.json").write_text("{}")
            artifacts.append(Artifact.local(tmp_path / name))
        version = runtime.activate_checkpoint(artifacts[0])
        beside = runtime.serve_artifact(artifacts[1], scenario="math")
        routes = [runtime.served_artifact(artifact.artifact_id, "math") for artifact in artifacts]
        assert routes[1] == beside and [route.url for route in routes] == [url, other_url]
        # Each route names the load of its own component, which admits requests under that name only.
        assert [route.model_name for route in routes] == [
            full_weight_name("Qwen/Qwen3-8B", each.pins.full_load_id) for each in (server, other)
        ]
        assert server.pins.full_load_id == version != other.pins.full_load_id
        runtime.shutdown()
        assert "close" not in events


@pytest.mark.parametrize(
    ("engine_name", "options"),
    [
        # Each engine's own cache, which the pause clears, with an option for KV outside the engine that is off.
        ("sglang", {"enable-hierarchical-cache": True, "enable-lmcache": False}),
        ("vllm", {"enable-prefix-caching": True, "kv-transfer-config": None}),
    ],
)
def test_prefix_sharing_keeps_the_cache_to_one_set_of_weights(component, server, options, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, services = component[0], component[1], component[3]
    runtime = RuntimeRegistry().build(
        serving_config(
            url,
            engine=server.engine,
            num_gpus=1,
            artifact_format=ArtifactFormat("full").to_dict(),
            prefix_sharing=True,
            options=options,
        ),
        model_path="toy",
        environ=TOKENS,
    )
    runtime.set_owner_sequence(1)
    runtime.open()
    # The component starts the engines with the pause that the runtime selected, not the in-place one, and with
    # their own caches.
    started = events[0][1]
    assert started["pause_mode"] == "retract"
    assert all(started["options"][name.replace("-", "_")] is value for name, value in options.items())
    engine = services[-1].engines
    artifacts = []
    for marker in "AB":
        (tmp_path / marker).mkdir()
        (tmp_path / marker / "marker.txt").write_text(marker)
        artifacts.append(Artifact.local(tmp_path / marker))

    def route(artifact):
        served = runtime.served_artifact(artifact.artifact_id, "math")
        assert served is not None
        return served.model_name

    def answers(name):
        # The first request fills the prefix cache, and the second can reuse it.
        replies = (pinned_request(url, server.engine, name), pinned_request(url, server.engine, name))
        return [(reply["text"], reply["meta_info"]["cached_tokens"]) for reply in replies]

    def refused(name):
        try:
            pinned_request(url, server.engine, name)
        except urllib.error.HTTPError as error:
            assert error.code == 421
            return True
        return False

    try:
        runtime.activate_checkpoint(artifacts[0])
        first = route(artifacts[0])
        assert answers(first) == [("AAA", 0), ("AAA", 1)]
        engine.generation_started.clear()
        with ThreadPoolExecutor(2) as pool:
            held = pool.submit(pinned_request, url, server.engine, first, hold=True)
            assert engine.generation_started.wait(5)
            # No pause has drained this request, so the component refuses a load.
            with pytest.raises(RuntimeError, match="409"):
                load_marker(runtime.backend, tmp_path, "B", "owner:9")
            pauses = events.count("pause")
            activation = pool.submit(runtime.activate_checkpoint, artifacts[1])
            try:
                # The activation first closes admission. Until then, the engines answer and cache each request.
                answered, deadline = 0, time.monotonic() + 5
                while not refused(first):
                    answered += 1
                    assert time.monotonic() < deadline, "the activation did not close admission"
                with pytest.raises(TimeoutError):
                    activation.result(timeout=0.1)
                # The activation then waits for the held request: the engines keep its weights and their cache.
                assert (events.count("pause"), engine.cache, engine.marker) == (pauses, 3 + answered, "A")
            finally:
                engine.release_generation.set()
            assert held.result(timeout=5)["text"] == "AAA"
            activation.result(timeout=5)
        assert refused(first)
        # The pause cleared the cache before the load: the first request on B reuses nothing from A.
        assert answers(route(artifacts[1])) == [("BBB", 0), ("BBB", 1)]
    finally:
        engine.release_generation.set()
        runtime.shutdown()


def test_a_runtime_whose_handler_fails_starts_no_engine(component, monkeypatch):
    url, events, _, _ = component

    def refuse(cls, upstream_url, **config):
        raise ReefError("handler refused its configuration")

    monkeypatch.setattr(SGLangInferenceHandler, "from_config", classmethod(refuse))
    with pytest.raises(ReefError, match="handler refused"):
        RuntimeRegistry().build(serving_config(url), model_path="Qwen/Qwen3-8B", environ=TOKENS)
    assert events == []


def test_the_serving_runtime_reloads_its_snapshot_into_the_engines_of_a_restarted_component(
    component, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, _, services = component
    runtime = RuntimeRegistry().build(
        serving_config(url, artifact_format=ArtifactFormat("full").to_dict()),
        model_path="Qwen/Qwen3-8B",
        environ=TOKENS,
    )
    runtime.set_owner_sequence(1)
    runtime.open()
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


def test_a_new_owner_starts_new_engines_and_loads_the_selected_adapter_again(component, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeEngines, "recover", lambda engines: 0)
    url, events, services = component[0], component[1], component[3]
    artifact = Artifact.local(saved_adapter(tmp_path / "job-1"))
    routes = []
    # A control restart, or a standby that takes over, opens with the next owner sequence.
    for sequence in (1, 2):
        runtime = RuntimeRegistry().build(
            serving_config(url, adapter_capacity=2), model_path="Qwen/Qwen3-8B", environ=TOKENS
        )
        runtime.set_owner_sequence(sequence)
        runtime.open()
        runtime.activate_checkpoint(artifact, scenario="math")
        routes.append(runtime.served_artifact(artifact.artifact_id, "math"))
        runtime.shutdown()
    # The engines' load IDs cannot prove which adapters they hold, so the new owner takes over nothing.
    assert len(services) == 2
    assert len([event for event in events if event[0] == "load"]) == 2
    assert None not in routes and routes[0].model_name != routes[1].model_name


def test_a_runtime_without_an_owner_sequence_starts_no_engine(component):
    runtime = RuntimeRegistry().build(serving_config(component[0]), model_path="toy", environ=TOKENS)
    with pytest.raises(RuntimeError, match="serving-owner sequence"):
        runtime.open()
    assert component[1] == []
