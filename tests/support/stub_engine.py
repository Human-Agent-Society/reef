"""A CPU stand-in for the native SGLang engines, behind Reef's own SGLang serving stack.

Selected with ``execution.rollout: tests.support.stub_engine:StubEngineExecutor`` under
``inference.backend: sglang``. The inference component then runs its real receiver and Reef's
``SGLangInferenceService``, ``SGLangControl`` and ``SGLangInferenceBackend``. Only the serving executor differs,
so no native engine starts. The stand-in checks the size of the deployment's reservation, loads Reef's adapter
files, and answers ``POST /generate`` with the identity of the weights that the request selects. A test can thus
check which weights a served model name reaches.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reef.execution.executor.placement import GpuBundles
from reef.inference.engine_control import EngineExecutor
from reef.inference.process import retire_engines
from reef.inference.sglang.chat import SGLANG_GENERATE_PATH
from reef.inference.sglang.config import SGLangSettings

#: The files of one adapter that Reef hands to the engines.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_weights_id(directory: str | Path) -> str:
    """The SHA-256 hex digest of ``adapter_model.safetensors`` in ``directory``.

    The ID is computed from the bytes of that file only. A stub engine answers a generation under an adapter with
    this ID.
    """
    return hashlib.sha256(Path(directory, "adapter_model.safetensors").read_bytes()).hexdigest()


class StubEngine:
    """One engine, as a Ray actor: the base model, a runtime load ID and the adapters that it holds.

    Values: ``model_path`` names the base model. ``version`` is "engine:0" until Reef sets a runtime load ID.
    ``adapters`` maps each loaded adapter name to the ``adapter_weights_id`` of the files that it was loaded from.
    A name keeps its weights until its unload.
    Purpose: ``StubEngineWorker`` starts one for each engine of the configuration. Reef's backend sets the load ID
    and unloads adapters on it directly, as on a native engine.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.version = "engine:0"
        self.adapters: dict[str, str] = {}

    def get_runtime_load_id(self) -> str:
        return self.version

    def set_runtime_load_id(self, runtime_load_id: str) -> dict[str, Any]:
        self.version = str(runtime_load_id)
        return {"success": True}

    def load_lora_adapter_from_disk(self, lora_name: str, lora_path: str) -> dict[str, Any]:
        if lora_name in self.adapters:
            return {"success": False, "message": f"adapter {lora_name!r} is already loaded"}
        missing = [name for name in ADAPTER_FILES if not Path(lora_path, name).is_file()]
        if missing:
            return {"success": False, "message": f"no {missing[0]} under {lora_path}"}
        self.adapters[lora_name] = adapter_weights_id(lora_path)
        return {"success": True}

    def unload_lora_adapter(self, lora_name: str) -> dict[str, Any]:
        self.adapters.pop(lora_name, None)
        return {"success": True}

    def generate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Answer one native generation with the identity of the weights that ``payload`` selects.

        ``text`` is the base model's path for a request without ``lora_path``, and the adapter's weights ID for a
        loaded adapter name. ``meta_info`` carries a finish reason and, as native SGLang does, the runtime load ID
        as ``weight_version``. A name that the engine does not hold gets an ``error`` object and no generation.
        """
        selected = payload.get("lora_path")
        if selected is None:
            weights = self.model_path
        elif selected in self.adapters:
            weights = self.adapters[selected]
        else:
            return {"error": {"message": f"the engine holds no adapter {selected!r}"}}
        return {"text": weights, "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": self.version}}

    def shutdown(self) -> None:
        """Nothing to stop: the engine holds no process. ``retire_engines`` calls this, then kills the actor."""


class StubEngineWorker:
    """The serving worker of the stub engines: the control operations that ``SGLangWorker`` offers.

    Values: ``config`` is the deployment's ``SGLangSettings``. ``engines`` holds the live ``StubEngine`` actors:
    ``config.engine_count`` of them after startup and after a recovery, and none after a termination or a shutdown.
    ``paused`` is True from a pause or a termination until the next resume. ``server`` answers at ``inference_url``
    until the shutdown.
    Purpose: ``StubEngineExecutor`` makes one inside Reef's inference control actor, which forwards each control
    operation to it.
    """

    def __init__(self, config: SGLangSettings, pg: GpuBundles) -> None:
        if len(pg.bundle_indices) != config.num_gpus:
            raise ValueError("the stub engine expects one reserved bundle per inference GPU")
        self.config = config
        self.engines = [self.start_engine() for _ in range(config.engine_count)]
        self.paused = False
        self.server = StubGenerateServer(self)
        threading.Thread(target=self.server.serve_forever, name="stub-engine-generate", daemon=True).start()

    def start_engine(self) -> Any:
        import ray

        # The stand-in takes no resource of the reservation, so an engine never waits for a CPU.
        return ray.remote(num_cpus=0)(StubEngine).remote(self.config.model_path)

    def check_health(self) -> None:
        """Raise if an engine actor does not answer."""
        import ray

        ray.get([engine.get_runtime_load_id.remote() for engine in self.engines], timeout=30)

    def inference_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def get_runtime_load_ids(self) -> list[str]:
        import ray

        return ray.get([engine.get_runtime_load_id.remote() for engine in self.engines])

    def updatable_engines(self) -> list[Any]:
        return self.engines

    def pause_generation_for_update(self) -> None:
        self.paused = True

    def continue_generation_after_update(self) -> None:
        self.paused = False

    def terminate_updatable_engines(self) -> int:
        count = len(self.engines)
        retire_engines(self.engines)
        self.engines = []
        self.paused = True
        return count

    def recover_updatable_engines(self) -> int:
        missing = self.config.engine_count - len(self.engines)
        self.engines = [*self.engines, *(self.start_engine() for _ in range(missing))]
        return missing

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> None:
        """Load one adapter directory into every engine, then give every engine ``runtime_load_id``."""
        import ray

        if not self.engines:
            raise RuntimeError("no stub engine runs that can load an adapter")
        results = ray.get(
            [
                engine.load_lora_adapter_from_disk.remote(lora_name=lora_name, lora_path=lora_path)
                for engine in self.engines
            ]
        )
        for result in results:
            if result.get("success") is not True:
                raise RuntimeError(f"stub engine refused adapter {lora_name!r}: {result}")
        if runtime_load_id is not None:
            ray.get([engine.set_runtime_load_id.remote(runtime_load_id) for engine in self.engines])

    def load_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> None:
        """Refuse: the stand-in serves adapters on one base model and replaces no full weights."""
        raise RuntimeError("the stub engines load adapters only, not full weights")

    def generate(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Answer one native generation with every engine; return the HTTP status and the JSON body.

        The status is 200 with the engines' common answer, and 400 when they refuse the selection. It is 503 while
        generation is paused or no engine runs, because Reef's receiver forwards requests only to resumed engines.
        It is 500 when the engines disagree: they then do not hold the same weights under the selected name.
        """
        import ray

        engines = self.engines
        if self.paused or not engines:
            return 503, {"error": {"message": "the stub engines do not generate now"}}
        answers = ray.get([engine.generate.remote(dict(payload)) for engine in engines], timeout=30)
        if any(answer != answers[0] for answer in answers):
            return 500, {"error": {"message": f"the stub engines disagree: {answers}"}}
        return (400 if "error" in answers[0] else 200), answers[0]

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        retire_engines(self.engines)
        self.engines = []


class StubEngineExecutor(EngineExecutor):
    """The serving executor that runs ``StubEngineWorker`` in place of the native SGLang worker.

    Values: its launch options are those that ``EngineControl`` gives each serving executor: ``config``, the
    deployment's ``SGLangSettings``, and ``pg``, the reserved inference bundles.
    Purpose: a deployment selects it by its import path as the ``rollout`` executor. Reef's SGLang control actor
    then makes it.
    """

    def create_worker(self) -> object:
        return StubEngineWorker(**dict(self.config.options))


class StubGenerateServer(ThreadingHTTPServer):
    """The HTTP endpoint of the stub engines, on a free loopback port.

    Values: ``worker`` is the ``StubEngineWorker`` that answers each generation.
    Purpose: Reef's receiver forwards the admitted requests to it, as to the native router.
    """

    def __init__(self, worker: StubEngineWorker) -> None:
        super().__init__(("127.0.0.1", 0), StubGenerateHandler)
        self.worker = worker


class StubGenerateHandler(BaseHTTPRequestHandler):
    """Answer ``POST /generate`` with the worker's status and JSON body; any other path gets 404.

    Values: ``server`` is the ``StubGenerateServer`` that received the request.
    Purpose: the server makes one for each request.
    """

    server: StubGenerateServer

    def log_message(self, format: str, *args: Any) -> None:
        """Write no access log."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == SGLANG_GENERATE_PATH:
            status, answer = self.server.worker.generate(json.loads(body))
        else:
            status, answer = 404, {"error": {"message": f"the stub engine has no route {self.path}"}}
        encoded = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
