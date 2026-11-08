"""A loopback endpoint that answers model calls from one artifact, for the rollouts that evaluate it.

An artifact evaluation runs its rollouts against an artifact that the scenario may not serve. A runtime that
answers each call from the artifact that the call names (``InferenceRuntime.serves_any_artifact``) can
answer them without loading the artifact: ``ArtifactEndpoint`` hands each call to the runtime's handler with that
artifact, under the runtime's inference admission. Reef's serving takes no part, and no call is a record.
"""

from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable, UpstreamStatusError
from reef.core.evaluation import EvaluationCall
from reef.core.served_models import ServedArtifact
from reef.inference.interfaces import InferenceArtifact, InferenceRuntime

#: The model call routes that the endpoint answers, as Reef's own inference routes do.
MODEL_CALL_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


class ArtifactEndpoint:
    """An OpenAI endpoint on loopback that answers every call from ``artifact`` with ``runtime``'s handler.

    Values: ``runtime`` is an ``InferenceRuntime`` that serves artifacts on request, and ``artifact``
    the materialized artifact that every call samples. ``served`` is the value that reaches an engine that answers
    for the artifact, or None for the runtime's own handler. A call to a path outside ``MODEL_CALL_PATHS`` gets 404, a
    body that is not a JSON object 400, a runtime that admits no call 503, a handler refusal its own status, and
    any other handler failure 502 with the reason. Purpose: the artifact evaluation service points its rollouts
    at ``url``. Use it as a context manager: it binds its loopback port and sets ``url`` when it is entered, and
    closes the port when it exits. Raise ``ValueError`` at construction when the runtime cannot serve
    ``artifact`` on request.
    """

    def __init__(self, runtime: InferenceRuntime, artifact: Artifact, served: ServedArtifact | None = None) -> None:
        if not runtime.serves_any_artifact:
            raise ValueError(f"the runtime serves only its current artifact, not {artifact.ref.revision!r}")
        self.runtime = runtime
        self.artifact = artifact
        self.served = served
        self.server: ThreadingHTTPServer | None = None
        self.url = ""

    async def answer(self, path: str, payload: dict[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """The status and the body that answer the call ``payload`` to ``path``."""
        try:
            admission = await self.runtime.acquire_inference()
        except InferenceUnavailable as exc:
            return 503, {"error": {"message": str(exc)}}
        try:
            if self.served is None:
                handler = self.runtime.inference_handler
            else:
                # A runtime that started the engine in another process has no handler of its own here.
                reached = self.runtime.handler_for(self.served.url)
                if reached is None:
                    raise ValueError(f"the runtime has no handler for {self.served.url}")
                # The candidate samples with the deployment's values, as a client's call to a served artifact does.
                body = reached.request_payload(payload, serving_config={})
                handler, payload = reached, reached.address_model(body, self.served.model_name)
            return 200, await handler.inference(InferenceArtifact.from_artifact(self.artifact), path, payload)
        except UpstreamStatusError as exc:
            return exc.status, {"error": {"message": str(exc)}}
        except (ValueError, TimeoutError, OSError) as exc:
            return 502, {"error": {"message": f"the runtime could not answer from the artifact: {exc}"}}
        finally:
            admission.release()

    def __enter__(self) -> ArtifactEndpoint:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path not in MODEL_CALL_PATHS:
                    self.reply(404, {"error": {"message": f"no route {self.path}"}})
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except json.JSONDecodeError as exc:
                    self.reply(400, {"error": {"message": f"the request is not JSON: {exc}"}})
                    return
                if not isinstance(payload, dict):
                    self.reply(400, {"error": {"message": "the request is not a JSON object"}})
                    return
                # Each call runs on its own server thread, so it runs the handler in its own event loop.
                self.reply(*asyncio.run(endpoint.answer(self.path, payload)))

            def reply(self, status: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="reef-artifact-endpoint", daemon=True).start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


class CandidateArtifacts(ABC):
    """Finds the candidate of an evaluation rollout, and keeps the rollout's calls.

    ``candidate(scenario, artifact_id, component)`` returns the runtime of ``scenario`` and the materialized
    artifact that ``artifact_id`` names for ``component``. It raises ``ReefError`` for a scenario without a
    runtime, and as ``Scenario.resolve_artifact`` does for an unknown artifact or component.
    ``candidate_task(scenario, task_name)`` returns the directory of one of its tasks here.
    ``add_evaluation_call(scenario, evaluation_id, artifact_id, call)`` keeps one call of a rollout to the candidate
    ``artifact_id`` as a record item of the scenario's owner (``reef.storage.record_items.evaluation_call_item``).

    Purpose: the dispatcher implements it; ``LocalCandidateEndpoints`` calls it for each rollout.
    """

    @abstractmethod
    def candidate_task(self, scenario: str, task_name: str) -> Path:
        """The directory of the Harbor task ``task_name`` of ``scenario`` on this process's file system, which the task
        file store fetches when another process stored it. ``ReefError`` for a task that the scenario does not use."""

    @abstractmethod
    def candidate(
        self, scenario: str, artifact_id: str, component: str | None
    ) -> tuple[InferenceRuntime, Artifact]: ...

    @abstractmethod
    def add_evaluation_call(
        self, scenario: str, evaluation_id: str, artifact_id: str, call: EvaluationCall
    ) -> None: ...


__all__ = ["ArtifactEndpoint", "CandidateArtifacts"]
