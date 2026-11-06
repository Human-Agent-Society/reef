from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from reef.artifact.artifact import Artifact
from reef.core.errors import UpstreamStatusError
from reef.core.evaluation import PolicyModel
from reef.core.served_models import ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.evaluation.artifact_endpoint import ArtifactEndpoint
from reef.inference.interfaces import InferenceArtifact, InferenceHandler, InferenceRuntime


class RefusingHandler(InferenceHandler):
    """Answers a call whose last message says ``ok``, refuses ``bad`` with 422, and fails on anything else."""

    async def inference(self, artifact, path, payload):
        content = payload["messages"][-1]["content"]
        if content == "ok":
            return {"artifact_id": artifact.artifact_id, "path": path}
        if content == "bad":
            raise UpstreamStatusError("the model refused the request", status=422)
        raise ValueError("the sampler returned nothing")


class Runtime(InferenceRuntime):
    def __init__(self, *, any_artifact: bool) -> None:
        super().__init__(base_url="http://runtime.invalid")
        self.any_artifact = any_artifact
        self.handler = RefusingHandler()

    @property
    def inference_handler(self):
        return self.handler

    @property
    def serves_any_artifact(self) -> bool:
        return self.any_artifact

    def artifact_model(self, artifact):
        return PolicyModel("base", artifact.ref.revision)

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"


def call(url: str, path: str, body: object) -> tuple[int, dict]:
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class EngineHandler(InferenceHandler):
    """An engine's handler, which sets its own sampling values and keeps each body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def request_payload(self, payload: Mapping[str, Any], *, serving_config: Mapping[str, Any]) -> dict[str, Any]:
        return {**payload, "temperature": 0.2}

    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(payload)
        return {"artifact_id": artifact.artifact_id}


class StartedElsewhereRuntime(Runtime):
    """A runtime that another process started an engine of: it has no handler of its own, as a Modal runtime has none
    before an activation."""

    def __init__(self) -> None:
        super().__init__(any_artifact=True)
        self.engine = EngineHandler()

    @property
    def inference_handler(self) -> InferenceHandler:
        raise RuntimeError("no engine serves yet")

    def handler_for(self, url: str) -> InferenceHandler | None:
        return self.engine


@pytest.mark.unit
def test_a_call_reaches_an_engine_that_another_process_started_with_the_engines_values() -> None:
    release = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = StartedElsewhereRuntime()
    served = ServedArtifact("artifact:r7", "https://engine.invalid", "engine-model")
    with ArtifactEndpoint(runtime, release, served) as endpoint:
        chat = {"model": "from-the-agent", "messages": [{"role": "user", "content": "ok"}]}
        assert call(endpoint.url, "/v1/chat/completions", chat) == (200, {"artifact_id": "artifact:r7"})
    assert runtime.engine.bodies == [{**chat, "model": "engine-model", "temperature": 0.2}]


@pytest.mark.unit
def test_the_endpoint_answers_every_call_from_its_release_and_maps_each_failure_to_a_status() -> None:
    release = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = Runtime(any_artifact=True)
    with ArtifactEndpoint(runtime, release) as endpoint:
        chat = {"messages": [{"role": "user", "content": "ok"}]}
        assert call(endpoint.url, "/v1/chat/completions", chat) == (
            200,
            {"artifact_id": "artifact:r7", "path": "/v1/chat/completions"},
        )
        assert call(endpoint.url, "/v1/embeddings", chat)[0] == 404
        assert call(endpoint.url, "/v1/chat/completions", [1, 2])[0] == 400
        assert call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "bad"}]})[0] == 422
        status, body = call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "?"}]})
        assert status == 502 and "the sampler returned nothing" in body["error"]["message"]
        assert runtime.inference_admission_status == {"open": True, "active": 0}, "every call releases admission"
        runtime.close_admission()
        assert call(endpoint.url, "/v1/chat/completions", chat)[0] == 503
    elsewhere = ServedArtifact("artifact:r7", "http://elsewhere.invalid", "m")
    with ArtifactEndpoint(Runtime(any_artifact=True), release, elsewhere) as endpoint:
        status, body = call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "ok"}]})
        assert status == 502 and "no handler" in body["error"]["message"]
    with pytest.raises(ValueError, match="serves only its current artifact"):
        ArtifactEndpoint(Runtime(any_artifact=False), release)
