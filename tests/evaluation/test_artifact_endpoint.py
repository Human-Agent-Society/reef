from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from reef.artifact.artifact import Artifact
from reef.core.artifact_ref import ArtifactRef
from reef.core.evaluation import PolicyModel
from reef.evaluation.artifact_endpoint import ArtifactEndpoint
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, UpstreamStatusError


class RefusingHandler(InferenceHandler):
    """Answers a call whose last message says ``ok``, refuses ``bad`` with 422, and fails on anything else."""

    async def inference(self, artifact, path, payload):
        content = payload["messages"][-1]["content"]
        if content == "ok":
            return {"release": artifact.ref.release_id, "path": path}
        if content == "bad":
            raise UpstreamStatusError("the model refused the request", status=422)
        raise ValueError("the sampler returned nothing")


class Runtime(InferenceRuntime):
    def __init__(self, *, serves_any_release: bool) -> None:
        super().__init__(base_url="http://runtime.invalid")
        self.serves_any_release = serves_any_release
        self.handler = RefusingHandler()

    @property
    def inference_handler(self):
        return self.handler

    def artifact_model(self, artifact):
        return PolicyModel("base", artifact.ref.release_id) if self.serves_any_release else None

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"

    def current_runtime_load_id(self):
        return "load-0"


def call(url: str, path: str, body: object) -> tuple[int, dict]:
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.unit
def test_the_endpoint_answers_every_call_from_its_release_and_maps_each_failure_to_a_status() -> None:
    release = Artifact(ArtifactRef(content_id="c7", release_id="r7", parent_release_id=None), None)
    runtime = Runtime(serves_any_release=True)
    with ArtifactEndpoint(runtime, release) as endpoint:
        chat = {"messages": [{"role": "user", "content": "ok"}]}
        assert call(endpoint.url, "/v1/chat/completions", chat) == (
            200,
            {"release": "r7", "path": "/v1/chat/completions"},
        )
        assert call(endpoint.url, "/v1/embeddings", chat)[0] == 404
        assert call(endpoint.url, "/v1/chat/completions", [1, 2])[0] == 400
        assert call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "bad"}]})[0] == 422
        status, body = call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "?"}]})
        assert status == 502 and "the sampler returned nothing" in body["error"]["message"]
        assert runtime.inference_admission_status == {"open": True, "active": 0}, "every call releases admission"
        runtime.close_admission()
        assert call(endpoint.url, "/v1/chat/completions", chat)[0] == 503
    with pytest.raises(ValueError, match="serves only its current artifact"):
        ArtifactEndpoint(Runtime(serves_any_release=False), release)
