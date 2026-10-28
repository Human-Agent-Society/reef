"""Inference admission: a closed runtime admits no request, and every request releases the handle it holds."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.artifact import Artifact, ArtifactRef
from reef.core.errors import InferenceUnavailable
from reef.observability.operations import OperationMetrics
from reef.runtime.interfaces import InferenceAdmissionController, InferenceHandler, InferenceStream
from reef.service.app import RequestService
from reef.service.streaming import stream_record
from reef.surface import RuntimeLoadMismatch, create_weight_surface


def test_closed_admission_refuses_new_requests_and_lets_admitted_ones_finish() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController()
        admitted = await controller.acquire()
        controller.close()
        with pytest.raises(InferenceUnavailable, match="shutting down"):
            await controller.acquire()
        assert controller.status == {"open": False, "active": 1}
        admitted.release()
        assert controller.status == {"open": False, "active": 0}

    asyncio.run(run())


class LocalReleases:
    """A release repository for request tests: every release materializes to the same local directory."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

    def materialize(self, ref: ArtifactRef) -> Artifact:
        return Artifact(ref, None, local_path=self.root)


def test_stream_and_failure_release_their_inference_admission_handles(tmp_path) -> None:
    runtime = StubInferenceRuntime()
    ref = ArtifactRef(content_id="content:current", release_id="release:0", parent_release_id=None)

    class Scenario:
        operations = OperationMetrics(("serve/request", "serve/admission"))
        recipe = "test"
        repository = LocalReleases(tmp_path / "releases")
        surface = create_weight_surface(kind="model")
        inference_handler = None

        @staticmethod
        def current_artifact_ref():
            return ref

        def __init__(self) -> None:
            self.runtime = runtime
            self.training_component = StubTrainingComponent()

    class Dispatcher:
        @staticmethod
        def get_or_create_scenario(*args, **kwargs):
            del args, kwargs
            return Scenario()

        @staticmethod
        def accept_record(item, **kwargs):
            del kwargs
            return item

    class StreamingBackend(InferenceHandler):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload

            async def chunks():
                yield b"data: [DONE]\n\n"

            return InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
                record_response={"metadata": {"runtime_load_id": "engine:0"}},
            )

    class FailingBackend(InferenceHandler):
        async def inference(self, artifact, path, payload):
            del artifact, path, payload
            raise RuntimeError("backend failed")

    class DeferredStreamingBackend(InferenceHandler):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload
            holder = {}

            async def chunks():
                yield b'data: {"content":"first token"}\n\n'
                holder["stream"].record_response = {
                    "metadata": {"runtime_load_id": "engine:0"},
                    "training": {"runtime_load_id": "engine:0"},
                }
                yield b"data: [DONE]\n\n"

            stream = InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
                record_response_pending=True,
            )
            holder["stream"] = stream
            return stream

    class UnverifiableStreamingBackend(InferenceHandler):
        async def inference(self, artifact, path, payload):
            raise AssertionError("streaming path expected")

        async def inference_stream(self, artifact, path, payload):
            del artifact, path, payload

            async def chunks():
                yield b"data: [DONE]\n\n"

            return InferenceStream(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                chunks=chunks(),
            )

    service = RequestService(Dispatcher())

    async def run() -> None:
        stream, pending = await service.start_stream(
            {"x-reef-scenario": "math"},
            {"stream": True},
            "/v1/chat/completions",
            StreamingBackend(),
        )
        assert runtime.inference_admission_status == {"open": True, "active": 0}
        assert pending.admission is None
        await stream.close()
        service.record_stream(pending, {"stream": True, "complete": True})
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        deferred, deferred_pending = await service.start_stream(
            {"x-reef-scenario": "math"},
            {"stream": True},
            "/v1/chat/completions",
            DeferredStreamingBackend(),
        )
        assert runtime.inference_admission_status == {"open": True, "active": 1}
        body = b"".join([chunk async for chunk in deferred.chunks])
        recorded = stream_record(deferred, body, complete=True)
        item = service.record_stream(deferred_pending, recorded)
        assert item.payload["runtime_load_id"] == "engine:0"
        assert Scenario.operations.snapshot()["serve/request/completed_total"] == 2
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        with pytest.raises(RuntimeLoadMismatch, match="atomic record_response"):
            await service.start_stream(
                {"x-reef-scenario": "math"},
                {"stream": True},
                "/v1/chat/completions",
                UnverifiableStreamingBackend(),
            )
        assert runtime.inference_admission_status == {"open": True, "active": 0}

        with pytest.raises(RuntimeError, match="backend failed"):
            await service.infer_with_data(
                {"x-reef-scenario": "math"},
                {},
                "/v1/chat/completions",
                FailingBackend(),
            )
        assert runtime.inference_admission_status == {"open": True, "active": 0}

    asyncio.run(run())
