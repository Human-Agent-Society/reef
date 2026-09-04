from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reef.core.records_types import RequestType
from reef.dispatcher import build_default_dispatcher
from reef.observability import InferenceObserver, InferenceTrace, ReportFeedback
from reef.observability.inference_factory import build_inference_observer
from reef.observability.langsmith import (
    LangSmithConfig,
    LangSmithInferenceObserver,
    langsmith_feedback_id,
    langsmith_run_id,
)
from reef.runtime.inference import InferenceBackend
from reef.service.request_service import RequestService


class _StubLangSmithClient:
    def __init__(self) -> None:
        self.created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.updated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.feedback: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.flushed = False
        self.closed = False

    def create_run(self, *args: Any, **kwargs: Any) -> None:
        self.created.append((args, kwargs))

    def update_run(self, *args: Any, **kwargs: Any) -> None:
        self.updated.append((args, kwargs))

    def create_feedback(self, *args: Any, **kwargs: Any) -> None:
        self.feedback.append((args, kwargs))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class _FailingLangSmithClient(_StubLangSmithClient):
    def create_run(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("rate limited")

    def create_feedback(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("network unavailable")

    def flush(self) -> None:
        raise TimeoutError("flush timed out")

    def close(self) -> None:
        raise RuntimeError("close failed")


class _CapturingObserver(InferenceObserver):
    def __init__(self, *, fail: bool = False) -> None:
        self.traces: list[InferenceTrace] = []
        self.feedback: list[ReportFeedback] = []
        self.fail = fail

    def record_inference(self, trace: InferenceTrace) -> None:
        self.traces.append(trace)
        if self.fail:
            raise RuntimeError("observer unavailable")

    def record_feedback(self, feedback: ReportFeedback) -> None:
        self.feedback.append(feedback)
        if self.fail:
            raise RuntimeError("observer unavailable")


class _ReplyBackend(InferenceBackend):
    def __init__(self) -> None:
        self.calls = 0

    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        self.calls += 1
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.unit
def test_receipts_and_report_edges_have_stable_deterministic_ids() -> None:
    first = langsmith_run_id("client:any/value")
    assert first == langsmith_run_id("client:any/value")
    assert first != langsmith_run_id("client:any/value:2")
    assert langsmith_feedback_id("report-1", "client:any/value") == langsmith_feedback_id(
        "report-1", "client:any/value"
    )
    assert langsmith_feedback_id("report-1", "client:any/value") != langsmith_feedback_id(
        "report-2", "client:any/value"
    )


@pytest.mark.unit
def test_default_factory_is_a_noop_and_does_not_construct_a_client() -> None:
    constructed = False

    def factory():
        nonlocal constructed
        constructed = True
        return _StubLangSmithClient()

    observer = build_inference_observer({}, client_factory=factory)
    observer.record_inference(
        InferenceTrace(
            agent_record_id="r",
            scenario="s",
            recipe="recipe",
            path="/v1/chat/completions",
            started_at=1,
            ended_at=2,
            retry_count=0,
            completion_state="complete",
            delivery_state="successful",
            record_accepted=True,
            streaming=False,
        )
    )
    observer.close()
    assert constructed is False


@pytest.mark.unit
def test_langsmith_export_redacts_payloads_and_attaches_feedback_to_every_reference() -> None:
    client = _StubLangSmithClient()
    observer = LangSmithInferenceObserver(
        LangSmithConfig(enabled=True, project="reef", flush_timeout_s=1),
        lambda: client,
    )
    observer.record_inference(
        InferenceTrace(
            agent_record_id="inference-1",
            scenario="prod",
            recipe="sao",
            path="/v1/chat/completions",
            started_at=1,
            ended_at=2,
            retry_count=2,
            completion_state="complete",
            delivery_state="successful",
            record_accepted=True,
            streaming=False,
            inputs={"messages": [{"content": "hello"}], "api_key": "input-secret"},
            outputs={"answer": "ok", "nested": {"token": "output-secret"}},
            metadata={"tags": {"environment": "prod"}, "authorization": "metadata-secret"},
            content_id="content-1",
            release_id="release-2",
            serving_runtime_load_id="load-3",
        )
    )
    observer.record_feedback(
        ReportFeedback(
            report_record_id="report-1",
            scenario="prod",
            references=("inference-1", "inference-2"),
            score=0.75,
            feedback={"rubric": "good", "api_key": "feedback-secret"},
            metadata={"token": "report-secret"},
        )
    )
    observer.close()

    assert client.flushed and client.closed
    [(create_args, create_kwargs)] = client.created
    assert create_args[:3] == ("reef.inference", {"messages": [{"content": "hello"}], "api_key": "[REDACTED]"}, "llm")
    assert create_kwargs["id"] == langsmith_run_id("inference-1")
    metadata = create_kwargs["extra"]["metadata"]
    assert metadata["reef.agent_record_id"] == "inference-1"
    assert metadata["reef.retry_count"] == 2
    assert metadata["reef.release_id"] == "release-2"
    assert metadata["reef.serving_runtime_load_id"] == "load-3"
    assert metadata["reef.tag.environment"] == "prod"
    assert metadata["reef.metadata"]["authorization"] == "[REDACTED]"
    assert create_kwargs["outputs"]["nested"]["token"] == "[REDACTED]"
    assert client.updated == []
    assert [args[0] for args, _ in client.feedback] == [
        langsmith_run_id("inference-1"),
        langsmith_run_id("inference-2"),
    ]
    assert [kwargs["feedback_id"] for _, kwargs in client.feedback] == [
        langsmith_feedback_id("report-1", "inference-1"),
        langsmith_feedback_id("report-1", "inference-2"),
    ]
    assert all(kwargs["value"]["api_key"] == "[REDACTED]" for _, kwargs in client.feedback)
    assert all(kwargs["source_info"]["reef.metadata"]["token"] == "[REDACTED]" for _, kwargs in client.feedback)


@pytest.mark.unit
def test_langsmith_privacy_switches_omit_inputs_outputs_and_client_metadata() -> None:
    client = _StubLangSmithClient()
    observer = LangSmithInferenceObserver(
        LangSmithConfig(
            enabled=True,
            project="reef",
            include_inputs=False,
            include_outputs=False,
            include_metadata=False,
            flush_timeout_s=1,
        ),
        lambda: client,
    )
    observer.record_inference(
        InferenceTrace(
            agent_record_id="inference-1",
            scenario="prod",
            recipe="recipe",
            path="/v1/messages",
            started_at=1,
            ended_at=2,
            retry_count=0,
            completion_state="complete",
            delivery_state="successful",
            record_accepted=True,
            streaming=False,
            inputs={"private": "prompt"},
            outputs={"private": "answer"},
            metadata={"private": "metadata"},
        )
    )
    observer.close()

    [(create_args, create_kwargs)] = client.created
    assert create_args[1] == {}
    assert "reef.metadata" not in create_kwargs["extra"]["metadata"]
    assert "outputs" not in create_kwargs
    assert client.updated == []


@pytest.mark.unit
def test_every_langsmith_client_failure_is_contained() -> None:
    observer = LangSmithInferenceObserver(
        LangSmithConfig(enabled=True, project="reef", flush_timeout_s=1),
        _FailingLangSmithClient,
    )
    observer.record_inference(
        InferenceTrace(
            agent_record_id="inference-1",
            scenario="prod",
            recipe="recipe",
            path="/v1/messages",
            started_at=1,
            ended_at=2,
            retry_count=0,
            completion_state="backend_error",
            delivery_state="failed",
            record_accepted=False,
            streaming=False,
            error="TimeoutError",
        )
    )
    observer.record_feedback(
        ReportFeedback(
            report_record_id="report-1",
            scenario="prod",
            references=("inference-1",),
            score=0.0,
        )
    )
    observer.close()


@pytest.mark.unit
def test_request_service_observes_only_after_acceptance_and_observer_failures_are_isolated() -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher()
        observer = _CapturingObserver(fail=True)
        service = RequestService(dispatcher, inference_observer=observer)
        response, inference = await service.infer_with_data(
            {"x-reef-scenario": "chat", "x-reef-tag-environment": "test"},
            {"messages": [{"role": "user", "content": "hello"}]},
            "/v1/chat/completions",
            _ReplyBackend(),
        )
        assert response["choices"][0]["message"]["content"] == "ok"
        assert dispatcher.get_or_create_scenario("chat").records.get("chat", inference.agent_record_id) is not None
        assert observer.traces[0].record_accepted is True
        assert observer.traces[0].metadata["tags"] == {"environment": "test"}

        report = service.accept(
            {"x-reef-scenario": "chat"},
            {"score": 1.0, "feedback": {"correct": True}, "references": [inference.agent_record_id]},
            request_type=RequestType.REPORT,
            agent_record_id="client-report-1",
        )
        assert dispatcher.get_or_create_scenario("chat").records.get("chat", report.agent_record_id) is not None
        assert observer.feedback[0].references == (inference.agent_record_id,)
        retried = service.accept(
            {"x-reef-scenario": "chat"},
            {"score": 1.0, "feedback": {"correct": True}, "references": [inference.agent_record_id]},
            request_type=RequestType.REPORT,
            agent_record_id="client-report-1",
        )
        assert retried.agent_record_id == report.agent_record_id
        assert len(observer.feedback) == 1

    asyncio.run(run())


@pytest.mark.unit
def test_client_selected_inference_id_replays_without_duplicate_backend_or_trace() -> None:
    async def run() -> None:
        observer = _CapturingObserver()
        backend = _ReplyBackend()
        service = RequestService(build_default_dispatcher(), inference_observer=observer)
        headers = {"x-reef-scenario": "chat", "x-reef-agent-record-id": "client/inference:1"}
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        first_response, first = await service.infer_with_data(headers, payload, "/v1/chat/completions", backend)
        second_response, second = await service.infer_with_data(headers, payload, "/v1/chat/completions", backend)

        assert first.agent_record_id == second.agent_record_id == "client/inference:1"
        assert first_response == second_response
        assert backend.calls == 1
        assert len(observer.traces) == 1

    asyncio.run(run())


@pytest.mark.unit
def test_stream_trace_distinguishes_disconnect_from_durable_acceptance() -> None:
    async def run() -> None:
        observer = _CapturingObserver()
        dispatcher = build_default_dispatcher()
        service = RequestService(dispatcher, inference_observer=observer)
        stream, pending = await service.start_stream(
            {"x-reef-scenario": "chat"},
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "/v1/chat/completions",
            _ReplyBackend(),
        )
        await stream.close()
        stored = service.record_stream(
            pending,
            {
                "stream": True,
                "status": 200,
                "headers": {"Content-Type": "text/event-stream"},
                "complete": False,
                "body": "partial",
                "error": "client disconnected",
            },
        )

        [trace] = observer.traces
        assert trace.agent_record_id == stored.agent_record_id
        assert trace.record_accepted is True
        assert trace.completion_state == "incomplete"
        assert trace.delivery_state == "disconnected"

    asyncio.run(run())


@pytest.mark.unit
def test_langsmith_configuration_is_sourced_from_observability_section() -> None:
    from reef.service.deploy.settings import service_settings_from_config

    settings = service_settings_from_config(
        {
            "reef": {"recipe": "recipe"},
            "observability": {
                "langsmith": {
                    "enabled": True,
                    "project": "reef-prod",
                    "endpoint": "https://eu.api.smith.langchain.com",
                    "include_inputs": False,
                }
            },
        }
    )
    assert settings.langsmith_config == {
        "enabled": True,
        "project": "reef-prod",
        "endpoint": "https://eu.api.smith.langchain.com",
        "include_inputs": False,
    }
