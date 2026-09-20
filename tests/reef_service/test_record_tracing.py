"""Record tracing: the observer contract, its OpenTelemetry exporter, and the dispatcher's isolation of it."""

from __future__ import annotations

import json

import pytest
from reef_service.runtime_stubs import runtime_bindings
from reef_service.test_commit_log import RecordingRuntime, TestPolicyRecipe, wait_for_step

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef
from reef.dispatcher import Dispatcher
from reef.observability import (
    CommittedStepEvent,
    NullRecordObserver,
    RecordObserver,
    TracingConfig,
    build_record_observer,
)
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.service.deploy.service_config import service_config_from_mapping
from reef.storage.sqlite import SQLiteScenarioStorage


def _inference(agent_record_id: str, scenario: str = "math", **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2], **payload},
        agent_record_id=agent_record_id,
    )


def _report(agent_record_id: str, *references: str, scenario: str = "math", **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": list(references), **payload},
        agent_record_id=agent_record_id,
    )


# -- configuration -----------------------------------------------------------


@pytest.mark.unit
def test_tracing_config_defaults_to_disabled_and_rejects_unknown_settings() -> None:
    assert TracingConfig.from_mapping(None) == TracingConfig()
    assert TracingConfig.from_mapping({}).enabled is False
    with pytest.raises(ValueError, match=r"unknown observability\.tracing settings: exporter"):
        TracingConfig.from_mapping({"enabled": True, "exporter": "jaeger"})
    with pytest.raises(ValueError, match="headers must map header names to strings"):
        TracingConfig.from_mapping({"headers": {"Authorization": 1}})
    with pytest.raises(ValueError, match="endpoint must be a non-empty string"):
        TracingConfig.from_mapping({"endpoint": " "})
    with pytest.raises(ValueError, match="must be a mapping"):
        TracingConfig.from_mapping("http://localhost:4318")
    with pytest.raises(ValueError, match=r"set observability\.tracing\.authorization"):
        TracingConfig.from_mapping({"headers": {"authorization": "Basic abc"}})


@pytest.mark.unit
def test_tracing_credential_comes_from_config_or_environment_and_stays_out_of_repr() -> None:
    configured = TracingConfig.from_mapping(
        {"authorization": " Basic abc "}, environ={"REEF_TRACING_AUTHORIZATION": "Basic env"}
    )
    assert configured.authorization == "Basic abc"
    assert configured.request_headers() == {"Authorization": "Basic abc"}
    assert "abc" not in repr(configured)

    from_environment = TracingConfig.from_mapping(
        {"headers": {"x-team": "ml"}}, environ={"REEF_TRACING_AUTHORIZATION": "Basic env"}
    )
    assert from_environment.authorization == "Basic env"
    assert from_environment.request_headers() == {"x-team": "ml", "Authorization": "Basic env"}

    # Nothing configured: the SDK reads OTEL_EXPORTER_OTLP_HEADERS itself.
    assert TracingConfig.from_mapping({}, environ={}).request_headers() is None
    assert TracingConfig.from_mapping({"authorization": ""}).authorization is None


@pytest.mark.unit
def test_tracing_config_parses_every_setting() -> None:
    config = TracingConfig.from_mapping(
        {
            "enabled": True,
            "endpoint": " http://collector:4318/v1/traces ",
            "authorization": "Basic abc",
            "headers": {"x-team": "ml"},
            "service_name": "reef-prod",
            "include_messages": True,
        }
    )
    assert config == TracingConfig(
        enabled=True,
        endpoint="http://collector:4318/v1/traces",
        authorization="Basic abc",
        headers={"x-team": "ml"},
        service_name="reef-prod",
        include_messages=True,
    )


@pytest.mark.unit
def test_service_config_carries_the_tracing_section() -> None:
    settings = service_config_from_mapping(
        {"reef": {"recipe": "recipe"}, "observability": {"tracing": {"enabled": True, "endpoint": "http://c:4318"}}}
    )
    assert settings.tracing_config == {"enabled": True, "endpoint": "http://c:4318"}
    assert service_config_from_mapping({"reef": {"recipe": "recipe"}}).tracing_config == {}


@pytest.mark.unit
def test_disabled_tracing_builds_the_null_observer_without_the_sdk() -> None:
    assert isinstance(build_record_observer(None), NullRecordObserver)
    assert isinstance(build_record_observer({"enabled": False, "endpoint": "http://c:4318"}), NullRecordObserver)


# -- OpenTelemetry exporter ----------------------------------------------------


def _observer(**overrides):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from reef.observability.open_telemetry import OpenTelemetryRecordObserver

    exporter = InMemorySpanExporter()
    observer = OpenTelemetryRecordObserver(TracingConfig(enabled=True, **overrides), span_exporter=exporter)
    return observer, exporter


def _spans(observer, exporter):
    observer.close()
    return {span.name: span for span in exporter.get_finished_spans()}


@pytest.mark.unit
def test_enabled_tracing_builds_the_opentelemetry_observer() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from reef.observability.open_telemetry import OpenTelemetryRecordObserver

    observer = build_record_observer({"enabled": True, "endpoint": "http://127.0.0.1:1/v1/traces"})
    try:
        assert isinstance(observer, OpenTelemetryRecordObserver)
    finally:
        observer.close()


@pytest.mark.unit
def test_inference_record_becomes_a_client_span_with_stable_ids_and_no_messages() -> None:
    from reef.observability.open_telemetry import record_span_context

    observer, exporter = _observer()
    served = LiveWeightArtifactRef(content_id="c1", release_id="v3", parent_release_id="v2", runtime_load_id="load-3")
    item = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={
            "model": "qwen",
            "messages": [{"role": "user", "content": "2+2?"}],
            "metadata": {"tags": ["eval", "smoke"]},
            "response": {
                "id": "chatcmpl-1",
                "model": "qwen-served",
                "choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            },
        },
        agent_record_id="i1",
        created_at=1_700_000_000.5,
        artifact_ref=served,
    )
    observer.record_accepted(item)
    span = _spans(observer, exporter)["chat qwen"]

    expected = record_span_context("math", "i1")
    assert span.context.trace_id == expected.trace_id
    assert span.context.span_id == expected.span_id
    assert span.parent is None
    assert span.kind.name == "CLIENT"
    assert span.start_time == span.end_time == 1_700_000_000_500_000_000
    assert span.resource.attributes["service.name"] == "reef"
    assert dict(span.attributes) == {
        "session.id": "math",
        "reef.scenario": "math",
        "reef.agent_record_id": "i1",
        "reef.request_type": "inference",
        "reef.release_id": "v3",
        "reef.content_id": "c1",
        "reef.runtime_load_id": "load-3",
        "reef.tags": ("eval", "smoke"),
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "qwen",
        "gen_ai.response.model": "qwen-served",
        "gen_ai.response.id": "chatcmpl-1",
        "gen_ai.usage.input_tokens": 7,
        "gen_ai.usage.output_tokens": 1,
        "gen_ai.response.finish_reasons": ("stop",),
    }


@pytest.mark.unit
def test_include_messages_exports_the_exchange_and_feedback_text() -> None:
    observer, exporter = _observer(include_messages=True)
    observer.record_accepted(
        _inference(
            "i1",
            model="qwen",
            system="be brief",
            messages=[{"role": "user", "content": "hi"}],
            response={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        )
    )
    observer.record_accepted(_report("r1", "i1", feedback={"note": "good"}))
    spans = _spans(observer, exporter)
    assert json.loads(spans["chat qwen"].attributes["gen_ai.input.messages"]) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert json.loads(spans["chat qwen"].attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "hello"}
    ]
    assert json.loads(spans["feedback"].attributes["reef.feedback"]) == {"note": "good"}


@pytest.mark.unit
def test_feedback_joins_the_first_referenced_trace_and_links_the_rest() -> None:
    from reef.observability.open_telemetry import record_span_context

    observer, exporter = _observer()
    observer.record_accepted(_inference("i1", model="qwen"))
    observer.record_accepted(_inference("i2", model="qwen"))
    observer.record_accepted(_report("r1", "i1", "i2", score=0.25, feedback="secret transcript"))
    observer.record_accepted(_report("r2"))
    observer.close()
    spans = {span.attributes["reef.agent_record_id"]: span for span in exporter.get_finished_spans()}

    scored = spans["r1"]
    first = record_span_context("math", "i1")
    assert scored.name == "feedback"
    assert scored.context.trace_id == first.trace_id == spans["i1"].context.trace_id
    assert scored.parent.span_id == first.span_id
    assert scored.context.span_id == record_span_context("math", "r1").span_id
    assert [link.context.trace_id for link in scored.links] == [spans["i2"].context.trace_id]
    assert scored.attributes["reef.score"] == 0.25
    assert scored.attributes["reef.references"] == ("i1", "i2")
    assert "reef.feedback" not in scored.attributes

    unreferenced = spans["r2"]
    assert unreferenced.parent is None
    assert unreferenced.context.trace_id == record_span_context("math", "r2").trace_id


@pytest.mark.unit
def test_committed_step_adds_a_child_span_below_every_consumed_record() -> None:
    from reef.observability.open_telemetry import commit_span_context, record_span_context

    observer, exporter = _observer()
    commit = CommittedStepEvent(
        scenario="math",
        step=3,
        artifact_ref=ArtifactRef(content_id="c3", release_id="v3", parent_release_id="v2"),
        operation="training",
        checkpoint=True,
        pending=False,
        consumed_ids=frozenset({"i1", "i2"}),
        compacted_ids=frozenset({"i1"}),
        recorded_at=1_700_000_100.0,
        metrics={"loss": 0.5, "selected": True, "label": "ok", "nan": float("nan"), "nested": {"x": 1}},
        training_job_id="job-3",
    )
    observer.record_committed(commit)
    observer.close()
    spans = exporter.get_finished_spans()

    (step,) = [span for span in spans if span.name == "training step 3"]
    assert step.context.trace_id == commit_span_context("math", 3).trace_id
    assert step.context.span_id == commit_span_context("math", 3).span_id
    assert step.parent is None
    assert step.attributes["reef.step"] == 3
    assert step.attributes["reef.release_id"] == "v3"
    assert step.attributes["reef.consumed_record_count"] == 2
    assert step.attributes["reef.compacted_record_count"] == 1
    assert step.attributes["reef.training_job_id"] == "job-3"
    assert step.attributes["reef.metrics.loss"] == 0.5
    assert step.attributes["reef.metrics.selected"] is True
    assert step.attributes["reef.metrics.label"] == "ok"
    assert "reef.metrics.nan" not in step.attributes
    assert "reef.metrics.nested" not in step.attributes

    consumed = {span.attributes["reef.agent_record_id"]: span for span in spans if span.name == "trained in step 3"}
    assert set(consumed) == {"i1", "i2"}
    for agent_record_id, span in consumed.items():
        expected_parent = record_span_context("math", agent_record_id)
        assert span.context.trace_id == expected_parent.trace_id
        assert span.parent.span_id == expected_parent.span_id
        assert [link.context.span_id for link in span.links] == [step.context.span_id]
        assert span.attributes["reef.release_id"] == "v3"


# -- dispatcher integration ---------------------------------------------------


class _CapturingObserver(RecordObserver):
    def __init__(self) -> None:
        self.accepted: list[str] = []
        self.committed: list[CommittedStepEvent] = []
        self.closed = False

    def record_accepted(self, item: AgentRecord) -> None:
        self.accepted.append(item.agent_record_id)

    def record_committed(self, commit: CommittedStepEvent) -> None:
        self.committed.append(commit)

    def close(self) -> None:
        self.closed = True


class _FailingObserver(_CapturingObserver):
    def record_accepted(self, item: AgentRecord) -> None:
        super().record_accepted(item)
        raise RuntimeError("collector down")

    def record_committed(self, commit: CommittedStepEvent) -> None:
        super().record_committed(commit)
        raise RuntimeError("collector down")


@pytest.mark.unit
@pytest.mark.parametrize("observer_type", [_CapturingObserver, _FailingObserver])
def test_dispatcher_reports_accepted_records_and_commits_without_depending_on_the_observer(
    tmp_path, observer_type
) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    observer = observer_type()
    dispatcher = Dispatcher(
        TestPolicyRecipe(
            **runtime_bindings(RecordingRuntime(served_version="w0")),
            batch_size=1,
            checkpoint_strategy=EveryNVersions(1000),
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        record_observer=observer,
    )
    try:
        dispatcher.accept_record(_inference("i1"))
        dispatcher.accept_record(_inference("i1"))
        dispatcher.accept_record(_report("r1", "i1"))
        wait_for_step(dispatcher, 1)
    finally:
        dispatcher.close()

    assert observer.accepted == ["i1", "r1"]
    (commit,) = observer.committed
    assert commit.scenario == "math"
    assert commit.step == 1
    assert commit.consumed_ids == frozenset({"i1", "r1"})
    assert observer.closed
