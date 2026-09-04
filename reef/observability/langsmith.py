"""Optional, asynchronous LangSmith adapter for Reef inference observations."""

from __future__ import annotations

import copy
import json
import logging
import queue
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any, Protocol

from reef.observability.inference import InferenceObserver, InferenceTrace, ReportFeedback

logger = logging.getLogger(__name__)

# This namespace is a public compatibility contract. Never change it after a
# release: any Reef process can reconstruct a LangSmith run id from a receipt.
REEF_LANGSMITH_RUN_NAMESPACE = uuid.UUID("da12da5a-abb9-5c19-a395-6f93f23f25ee")
REEF_LANGSMITH_FEEDBACK_NAMESPACE = uuid.UUID("b6dc876a-0d16-5710-8fad-9e10b668dbb3")

_DEFAULT_REDACT_KEYS = (
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "api_key",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "cookie",
    "set-cookie",
)
_REDACTED = "[REDACTED]"


class LangSmithClientFactory(Protocol):
    """Construct the SDK client lazily on the exporter thread."""

    def __call__(self) -> Any: ...


def langsmith_run_id(agent_record_id: str) -> uuid.UUID:
    """Deterministically map any Reef receipt string to a LangSmith UUID."""

    if not isinstance(agent_record_id, str) or not agent_record_id:
        raise ValueError("agent_record_id must be a non-empty string")
    return uuid.uuid5(REEF_LANGSMITH_RUN_NAMESPACE, agent_record_id)


def langsmith_feedback_id(report_record_id: str, reference: str) -> uuid.UUID:
    """Stable id for one report/reference edge, making retries idempotent."""

    return uuid.uuid5(REEF_LANGSMITH_FEEDBACK_NAMESPACE, f"{report_record_id}\0{reference}")


def _bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"observability.langsmith.{field_name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class LangSmithConfig:
    enabled: bool = False
    project: str | None = None
    endpoint: str | None = None
    include_inputs: bool = True
    include_outputs: bool = True
    include_metadata: bool = True
    redact_keys: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_REDACT_KEYS)
    queue_size: int = 1024
    batch_size: int = 32
    flush_timeout_s: float = 2.0

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.project)

    @classmethod
    def from_mapping(cls, value: object) -> LangSmithConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("observability.langsmith must be an object")
        enabled = _bool(value.get("enabled"), field_name="enabled", default=False)
        project_value = value.get("project")
        if project_value is not None and not isinstance(project_value, str):
            raise ValueError("observability.langsmith.project must be a string")
        project = project_value.strip() if isinstance(project_value, str) and project_value.strip() else None
        endpoint_value = value.get("endpoint")
        if endpoint_value is not None and not isinstance(endpoint_value, str):
            raise ValueError("observability.langsmith.endpoint must be a string")
        endpoint = endpoint_value.strip() if isinstance(endpoint_value, str) and endpoint_value.strip() else None
        raw_redact_keys = value.get("redact_keys", _DEFAULT_REDACT_KEYS)
        if (
            isinstance(raw_redact_keys, str)
            or not isinstance(raw_redact_keys, Sequence)
            or any(not isinstance(key, str) or not key.strip() for key in raw_redact_keys)
        ):
            raise ValueError("observability.langsmith.redact_keys must be a list of non-empty strings")
        queue_size = int(value.get("queue_size", 1024))
        batch_size = int(value.get("batch_size", 32))
        flush_timeout_s = float(value.get("flush_timeout_s", 2.0))
        if queue_size <= 0 or batch_size <= 0 or flush_timeout_s < 0:
            raise ValueError(
                "observability.langsmith requires positive queue_size/batch_size and non-negative flush_timeout_s"
            )
        return cls(
            enabled=enabled,
            project=project,
            endpoint=endpoint,
            include_inputs=_bool(value.get("include_inputs"), field_name="include_inputs", default=True),
            include_outputs=_bool(value.get("include_outputs"), field_name="include_outputs", default=True),
            include_metadata=_bool(value.get("include_metadata"), field_name="include_metadata", default=True),
            redact_keys=tuple(dict.fromkeys(key.strip().lower() for key in raw_redact_keys)),
            queue_size=queue_size,
            batch_size=batch_size,
            flush_timeout_s=flush_timeout_s,
        )


def _redact(value: Any, keys: frozenset[str]) -> Any:
    """Copy JSON-like data while recursively redacting configured key names."""

    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if str(key).lower() in keys else _redact(item, keys) for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(item, keys) for item in value]
    return copy.deepcopy(value)


def _utc(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


class LangSmithInferenceObserver(InferenceObserver):
    """Bounded background exporter using only the LangSmith client surface."""

    def __init__(self, config: LangSmithConfig, client_factory: LangSmithClientFactory) -> None:
        self._config = config
        self._client_factory = client_factory
        self._queue: queue.Queue[InferenceTrace | ReportFeedback] = queue.Queue(maxsize=config.queue_size)
        self._closing = Event()
        self._closed = Event()
        self._thread = Thread(target=self._run, name="reef-langsmith-export", daemon=True)
        self._thread.start()

    def _enqueue(self, value: InferenceTrace | ReportFeedback) -> None:
        if self._closing.is_set():
            return
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            logger.warning("LangSmith observation queue is full; dropping one event")

    def record_inference(self, trace: InferenceTrace) -> None:
        try:
            self._enqueue(copy.deepcopy(trace))
        except Exception as exc:
            logger.warning("LangSmith observer rejected an inference snapshot (%s)", type(exc).__name__)

    def record_feedback(self, feedback: ReportFeedback) -> None:
        try:
            self._enqueue(copy.deepcopy(feedback))
        except Exception as exc:
            logger.warning("LangSmith observer rejected a feedback snapshot (%s)", type(exc).__name__)

    def _run(self) -> None:
        client = None
        try:
            try:
                client = self._client_factory()
            except Exception as exc:
                logger.warning(
                    "LangSmith client initialization failed (%s); observations will be dropped", type(exc).__name__
                )
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    first = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                batch = [first]
                while len(batch) < self._config.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:  # noqa: PERF203 - non-blocking bounded queue drain
                        break
                for event in batch:
                    try:
                        if client is not None:
                            if isinstance(event, InferenceTrace):
                                self._export_inference(client, event)
                            else:
                                self._export_feedback(client, event)
                    except Exception as exc:  # noqa: PERF203 - isolate each provider call
                        logger.warning("LangSmith export failed (%s); Reef state is unaffected", type(exc).__name__)
                    finally:
                        self._queue.task_done()
        finally:
            if client is not None:
                try:
                    flush = getattr(client, "flush", None)
                    if callable(flush):
                        flush()
                except Exception as exc:
                    logger.warning("LangSmith flush failed (%s)", type(exc).__name__)
                try:
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
                except Exception as exc:
                    logger.warning("LangSmith client close failed (%s)", type(exc).__name__)
            self._closed.set()

    def _metadata(self, trace: InferenceTrace) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "reef.agent_record_id": trace.agent_record_id,
            "reef.scenario": trace.scenario,
            "reef.request_type": "inference",
            "reef.path": trace.path,
            "reef.retry_count": trace.retry_count,
            "reef.completion_state": trace.completion_state,
            "reef.delivery_state": trace.delivery_state,
            "reef.record_accepted": trace.record_accepted,
            "reef.streaming": trace.streaming,
        }
        optional = {
            "reef.recipe": trace.recipe,
            "reef.content_id": trace.content_id,
            "reef.release_id": trace.release_id,
            "reef.serving_runtime_load_id": trace.serving_runtime_load_id,
        }
        metadata.update({key: value for key, value in optional.items() if value is not None})
        if self._config.include_metadata:
            sanitized = _redact(trace.metadata, frozenset(self._config.redact_keys))
            if isinstance(sanitized, Mapping):
                metadata["reef.metadata"] = dict(sanitized)
                tags = sanitized.get("tags")
                if isinstance(tags, Mapping):
                    metadata.update({f"reef.tag.{key}": value for key, value in tags.items()})
        return metadata

    def _export_inference(self, client: Any, trace: InferenceTrace) -> None:
        run_id = langsmith_run_id(trace.agent_record_id)
        keys = frozenset(self._config.redact_keys)
        inputs = _redact(trace.inputs, keys) if self._config.include_inputs else {}
        outputs = None
        if trace.outputs is not None and self._config.include_outputs:
            outputs = _redact(trace.outputs, keys)
        metadata = self._metadata(trace)
        create: dict[str, Any] = {
            "id": run_id,
            "start_time": _utc(trace.started_at),
            "end_time": _utc(trace.ended_at),
            "project_name": self._config.project,
            "extra": {"metadata": metadata},
        }
        if outputs is not None:
            create["outputs"] = outputs
        if trace.error is not None:
            create["error"] = trace.error
        # A complete run is one batchable operation. This avoids a PATCH racing
        # ahead of the SDK's asynchronous create queue while retaining exact
        # start/end timestamps for the backend and durable-acceptance span.
        client.create_run("reef.inference", inputs, "llm", **create)

    def _export_feedback(self, client: Any, feedback: ReportFeedback) -> None:
        keys = frozenset(self._config.redact_keys)
        value = _redact(feedback.feedback, keys) if feedback.feedback is not None else None
        source_info: dict[str, Any] = {
            "reef.report_record_id": feedback.report_record_id,
            "reef.scenario": feedback.scenario,
        }
        if self._config.include_metadata and feedback.metadata:
            source_info["reef.metadata"] = _redact(feedback.metadata, keys)
        for reference in feedback.references:
            kwargs: dict[str, Any] = {
                "key": "reef.report",
                "feedback_id": langsmith_feedback_id(feedback.report_record_id, reference),
                "source_info": source_info,
            }
            if feedback.score is not None:
                kwargs["score"] = feedback.score
            if value is not None:
                kwargs["value"] = value
                if isinstance(value, Mapping):
                    kwargs["comment"] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            client.create_feedback(langsmith_run_id(reference), **kwargs)

    def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        self._thread.join(timeout=self._config.flush_timeout_s)
        if self._thread.is_alive():
            logger.warning("LangSmith observer shutdown timed out; remaining events will be dropped")


__all__ = [
    "REEF_LANGSMITH_FEEDBACK_NAMESPACE",
    "REEF_LANGSMITH_RUN_NAMESPACE",
    "LangSmithClientFactory",
    "LangSmithConfig",
    "LangSmithInferenceObserver",
    "langsmith_feedback_id",
    "langsmith_run_id",
]
