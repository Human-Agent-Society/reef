"""Transport-independent request handling behind the aiohttp routes.

``RequestService`` normalizes typed Reef payloads, resolves the artifact
version before every provider call so concurrent publication cannot change
what gets recorded, stores each exchange as a record, and applies the
scenario surface's request/response checks. No aiohttp types appear here;
``reef.service.routes`` adapts these methods to HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactNotFound, ArtifactRef
from reef.core.errors import ReefError, UnknownScenario
from reef.core.records_types import RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import available_adapters, get_adapter
from reef.observability import InferenceObserver, InferenceTrace, NullInferenceObserver, ReportFeedback
from reef.recipe.errors import RecipeConfigError
from reef.records import AgentRecord, RecordConflict
from reef.runtime.base import InferenceAdmissionHandle, TrainingRuntime
from reef.runtime.inference import InferenceBackend, InferenceStream
from reef.scenario.scenario import Scenario
from reef.service.install_script import render_install_script
from reef.service.wire import SCENARIO_HEADER, ReportPayload, RequestHeaders, parse_request_headers
from reef.surface.base import InferenceLease, LeasingInferenceHooks, Surface
from reef.surface.weights import RuntimeLoadMismatch, reported_runtime_load_id, reported_runtime_load_spans

logger = logging.getLogger(__name__)


def _random_harness_scenario_name() -> str:
    return f"harness-{uuid.uuid4().hex[:12]}"


def _inference_aborted(response: Mapping[str, Any]) -> bool:
    def aborted(value: Any) -> bool:
        return value == "abort" or (isinstance(value, Mapping) and value.get("type") == "abort")

    meta = response.get("meta_info")
    if aborted(response.get("finish_reason")) or (isinstance(meta, Mapping) and aborted(meta.get("finish_reason"))):
        return True
    training = response.get("training")
    if isinstance(training, Mapping) and aborted(training.get("finish_reason")):
        return True
    choices = response.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, Mapping)
        and (
            aborted(choice.get("finish_reason"))
            or (isinstance(choice.get("meta_info"), Mapping) and aborted(choice["meta_info"].get("finish_reason")))
        )
        for choice in choices
    )


class RequestPayloadNormalizer:
    """Normalize only typed Reef payloads while preserving native bodies."""

    def __init__(self) -> None:
        self._normalizers: dict[
            RequestType,
            Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], tuple[str, ...]]],
        ] = {
            RequestType.REPORT: self._normalize_report,
        }

    def normalize(
        self,
        request_type: RequestType,
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        normalizer = self._normalizers.get(request_type)
        if normalizer is None:
            return dict(payload), ()
        return normalizer(payload)

    @staticmethod
    def _normalize_report(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        report = ReportPayload.from_dict(payload)
        return report.to_dict(), report.references


@dataclass(frozen=True)
class PendingInference:
    item: AgentRecord
    release_id: str | None
    admission: InferenceAdmissionHandle | None = None
    lease: InferenceLease | None = None
    deferred_prepared: PreparedInference | None = None
    path: str | None = None
    started_at: float = 0.0
    recipe: str | None = None


@dataclass(frozen=True)
class PreparedInference:
    """Everything one inference attempt froze before calling the provider."""

    parsed: RequestHeaders
    artifact: Artifact
    backend: InferenceBackend
    surface: Surface
    #: True when a training runtime serves the scenario: the recorded payload
    #: must then carry the engine-confirmed runtime load ID.
    durable: bool
    recipe: str
    admission: InferenceAdmissionHandle | None = None
    #: Releases serving state the surface held for this attempt (an adapter
    #: lease); called exactly once when the attempt ends.
    lease: InferenceLease | None = None

    def release(self) -> None:
        try:
            if self.lease is not None:
                self.lease.release()
        finally:
            if self.admission is not None:
                self.admission.release()


@dataclass(frozen=True)
class InferenceRetryPolicy:
    initial_s: float = 0.05
    max_s: float = 1.0
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not 0 < self.initial_s <= self.max_s or self.timeout_s <= 0:
            raise ValueError("inference retry policy requires 0 < initial_s <= max_s and timeout_s > 0")


class InferenceRetryTimeout(ReefError):
    """Inference attempts ending with a backend ``abort`` exhausted their retry deadline."""


class RequestService:
    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        retry_policy: InferenceRetryPolicy | None = None,
        inference_observer: InferenceObserver | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._payload_normalizer = RequestPayloadNormalizer()
        self._retry_policy = retry_policy or InferenceRetryPolicy()
        self._inference_observer = inference_observer or NullInferenceObserver()

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    def close(self) -> None:
        """Best-effort observer flush; durable service shutdown never depends on it."""

        try:
            self._inference_observer.close()
        except Exception as exc:
            logger.warning("inference observer close failed (%s)", type(exc).__name__)

    def _observe_inference(self, trace: InferenceTrace) -> None:
        try:
            self._inference_observer.record_inference(trace)
        except Exception as exc:
            logger.warning("inference observer failed (%s)", type(exc).__name__)

    def _observe_feedback(self, feedback: ReportFeedback) -> None:
        try:
            self._inference_observer.record_feedback(feedback)
        except Exception as exc:
            logger.warning("inference observer feedback failed (%s)", type(exc).__name__)

    @staticmethod
    def _require_inference(headers: Mapping[str, str]) -> RequestHeaders:
        return parse_request_headers(headers, RequestType.INFERENCE)

    def accept(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        *,
        request_type: RequestType,
        agent_record_id: str | None = None,
    ) -> AgentRecord:
        if request_type is RequestType.INFERENCE:
            raise ValueError("inference requests must use infer()")
        parsed = parse_request_headers(headers, request_type)
        existing = None
        if agent_record_id is not None:
            scenario = self._dispatcher.get_or_create_scenario(
                parsed.scenario,
                release_id=parsed.release_id,
            )
            if scenario is not None:
                existing = scenario.records.get(parsed.scenario, agent_record_id)
        item = self._accept(parsed, payload, agent_record_id=agent_record_id)
        if existing is None and item.request_type is RequestType.REPORT and item.references:
            self._observe_feedback(
                ReportFeedback(
                    report_record_id=item.agent_record_id,
                    scenario=item.scenario,
                    references=item.references,
                    score=item.payload.get("score"),
                    feedback=item.payload.get("feedback"),
                    metadata=item.payload.get("metadata", {}),
                )
            )
        return item

    async def infer(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        backend: InferenceBackend,
    ) -> dict[str, Any]:
        response, _ = await self.infer_with_data(headers, payload, path, backend)
        return response

    async def infer_with_data(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        backend: InferenceBackend | None = None,
    ) -> tuple[dict[str, Any], AgentRecord]:
        original_payload = dict(payload)
        requested = self._require_inference(headers)
        agent_record_id = requested.agent_record_id or uuid.uuid4().hex
        trace_started_at = time.time()
        retry_delay = self._retry_policy.initial_s
        loop = asyncio.get_running_loop()
        remaining_budget = self._retry_policy.timeout_s
        timeout_error = f"inference retry deadline exceeded ({self._retry_policy.timeout_s:g}s)"
        attempt = 0
        prepared: PreparedInference | None = None
        try:
            while True:
                attempt += 1
                prepared, payload = await self._prepare_request(headers, original_payload, path, backend)
                try:
                    if prepared.durable:
                        payload = {**payload, "return_meta_info": True}
                    if requested.agent_record_id is not None:
                        replayed = self._replay_inference(prepared, payload, requested.agent_record_id)
                        if replayed is not None:
                            return replayed
                    if remaining_budget <= 0:
                        raise InferenceRetryTimeout(timeout_error)
                    started = loop.time()
                    try:
                        response = await asyncio.wait_for(
                            prepared.backend.inference(prepared.artifact, path, payload),
                            timeout=remaining_budget,
                        )
                    except TimeoutError as exc:
                        logger.warning(
                            "inference for scenario %r timed out after %d attempt(s) at artifact %r",
                            prepared.parsed.scenario,
                            attempt,
                            prepared.artifact.ref.release_id,
                        )
                        raise InferenceRetryTimeout(timeout_error) from exc
                    finally:
                        remaining_budget -= loop.time() - started
                    interrupted = _inference_aborted(response)
                    if not interrupted:
                        # A completed response with invalid runtime-load-ID information is a
                        # backend contract error, not a retryable inference abort.
                        if prepared.surface.inference is not None:
                            prepared.surface.inference.verify_response(prepared.artifact, path, response)
                        self._stamp_durable_runtime_load_id(prepared, payload, response)
                        item = await asyncio.to_thread(
                            self._accept,
                            prepared.parsed,
                            {**payload, "response": response},
                            agent_record_id=agent_record_id,
                            artifact_ref=prepared.artifact.ref,
                        )
                        client_response = client_inference_response(response)
                        self._observe_inference(
                            self._completed_trace(
                                item=item,
                                prepared=prepared,
                                path=path,
                                started_at=trace_started_at,
                                retry_count=attempt - 1,
                                inputs=original_payload,
                                outputs=client_response,
                                streaming=False,
                            )
                        )
                        return client_response, item
                    # A backend ``abort`` finish reason makes the attempt unusable.
                    # Restart the request against the latest artifact and never record it.
                    logger.info(
                        "retrying backend-aborted inference for scenario %r (attempt %d): frozen artifact %r, "
                        "engine reported runtime load ID %r",
                        prepared.parsed.scenario,
                        attempt,
                        prepared.artifact.ref.release_id,
                        reported_runtime_load_id(response),
                    )
                finally:
                    prepared.release()
                if remaining_budget <= 0:
                    raise InferenceRetryTimeout(timeout_error)
                sleep_for = min(retry_delay, remaining_budget)
                await asyncio.sleep(sleep_for)
                remaining_budget -= sleep_for
                retry_delay = min(retry_delay * 2, self._retry_policy.max_s)
        except BaseException as exc:
            self._observe_inference(
                self._failed_trace(
                    agent_record_id=agent_record_id,
                    headers=headers,
                    prepared=prepared,
                    path=path,
                    started_at=trace_started_at,
                    retry_count=max(0, attempt - 1),
                    inputs=original_payload,
                    error=type(exc).__name__,
                    streaming=False,
                )
            )
            raise

    async def start_stream(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        backend: InferenceBackend | None = None,
    ) -> tuple[InferenceStream, PendingInference]:
        original_payload = dict(payload)
        requested = self._require_inference(headers)
        agent_record_id = requested.agent_record_id or uuid.uuid4().hex
        trace_started_at = time.time()
        prepared: PreparedInference | None = None
        try:
            prepared, payload = await self._prepare_request(headers, payload, path, backend)
        except BaseException as exc:
            self._observe_inference(
                self._failed_trace(
                    agent_record_id=agent_record_id,
                    headers=headers,
                    prepared=prepared,
                    path=path,
                    started_at=trace_started_at,
                    retry_count=0,
                    inputs=original_payload,
                    error=type(exc).__name__,
                    streaming=True,
                )
            )
            raise
        admission = prepared.admission
        lease = prepared.lease
        try:
            stream = await prepared.backend.inference_stream(prepared.artifact, path, payload)
            record_response = getattr(stream, "record_response", None)
            record_response_pending = bool(getattr(stream, "record_response_pending", False))
            if record_response is not None:
                if prepared.surface.inference is not None:
                    prepared.surface.inference.verify_response(prepared.artifact, path, record_response)
                self._stamp_durable_runtime_load_id(prepared, payload, record_response)
                # Buffered streaming backends have already finished model
                # execution. Downstream client backpressure must not leave a
                # stale admission handle across the colocated pause lifecycle.
                if admission is not None:
                    admission.release()
                    admission = None
                if lease is not None:
                    lease.release()
                    lease = None
            elif prepared.durable and not record_response_pending:
                raise RuntimeLoadMismatch(
                    "durable streaming inference requires an atomic record_response with serving runtime load IDs"
                )
        except BaseException as exc:
            try:
                if "stream" in locals():
                    await stream.close()
            finally:
                try:
                    if lease is not None:
                        lease.release()
                finally:
                    if admission is not None:
                        admission.release()
            self._observe_inference(
                self._failed_trace(
                    agent_record_id=agent_record_id,
                    headers=headers,
                    prepared=prepared,
                    path=path,
                    started_at=trace_started_at,
                    retry_count=0,
                    inputs=original_payload,
                    error=type(exc).__name__,
                    streaming=True,
                )
            )
            raise
        pending = PendingInference(
            item=AgentRecord.create(
                scenario=prepared.parsed.scenario,
                request_type=RequestType.INFERENCE,
                payload=_with_tags(payload, prepared.parsed),
                agent_record_id=agent_record_id,
                artifact_ref=prepared.artifact.ref,
            ),
            release_id=prepared.parsed.release_id,
            admission=admission,
            lease=lease,
            deferred_prepared=prepared if record_response_pending else None,
            path=path,
            started_at=trace_started_at,
            recipe=prepared.recipe,
        )
        return stream, pending

    def record_stream(self, pending: PendingInference, response: Mapping[str, Any]) -> AgentRecord:
        stored: AgentRecord | None = None
        try:
            payload = dict(pending.item.payload)
            # A token-native streaming backend fills record_response only when
            # the upstream generation finishes. Validate that final capture
            # here, after the route has drained the stream but before it can
            # become a training record. Incomplete/disconnected streams have
            # no training block and remain delivery diagnostics only.
            if pending.deferred_prepared is not None and isinstance(response.get("training"), Mapping):
                if pending.path is None:
                    raise ReefError("deferred inference response has no request path")
                hooks = pending.deferred_prepared.surface.inference
                if hooks is not None:
                    hooks.verify_response(
                        pending.deferred_prepared.artifact,
                        pending.path,
                        response,
                    )
                self._stamp_durable_runtime_load_id(pending.deferred_prepared, payload, response)
            item = replace(
                pending.item,
                payload={**payload, "response": dict(response)},
            )
            stored = self._dispatcher.accept_record(
                item,
                release_id=pending.release_id,
            )
            self._observe_inference(self._stream_trace(pending, stored, response, record_accepted=True))
            return stored
        except Exception:
            logger.exception(
                "dispatcher rejected the stream record for scenario %r (record %s)",
                pending.item.scenario,
                pending.item.agent_record_id,
            )
            self._observe_inference(self._stream_trace(pending, pending.item, response, record_accepted=False))
            raise
        finally:
            try:
                if pending.lease is not None:
                    pending.lease.release()
            finally:
                if pending.admission is not None:
                    pending.admission.release()

    async def _prepare_request(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        path: str,
        backend: InferenceBackend | None,
    ) -> tuple[PreparedInference, dict[str, Any]]:
        """The shared first half of every inference: freeze the serving state
        (headers, scenario, artifact, backend, surface) and let the surface
        transform the request payload."""
        parsed = self._require_inference(headers)
        initial = await asyncio.to_thread(
            self._dispatcher.get_or_create_scenario,
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if initial is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        admission = await initial.runtime.acquire_inference() if initial.runtime is not None else None
        try:
            # Re-resolve after admission: a queued request must freeze the head
            # committed by the weight update that released it, never the head it
            # observed before waiting.
            prepared = await asyncio.to_thread(self._prepare_inference, parsed, backend, admission)
            hooks = prepared.surface.inference
            transformed = (
                dict(payload)
                if hooks is None
                else await asyncio.to_thread(
                    hooks.prepare_request,
                    prepared.artifact,
                    path,
                    dict(payload),
                )
            )
            if isinstance(hooks, LeasingInferenceHooks):
                # Freeze the served adapter for the attempt: the surface has
                # named it, so it must stay resident until the attempt ends.
                lease = await asyncio.to_thread(hooks.begin_request, prepared.artifact, path)
                prepared = replace(prepared, lease=lease)
            return prepared, transformed
        except BaseException:
            if admission is not None:
                admission.release()
            raise

    @staticmethod
    def _stamp_durable_runtime_load_id(
        prepared: PreparedInference,
        payload: dict[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        """Record which engine weights answered a training-scenario request."""
        if not prepared.durable:
            return
        spans = reported_runtime_load_spans(response)
        if spans:
            payload["runtime_load_spans"] = [
                {"start": span.start, "end": span.end, "runtime_load_id": span.runtime_load_id} for span in spans
            ]
            versions = {span.runtime_load_id for span in spans}
            if len(versions) == 1:
                payload["runtime_load_id"] = versions.pop()
            else:
                payload.pop("runtime_load_id", None)
            return
        if (version := reported_runtime_load_id(response)) is None:
            raise RuntimeLoadMismatch("durable training response reports no runtime_load_id")
        payload["runtime_load_id"] = version

    def _prepare_inference(
        self,
        parsed: RequestHeaders,
        backend: InferenceBackend | None,
        admission: InferenceAdmissionHandle | None,
    ) -> PreparedInference:
        scenario = self._dispatcher.get_or_create_scenario(
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        selected_backend = backend if backend is not None else scenario.inference_backend
        if selected_backend is None:
            raise RecipeConfigError("the served recipe has no inference backend")
        ref = scenario.current_artifact_ref()
        return PreparedInference(
            parsed=parsed,
            artifact=Artifact(ref, scenario.repository),
            backend=selected_backend,
            surface=scenario.surface,
            durable=isinstance(scenario.runtime, TrainingRuntime),
            recipe=self._dispatcher.recipe_name,
            admission=admission,
        )

    def _replay_inference(
        self,
        prepared: PreparedInference,
        payload: Mapping[str, Any],
        agent_record_id: str,
    ) -> tuple[dict[str, Any], AgentRecord] | None:
        """Return an accepted client-id retry without calling the provider.

        The stored response and Reef-generated runtime-load fields are not
        part of request equality. A reused id with different request content
        retains the record store's conflict semantics.
        """

        scenario = self._dispatcher.get_or_create_scenario(
            prepared.parsed.scenario,
            release_id=prepared.parsed.release_id,
        )
        if scenario is None:
            return None
        stored = scenario.records.get(prepared.parsed.scenario, agent_record_id)
        if stored is None:
            return None
        if stored.request_type is not RequestType.INFERENCE:
            raise RecordConflict(f"agent_record_id {agent_record_id!r} already has different content")
        stored_request = dict(stored.payload)
        response = stored_request.pop("response", None)
        stored_request.pop("runtime_load_id", None)
        stored_request.pop("runtime_load_spans", None)
        expected = dict(_with_tags(payload, prepared.parsed))
        if stored_request != expected or not isinstance(response, Mapping):
            raise RecordConflict(f"agent_record_id {agent_record_id!r} already has different content")
        # The original acceptance already exported this deterministic run id.
        # Skipping observer emission also avoids duplicate client calls.
        return client_inference_response(response), stored

    @staticmethod
    def _trace_metadata(item: AgentRecord) -> Mapping[str, Any]:
        metadata = item.payload.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    @staticmethod
    def _serving_runtime_load_id(item: AgentRecord) -> str | None:
        value = item.payload.get("runtime_load_id")
        return value if isinstance(value, str) else None

    def _completed_trace(
        self,
        *,
        item: AgentRecord,
        prepared: PreparedInference,
        path: str,
        started_at: float,
        retry_count: int,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        streaming: bool,
    ) -> InferenceTrace:
        return InferenceTrace(
            agent_record_id=item.agent_record_id,
            scenario=item.scenario,
            recipe=prepared.recipe,
            path=path,
            started_at=started_at,
            ended_at=time.time(),
            retry_count=retry_count,
            completion_state="complete",
            delivery_state="successful",
            record_accepted=True,
            streaming=streaming,
            inputs=inputs,
            outputs=outputs,
            metadata=self._trace_metadata(item),
            content_id=prepared.artifact.ref.content_id,
            release_id=prepared.artifact.ref.release_id,
            serving_runtime_load_id=self._serving_runtime_load_id(item),
        )

    def _failed_trace(
        self,
        *,
        agent_record_id: str,
        headers: Mapping[str, str],
        prepared: PreparedInference | None,
        path: str,
        started_at: float,
        retry_count: int,
        inputs: Mapping[str, Any],
        error: str,
        streaming: bool,
    ) -> InferenceTrace:
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        scenario = prepared.parsed.scenario if prepared is not None else normalized_headers.get(SCENARIO_HEADER, "")
        tags = prepared.parsed.tags if prepared is not None else {}
        ref = None if prepared is None else prepared.artifact.ref
        return InferenceTrace(
            agent_record_id=agent_record_id,
            scenario=scenario,
            recipe=None if prepared is None else prepared.recipe,
            path=path,
            started_at=started_at,
            ended_at=time.time(),
            retry_count=retry_count,
            completion_state="backend_error",
            delivery_state="failed",
            record_accepted=False,
            streaming=streaming,
            inputs=inputs,
            metadata={"tags": dict(tags)} if tags else {},
            content_id=None if ref is None else ref.content_id,
            release_id=None if ref is None else ref.release_id,
            error=error,
        )

    def _stream_trace(
        self,
        pending: PendingInference,
        item: AgentRecord,
        response: Mapping[str, Any],
        *,
        record_accepted: bool,
    ) -> InferenceTrace:
        delivery = response.get("stream_delivery")
        delivery_data = delivery if isinstance(delivery, Mapping) else response
        complete = delivery_data.get("complete") is True
        error_value = delivery_data.get("error")
        error = error_value if isinstance(error_value, str) else None
        if not record_accepted:
            completion_state, delivery_state = "rejected", "failed"
        elif complete:
            completion_state, delivery_state = "complete", "successful"
        elif error == "client disconnected":
            completion_state, delivery_state = "incomplete", "disconnected"
        elif error:
            completion_state, delivery_state = "backend_error", "failed"
        else:
            completion_state, delivery_state = "incomplete", "incomplete"
        ref = item.artifact_ref
        return InferenceTrace(
            agent_record_id=item.agent_record_id,
            scenario=item.scenario,
            recipe=pending.recipe,
            path=pending.path or "",
            started_at=pending.started_at,
            ended_at=time.time(),
            retry_count=0,
            completion_state=completion_state,
            delivery_state=delivery_state,
            record_accepted=record_accepted,
            streaming=True,
            inputs={key: value for key, value in pending.item.payload.items() if key != "metadata"},
            outputs=client_inference_response(response),
            metadata=self._trace_metadata(item),
            content_id=None if ref is None else ref.content_id,
            release_id=None if ref is None else ref.release_id,
            serving_runtime_load_id=self._serving_runtime_load_id(item),
            error=error if record_accepted else "RecordRejected",
        )

    def harness_manifest(self, headers: Mapping[str, str], release_id: str | None = None) -> dict[str, Any]:
        """The served tree plus its parent release and gate metrics.

        ``release_id`` addresses one catalog release instead of the
        serving head, so a consumer can pin or roll back by pulling an older
        tree; an unknown or unrestorable release raises ArtifactNotFound
        naming it. The gate field carries the metrics of the training step
        that published the served release, so a consumer can audit what a
        pulled tree changed and why it was admitted before running it.
        Read-only: never creates a scenario.
        """
        scenario = self._file_scenario(headers)
        return self._harness_manifest_for_scenario(scenario, release_id)

    @staticmethod
    def _harness_manifest_for_scenario(
        scenario: Scenario,
        release_id: str | None = None,
    ) -> dict[str, Any]:
        artifact, gate = scenario.artifact_snapshot(release_id)
        tree = scenario.surface.files
        if tree is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: the deployment's recipe "
                "carries no harness surface (record-only or weight-training recipes have "
                "no file tree). Point 'reef.recipe' at a harness_evolve recipe."
            )
        files = tree.read_files(artifact)
        if files is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: no harness composition has "
                "been published yet. The scenario's initial artifact carries no files "
                "until the trainer publishes its first step (see docs/user-guide/evolve-your-harness)."
            )
        return {
            "release_id": artifact.ref.release_id,
            "parent_release_id": artifact.ref.parent_release_id,
            "content_id": artifact.ref.content_id,
            "files": dict(files),
            "gate": gate,
        }

    def harness_releases(self, headers: Mapping[str, str]) -> dict[str, Any]:
        """The scenario's release catalog with per-release gate metrics, newest last.

        The list side of the update channel: every committed release stays
        addressable through the manifest read's ``release_id``, and each
        training row carries the metrics of the step that published it, so an
        update is a decision over numbers rather than a blind pull. Same
        read-only rules as ``harness_manifest``.
        """
        scenario = self._file_scenario(headers)
        return {
            "scenario": scenario.name,
            "releases": list(reversed(scenario.releases())),
        }

    def harness_install_script(
        self,
        headers: Mapping[str, str],
        adapter: str | None,
        release_id: str | None = None,
    ) -> str:
        """A self-contained install script over one served manifest.

        The manifest side is adapter-agnostic files, addressed exactly like
        ``harness_manifest`` (head by default, any catalog release through
        ``release_id``); the named ``adapter`` contributes only its
        descriptor's install section, which the script uses to ensure the
        pinned binary through the vendor's own channel. An unknown adapter
        raises ArtifactNotFound naming it, mirroring the unknown-release
        behavior; a known adapter whose descriptor declares no install
        section raises DescriptorError (HTTP 400) naming it. Unlike the other
        harness reads, a missing or empty scenario header creates a new,
        randomly named file-serving scenario.
        """
        if not adapter:
            raise ReefError("the harness install route requires an 'adapter' query parameter naming an adapter")
        known = available_adapters()
        if adapter not in known:
            raise ArtifactNotFound(f"unknown harness adapter {adapter!r}; known adapters: {', '.join(known)}")
        scenario = self._file_scenario(
            headers,
            create_if_missing=True,
            release_id=release_id,
        )
        manifest = self._harness_manifest_for_scenario(scenario, release_id)
        return render_install_script(
            descriptor=get_adapter(adapter),
            files=manifest["files"],
            release_id=manifest["release_id"],
            content_id=manifest["content_id"],
            scenario=scenario.name,
        )

    def _file_scenario(
        self,
        headers: Mapping[str, str],
        *,
        create_if_missing: bool = False,
        release_id: str | None = None,
    ) -> Scenario:
        """Resolve a file-serving scenario, optionally creating a randomly named one."""
        normalized = {key.lower(): value.strip() for key, value in headers.items()}
        if create_if_missing and not normalized.get(SCENARIO_HEADER):
            if not self._dispatcher.recipe_has_files():
                raise ArtifactNotFound("no harness recipes are available")

            scenario_name = _random_harness_scenario_name()
            scenario = self._dispatcher.get_or_create_scenario(
                scenario_name,
                release_id=release_id,
                allow_implicit_creation=True,
            )
            if scenario is None:
                raise ReefError("implicit harness scenario creation returned no scenario")
            return scenario

        parsed = self._require_inference(headers)
        if not self._dispatcher.has_scenario(parsed.scenario):
            raise ArtifactNotFound(f"unknown scenario {parsed.scenario!r}")
        scenario = self._dispatcher.get_or_create_scenario(parsed.scenario, release_id=parsed.release_id)
        if scenario is None:
            raise ReefError(f"scenario {parsed.scenario!r} disappeared during lookup")
        if scenario.surface.files is None:
            raise ArtifactNotFound(
                f"scenario {parsed.scenario!r} serves no files: the deployment's recipe "
                "carries no harness surface (record-only or weight-training recipes have "
                "no file tree). Point 'reef.recipe' at a harness_evolve recipe."
            )
        return scenario

    def _accept(
        self,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        *,
        agent_record_id: str | None = None,
        artifact_ref: ArtifactRef | None = None,
    ) -> AgentRecord:
        normalized_payload, references = self._payload_normalizer.normalize(parsed.request_type, payload)
        normalized_payload = _with_tags(normalized_payload, parsed)
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=normalized_payload,
            agent_record_id=agent_record_id,
            references=references,
            artifact_ref=artifact_ref,
        )
        try:
            return self._dispatcher.accept_record(
                item,
                release_id=parsed.release_id,
            )
        except Exception as exc:
            logger.warning(
                "dispatcher rejected %s record for scenario %r (record %s): %s: %s",
                parsed.request_type.value,
                parsed.scenario,
                item.agent_record_id,
                type(exc).__name__,
                exc,
            )
            raise


def _with_tags(payload: Mapping[str, Any], parsed: RequestHeaders) -> Mapping[str, Any]:
    """Carry ``x-reef-tag-*`` through to the INFERENCE record's metadata.

    Only inference: a tag is context about a served exchange, and the
    processors that read one correlate on the inference side. The service
    never interprets a value — it stores the pair and moves on
    (method-integration RFC §3.2).
    """
    if parsed.request_type is not RequestType.INFERENCE or not parsed.tags:
        return payload
    metadata = dict(payload.get("metadata") or {})
    metadata["tags"] = {**(metadata.get("tags") or {}), **parsed.tags}
    return {**payload, "metadata": metadata}


def client_inference_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Remove Reef-private training tensors from a buffered client response.

    The record keeps the block; no client ever sees it. (The
    ``x-reef-return-training`` opt-in existed for the external OpenClaw-RL
    grader, whose judging now runs in-processor off the records.)
    """

    client_response = dict(response)
    client_response.pop("training", None)
    return client_response


__all__ = [
    "InferenceRetryPolicy",
    "InferenceRetryTimeout",
    "PendingInference",
    "PreparedInference",
    "RequestPayloadNormalizer",
    "RequestService",
    "client_inference_response",
]
