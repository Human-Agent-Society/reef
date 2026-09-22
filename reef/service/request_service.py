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
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactError, ArtifactNotFound, ArtifactRef
from reef.core.errors import ReefError, UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.core.requirements import ancestor_requiring_nothing, required_by
from reef.core.training_request import TrainingRequest
from reef.dispatcher import Dispatcher
from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError
from reef.harness.tree.mutations import Mutation, MutationError
from reef.harness.tree.render import RenderError, render_composition
from reef.observability.operations import OperationMeasurement, OperationMetrics
from reef.recipe.errors import RecipeConfigError
from reef.runtime.interfaces import InferenceAdmissionHandle, InferenceHandler, InferenceStream
from reef.scenario.scenario import Scenario
from reef.service.install_script import TOKEN_PLACEHOLDER, render_install_script
from reef.service.release_page import before_release_id, build_release_page, result_of
from reef.service.request_page import STATE_WORDS, build_request_page, request_state, settled_step
from reef.service.wire import SCENARIO_HEADER, ProposalPayload, ReportPayload, RequestHeaders, parse_request_headers
from reef.surface.base import InferenceLease, LeasingInferenceHooks, Surface
from reef.surface.weights import RuntimeLoadMismatch, reported_runtime_load_id, reported_runtime_load_spans
from reef.train.cordis_backend.contracts import ProposalValidator, StepProgressReader, StepRecords
from reef.train.cordis_backend.proposals import ProposalInbox

logger = logging.getLogger(__name__)


def page_headers(headers: Mapping[str, str], query: Mapping[str, str]) -> dict[str, str]:
    """The headers a page route reads, ``?scenario=`` standing in for ``x-reef-scenario`` when that header is absent.

    A page is a link a person opens in a browser, which sends no ``x-reef-*``
    header; the header wins when both are present. The token has the same
    fallback in :mod:`reef.service.auth`, for the page routes alone.
    """
    merged = dict(headers.items())
    scenario = query.get("scenario", "").strip()
    if scenario and not any(key.lower() == SCENARIO_HEADER for key in merged):
        merged[SCENARIO_HEADER] = scenario
    return merged


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


def normalize_request_payload(
    request_type: RequestType,
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Normalize a typed Reef payload; a native provider body passes through."""
    if request_type is RequestType.TRAIN:
        request = TrainingRequest.from_dict(payload)
        return request.to_dict(), ()
    if request_type is not RequestType.REPORT:
        return dict(payload), ()
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
    measurement: OperationMeasurement | None = None


@dataclass(frozen=True)
class PreparedInference:
    """Everything one inference attempt froze before calling the provider."""

    parsed: RequestHeaders
    artifact: Artifact
    handler: InferenceHandler
    surface: Surface
    #: True when a training runtime serves the scenario: the recorded payload
    #: must then carry the engine-confirmed runtime load ID.
    durable: bool
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
    """Inference attempts ending with a handler ``abort`` exhausted their retry deadline."""


class RequestService:
    def __init__(self, dispatcher: Dispatcher, *, retry_policy: InferenceRetryPolicy | None = None) -> None:
        self._dispatcher = dispatcher
        self._retry_policy = retry_policy or InferenceRetryPolicy()

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

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
        return self._accept(parsed, payload, agent_record_id=agent_record_id)

    async def infer(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        handler: InferenceHandler,
    ) -> dict[str, Any]:
        response, _ = await self.infer_with_data(headers, payload, path, handler)
        return response

    async def infer_with_data(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        handler: InferenceHandler | None = None,
    ) -> tuple[dict[str, Any], AgentRecord]:
        operations = await self.inference_operations(headers)
        measurement = operations.start("serve/request")
        succeeded = False
        try:
            original_payload = dict(payload)
            retry_delay = self._retry_policy.initial_s
            loop = asyncio.get_running_loop()
            remaining_budget = self._retry_policy.timeout_s
            timeout_error = f"inference retry deadline exceeded ({self._retry_policy.timeout_s:g}s)"
            attempt = 0
            while True:
                attempt += 1
                if attempt > 1:
                    operations.increment("serve/retries_total")
                prepared, payload = await self._prepare_request(headers, original_payload, path, handler)
                try:
                    if prepared.durable:
                        payload = {**payload, "return_meta_info": True}
                    if remaining_budget <= 0:
                        raise InferenceRetryTimeout(timeout_error)
                    started = loop.time()
                    try:
                        response = await asyncio.wait_for(
                            prepared.handler.inference(prepared.artifact, path, payload),
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
                        # handler contract error, not a retryable inference abort.
                        if prepared.surface.inference is not None:
                            prepared.surface.inference.verify_response(prepared.artifact, path, response)
                        self._stamp_durable_runtime_load_id(prepared, payload, response)
                        item = await asyncio.to_thread(
                            self._accept,
                            prepared.parsed,
                            {**payload, "response": response},
                            artifact_ref=prepared.artifact.ref,
                        )
                        succeeded = True
                        return client_inference_response(response), item
                    # A handler ``abort`` finish reason makes the attempt unusable.
                    # Restart the request against the latest artifact and never record it.
                    logger.info(
                        "retrying handler-aborted inference for scenario %r (attempt %d): frozen artifact %r, "
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
        except RuntimeLoadMismatch:
            operations.increment("serve/version_mismatch_total")
            raise
        except InferenceRetryTimeout:
            operations.increment("serve/timeouts_total")
            raise
        finally:
            measurement.finish(succeeded=succeeded)

    async def start_stream(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        handler: InferenceHandler | None = None,
    ) -> tuple[InferenceStream, PendingInference]:
        operations = await self.inference_operations(headers)
        measurement = operations.start("serve/request")
        try:
            prepared, payload = await self._prepare_request(headers, payload, path, handler)
            admission = prepared.admission
            lease = prepared.lease
            try:
                stream = await prepared.handler.inference_stream(prepared.artifact, path, payload)
                record_response = stream.record_response
                record_response_pending = stream.record_response_pending
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
            except BaseException:
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
                raise
            pending = PendingInference(
                item=AgentRecord.create(
                    scenario=prepared.parsed.scenario,
                    request_type=RequestType.INFERENCE,
                    payload=_with_tags(payload, prepared.parsed),
                    artifact_ref=prepared.artifact.ref,
                ),
                release_id=prepared.parsed.release_id,
                measurement=measurement,
                admission=admission,
                lease=lease,
                deferred_prepared=prepared if record_response_pending else None,
                path=path if record_response_pending else None,
            )
            return stream, pending
        except BaseException as exc:
            measurement.finish(succeeded=False)
            if isinstance(exc, RuntimeLoadMismatch):
                operations.increment("serve/version_mismatch_total")
            raise

    async def relay_multimodal(
        self, headers: Mapping[str, str], payload: dict[str, Any], path: str
    ) -> InferenceStream:
        """A multimodal call for a scenario, relayed by the deployment's recipe to the provider it configured;
        nothing is recorded. The relay is the deployment's: its key and gateway, whatever model a scenario chats
        with."""
        parsed = parse_request_headers(headers, RequestType.INFERENCE)
        scenario = await asyncio.to_thread(
            self._dispatcher.get_or_create_scenario,
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        relay = self._dispatcher.recipe.multimodal_relay
        if relay is None:
            raise NotImplementedError(f"the served recipe relays no multimodal calls, so it serves no {path}")
        return await relay.relay(path, payload)

    def record_stream(self, pending: PendingInference, response: Mapping[str, Any]) -> AgentRecord:
        succeeded = False
        try:
            payload = dict(pending.item.payload)
            # A token-native streaming handler fills record_response only when
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
            delivery = response.get("stream_delivery", response)
            succeeded = (
                isinstance(delivery, Mapping) and delivery.get("complete") is True and not delivery.get("error")
            )
            return stored
        except Exception as exc:
            if isinstance(exc, RuntimeLoadMismatch) and pending.measurement is not None:
                pending.measurement.metrics.increment("serve/version_mismatch_total")
            logger.exception(
                "dispatcher rejected the stream record for scenario %r (record %s)",
                pending.item.scenario,
                pending.item.agent_record_id,
            )
            raise
        finally:
            if pending.measurement is not None:
                pending.measurement.finish(succeeded=succeeded)
            try:
                if pending.lease is not None:
                    pending.lease.release()
            finally:
                if pending.admission is not None:
                    pending.admission.release()

    async def inference_operations(self, headers: Mapping[str, str]) -> OperationMetrics:
        """Resolve the scenario before measuring its inference request lifetime."""
        parsed = parse_request_headers(headers, RequestType.INFERENCE)
        scenario = await asyncio.to_thread(
            self._dispatcher.get_or_create_scenario,
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        return scenario.operations

    async def _prepare_request(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        path: str,
        handler: InferenceHandler | None,
    ) -> tuple[PreparedInference, dict[str, Any]]:
        """The shared first half of every inference: freeze the serving state
        (headers, scenario, artifact, handler, surface) and let the surface
        transform the request payload."""
        parsed = parse_request_headers(headers, RequestType.INFERENCE)
        initial = await asyncio.to_thread(
            self._dispatcher.get_or_create_scenario,
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if initial is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        if initial.runtime is not None:
            with initial.operations.measure("serve/admission"):
                admission = await initial.runtime.acquire_inference()
        else:
            admission = None
        try:
            # Re-resolve after admission: a queued request must freeze the head
            # committed by the weight update that released it, never the head it
            # observed before waiting.
            prepared = await asyncio.to_thread(self._prepare_inference, parsed, handler, admission)
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
        handler: InferenceHandler | None,
        admission: InferenceAdmissionHandle | None,
    ) -> PreparedInference:
        scenario = self._dispatcher.get_or_create_scenario(
            parsed.scenario,
            release_id=parsed.release_id,
        )
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        selected_handler = handler if handler is not None else scenario.inference_handler
        if selected_handler is None:
            raise RecipeConfigError("the served recipe has no inference handler")
        ref = scenario.current_artifact_ref()
        return PreparedInference(
            parsed=parsed,
            artifact=Artifact(ref, scenario.repository),
            handler=selected_handler,
            surface=scenario.surface,
            durable=scenario.training_runtime is not None,
            admission=admission,
        )

    def harness_manifest(self, headers: Mapping[str, str], release_id: str | None = None) -> dict[str, Any]:
        """The served tree plus its parent release and evaluation metrics.

        ``release_id`` addresses one catalog release instead of the
        serving head, so a consumer can pin or roll back by pulling an older
        tree; an unknown or unrestorable release raises ArtifactNotFound
        naming it. The ``evaluation`` field carries the metrics of the training step
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
        artifact, evaluation_metrics = scenario.artifact_with_metrics(release_id)
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
            "evaluation": evaluation_metrics,
            "gate": evaluation_metrics,  # Legacy clients read this manifest field.
            # The union over the chain, not this evaluation's list: a release whose request named nothing still installs an earlier extension.
            "requires": required_by(list(reversed(scenario.releases())), artifact.ref.release_id),
        }

    def harness_head(self, headers: Mapping[str, str]) -> str | None:
        """The release ``GET /reef/harness`` serves the request's scenario, or None when it serves no files."""
        try:
            scenario = self._file_scenario(headers)
        except ArtifactNotFound:
            return None
        return scenario.repository.require_current_artifact().release_id

    def harness_propose(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]:
        """Admit one agent proposal against the head release's entries and hold it for the next evolve step.

        The admission is the backend's own (``admit_mutations`` over a fresh
        loader), never a touch of its live tree, and it runs again on the
        training thread when the step takes the proposal, since the head may
        have moved. The answer names the proposal, whether it was admitted,
        the refusal when not, and the head it was admitted against.
        """
        proposal = ProposalPayload.from_dict(payload)
        scenario = self._file_scenario(headers)
        backend = scenario.trainer.candidate_backend
        if not isinstance(backend, ProposalValidator) or backend.proposals is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} takes no proposals: the deployment's recipe is not a harness "
                "evolution recipe with a proposal inbox"
            )
        head = scenario.repository.require_current_artifact().release_id
        proposal_id = ProposalInbox.new_id()
        # Only an automatic step claims the inbox, and a manual scenario runs instruction steps only.
        if scenario.trainer.training_mode == "manual":
            return {
                "proposal_id": proposal_id,
                "admitted": False,
                "reason": "manual mode takes instructions only",
                "release_id": head,
            }
        # The entries the head commit logged, which the served tree.json carries too, not the trainer's live
        # state, which a step in flight has already moved; the seed before the first commit.
        logged = scenario.entries_for_version(head)
        info = scenario.surface.harness
        if logged is None and info is not None:
            logged = info.seed_entries
        entries = [dict(entry) for entry in logged or ()]
        try:
            mutations = [Mutation(str(m["op"]), str(m["id"]), m.get("options")) for m in proposal.mutations]
            _, refusal = backend.admit(entries, mutations)
        except MutationError as error:
            refusal = str(error)
        if refusal is None:
            refusal = backend.proposals.submit(proposal_id, {**proposal.to_dict(), "head_release_id": head})
        return {"proposal_id": proposal_id, "admitted": refusal is None, "reason": refusal, "release_id": head}

    def harness_releases(self, headers: Mapping[str, str]) -> dict[str, Any]:
        """The scenario's release catalog with per-release evaluation metrics, newest last.

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

    def harness_step_records(self, headers: Mapping[str, str], step: int, relative: str | None) -> dict[str, Any]:
        """Raw retained files for a catalog row; presentation belongs to the caller."""
        scenario = self._file_scenario(headers)
        rows = list(reversed(scenario.releases()))
        if not 0 <= step < len(rows):
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no step {step}")
        directory = (rows[step].get("metrics") or {}).get("step_record")
        backend = scenario.trainer.candidate_backend
        if not directory or not isinstance(backend, StepRecords):
            return {"status": "not_recorded", "files": []}
        if not isinstance(directory, str):
            raise ValueError("invalid step record directory")
        try:
            return backend.read_step_records(directory, relative)
        except FileNotFoundError as error:
            raise ArtifactNotFound("record file is not retained") from error

    def harness_release_page(
        self, headers: Mapping[str, str], step: int, link_query: Mapping[str, str] | None = None
    ) -> str:
        """One HTML page for the catalog row at ``step``, counted oldest first with the creation row as 0.

        The rows are the ones ``harness_releases`` answers, so the step a
        client counts there is the step this page names. The release the
        step ran on (the parent of a win, the head a rejected or skipped
        step ran on) comes through the artifact snapshot when it is
        restorable, so an extension update shows as a diff, else as its new
        text. ``link_query`` is carried to the Chain's links, so a page
        opened through query parameters links pages that open the same way.
        An unknown step raises ArtifactNotFound naming the range.
        """
        scenario = self._file_scenario(headers)
        rows = list(reversed(scenario.releases()))
        if not 0 <= step < len(rows):
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} has no step {step}: the catalog holds steps 0 to {len(rows) - 1}"
            )
        before = before_release_id(rows[step])
        before_entries: Sequence[Mapping[str, Any]] = ()
        before_files: Mapping[str, str] | None = None
        if before is not None:
            info = scenario.surface.harness
            logged = scenario.entries_for_version(before)
            if logged is None and info is not None:
                logged = info.seed_entries
            before_entries = logged or ()
            tree = scenario.surface.files
            try:
                artifact, _ = scenario.artifact_with_metrics(before)
                before_files = None if tree is None else tree.read_files(artifact)
            except ArtifactError:
                before_files = None
        descriptor = getattr(scenario.trainer.candidate_backend, "descriptor", None)
        return build_release_page(
            step,
            rows,
            before_entries=before_entries,
            before_files=before_files,
            node_paths=None if descriptor is None else descriptor.node_paths,
            link_query=link_query,
        )

    def harness_request_page(
        self, headers: Mapping[str, str], record_id: str, link_query: Mapping[str, str] | None = None
    ) -> str:
        """One HTML page for the harness request stored as agent record ``record_id``, live until its step settles.

        The record is the ``POST /reef/train`` instruction as stored; the
        catalog row whose ``training_request.id`` names it settles the page.
        Until then the page reads the running step's progress from the
        scenario's candidate backend, when the backend reports one, and
        whether the trainer holds the request in its reserved batch.
        ``link_query`` is carried to the version page link, so a page opened
        through query parameters links one that opens the same way. An
        unknown id, or one that is not a training instruction, raises
        ArtifactNotFound naming it.
        """
        scenario = self._file_scenario(headers)
        record = self._dispatcher.read_record(scenario.name, record_id)
        if record is None or record.get("request_type") != RequestType.TRAIN.value:
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no harness request {record_id!r}")
        rows = list(reversed(scenario.releases()))
        backend = scenario.trainer.candidate_backend
        progress = backend.step_progress if isinstance(backend, StepProgressReader) else None
        reserved = scenario.trainer.pending_batch
        consumed = reserved is not None and reserved.request is not None and reserved.request.id == record_id
        return build_request_page(record, rows, progress=progress, consumed=consumed, link_query=link_query)

    def harness_request_progress(self, headers: Mapping[str, str], record_id: str) -> dict[str, Any]:
        """Where a filed request stands, as JSON, for a client with no browser to open its page.

        The same reading the page renders: the state
        (``queued``, ``proposing``, ``evaluating``, ``running``, ``settling``,
        else the settled row's result), what it means in the page's words,
        and, once a row answers the request, the step it landed as. A client
        polls this to show a phase; the page itself stays the readable view.
        An unknown id, or one that is not a training instruction, raises
        ArtifactNotFound naming it.
        """
        scenario = self._file_scenario(headers)
        record = self._dispatcher.read_record(scenario.name, record_id)
        if record is None or record.get("request_type") != RequestType.TRAIN.value:
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no harness request {record_id!r}")
        rows = list(reversed(scenario.releases()))
        step = settled_step(rows, record_id)
        if step is not None:
            return {
                "request_id": record_id,
                "settled": True,
                "step": step,
                "state": result_of(rows[step], rows),
                "meaning": None,
                "started_at": None,
                "episodes_total": None,
                "step_record": None,
                "activity": [],
            }
        backend = scenario.trainer.candidate_backend
        progress = backend.step_progress if isinstance(backend, StepProgressReader) else None
        reserved = scenario.trainer.pending_batch
        consumed = reserved is not None and reserved.request is not None and reserved.request.id == record_id
        state = request_state(record, progress, consumed)
        mine = progress if progress is not None and progress.request_id == record_id else None
        return {
            "request_id": record_id,
            "settled": False,
            "step": None,
            "state": state,
            "meaning": STATE_WORDS.get(state),
            # The step's own clock, so a client shows the time in the step and not the time since it asked.
            "started_at": None if mine is None else mine.started_at,
            "episodes_total": None if mine is None else mine.episodes_total,
            "step_record": None if mine is None else mine.step_record,
            # What the proposer has done so far, oldest first: {at, kind, text, failed?}.
            "activity": [] if mine is None else [dict(line) for line in mine.activity],
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
        descriptor = get_adapter(adapter)
        return render_install_script(
            descriptor=descriptor,
            files=manifest["files"],
            release_id=manifest["release_id"],
            content_id=manifest["content_id"],
            scenario=scenario.name,
            binding_files=self._install_binding(scenario, manifest, descriptor, headers),
            requires=manifest["requires"],
            fallback_release_id=ancestor_requiring_nothing(
                list(reversed(scenario.releases())), manifest["release_id"]
            ),
        )

    def _install_binding(
        self, scenario: Scenario, manifest: Mapping[str, Any], descriptor: Any, headers: Mapping[str, str]
    ) -> dict[str, str]:
        """The adapter's config targets re-rendered with a binding at the Reef this request reached.

        The served composition never carries an endpoint or a credential, so
        an installed tree needs one written beside it: the release's own
        entries (the recipe's seed for the base release no step published)
        plus the descriptor's binding template, the base URL taken from the
        request's Host, the model from the evaluation the release ran against (the
        recipe's served model for the base release), and the token left as a
        placeholder the script fills from the client's environment. Empty
        when any of those is unknown, and the script then installs the
        composition as before.
        """
        normalized = {key.lower(): value.strip() for key, value in headers.items()}
        # A gateway in front of Reef names the address the client reached in the forwarded
        # headers; the binding goes there, so the installed harness calls back through it.
        host = normalized.get("x-forwarded-host") or normalized.get("host")
        evaluation_metrics = manifest.get("evaluation", manifest.get("gate")) or {}
        model = (
            (evaluation_metrics.get("evaluation_context", evaluation_metrics.get("gated_against")) or {}).get("model")
            if isinstance(evaluation_metrics, Mapping)
            else None
        )
        info = scenario.surface.harness
        if not isinstance(model, str) or not model:
            model = None if info is None else info.served_model
        entries = scenario.entries_for_version(manifest["release_id"])
        if entries is None and info is not None:
            entries = info.seed_entries
        if not host or not model or not entries:
            return {}
        scheme = normalized.get("x-forwarded-proto") or "http"
        api = "openai" if info is None else info.served_api
        client_models = () if info is None else info.client_models
        override = scenario.model_config.runtime
        if override is not None:
            selected = ModelBinding.from_runtime(override)
            model, api = selected.model, selected.api
            client_models = ()
        binding = ModelBinding(base_url=f"{scheme}://{host}", model=model, api_key=TOKEN_PLACEHOLDER, api=api)
        nodes = [(str(entry["name"]), entry.get("config")) for entry in entries if not entry.get("disabled")]
        try:
            bound = binding.compose_nodes(descriptor, models=client_models)
            files = render_composition((*nodes, *bound), descriptor)
        except (ModelBindingError, RenderError, KeyError, TypeError):
            return {}
        targets = {descriptor.config_targets[str(config.get("target", "primary"))].path for _, config in bound}
        return {path: files[path] for path in sorted(targets) if path in files}

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

        parsed = parse_request_headers(headers, RequestType.INFERENCE)
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
        normalized_payload, references = normalize_request_payload(parsed.request_type, payload)
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
    "RequestService",
    "client_inference_response",
    "normalize_request_payload",
    "page_headers",
]
