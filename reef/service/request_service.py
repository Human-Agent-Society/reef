"""Transport-independent request handling behind the aiohttp routes.

``RequestService`` normalizes typed Reef payloads, resolves the artifact
version before every provider call so concurrent publication cannot change
what gets recorded, stores each exchange as a record, and applies the
scenario surface's request/response checks. No aiohttp types appear here;
``reef.service.routes`` adapts these methods to HTTP.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from reef.artifact.artifact import ArtifactError, ArtifactNotFound, ArtifactRef
from reef.core.errors import ReefError, UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.core.requirements import ancestor_requiring_nothing, required_by
from reef.core.training_request import normalize_training_request_payload
from reef.dispatcher import Dispatcher
from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.rollouts.model_binding import ModelBinding, ModelBindingError
from reef.harness.tree.mutations import Mutation, MutationError
from reef.harness.tree.render import RenderError, render_composition
from reef.scenario.scenario import Scenario
from reef.service.install_script import TOKEN_PLACEHOLDER, render_install_script
from reef.service.release_page import before_release_id, build_release_page, build_running_step_page, result_of
from reef.service.request_page import STATE_WORDS, build_request_page, request_state, settled_step
from reef.service.wire import (
    SCENARIO_HEADER,
    ProposalPayload,
    ReportPayload,
    RequestHeaders,
    parse_request_headers,
    with_request_tags,
)
from reef.train.backend import CandidateBackend
from reef.train.cordis_backend.contracts import ProposalValidator, StepProgressReader, StepRecords
from reef.train.cordis_backend.proposals import ProposalInbox
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)


def page_headers(headers: Mapping[str, str], query: Mapping[str, str]) -> dict[str, str]:
    """The headers a page route reads, ``?scenario=`` standing in for ``x-reef-scenario`` when that header is absent.

    A page is a link a person opens in a browser, which sends no ``x-reef-*``
    header; the header wins when both are present. The page key stands in for
    the token in :mod:`reef.service.auth`, for the page routes alone.
    """
    merged = dict(headers.items())
    scenario = query.get("scenario", "").strip()
    if scenario and not any(key.lower() == SCENARIO_HEADER for key in merged):
        merged[SCENARIO_HEADER] = scenario
    return merged


def _random_harness_scenario_name() -> str:
    return f"harness-{uuid.uuid4().hex[:12]}"


def install_service_url(headers: Mapping[str, str]) -> str | None:
    """The address an install request reached Reef at, from its Host; None without one.

    A gateway in front of Reef names the address the client reached in the
    forwarded headers; the binding goes there, so the installed harness
    calls back through it."""
    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    host = normalized.get("x-forwarded-host") or normalized.get("host")
    if not host:
        return None
    return f"{normalized.get('x-forwarded-proto') or 'http'}://{host}"


def normalize_request_payload(
    request_type: RequestType,
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Normalize a typed Reef payload; a native provider body passes through."""
    if request_type is RequestType.TRAIN:
        return normalize_training_request_payload(payload), ()
    if request_type is not RequestType.REPORT:
        return dict(payload), ()
    report = ReportPayload.from_dict(payload)
    return report.to_dict(), report.references


class RequestService:
    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatcher = dispatcher
        # The harness head of a scenario with several components, by scenario step and active release: it is
        # read on every inference answer and changes only when a commit lands.
        self.harness_heads: dict[str, tuple[int, str, str]] = {}

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

    def import_record(self, headers: Mapping[str, str], body: Mapping[str, object]) -> AgentRecord:
        """Persist an existing inference or report through ordinary admission.

        A stable client ID makes uploads retryable. The scenario comes from
        the headers, and importing never invokes an inference handler.
        """
        parsed, item = self.prepare_import_record(headers, body)
        return self._dispatcher.accept_record(item, release_id=parsed.release_id)

    def import_records(self, headers: Mapping[str, str], body: Mapping[str, object]) -> tuple[AgentRecord, ...]:
        """Import a bounded, atomic batch using the existing record contract."""
        records = body.get("records")
        if body.keys() != {"records"} or not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise ValueError("body must contain only 'records', an array of 1 to 1000 record objects")
        items: list[AgentRecord] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("each record must be an object")
            parsed, item = self.prepare_import_record(headers, record)
            items.append(item)
        return self._dispatcher.accept_records(items, release_id=parsed.release_id)

    def prepare_import_record(
        self, headers: Mapping[str, str], body: Mapping[str, object]
    ) -> tuple[RequestHeaders, AgentRecord]:
        if body.keys() - {"agent_record_id", "request_type", "payload"}:
            raise ValueError("record fields must be agent_record_id, request_type and payload")
        record_id = body.get("agent_record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("agent_record_id must be a non-empty string")
        request_type = body.get("request_type")
        if not isinstance(request_type, str) or request_type not in ("inference", "report"):
            raise ValueError("import request_type must be 'inference' or 'report'; use /reef/train for instructions")
        payload = body.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("record payload must be an object")
        parsed = parse_request_headers(headers, RequestType(request_type))
        normalized, references = normalize_request_payload(parsed.request_type, payload)
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=with_request_tags(normalized, parsed),
            agent_record_id=record_id,
            references=references,
        )
        return parsed, item

    def harness_manifest(self, headers: Mapping[str, str], release_id: str | None = None) -> dict[str, Any]:
        """The served tree plus its parent release and evaluation metrics.

        ``release_id`` addresses one catalog release instead of the
        serving head, so a consumer can pin or go back by pulling an older
        tree; an unknown or unrestorable release raises ArtifactNotFound
        naming it. The ``evaluation`` field carries the metrics of the training step
        that published the active release, so a consumer can audit what a
        pulled tree changed and why it was admitted before running it.
        Read-only: never creates a scenario.
        """
        scenario = self._file_scenario(headers)
        return self._harness_manifest_for_scenario(scenario, release_id or self.harness_release_id(scenario))

    @staticmethod
    def files_trainer(scenario: Scenario) -> Trainer:
        """The trainer evolving the component a client pulls; a flat scenario's only trainer."""
        return scenario.trainer_for(scenario.surface.files_component)

    @staticmethod
    def _harness_manifest_for_scenario(
        scenario: Scenario,
        release_id: str | None = None,
    ) -> dict[str, Any]:
        artifact, evaluation_metrics = scenario.release_queries.artifact_with_metrics(release_id)
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
        manifest = {
            "release_id": artifact.ref.release_id,
            "parent_release_id": artifact.ref.parent_release_id,
            "content_id": artifact.ref.content_id,
            "files": dict(files),
            "evaluation": evaluation_metrics,
            # The union over the chain, not this evaluation's list: a release whose request named nothing still installs an earlier extension.
            "requires": required_by(list(reversed(scenario.releases())), artifact.ref.release_id),
        }
        components = artifact.materialize().components
        if components is not None:
            # The whole combination the pulled tree belongs to, by component content id.
            manifest["components"] = components.content_ids
        return manifest

    @staticmethod
    def harness_lineage(scenario: Scenario) -> tuple[list[dict[str, Any]], str | None]:
        """The catalog rows a client pulls, newest first, and the release of the active harness artifact.

        A row names the harness artifact it holds or ran on: ``artifact_id``
        and ``component`` from ``Scenario.artifacts``, with ``active`` and
        ``activated``. ``active`` is true on exactly one row, the row that
        made the active harness artifact. In a scenario of several
        components, a release that minted no harness artifact (another
        component's step, an activation) is not listed. A listed row names
        the release that minted its artifact as ``release_id`` and the
        release that minted its parent's harness artifact as
        ``parent_release_id``; the ids it had are kept beside them as
        ``composed_release_id`` and ``composed_parent_release_id``. A step
        that made no artifact is listed when the harness trainer made it.
        """
        files_component = scenario.surface.files_component
        single = scenario.surface.single
        harness = [row for row in scenario.artifacts() if single or row["component"] == files_component]
        by_release = {row["release_id"]: row for row in harness}
        by_artifact = {row["artifact_id"]: row for row in harness}
        manifests = scenario.release_queries.manifests(scenario.scenario_step)

        def artifact_of(release_id: str | None) -> dict[str, Any] | None:
            """The harness artifact that the release ``release_id`` binds, or ``None`` when it is not known."""
            if release_id is None or release_id in by_release:
                return None if release_id is None else by_release[release_id]
            return by_artifact.get((manifests.get(release_id) or {}).get(str(files_component)))

        kept: list[dict[str, Any]] = []
        for row in scenario.releases():
            if row["checkpoint"] and row["release_id"] not in by_release:
                continue
            if not row["checkpoint"] and row.get("component") not in (None, files_component):
                continue
            artifact = artifact_of(row["release_id"])
            if artifact is None:
                continue
            listed = {
                **row,
                "artifact_id": artifact["artifact_id"],
                "component": artifact["component"],
                "active": row["checkpoint"] and artifact["active"],
                "activated": artifact["activated"],
            }
            if artifact["release_id"] != row["release_id"]:
                listed["composed_release_id"] = row["release_id"]
                listed["release_id"] = artifact["release_id"]
            parent = artifact_of(artifact["parent_release_id"])
            if parent is not None and parent["release_id"] != row["parent_release_id"]:
                listed["composed_parent_release_id"] = row["parent_release_id"]
                listed["parent_release_id"] = parent["release_id"]
            kept.append(listed)
        return kept, next((row["release_id"] for row in harness if row["active"]), None)

    @classmethod
    def harness_rows(cls, scenario: Scenario) -> list[dict[str, Any]]:
        """The catalog rows a client pulls, newest first; see ``harness_lineage``."""
        rows, _ = cls.harness_lineage(scenario)
        return rows

    def harness_release_id(self, scenario: Scenario) -> str:
        """The release that a client pulls: the release that minted the active harness artifact."""
        current = scenario.repository.require_current_artifact().release_id
        if scenario.surface.single or scenario.surface.files_component is None:
            return current
        step = scenario.scenario_step
        cached = self.harness_heads.get(scenario.name)
        if cached is not None and cached[0] == step and cached[1] == current:
            return cached[2]
        _, head = self.harness_lineage(scenario)
        if head is None:
            head = current
        self.harness_heads[scenario.name] = (step, current, head)
        return head

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
        backend = self.files_trainer(scenario).candidate_backend
        if not isinstance(backend, ProposalValidator) or backend.proposals is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} takes no proposals: the deployment's recipe is not a harness "
                "evolution recipe with a proposal inbox"
            )
        head = self.harness_release_id(scenario)
        proposal_id = ProposalInbox.new_id()
        # Only an automatic step claims the inbox, and a manual scenario runs instruction steps only.
        if self.files_trainer(scenario).pipeline.training_mode == "manual":
            return {
                "proposal_id": proposal_id,
                "admitted": False,
                "reason": "manual mode takes instructions only",
                "release_id": head,
            }
        # The entries of the head, which the served tree.json carries too, not the trainer's live state, which a
        # step in flight has already moved; the base artifact's before the first commit.
        logged = scenario.release_queries.entries_for_version(head, scenario.surface.files_component)
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
        update is a decision over numbers rather than a blind pull. A scenario
        with several components lists the releases that changed the pulled
        tree (see ``harness_lineage``); the others stay addressable by id.
        Same read-only rules as ``harness_manifest``.
        """
        scenario = self._file_scenario(headers)
        return {
            "scenario": scenario.name,
            "releases": list(reversed(self.harness_rows(scenario))),
        }

    def harness_step_records(self, headers: Mapping[str, str], step: int, relative: str | None) -> dict[str, Any]:
        """Raw retained files for a catalog row; presentation belongs to the caller."""
        scenario = self._file_scenario(headers)
        rows = list(reversed(self.harness_rows(scenario)))
        if not 0 <= step < len(rows):
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no step {step}")
        directory = (rows[step].get("metrics") or {}).get("step_record")
        backend = self.files_trainer(scenario).candidate_backend
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
        rows = list(reversed(self.harness_rows(scenario)))
        if step == len(rows):
            # The next step, while a request runs it: the catalog has no row yet, so the page says so and links the
            # request's page, which follows the step live.
            running = self.running_request_id(scenario)
            if running is not None:
                return build_running_step_page(step, running, link_query)
        if not 0 <= step < len(rows):
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} has no step {step}: the catalog holds steps 0 to {len(rows) - 1}"
            )
        before = before_release_id(rows[step])
        before_entries: Sequence[Mapping[str, Any]] = ()
        before_files: Mapping[str, str] | None = None
        if before is not None:
            before_entries = (
                scenario.release_queries.entries_for_version(before, scenario.surface.files_component) or ()
            )
            tree = scenario.surface.files
            try:
                artifact, _ = scenario.release_queries.artifact_with_metrics(before)
                before_files = None if tree is None else tree.read_files(artifact)
            except ArtifactError:
                before_files = None
        backend = self.files_trainer(scenario).candidate_backend
        return build_release_page(
            step,
            rows,
            before_entries=before_entries,
            before_files=before_files,
            node_paths=None if backend is None else backend.harness_node_paths,
            link_query=link_query,
            adapter=self.harness_adapter(backend),
        )

    @staticmethod
    def harness_adapter(backend: CandidateBackend | None) -> str:
        """The adapter a harness page names in its commands: the backend's, pi for a backend that names none."""
        adapter = None if backend is None else backend.harness_adapter
        return "pi" if adapter is None else adapter

    @classmethod
    def running_request_id(cls, scenario: Scenario) -> str | None:
        """The id of the request the scenario's step is running: the reserved job data's, else the backend's progress."""
        trainer = cls.files_trainer(scenario)
        reserved = trainer.pending_job_data
        if reserved is not None and reserved.request is not None:
            return str(reserved.request.id)
        backend = trainer.candidate_backend
        progress = backend.step_progress if isinstance(backend, StepProgressReader) else None
        return None if progress is None or progress.request_id is None else str(progress.request_id)

    def harness_request_page(
        self, headers: Mapping[str, str], record_id: str, link_query: Mapping[str, str] | None = None
    ) -> str:
        """One HTML page for the harness request stored as agent record ``record_id``, live until its step settles.

        The record is the ``POST /reef/train`` instruction as stored; the
        catalog row whose ``training_request.id`` names it settles the page.
        Until then the page reads the running step's progress from the
        scenario's candidate backend, when the backend reports one, and
        whether the trainer holds the request in its reserved job data.
        ``link_query`` is carried to the version page link, so a page opened
        through query parameters links one that opens the same way. An
        unknown id, or one that is not a training instruction, raises
        ArtifactNotFound naming it.
        """
        scenario = self._file_scenario(headers)
        record = self._dispatcher.read_record(scenario.name, record_id)
        if record is None or record.get("request_type") != RequestType.TRAIN.value:
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no harness request {record_id!r}")
        # The step a request settled as counts the catalog rows, the ones the release page opens.
        rows = list(reversed(self.harness_rows(scenario)))
        backend = self.files_trainer(scenario).candidate_backend
        progress = backend.step_progress if isinstance(backend, StepProgressReader) else None
        reserved = self.files_trainer(scenario).pending_job_data
        consumed = reserved is not None and reserved.request is not None and reserved.request.id == record_id
        return build_request_page(
            record,
            rows,
            progress=progress,
            consumed=consumed,
            link_query=link_query,
            adapter=self.harness_adapter(backend),
        )

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
        rows = list(reversed(self.harness_rows(scenario)))
        step = settled_step(rows, record_id)
        if step is not None:
            return {
                "request_id": record_id,
                "settled": True,
                "step": step,
                "state": result_of(rows[step]),
                "meaning": None,
                "started_at": None,
                "rollouts_total": None,
                "step_record": None,
                "activity": [],
            }
        backend = self.files_trainer(scenario).candidate_backend
        progress = backend.step_progress if isinstance(backend, StepProgressReader) else None
        reserved = self.files_trainer(scenario).pending_job_data
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
            "rollouts_total": None if mine is None else mine.rollouts_total,
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
        ``harness_manifest`` (the harness head by default, any release through
        ``release_id``); the named ``adapter`` contributes only its
        descriptor's install section, which the script uses to ensure the
        pinned binary through the vendor's own channel. An unknown adapter
        raises ArtifactNotFound naming it, mirroring the unknown-release
        behavior; a known adapter whose descriptor declares no install
        section raises DescriptorError (HTTP 400) naming it. Unlike the other
        harness reads, a missing or empty scenario header creates a new,
        randomly named file-serving scenario, unless the deployment turned
        implicit scenario creation off.
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
        manifest = self._harness_manifest_for_scenario(scenario, release_id or self.harness_release_id(scenario))
        descriptor = get_adapter(adapter)
        binding_files = self._install_binding(scenario, manifest, descriptor, headers)
        return render_install_script(
            descriptor=descriptor,
            files=manifest["files"],
            release_id=manifest["release_id"],
            content_id=manifest["content_id"],
            scenario=scenario.name,
            binding_files=binding_files,
            # The install record keeps the address the binding names, for a wrapper whose binding a session changed.
            service_url=install_service_url(headers) if binding_files else None,
            requires=manifest["requires"],
            # The release the script names for a first install must be one the catalog lists.
            fallback_release_id=ancestor_requiring_nothing(
                list(reversed(self.harness_rows(scenario))), manifest["release_id"]
            ),
            # A request that carried a token reached a service that wants one, so the harness needs it too.
            is_token_expected=any(name.lower() in ("authorization", "x-api-key") for name in headers),
        )

    def _install_binding(
        self, scenario: Scenario, manifest: Mapping[str, Any], descriptor: Any, headers: Mapping[str, str]
    ) -> dict[str, str]:
        """The adapter's config targets re-rendered with a binding at the Reef this request reached.

        The served composition never carries an endpoint or a credential, so
        an installed tree needs one written beside it: the release's own
        entries (the base artifact's for the creation) plus the descriptor's
        binding template, the base URL taken from the request's Host, the model from the evaluation the release ran against (the
        recipe's served model for the base release), and the token left as a
        placeholder the script fills from the client's environment. Empty
        when any of those is unknown, and the script then installs the
        composition as before.
        """
        service_url = install_service_url(headers)
        evaluation_metrics = manifest.get("evaluation") or {}
        model = (
            (evaluation_metrics.get("evaluation_context") or {}).get("model")
            if isinstance(evaluation_metrics, Mapping)
            else None
        )
        info = scenario.surface.harness
        if not isinstance(model, str) or not model:
            model = None if info is None else info.served_model
        entries = scenario.release_queries.entries_for_version(
            manifest["release_id"], scenario.surface.files_component
        )
        if not service_url or not model or not entries:
            return {}
        api = "openai" if info is None else info.served_api
        client_models = () if info is None else info.client_models
        metadata = None if info is None else info.model_metadata.get(model)
        if metadata is None and info is not None and model == info.served_model:
            metadata = info.served_metadata
        override = scenario.model_config.runtime
        if override is not None:
            selected = ModelBinding.from_runtime(override)
            model, api = selected.model, selected.api
            if descriptor.name == "codex":
                selected = selected.with_metadata(None if info is None else info.model_metadata.get(model))
            metadata = selected.metadata
            client_models = ()
        binding = ModelBinding(
            base_url=service_url, model=model, api_key=TOKEN_PLACEHOLDER, api=api, metadata=metadata
        )
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
            scenario = self._dispatcher.get_or_create_scenario(scenario_name, release_id=release_id)
            if scenario is None:
                raise UnknownScenario(
                    "this deployment creates no scenario implicitly: create one with POST /reef/scenarios "
                    f"and name it in the {SCENARIO_HEADER} header"
                )
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
        normalized_payload = with_request_tags(normalized_payload, parsed)
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


__all__ = [
    "RequestService",
    "normalize_request_payload",
    "page_headers",
]
