"""Transport-independent request handling behind the aiohttp routes.

``RequestService`` normalizes typed Reef payloads, resolves the artifact
version before every provider call so concurrent publication cannot change
what gets recorded, stores each exchange as a record, and applies the
scenario surface's request/response checks. No aiohttp types appear here;
``reef.service.routes`` adapts these methods to HTTP.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactError, ArtifactNotFound, ArtifactRef
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.requirements import required_by
from reef.core.training_request import normalize_training_request_payload
from reef.dispatcher import Dispatcher
from reef.harness.sync import import_files, layer_install, merge_trees
from reef.harness.tree import HarnessTree, validate_tree
from reef.harness.tree.checks import clip_record_value
from reef.scenario.release_files import component_files
from reef.scenario.scenario import Scenario
from reef.service.release_page import before_release_id, build_release_page, build_running_step_page, result_of
from reef.service.request_page import STATE_WORDS, build_request_page, request_state, settled_step
from reef.service.wire import SCENARIO_HEADER, ReportPayload, RequestHeaders, parse_request_headers, with_request_tags
from reef.train.harness_backend.backend import HarnessBackend
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

    def harness_manifest(
        self,
        headers: Mapping[str, str],
        release_id: str | None = None,
        form: str = "rendered",
        service_url: str | None = None,
        root: str | None = None,
    ) -> dict[str, Any]:
        """The served tree plus its parent release and evaluation metrics.

        ``release_id`` addresses one catalog release instead of the
        active release, so a consumer can pin or go back by pulling an older
        tree; an unknown or unrestorable release raises ArtifactNotFound
        naming it. ``form`` chooses the ``files``: ``rendered`` (the default)
        is the release rendered for the scenario's adapter, ``source`` is
        its source tree, the paths a patch names, and ``client`` is the
        release as a layer for a client that installs it at ``root``, bound
        to ``service_url``'s route of the scenario
        (``reef.harness.sync.layer_install``). The client form adds ``run``,
        the command that runs the agent on the files (``binary``, ``args``,
        ``env``), the agent's commands that record and remove the install
        (``register``, ``unregister``) and the agent's pinned ``install``, and ``synced``, the render templates of the files a client
        sends back. The ``evaluation`` field carries the metrics of the
        training step that published the release, so a consumer can audit
        what a pulled tree changed and why it was admitted before running
        it. Read-only: never creates a scenario.
        """
        if form not in ("rendered", "source", "client"):
            raise ValueError(f"form must be 'rendered', 'source' or 'client', got {form!r}")
        if form == "client" and not (service_url and root):
            raise ValueError(
                "form=client needs service_url, the address the client reaches Reef at, and root, the directory "
                "the client installs the files in"
            )
        scenario = self._file_scenario(headers)
        manifest = self._harness_manifest_for_scenario(
            scenario, release_id or self.harness_release_id(scenario), source=form != "rendered"
        )
        if form == "client" and service_url is not None and root is not None:
            backend = self.harness_backend(scenario)
            scenario_url = f"{service_url.rstrip('/')}/reef/scenarios/{scenario.name}"
            layer = layer_install(
                HarnessTree(manifest["files"]), backend.descriptor, backend.models.served, scenario_url, root
            )
            manifest["files"] = dict(layer.files)
            install = backend.descriptor.install
            manifest["run"] = {
                "binary": backend.descriptor.binary,
                "args": list(layer.args),
                "env": dict(layer.env),
                "register": [list(command) for command in layer.register],
                "unregister": [list(command) for command in layer.unregister],
                # The pinned vendor install, so a client can install the agent the rollouts run when it has none.
                "install": None if install is None else asdict(install),
            }
            manifest["synced"] = list(layer.synced)
        return manifest

    def harness_import(self, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]:
        """Commit a sync of a person's harness files: ``POST /reef/harness/import``.

        ``payload["files"]`` maps each file's path under the adapter's render
        roots to its text. ``payload["base_release_id"]`` is the release the
        client installed. The files become the tree that
        ``reef.harness.sync.import_files`` reads against the base, and
        ``merge_trees`` puts the client's changes on the tree of the active
        release (``harness_release_id``), not on the training start: a kept
        tree that the activation rule held stays inactive, so a sync never
        serves a change that waits for review. A file that the active tree
        and the client both changed in a way its part's ``merge`` refuses
        raises ``ArtifactConflict`` naming it. A merged tree that fails
        validation raises ``ValueError`` naming every problem. A sync that
        changes nothing publishes nothing. Otherwise the merged tree becomes
        the active release and the harness trainer's training start in one
        ``"sync"`` commit (``ScenarioCommitter.sync_release``). The answer
        names the active release after the sync, the release it was merged
        onto, whether it changed, whether it merged onto a newer release than
        the base, the tree's paths, and the client's paths no part takes.
        """
        files = payload.get("files")
        if not isinstance(files, Mapping) or not all(
            isinstance(path, str) and isinstance(text, str) for path, text in files.items()
        ):
            raise ValueError("files must map each path under the adapter's roots to the file's text")
        base = payload.get("base_release_id")
        if not isinstance(base, str) or not base:
            raise ValueError("base_release_id must be the release the client installed")
        scenario = self._file_scenario(headers)
        backend = self.harness_backend(scenario)
        component = scenario.surface.files_component
        if component is None:
            raise ArtifactNotFound(f"scenario {scenario.name!r} serves no harness component")
        current_id = scenario.repository.require_current_artifact().release_id
        active_id = self.harness_release_id(scenario)
        active = HarnessTree(self._harness_manifest_for_scenario(scenario, active_id, source=True)["files"])
        if base == active_id:
            base_tree = active
        else:
            base_tree = HarnessTree(self._harness_manifest_for_scenario(scenario, base, source=True)["files"])
        imported = import_files(files, base_tree, backend.descriptor)
        merge = merge_trees(base_tree, active, imported.tree, backend.descriptor.layout)
        if merge.tree is None:
            located = [(path, backend.descriptor.layout.locate(path)) for path in merge.conflicts]
            # The person knows each file at its render path, such as plugin/rules.md for rules.md.
            named = [
                path if location is None or location.part.renders is None else location.part.render_path(location.name)
                for path, location in located
            ]
            raise ArtifactConflict(
                f"release {active_id} and your files both changed {', '.join(named)} since release {base}: "
                "make each one match one side, or drop yours with sync --discard, then sync again"
            )
        tree = merge.tree
        answer = {
            "parent_release_id": active_id,
            "merged": base_tree != active,
            "files": sorted(tree.files),
            "ignored": list(imported.ignored),
        }
        if tree == active:
            return {**answer, "release_id": active_id, "changed": False}
        validation = validate_tree(tree, backend.descriptor)
        if not validation.passed:
            raise ValueError(f"the synced harness fails validation: {validation.message()}")
        patch = active.diff(tree)
        directory = Path(tempfile.mkdtemp(prefix="reef-harness-sync-"))
        tree.write(directory)
        artifact = Artifact.local(directory)
        try:
            published = self._dispatcher.sync_release(
                scenario.name,
                artifact,
                component=component,
                expected_release_id=current_id,
                metrics={
                    "sync": {
                        "files": len(tree.files),
                        "ignored": list(imported.ignored),
                        "base_release_id": base,
                    },
                    "patch": clip_record_value(patch.to_json()),
                    "patch_parts": backend.descriptor.layout.part_names(patch.changes),
                },
            )
        finally:
            artifact.discard()
        return {**answer, "release_id": published.release_id, "changed": True}

    def harness_backend(self, scenario: Scenario) -> HarnessBackend:
        """The backend evolving the harness a client pulls; ArtifactNotFound when the recipe evolves no harness."""
        backend = self.files_trainer(scenario).candidate_backend
        if not isinstance(backend, HarnessBackend):
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} evolves no harness: the deployment's recipe is not a harness "
                "evolution recipe"
            )
        return backend

    @staticmethod
    def files_trainer(scenario: Scenario) -> Trainer:
        """The trainer evolving the component a client pulls; a flat scenario's only trainer."""
        return scenario.trainer_for(scenario.surface.files_component)

    @staticmethod
    def _harness_manifest_for_scenario(
        scenario: Scenario, release_id: str | None = None, *, source: bool = False
    ) -> dict[str, Any]:
        artifact, evaluation_metrics = scenario.release_queries.artifact_with_metrics(release_id)
        tree = scenario.surface.files
        component = scenario.surface.files_component
        if tree is None or component is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: the deployment's recipe "
                "carries no harness surface (record-only or weight-training recipes have "
                "no file tree). Point 'recipe.implementation' at a harness evolution recipe."
            )
        files = component_files(scenario.surface, artifact, component) if source else tree.read_files(artifact)
        if files is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: release {artifact.ref.release_id!r} holds no "
                "harness tree. The base release holds the recipe's evolution.seed, so an empty seed serves "
                "nothing until a step publishes a tree."
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

    def harness_head(self, headers: Mapping[str, str]) -> str | None:
        """The release ``GET /reef/harness`` serves the request's scenario, or None when it serves no files."""
        try:
            scenario = self._file_scenario(headers)
        except ArtifactNotFound:
            return None
        return self.harness_release_id(scenario)

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
        if not directory or not isinstance(backend, HarnessBackend):
            return {"status": "not_recorded", "files": []}
        if not isinstance(directory, str):
            raise ValueError("invalid step record directory")
        try:
            return backend.records.read(directory, relative)
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
        restorable, so an updated file shows as a diff, else as its new text.
        ``link_query`` is carried to the Chain's links, so a page
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
        component = scenario.surface.files_component
        before_files: Mapping[str, str] | None = None
        if before is not None and component is not None:
            try:
                artifact, _ = scenario.release_queries.artifact_with_metrics(before)
                before_files = component_files(scenario.surface, artifact, component)
            except ArtifactError:
                before_files = None
        return build_release_page(step, rows, before_files=before_files, link_query=link_query)

    @classmethod
    def running_request_id(cls, scenario: Scenario) -> str | None:
        """The id of the request the scenario's step is running: the reserved job data's, else the backend's progress."""
        trainer = cls.files_trainer(scenario)
        reserved = trainer.pending_job_data
        if reserved is not None and reserved.request is not None:
            return str(reserved.request.id)
        backend = trainer.candidate_backend
        progress = backend.step_progress if isinstance(backend, HarnessBackend) else None
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
        progress = backend.step_progress if isinstance(backend, HarnessBackend) else None
        reserved = self.files_trainer(scenario).pending_job_data
        consumed = reserved is not None and reserved.request is not None and reserved.request.id == record_id
        return build_request_page(
            record, rows, scenario=scenario.name, progress=progress, consumed=consumed, link_query=link_query
        )

    def harness_request_progress(self, headers: Mapping[str, str], record_id: str) -> dict[str, Any]:
        """Where a filed request stands, as JSON, for a client with no browser to open its page.

        The same reading the page renders: the state
        (``queued``, ``proposing``, ``evaluating``, ``running``, else the
        settled row's result), what it means in the page's words,
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
                "runs_total": None,
                "step_record": None,
                "activity": [],
            }
        backend = self.files_trainer(scenario).candidate_backend
        progress = backend.step_progress if isinstance(backend, HarnessBackend) else None
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
            "runs_total": None if mine is None else mine.runs_total,
            "step_record": None if mine is None else mine.step_record,
            # What the method has done so far, oldest first: {at, kind, text, failed?}.
            "activity": [] if mine is None else [dict(line) for line in mine.activity],
        }

    def _file_scenario(self, headers: Mapping[str, str]) -> Scenario:
        """The existing file-serving scenario the headers name; ArtifactNotFound for an unknown one or one without files."""
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
                "no file tree). Point 'recipe.implementation' at a harness evolution recipe."
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
