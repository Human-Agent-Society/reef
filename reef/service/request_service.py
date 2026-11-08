"""Transport-independent request handling behind the aiohttp routes.

``RequestService`` normalizes typed Reef payloads, resolves the artifact
version before every provider call so concurrent publication cannot change
what gets recorded, and stores each exchange as a record. No aiohttp types appear here;
``reef.service.routes`` adapts these methods to HTTP.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactError, ArtifactNotFound
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.requirements import required_by
from reef.dispatcher import Dispatcher
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.sync import HarnessFiles, import_files, layer_install, merge_trees
from reef.harness.tree import HarnessTree, validate_tree
from reef.harness.tree.checks import clip_record_value
from reef.scenario.scenario import Scenario
from reef.scenario.snapshot_files import component_files
from reef.service.artifact_page import before_artifact_id, build_artifact_page, build_running_step_page, result_of
from reef.service.lifecycle import PendingWrites
from reef.service.request_page import STATE_WORDS, build_request_page, request_state, settled_step
from reef.service.routed_inference import (
    DispatcherRecords,
    InferenceRetryPolicy,
    InferenceRetryTimeout,
    RoutedInference,
)
from reef.service.routing import RoutingAdapter
from reef.service.routing_preview import RoutingPreviewClient
from reef.service.wire import (
    SCENARIO_HEADER,
    ReportPayload,
    RequestHeaders,
    parse_request_headers,
    scenario_header,
    with_request_tags,
)
from reef.train.harness_backend.backend import StepProgress
from reef.train.harness_backend.component import PROGRESS_FILE, STEPS_DIR, read_step_progress
from reef.train.harness_backend.records import read_step_records
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)
WriteResult = TypeVar("WriteResult")


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
    if request_type is not RequestType.REPORT:
        return dict(payload), ()
    report = ReportPayload.from_dict(payload)
    return report.to_dict(), report.references


def request_view(scenario: Scenario, request_id: str, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The training request ``request_id`` of ``scenario`` as the request page reads it: ``agent_record_id``,
    ``payload`` and ``created_at``.

    A queued or recent request comes from its queue item. One whose item control deleted after the work retention
    comes from the catalog row that settled it, which records the request (``metrics["training_request"]``), without
    ``created_at``. Raise ``ArtifactNotFound`` when neither holds it.
    """
    item = None if scenario.training_requests is None else scenario.training_requests.get(request_id)
    if item is not None:
        return {"agent_record_id": item.id, "payload": dict(item.payload), "created_at": item.created_at}
    step = settled_step(rows, request_id)
    metrics = rows[step].get("metrics") if step is not None else None
    request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
    if not isinstance(request, Mapping):
        raise ArtifactNotFound(f"scenario {scenario.name!r} has no harness request {request_id!r}")
    return {
        "agent_record_id": request_id,
        "payload": {key: value for key, value in request.items() if key != "id"},
        "created_at": None,
    }


class RequestService:
    """The request handling of one service app over ``dispatcher``.

    Values: ``inference_url`` is the address at which clients reach the inference service
    (``harness.inference_url``), or ``None``; a client's harness layer sends its model calls there, else to the
    address that the client gave.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        inference_url: str | None = None,
        routing: RoutingAdapter | None = None,
        writes: PendingWrites | None = None,
        preview: RoutingPreviewClient | None = None,
        retry_policy: InferenceRetryPolicy | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self.routing = routing
        self.preview = preview
        self.routed = RoutedInference(
            DispatcherRecords(dispatcher), preview, retry_policy or InferenceRetryPolicy(), writes
        )
        self.inference_url = None if inference_url is None else inference_url.rstrip("/")

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    async def persist(self, write: Awaitable[WriteResult]) -> WriteResult:
        """Retain a managed child's submitted writes through request cancellation and drain."""
        return await self.routed.persist(write)

    def accept_request(
        self, headers: Mapping[str, str], payload: Mapping[str, Any], *, request_id: str | None = None
    ) -> tuple[str, str]:
        """Queue a training request (``Dispatcher.accept_request``); return its scenario and ID."""
        item = self._dispatcher.accept_request(scenario_header(headers), payload, request_id=request_id)
        return item.scenario, item.id

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
        return self._dispatcher.accept_record(self.prepare_import_record(headers, body))

    def import_records(self, headers: Mapping[str, str], body: Mapping[str, object]) -> tuple[AgentRecord, ...]:
        """Import a bounded, atomic batch using the existing record contract."""
        records = body.get("records")
        if body.keys() != {"records"} or not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise ValueError("body must contain only 'records', an array of 1 to 1000 record objects")
        items: list[AgentRecord] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("each record must be an object")
            items.append(self.prepare_import_record(headers, record))
        return self._dispatcher.accept_records(items)

    def prepare_import_record(self, headers: Mapping[str, str], body: Mapping[str, object]) -> AgentRecord:
        if body.keys() - {"agent_record_id", "request_type", "payload"}:
            raise ValueError("record fields must be agent_record_id, request_type and payload")
        record_id = body.get("agent_record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("agent_record_id must be a non-empty string")
        request_type = body.get("request_type")
        if not isinstance(request_type, str) or request_type not in ("inference", "report"):
            raise ValueError("import request_type must be 'inference' or 'report'")
        payload = body.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("record payload must be an object")
        parsed = parse_request_headers(headers, RequestType(request_type))
        normalized, references = normalize_request_payload(parsed.request_type, payload)
        return AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=with_request_tags(normalized, parsed),
            agent_record_id=record_id,
            references=references,
            artifact_id=normalized.get("artifact_id"),
        )

    def harness_manifest(
        self,
        headers: Mapping[str, str],
        form: str = "rendered",
        service_url: str | None = None,
        root: str | None = None,
        *,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """The served tree plus its parent snapshot and evaluation metrics.

        ``artifact_id`` addresses one harness artifact instead of the
        active selection, so a consumer can pin or go back by pulling an older
        tree; an unknown or unrestorable artifact raises ArtifactNotFound
        naming it. ``form`` chooses the ``files``: ``rendered`` (the default)
        is the snapshot rendered for the scenario's adapter, ``source`` is
        its source tree, the paths a patch names, and ``client`` is the
        snapshot as a layer for a client that installs it at ``root``, bound
        to the scenario's route at ``inference_url``, else at ``service_url``
        (``reef.harness.sync.layer_install``). The client form adds ``run``,
        the command that runs the agent on the files (``binary``, ``args``,
        ``env``), the agent's commands that record and remove the install
        (``register``, ``unregister``) and the agent's pinned ``install``, and ``synced``, the render templates of the files a client
        sends back. The ``evaluation`` field carries the metrics of the
        training step that published the snapshot, so a consumer can audit
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
        selected = self.harness_artifact(scenario, artifact_id)
        manifest = self.harness_manifest_for_artifact(scenario, selected, source=form != "rendered")
        if form == "client" and service_url is not None and root is not None:
            descriptor = self.harness_descriptor(scenario)
            served = self._dispatcher.recipe.harness_served_model(
                scenario.name, str(scenario.components.files_component)
            )
            if served is None:
                raise ArtifactNotFound(f"scenario {scenario.name!r} evolves no harness that calls a model")
            # The layer's model calls go to the inference service when the deployment names its address.
            model_url = service_url.rstrip("/") if self.inference_url is None else self.inference_url
            scenario_url = f"{model_url}/reef/scenarios/{scenario.name}"
            layer = layer_install(HarnessTree(manifest["files"]), descriptor, served, scenario_url, root)
            manifest["files"] = dict(layer.files)
            install = descriptor.install
            manifest["run"] = {
                "binary": descriptor.binary,
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
        roots to its text. ``payload["base_artifact_id"]`` identifies the installed artifact. The files become the tree that
        ``reef.harness.sync.import_files`` reads against the base, and
        ``merge_trees`` puts the client's changes on the tree of the active
        artifact, not on the training start: a kept
        tree that the activation rule held stays inactive, so a sync never
        serves a change that waits for review. A file that the active tree
        and the client both changed in a way its part's ``merge`` refuses
        raises ``ArtifactConflict`` naming it. A merged tree that fails
        validation raises ``ValueError`` naming every problem. A sync that
        changes nothing publishes nothing. Otherwise the merged tree becomes
        the active selection and the harness trainer's training start in one
        ``"sync"`` commit (``ScenarioCommitter.sync_artifact``). The answer
        names the active selection after the sync, the snapshot it was merged
        onto, whether it changed, whether it merged onto a newer snapshot than
        the base, the tree's paths, and the client's paths no part takes.
        """
        files = payload.get("files")
        if not isinstance(files, Mapping) or not all(
            isinstance(path, str) and isinstance(text, str) for path, text in files.items()
        ):
            raise ValueError("files must map each path under the adapter's roots to the file's text")
        scenario = self._file_scenario(headers)
        component = scenario.components.files_component
        if component is None:
            raise ArtifactNotFound(f"scenario {scenario.name!r} serves no harness component")
        if payload.get("base_artifact_id") is None:
            raise ValueError("base_artifact_id must be the artifact the client installed")
        base_artifact = self.harness_artifact(scenario, payload.get("base_artifact_id"))
        base_artifact_id = scenario.components.component_artifact(base_artifact, component).artifact_id
        descriptor = self.harness_descriptor(scenario)
        expected_selection = scenario.artifact_selection()
        active_id = expected_selection.components[component]
        _, active_artifact = scenario.resolve_artifact(active_id, component)
        active_manifest = self.harness_manifest_for_artifact(scenario, active_artifact, source=True)
        active = HarnessTree(active_manifest["files"])
        if base_artifact_id == active_id:
            base_tree = active
        else:
            base_manifest = self.harness_manifest_for_artifact(scenario, base_artifact, source=True)
            base_tree = HarnessTree(base_manifest["files"])
        imported = import_files(files, base_tree, descriptor)
        merge = merge_trees(base_tree, active, imported.tree, descriptor.layout)
        if merge.tree is None:
            located = [(path, descriptor.layout.locate(path)) for path in merge.conflicts]
            # The person knows each file at its render path, such as plugin/rules.md for rules.md.
            named = [
                path if location is None or location.part.renders is None else location.part.render_path(location.name)
                for path, location in located
            ]
            raise ArtifactConflict(
                f"artifact {active_id} and your files both changed {', '.join(named)} since artifact {base_artifact_id}: "
                "make each one match one side, or drop yours with sync --discard, then sync again"
            )
        tree = merge.tree
        answer = {
            "parent_artifact_id": active_manifest["artifact_id"],
            "merged": base_tree != active,
            "files": sorted(tree.files),
            "ignored": list(imported.ignored),
        }
        if tree == active:
            return {
                **answer,
                "artifact_id": active_manifest["artifact_id"],
                "changed": False,
            }
        validation = validate_tree(tree, descriptor)
        if not validation.passed:
            raise ValueError(f"the synced harness fails validation: {validation.message()}")
        patch = active.diff(tree)
        directory = Path(tempfile.mkdtemp(prefix="reef-harness-sync-"))
        tree.write(directory)
        artifact = Artifact.local(directory)
        try:
            published = self._dispatcher.sync_artifact(
                scenario.name,
                artifact,
                component=component,
                expected_selection=expected_selection,
                metrics={
                    "sync": {
                        "files": len(tree.files),
                        "ignored": list(imported.ignored),
                        "base_artifact_id": base_artifact_id,
                    },
                    "patch": clip_record_value(patch.to_json()),
                    "patch_parts": descriptor.layout.part_names(patch.changes),
                },
            )
        finally:
            artifact.discard()
        _, selected = scenario.resolve_artifact(published.artifact_id, published.component)
        manifest = self.harness_manifest_for_artifact(scenario, selected, source=True)
        return {
            **answer,
            "artifact_id": manifest["artifact_id"],
            "changed": True,
        }

    @staticmethod
    def harness_descriptor(scenario: Scenario) -> AdapterDescriptor:
        """The adapter of the harness a client pulls; ArtifactNotFound when the scenario serves no harness tree."""
        component = scenario.components.files_component
        files = None if component is None else scenario.components.components[component].files
        if not isinstance(files, HarnessFiles):
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} evolves no harness: the deployment's recipe is not a harness "
                "evolution recipe"
            )
        return files.descriptor

    def harness_state_dir(self, scenario: Scenario) -> Path | None:
        """The state directory of the harness a client pulls (``Recipe.harness_state_dir``), or None."""
        component = scenario.components.files_component
        return None if component is None else self._dispatcher.recipe.harness_state_dir(scenario.name, component)

    def step_progress(self, scenario: Scenario) -> StepProgress | None:
        """The progress of the harness step that runs now, which the training component writes; None for none."""
        root = self.harness_state_dir(scenario)
        return None if root is None else read_step_progress(root / PROGRESS_FILE)

    @staticmethod
    def files_trainer(scenario: Scenario) -> Trainer:
        """The trainer evolving the component a client pulls; a flat scenario's only trainer."""
        return scenario.trainer_for(scenario.components.files_component)

    @staticmethod
    def harness_artifact(scenario: Scenario, artifact_id: object) -> Artifact:
        """The snapshot that holds the harness that a client names by ``artifact_id``; the active harness when it is
        None. Raise ``ValueError`` when it is not a non-empty string.
        """
        if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
            raise ValueError("artifact_id must be a non-empty string")
        component = scenario.components.files_component
        if not isinstance(artifact_id, str):
            if component is None:
                raise ArtifactNotFound(f"scenario {scenario.name!r} serves no harness component")
            artifact_id = scenario.artifact_selection().components[component]
        _, selected = scenario.resolve_artifact(artifact_id, component)
        return selected

    @staticmethod
    def harness_manifest_for_artifact(
        scenario: Scenario, artifact: Artifact, *, source: bool = False
    ) -> dict[str, Any]:
        component = scenario.components.files_component
        if component is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: the deployment's recipe "
                "declares no harness component (record-only or weight-training recipes have "
                "no file tree). Point 'recipe.implementation' at a harness evolution recipe."
            )
        files = (
            component_files(scenario.components, artifact, component)
            if source
            else scenario.components.read_files(artifact)
        )
        if files is None:
            raise ArtifactNotFound(
                f"scenario {scenario.name!r} serves no files: snapshot {artifact.ref.revision!r} holds no "
                "harness tree. A root snapshot made from the default start holds the recipe's evolution.seed, so an empty seed serves "
                "nothing until a step publishes a tree."
            )
        artifact_id = scenario.components.component_artifact(artifact, component).artifact_id
        catalog = scenario.artifacts()
        row = next((row for row in catalog if row["artifact_id"] == artifact_id), None)
        manifest = {
            "artifact_id": artifact_id,
            "parent_artifact_id": None if row is None else row["parent_artifact_id"],
            "files": dict(files),
            "evaluation": None if row is None or row["operation"] != "training" else row.get("metrics"),
            # The union over the chain, not this evaluation's list: a snapshot whose request named nothing still installs an earlier extension.
            "requires": required_by(list(reversed(catalog)), artifact_id),
        }
        components = artifact.materialize().components
        if components is not None:
            manifest["components"] = components.artifact_ids(artifact.ref.revision)
        return manifest

    def harness_artifacts(self, headers: Mapping[str, str]) -> dict[str, Any]:
        """The harness artifacts, oldest first, addressed by artifact ID, with the metrics of the step that published
        each one, so an update is a decision over numbers. A scenario with several components lists the artifacts of
        its harness component. Same read-only rules as ``harness_manifest``."""
        scenario = self._file_scenario(headers)
        return {"scenario": scenario.name, "artifacts": list(reversed(scenario.harness_rows()))}

    def harness_step_records(self, headers: Mapping[str, str], step: int, relative: str | None) -> dict[str, Any]:
        """Raw retained files for a catalog row; presentation belongs to the caller."""
        scenario = self._file_scenario(headers)
        rows = list(reversed(scenario.harness_rows()))
        if not 0 <= step < len(rows):
            raise ArtifactNotFound(f"scenario {scenario.name!r} has no step {step}")
        directory = (rows[step].get("metrics") or {}).get("step_record")
        root = self.harness_state_dir(scenario)
        if not directory or root is None:
            return {"status": "not_recorded", "files": []}
        if not isinstance(directory, str):
            raise ValueError("invalid step record directory")
        try:
            return read_step_records(root / STEPS_DIR, directory, relative)
        except FileNotFoundError as error:
            raise ArtifactNotFound("record file is not retained") from error

    def harness_artifact_page(
        self, headers: Mapping[str, str], step: int, link_query: Mapping[str, str] | None = None
    ) -> str:
        """One HTML page for the catalog row at ``step``, counted oldest first with the creation row as 0.

        The rows are the ones ``harness_artifacts`` answers, so the step a
        client counts there is the step this page names. The snapshot the
        step ran on (the parent of a win, the head a rejected or skipped
        step ran on) comes through the artifact snapshot when it is
        restorable, so an updated file shows as a diff, else as its new text.
        ``link_query`` is carried to the Chain's links, so a page
        opened through query parameters links pages that open the same way.
        An unknown step raises ArtifactNotFound naming the range.
        """
        scenario = self._file_scenario(headers)
        rows = list(reversed(scenario.harness_rows()))
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
        before = before_artifact_id(rows[step])
        component = scenario.components.files_component
        before_files: Mapping[str, str] | None = None
        if before is not None and component is not None:
            try:
                _, artifact = scenario.resolve_artifact(before, component)
                before_files = component_files(scenario.components, artifact, component)
            except ArtifactError:
                before_files = None
        return build_artifact_page(step, rows, before_files=before_files, link_query=link_query)

    def running_request_id(self, scenario: Scenario) -> str | None:
        """The id of the request the scenario's step is running: the reserved job data's, else the step's progress."""
        reserved = self.files_trainer(scenario).pending_job_data
        if reserved is not None and reserved.request is not None:
            return str(reserved.request.id)
        progress = self.step_progress(scenario)
        return None if progress is None or progress.request_id is None else str(progress.request_id)

    def harness_request_page(
        self, headers: Mapping[str, str], record_id: str, link_query: Mapping[str, str] | None = None
    ) -> str:
        """One HTML page for the harness request stored as agent record ``record_id``, live until its step settles.

        The record is the ``POST /reef/train`` request as stored; the
        catalog row whose ``training_request.id`` names it settles the page.
        Until then the page reads the running step's progress from the
        progress file of the harness training component, when a step runs, and
        whether the trainer holds the request in its reserved job data.
        ``link_query`` is carried to the version page link, so a page opened
        through query parameters links one that opens the same way. An
        unknown id, or one that is not a training request, raises
        ArtifactNotFound naming it.
        """
        scenario = self._file_scenario(headers)
        # The step a request settled as counts the catalog rows, the ones the snapshot page opens.
        rows = list(reversed(scenario.harness_rows()))
        record = request_view(scenario, record_id, rows)
        progress = self.step_progress(scenario)
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
        An unknown id, or one that is not a training request, raises
        ArtifactNotFound naming it.
        """
        scenario = self._file_scenario(headers)
        rows = list(reversed(scenario.harness_rows()))
        record = request_view(scenario, record_id, rows)
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
        progress = self.step_progress(scenario)
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
        scenario = self._dispatcher.get_or_create_scenario(parsed.scenario)
        if scenario is None:
            raise ReefError(f"scenario {parsed.scenario!r} disappeared during lookup")
        if scenario.components.files_component is None:
            raise ArtifactNotFound(
                f"scenario {parsed.scenario!r} serves no files: the deployment's recipe "
                "declares no harness component (record-only or weight-training recipes have "
                "no file tree). Point 'recipe.implementation' at a harness evolution recipe."
            )
        return self._dispatcher.at_newest_commit(scenario)

    def _accept(
        self,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        *,
        agent_record_id: str | None = None,
    ) -> AgentRecord:
        normalized_payload, references = normalize_request_payload(parsed.request_type, payload)
        normalized_payload = with_request_tags(normalized_payload, parsed)
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=normalized_payload,
            agent_record_id=agent_record_id,
            references=references,
            artifact_id=normalized_payload.get("artifact_id"),
        )
        try:
            return self._dispatcher.accept_record(item)
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
    "InferenceRetryPolicy",
    "InferenceRetryTimeout",
    "RequestService",
    "normalize_request_payload",
    "page_headers",
]
