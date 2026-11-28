"""The harness operations: the tree that a client pulls, the sync of a person's files, and the steps of a scenario.

``harness_manifest`` answers ``GET /reef/harness``, ``harness_import`` commits ``POST /reef/harness/import``,
``harness_artifacts`` lists the harness artifacts, ``harness_step_records`` reads the files that a step kept, and
``harness_request_progress`` says where a filed request stands. Each one takes the dispatcher and the name of the
scenario, and reads the scenario at its newest commit (``file_scenario``). None creates a scenario.

The row readers (``result_of``, ``before_artifact_id``, ``settled_step``, ``request_state``) read one row of the
harness artifacts or one request. The harness pages in ``reef.api`` use them too, so a page and the JSON answer name
one result the same way.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactConflictError, ArtifactNotFoundError
from reef.control.dispatcher import Dispatcher
from reef.control.scenario.live_objects import harness_served_model
from reef.control.scenario.scenario import Scenario
from reef.control.scenario.snapshot_files import component_files
from reef.core.errors import ReefError
from reef.core.requirements import required_by
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.sync import HarnessFiles, import_files, layer_install, merge_trees
from reef.harness.tree import HarnessTree, validate_tree
from reef.harness.tree.checks import clip_record_value
from reef.train.harness_backend.backend import JobProgress
from reef.train.harness_backend.records import read_job_records
from reef.train.job_queue import job_records_content
from reef.train.jobs import TRAINING_JOB_KEY, JobLoopState
from reef.train.trainer import Trainer

#: What each state means, in the words the page prints beside it.
STATE_WORDS = {
    "queued": "no step has taken the request yet; the trainer runs one step per request, oldest first",
    "proposing": "the method is making the change; the activity shows what it has done so far",
    "evaluating": "the evaluation is running the candidate through its runs",
    "running": "the step holds the request and reports no phase; its row follows",
}


def harness_manifest(
    dispatcher: Dispatcher,
    scenario_name: str,
    form: str = "rendered",
    service_url: str | None = None,
    root: str | None = None,
    *,
    artifact_id: str | None = None,
    inference_url: str | None = None,
) -> dict[str, Any]:
    """The served tree plus its parent snapshot and evaluation metrics.

    ``artifact_id`` addresses one harness artifact instead of the
    active selection, so a consumer can pin or go back by pulling an older
    tree; an unknown or unrestorable artifact raises ArtifactNotFoundError
    naming it. ``form`` chooses the ``files``: ``rendered`` (the default)
    is the snapshot rendered for the scenario's adapter, ``source`` is
    its source tree, the paths a patch names, and ``client`` is the
    snapshot as a layer for a client that installs it at ``root``, bound
    to the scenario's route at ``inference_url`` (``harness.inference_url``, the address at which clients reach
    the inference service), else at ``service_url``
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
    scenario = file_scenario(dispatcher, scenario_name)
    selected = harness_artifact(scenario, artifact_id)
    manifest = harness_manifest_for_artifact(scenario, selected, source=form != "rendered")
    if form == "client" and service_url is not None and root is not None:
        descriptor = harness_descriptor(scenario)
        served = harness_served_model(
            dispatcher.recipe,
            dispatcher.live,
            scenario.name,
            str(scenario.components.files_component),
        )
        if served is None:
            raise ArtifactNotFoundError(f"scenario {scenario.name!r} evolves no harness that calls a model")
        # The layer's model calls go to the inference service when the deployment names its address.
        model_url = (service_url if inference_url is None else inference_url).rstrip("/")
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


def harness_import(dispatcher: Dispatcher, scenario_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Commit a sync of a person's harness files: ``POST /reef/harness/import``.

    ``payload["files"]`` maps each file's path under the adapter's render
    roots to its text. ``payload["base_artifact_id"]`` identifies the installed artifact. The files become the tree that
    ``reef.harness.sync.import_files`` reads against the base, and
    ``merge_trees`` puts the client's changes on the tree of the active
    artifact, not on the training start: a kept
    tree that the activation rule held stays inactive, so a sync never
    serves a change that waits for review. A file that the active tree
    and the client both changed in a way its part's ``merge`` refuses
    raises ``ArtifactConflictError`` naming it. A merged tree that fails
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
    scenario = file_scenario(dispatcher, scenario_name)
    component = scenario.components.files_component
    if component is None:
        raise ArtifactNotFoundError(f"scenario {scenario.name!r} serves no harness component")
    if payload.get("base_artifact_id") is None:
        raise ValueError("base_artifact_id must be the artifact the client installed")
    base_artifact = harness_artifact(scenario, payload.get("base_artifact_id"))
    base_artifact_id = scenario.components.component_artifact(base_artifact, component).artifact_id
    descriptor = harness_descriptor(scenario)
    expected_selection = scenario.artifact_selection()
    active_id = expected_selection.components[component]
    _, active_artifact = scenario.resolve_artifact(active_id, component)
    active_manifest = harness_manifest_for_artifact(scenario, active_artifact, source=True)
    active = HarnessTree(active_manifest["files"])
    if base_artifact_id == active_id:
        base_tree = active
    else:
        base_manifest = harness_manifest_for_artifact(scenario, base_artifact, source=True)
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
        raise ArtifactConflictError(
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
        published = dispatcher.sync_artifact(
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
    manifest = harness_manifest_for_artifact(scenario, selected, source=True)
    return {
        **answer,
        "artifact_id": manifest["artifact_id"],
        "changed": True,
    }


def harness_artifacts(dispatcher: Dispatcher, scenario_name: str) -> dict[str, Any]:
    """The harness artifacts, oldest first, addressed by artifact ID, with the metrics of the step that published
    each one, so an update is a decision over numbers. A scenario with several components lists the artifacts of
    its harness component. Same read-only rules as ``harness_manifest``."""
    scenario = file_scenario(dispatcher, scenario_name)
    return {"scenario": scenario.name, "artifacts": list(reversed(scenario.harness_rows()))}


def harness_step_records(
    dispatcher: Dispatcher, scenario_name: str, step: int, relative: str | None
) -> dict[str, Any]:
    """Raw retained files for a catalog row; presentation belongs to the caller."""
    scenario = file_scenario(dispatcher, scenario_name)
    rows = list(reversed(scenario.harness_rows()))
    if not 0 <= step < len(rows):
        raise ArtifactNotFoundError(f"scenario {scenario.name!r} has no step {step}")
    metrics = rows[step].get("metrics") or {}
    job = metrics.get(TRAINING_JOB_KEY)
    if not metrics.get("step_record") or not isinstance(job, Mapping):
        return {"status": "not_recorded", "files": []}
    # The training service stored the job's record directory in the artifact store when the job ended.
    try:
        directory = scenario.repository.backend.materialize_content(job_records_content(str(job["job_id"])))
    except ArtifactNotFoundError:
        return {"status": "missing", "files": []}
    try:
        return read_job_records(directory.parent, str(directory), relative)
    except FileNotFoundError as error:
        raise ArtifactNotFoundError("record file is not retained") from error


def harness_request_progress(dispatcher: Dispatcher, scenario_name: str, record_id: str) -> dict[str, Any]:
    """Where a filed request stands, as JSON, for a client with no browser to open its page.

    The same reading the page renders: the state
    (``queued``, ``proposing``, ``evaluating``, ``running``, else the
    settled row's result), what it means in the page's words,
    and, once a row answers the request, the step it landed as. A client
    polls this to show a phase; the page itself stays the readable view.
    An unknown id, or one that is not a training request, raises
    ArtifactNotFoundError naming it.
    """
    scenario = file_scenario(dispatcher, scenario_name)
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
    progress = step_progress(scenario)
    reserved = files_trainer(scenario).pending_job_data
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
        "step_record": None if mine is None else mine.job_record,
        # What the method has done so far, oldest first: {at, kind, text, failed?}.
        "activity": [] if mine is None else [dict(line) for line in mine.activity],
    }


def file_scenario(dispatcher: Dispatcher, scenario_name: str) -> Scenario:
    """The existing file-serving scenario ``scenario_name``, at its newest commit; ``ArtifactNotFoundError`` for an
    unknown one or one without files. It never creates a scenario."""
    if not dispatcher.has_scenario(scenario_name):
        raise ArtifactNotFoundError(f"unknown scenario {scenario_name!r}")
    scenario = dispatcher.get_or_create_scenario(scenario_name)
    if scenario is None:
        raise ReefError(f"scenario {scenario_name!r} disappeared during lookup")
    if scenario.components.files_component is None:
        raise ArtifactNotFoundError(
            f"scenario {scenario_name!r} serves no files: the deployment's recipe "
            "declares no harness component (record-only or weight-training recipes have "
            "no file tree). Point 'recipe.implementation' at a harness evolution recipe."
        )
    return dispatcher.at_newest_commit(scenario)


def harness_descriptor(scenario: Scenario) -> AdapterDescriptor:
    """The adapter of the harness a client pulls. Raise ``ArtifactNotFoundError`` when the scenario serves no
    harness tree."""
    component = scenario.components.files_component
    files = None if component is None else scenario.components.components[component].files
    if not isinstance(files, HarnessFiles):
        raise ArtifactNotFoundError(
            f"scenario {scenario.name!r} evolves no harness: the deployment's recipe is not a harness "
            "evolution recipe"
        )
    return files.descriptor


def files_trainer(scenario: Scenario) -> Trainer:
    """The trainer evolving the component a client pulls; a flat scenario's only trainer."""
    return scenario.trainer_for(scenario.components.files_component)


def harness_artifact(scenario: Scenario, artifact_id: object) -> Artifact:
    """The snapshot that holds the harness that a client names by ``artifact_id``; the active harness when it is
    None. Raise ``ValueError`` when it is not a non-empty string.
    """
    if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
        raise ValueError("artifact_id must be a non-empty string")
    component = scenario.components.files_component
    if not isinstance(artifact_id, str):
        if component is None:
            raise ArtifactNotFoundError(f"scenario {scenario.name!r} serves no harness component")
        artifact_id = scenario.artifact_selection().components[component]
    _, selected = scenario.resolve_artifact(artifact_id, component)
    return selected


def harness_manifest_for_artifact(scenario: Scenario, artifact: Artifact, *, source: bool = False) -> dict[str, Any]:
    component = scenario.components.files_component
    if component is None:
        raise ArtifactNotFoundError(
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
        raise ArtifactNotFoundError(
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
    # The snapshot carries its manifest: no file is read for it.
    components = artifact.components
    if components is not None:
        manifest["components"] = components.artifact_ids(artifact.ref.revision)
    return manifest


def step_progress(scenario: Scenario) -> JobProgress | None:
    """The progress of the harness step that runs now, or None for none.

    The training worker writes it into the item of the running job (``TrainingComponent.job_progress``), and
    this reads it there: the first running job of the harness trainer that reports one.
    """
    trainer = files_trainer(scenario)
    loop = trainer.job_loop
    if loop is None:
        return None
    for job in JobLoopState.from_dict(trainer.state).jobs:
        value = loop.component.job_progress(job.job_id)
        if value is not None:
            return JobProgress.from_json(value)
    return None


def running_request_id(scenario: Scenario) -> str | None:
    """The id of the request the scenario's step is running: the reserved job data's, else the step's progress."""
    reserved = files_trainer(scenario).pending_job_data
    if reserved is not None and reserved.request is not None:
        return str(reserved.request.id)
    progress = step_progress(scenario)
    return None if progress is None or progress.request_id is None else str(progress.request_id)


def request_view(scenario: Scenario, request_id: str, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The training request ``request_id`` of ``scenario`` as the request page reads it: ``agent_record_id``,
    ``payload`` and ``created_at``.

    A queued or recent request comes from its queue item. One whose item control deleted after the work retention
    comes from the catalog row that settled it, which records the request (``metrics["training_request"]``), without
    ``created_at``. Raise ``ArtifactNotFoundError`` when neither holds it.
    """
    item = None if scenario.training_requests is None else scenario.training_requests.get(request_id)
    if item is not None:
        return {"agent_record_id": item.id, "payload": dict(item.payload), "created_at": item.created_at}
    step = settled_step(rows, request_id)
    metrics = rows[step].get("metrics") if step is not None else None
    request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
    if not isinstance(request, Mapping):
        raise ArtifactNotFoundError(f"scenario {scenario.name!r} has no harness request {request_id!r}")
    return {
        "agent_record_id": request_id,
        "payload": {key: value for key, value in request.items() if key != "id"},
        "created_at": None,
    }


def result_of(row: Mapping[str, Any]) -> str:
    """The result of one ``GET /reef/harness/artifacts`` row, one of:

    - ``rejected``: ``metrics.selected`` is false.
    - ``inactive``: ``metrics.selected`` is true, the row made an artifact
      (``checkpoint``) and that artifact was never active (``activated`` is
      false). A person must activate it.
    - ``selected``: ``metrics.selected`` is true otherwise.
    - ``failed``: ``metrics.selected`` is absent, ``metrics.skipped`` is set and
      ``metrics.error`` is non-blank text: the step raised and published nothing.
    - ``skipped``: ``metrics.selected`` is absent and ``metrics.skipped`` is set otherwise.
    - ``failed`` also for the end of a training job whose ``metrics.training_job.event`` is ``failed``: the step's
      job raised and published nothing.
    - the row's ``operation`` (``creation``, ``training`` or ``recovery``) for
      any other row, ``unknown`` when the row names none.

    The harness pages, the request page and the tutorial drivers read this function."""
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    selected = metrics.get("selected")
    if selected is False:
        return "rejected"
    if selected is True:
        return "inactive" if row.get("checkpoint") and not row.get("activated") else "selected"
    if metrics.get("skipped"):
        error = metrics.get("error")
        return "failed" if isinstance(error, str) and error.strip() else "skipped"
    job = metrics.get("training_job")
    if isinstance(job, Mapping) and job.get("event") == "failed":
        return "failed"
    return str(row.get("operation") or "unknown")


def before_artifact_id(row: Mapping[str, Any]) -> str | None:
    """The snapshot the step ran on: the parent of a snapshot the step made, the snapshot a rejected or skipped step
    ran on.

    A rejected or skipped step's row carries the id of the snapshot it ran
    on, so its parent would be one snapshot too far back."""
    selection_result = result_of(row)
    if selection_result in ("selected", "inactive"):
        parent = row.get("parent_artifact_id")
        return str(parent) if parent else None
    if selection_result in ("rejected", "skipped", "failed"):
        return str(row.get("artifact_id") or "") or None
    return None


def settled_step(rows: Sequence[Mapping[str, object]], record_id: str) -> int | None:
    """The step whose row answered the request: the one whose ``metrics.training_request.id`` is ``record_id``."""
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
        if isinstance(request, Mapping) and request.get("id") == record_id:
            return index
    return None


def request_state(record: Mapping[str, object], progress: JobProgress | None, consumed: bool) -> str:
    """Read the backend phase or reserved job data; an unreserved request is queued."""
    if progress is not None and progress.request_id == record["agent_record_id"]:
        return progress.phase
    if consumed:
        return "running"
    return "queued"
