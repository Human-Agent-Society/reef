"""Training jobs through the jobs queue: control adds jobs, the training service runs them.

Control builds no training component. ``JobQueueComponent`` is the component that a trainer's ``JobLoop`` sees. It
holds the backend's facts (``TrainingBackendFacts``) and the work queue: it answers from the facts, adds each
``TrainingJob`` to the ``jobs`` queue, and reads the job's ``JobStatus`` back from the queue item.
``TrainingJobWorker`` runs in the training service: it takes the jobs of its component, has the component run each
one, stores the files of each saved checkpoint in the artifact store, records the checkpoint in the item's
progress, and ends the item with the job.

What only the component can answer, control asks through the ``training_checks`` queue, and
``TrainingCheckWorker`` answers in the training service. An item of that queue:

- ``kind`` is ``training_check:<component name>``;
- ``payload`` is ``{"check": "job", "job_id", "job"}``, where ``job`` is a ``job_payload``, for
  ``TrainingComponent.check_job``; or ``{"check": "artifact", "names"}`` for
  ``TrainingComponent.check_remote_artifact``; or ``{"check": "archive"}`` for
  ``TrainingComponent.archive_scenario`` of the item's scenario, which control adds when it deletes the scenario
  and whose answer nobody reads;
- ``result`` is ``{"refusal": <message>}`` when the component raised ``ValueError``, else ``{"refusal": null}``;
- ``id`` is ``job:<job ID>`` for a job check, so control finds the check of a job in the queue,
  ``artifact:<random>`` for an artifact check, and ``archive:<scenario>:<random>`` for an archive;
- ``metadata["asked_at"]`` is the time in seconds when control last asked for the answer, and
  ``metadata["answer_read"]`` is true once control read it.

What bounds the life of a check item: the check worker fails a check that control did not ask for within
``UNASKED_CHECK_SECONDS`` instead of answering it, control cancels an artifact check that it stops waiting for,
and control's sweep deletes every final item after the work retention (``EndedWorkSweeper``).

A job's start can name a snapshot of its scenario (``SnapshotStart``). Both workers replace it by the component's
own start before the component reads the job (``resolved_job``), so the training service reads the scenario's
artifact store.

A jobs-queue item:

- ``id`` is the job ID, which ``JobLoop.job_id`` makes from the job's inputs;
- ``kind`` is ``training_job:<component name>``, so a worker takes only the jobs of its component;
- ``scenario`` is the job's scenario; the item has no ``serial_key``, so a free worker takes it at once;
- ``payload`` is the job (``job_payload``): its backend, start, parameters, component and previous job, the reference of its job data
  directory (``JobDataRef.to_dict``), and the summary of its training request for the job listing;
- each ``progress`` entry is one ``SavedCheckpoint``, in step order, with the name of its stored content and the
  parts that it changed (``TrainingComponent.changed_parts``);
- ``result`` is ``{"state": "finished", "metrics": {...}}``;
- ``metadata["progress"]`` is what the component reports about the running job
  (``TrainingComponent.job_progress``), or None; the training worker writes it whenever it changes;
- ``metadata["experiment"]``, when present, is the ``ExperimentRun`` that records the job, as ``to_dict`` gives
  it. Control writes it (``JobExperimentRecorder``) when its experiment tracker records a commit of the job, so the
  job listing of any service reads it from the item.

A checkpoint's ``path`` is a path of the training service, which only the component reads. Control reads no
file of a checkpoint: the training worker stores the files of the checkpoint's ``artifact_path`` as content of the
scenario's artifact store (``RepositoryBackend.store_content``) under ``checkpoint_content``, and control commits
that content by name (``JobLoop.publish``). Before it stores a checkpoint, the worker writes the content's name
into the item's ``metadata["stored_content"]``, so the item names all content of its job, also content that no
progress entry names because the worker stopped or lost the job in between. The committer discards the content of
a checkpoint once its commit is recorded, and control's sweep discards every named content of an ended job before
it deletes the job's item (``discard_ended_content``), so content that no commit names lives at most the work
retention longer than its job.

Control reads no other file of the training service either. The files that record how a job ran
(``TrainingComponent.job_records``) are stored content too, under ``job_records_content``, which the worker
stores when the job ends and which stays for the pages of control. A scenario's own directories of a component
are moved aside by the training service, when control asks (``archive``).

The job's data is not in the item. Control writes it as a job data directory in the data store
(``reef.storage.job_data``) before it adds the item, and the training worker opens the directory from the data
store before the component runs the job.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from reef.artifact.artifact import Artifact, ArtifactError
from reef.artifact.repository import Repository, RepositoryBackendFactory
from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import InternalError
from reef.core.weight_layout import ArtifactFormat
from reef.execution.workers import ItemFailedError, QueueWorker, WorkerStoppedError
from reef.observability.base import (
    ExperimentLogger,
    ExperimentRun,
    ExperimentTracker,
    TrainFromExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.storage.data import DataStore
from reef.storage.job_data import JobDataKind
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLostError, WorkQueue
from reef.train.component import (
    JobCheckPendingError,
    JobStatus,
    SavedCheckpoint,
    ScenarioRoot,
    SnapshotStart,
    TrainingBackendFacts,
    TrainingComponent,
    TrainingJob,
)
from reef.train.training_requests import REQUEST_FAILURE_KEY, TRAINING_REQUESTS_QUEUE

logger = logging.getLogger(__name__)

#: The queue of training jobs.
JOBS_QUEUE = "jobs"
#: The queue of the questions that control asks the training service.
CHECKS_QUEUE = "training_checks"
#: The key of a check item's ``metadata`` that holds when control last asked for its answer, in seconds.
ASKED_AT_KEY = "asked_at"
#: The key of a check item's ``metadata`` that is true once control read its answer.
ANSWER_READ_KEY = "answer_read"
#: The seconds after which the check worker answers no check that control did not ask for again.
UNASKED_CHECK_SECONDS = 600.0
#: The error of a check that control stopped asking for.
UNASKED_CHECK_ERROR = "no trainer asked for this check any more"
#: The key of a job item's ``metadata`` that holds the names of the content that the training worker stored, or
#: was about to store, for the job's checkpoints: a list of strings.
STORED_CONTENT_KEY = "stored_content"
#: The key of a job item's ``metadata`` that holds the running job's progress, as its component reports it.
PROGRESS_METADATA_KEY = "progress"
#: The key of a job item's ``metadata`` that holds the experiment run that records the job.
EXPERIMENT_METADATA_KEY = "experiment"


def job_kind(component_name: str) -> str:
    """The work-item kind of the jobs of the component named ``component_name``."""
    return f"training_job:{component_name}"


def check_kind(component_name: str) -> str:
    """The work-item kind of the checks that the component named ``component_name`` answers."""
    return f"training_check:{component_name}"


def job_payload(job: TrainingJob, request: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    """The jobs-queue payload of ``job``: its backend, start, parameters and data reference, and ``request``, the
    ``{id, text, parameters}`` of its training request or None, as JSON."""
    return cast(
        dict[str, JsonValue],
        {
            "backend": job.backend,
            "start": job.start,
            "data": job.data.to_dict(),
            "parameters": dict(job.parameters),
            "request": None if request is None else dict(request),
            "component": job.component,
            "previous_job_id": job.previous_job_id,
            "train_from_artifact_id": job.train_from_artifact_id,
        },
    )


def json_object(value: JsonValue, name: str) -> Mapping[str, JsonValue]:
    """``value`` as a JSON object; ``ValueError`` when it is another JSON value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is a JSON object")
    return value


def json_text(value: JsonValue, name: str) -> str:
    """``value`` as a non-empty string; ``ValueError`` when it is another JSON value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is a non-empty string")
    return value


def job_from_item(item: WorkItem, data_store: DataStore) -> TrainingJob:
    """The ``TrainingJob`` that ``job_payload`` wrote into the jobs-queue ``item``, with its data directory opened
    in ``data_store``; ``ValueError`` for another payload."""
    return job_from_payload(item.id, item.scenario, item.payload, data_store)


def job_from_payload(
    job_id: str, scenario: str, payload: Mapping[str, JsonValue], data_store: DataStore
) -> TrainingJob:
    """The ``TrainingJob`` ``job_id`` of ``scenario`` that ``job_payload`` wrote as ``payload``, with its data
    directory opened in ``data_store``; ``ValueError`` for another payload."""
    data = json_object(payload.get("data"), "a job's data")
    request = payload_request(payload)
    ref = data_store.open_job_directory(data)
    return TrainingJob(
        job_id,
        json_text(payload.get("backend"), "a job's backend"),
        json_text(payload.get("start"), "a job's start"),
        ref,
        json_object(payload.get("parameters"), "a job's parameters"),
        scenario=scenario,
        request_id=None if request is None else cast(str, request["id"]),
        component=optional_text(payload.get("component"), "a job's component"),
        previous_job_id=optional_text(payload.get("previous_job_id"), "a job's previous job"),
        train_from_artifact_id=optional_text(payload.get("train_from_artifact_id"), "a job's start artifact"),
    )


def optional_text(value: JsonValue, name: str) -> str | None:
    """``value`` as a non-empty string, or None for null or a missing key; ``ValueError`` for another value."""
    return None if value is None else json_text(value, name)


def request_summary(item: WorkItem) -> dict[str, JsonValue] | None:
    """The ``{id, text, parameters}`` of the training request of the job in ``item``, or None."""
    return payload_request(item.payload)


def payload_request(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    """The ``{id, text, parameters}`` of the training request in a job's ``payload``, or None."""
    value = payload.get("request")
    if value is None:
        return None
    given = json_object(value, "a job's request")
    text = given.get("text")
    if not isinstance(text, str):
        raise ValueError("a request's text is a string")
    return {
        "id": json_text(given.get("id"), "a request's id"),
        "text": text,
        "parameters": dict(json_object(given.get("parameters", {}), "a request's parameters")),
    }


def checkpoint_content(job_id: str, step: int) -> str:
    """The name of the stored content of the checkpoint that the job ``job_id`` saved at ``step``. It is the same
    for every attempt of the job, so a worker that stores the checkpoint again replaces the content."""
    return f"{job_id}:{step}"


def job_records_content(job_id: str) -> str:
    """The name of the stored content that holds the record files of the job ``job_id``
    (``TrainingComponent.job_records``)."""
    return f"{job_id}:records"


def checkpoint_json(checkpoint: SavedCheckpoint) -> dict[str, JsonValue]:
    """``checkpoint``, whose files the training service stored, as a progress entry:
    ``{step, path, content, metrics, changed_parts}``, with the changed parts as a sorted list. Raise
    ``ValueError`` for a checkpoint with no ``content``."""
    if checkpoint.content is None:
        raise ValueError("a progress entry holds a checkpoint whose files are stored content")
    return {
        "step": checkpoint.step,
        "path": checkpoint.path,
        "content": checkpoint.content,
        "metrics": cast(JsonValue, dict(checkpoint.metrics)),
        "changed_parts": cast(JsonValue, sorted(checkpoint.changed_parts)),
    }


def checkpoint_from_json(entry: Mapping[str, JsonValue]) -> SavedCheckpoint:
    """The ``SavedCheckpoint`` that ``checkpoint_json`` wrote; ``ValueError`` for another object."""
    step = entry.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("a checkpoint's step is an integer")
    parts = entry.get("changed_parts")
    if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
        raise ValueError("a checkpoint's changed_parts is a list of strings")
    return SavedCheckpoint(
        step,
        json_text(entry.get("path"), "a checkpoint's path"),
        None,
        json_object(entry.get("metrics"), "a checkpoint's metrics"),
        frozenset(cast(list[str], parts)),
        json_text(entry.get("content"), "a checkpoint's content"),
    )


def resolved_job(
    component: TrainingComponent,
    job: TrainingJob,
    backends: RepositoryBackendFactory | None,
    local_artifact_dir: Path | None,
) -> TrainingJob:
    """``job`` with the start that ``component`` reads.

    A job whose start is the component's own is returned as it is. A start that names a snapshot
    (``SnapshotStart``) is replaced: the snapshot is read from the artifact store of the job's scenario
    (``backends``) and made local under ``local_artifact_dir``, and the component makes its start of the job's
    component of it, with ``root_start`` for a scenario root and ``snapshot_start`` for any other snapshot. The
    component can write the start into its own state, and the same snapshot gives the same start again.

    Raise ``ValueError`` when the component refuses the snapshot, or when ``backends`` is None. Raise
    ``InternalError`` when the component returns a start that names a snapshot.
    """
    named = SnapshotStart.from_text(job.start)
    if named is None:
        return job
    if backends is None:
        raise ValueError(f"job {job.job_id} starts from a snapshot, and this worker opens no artifact store")
    backend = backends(job.scenario)
    snapshot = Artifact(
        backend.resolve_snapshot(named.revision),
        Repository(backend, backend.current(), local_dir=local_artifact_dir),
    )
    materialized = snapshot.materialize()
    manifest = materialized.components
    # A snapshot of one component is that component, whatever the trainer calls it.
    whole = job.component is None or manifest is None or manifest.single
    artifact = materialized if whole else materialized.component(cast(str, job.component))
    if named.root is None:
        start = component.snapshot_start(artifact)
    else:
        start = component.root_start(ScenarioRoot(artifact, named.root))
    if SnapshotStart.from_text(start) is not None:
        raise InternalError(f"the {component.name} training component made a start that names a snapshot")
    return replace(job, start=start)


#: The metadata entry of a failed job's item that holds the metrics the component reported for it.
FAILED_METRICS_KEY = "failed_metrics"


def status_from_item(item: WorkItem) -> JobStatus:
    """The ``JobStatus`` of the job in ``item``: running until the item is final, then finished or failed."""
    checkpoints = tuple(checkpoint_from_json(entry) for entry in item.progress)
    if item.state == "finished" and item.result is not None:
        return JobStatus(
            item.id, "finished", checkpoints, metrics=json_object(item.result.get("metrics", {}), "metrics")
        )
    if item.state == "failed":
        # A failed item always carries its error (``WorkItem``); a job that failed with metrics keeps them in the
        # item's metadata, since a failed item has no result.
        metrics = json_object(item.metadata.get(FAILED_METRICS_KEY, {}), "a failed job's metrics")
        return JobStatus(item.id, "failed", checkpoints, metrics=metrics, error=item.error)
    return JobStatus(item.id, "running", checkpoints)


#: The error of a job that a client cancelled.
JOB_CANCELLED_ERROR = "cancelled by request"


def job_json(item: WorkItem) -> dict[str, JsonValue]:
    """The jobs-queue ``item`` as the JSON object that the job listing gives; ``ValueError`` for an item whose
    payload ``job_payload`` did not write.

    The object has these keys:

    - ``id``, ``state``, ``attempt_count`` and ``error``: the item's fields (``WorkItem``).
    - ``checkpoints``: the saved checkpoints that the job reported, in step order, each as ``checkpoint_json``
      gives it, with its ``changed_parts``. A progress entry that another release of Reef wrote in another form
      is left out: such a job reads as failed (``JobQueueComponent.status``).
    - ``result``: ``{"state": "finished", "metrics": {...}}`` once the job finished, else None.
    - ``parameters``: the job's parameters, a JSON object: the recipe's parameters for the kind of the job's
      samples, with the request's parameters merged in. They hold the training ``method`` and the
      ``scheduling`` when the recipe or the request gives them.
    - ``sample_count``: the number of samples in the job's data, an integer from 0.
    - ``sample_kind``: ``"trajectory"`` or ``"task"``, the kind of the job's samples, which one job never mixes;
      None for data with no sample.
    - ``request``: ``{id, text, parameters}`` of the training request (``POST /reef/train``) that the job
      trains, where ``parameters`` is the JSON object that the request gave, ``{}`` when it gave none; None for
      a job that the training mode ``auto`` made without a request.
    - ``train_from_artifact_id``: the artifact that the job started from.
    - ``created_at``, ``started_at`` and ``ended_at``: ISO 8601 UTC times of the item, when control added the job,
      when a worker first took it, and when it finished or failed. ``started_at`` is None until a worker takes the
      job, and ``ended_at`` until the job ends.
    - ``experiment``: ``{tracker, project, entity, run_id, url}`` of the experiment run that records the job
      (``ExperimentRun.to_dict``), once control's tracker recorded a commit of the job; else None, and always None
      without an experiment tracker.
    """
    payload = item.payload
    data = json_object(payload.get("data"), "a job's data")
    sample_count, sample_kind = data.get("sample_count"), data.get("kind")
    experiment = item.metadata.get(EXPERIMENT_METADATA_KEY)
    return {
        "id": item.id,
        "state": item.state,
        "attempt_count": item.attempt_count,
        "checkpoints": [checkpoint_json(checkpoint) for checkpoint in listed_checkpoints(item)],
        "result": None if item.result is None else dict(item.result),
        "error": item.error,
        "parameters": dict(json_object(payload.get("parameters"), "a job's parameters")),
        "sample_count": sample_count,
        "sample_kind": sample_kind,
        "request": cast(JsonValue, request_summary(item)),
        "train_from_artifact_id": optional_text(payload.get("train_from_artifact_id"), "a job's start artifact"),
        "created_at": utc_time(item.created_at),
        "started_at": None if item.started_at is None else utc_time(item.started_at),
        "ended_at": None if item.ended_at is None else utc_time(item.ended_at),
        "experiment": (
            None
            if experiment is None
            else cast(JsonValue, ExperimentRun.from_dict(json_object(experiment, "a job's experiment")).to_dict())
        ),
    }


def listed_checkpoints(item: WorkItem) -> list[SavedCheckpoint]:
    """The checkpoints of the progress entries of ``item`` that have the current form."""
    checkpoints = []
    for entry in item.progress:
        try:
            checkpoints.append(checkpoint_from_json(entry))
        except ValueError:
            continue
    return checkpoints


def utc_time(seconds: float) -> str:
    """The time ``seconds`` after the epoch as an ISO 8601 string in UTC, such as ``2026-10-31T08:15:00.250000+00:00``."""
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def scenario_jobs(queue: WorkQueue, scenario: str) -> list[dict[str, JsonValue]]:
    """The training jobs of ``scenario`` in ``queue``, oldest first, each as ``job_json`` gives it."""
    return [job_json(item) for item in queue.items(JOBS_QUEUE, scenario=scenario)]


def cancel_scenario_job(queue: WorkQueue, scenario: str, job_id: str) -> dict[str, JsonValue] | None:
    """Cancel the training job ``job_id`` of ``scenario``; return the job, as ``job_json`` gives it, after the cancel.

    A job that has not ended fails with ``JOB_CANCELLED_ERROR``: a queued job at once, and a running job's training
    worker cancels the run at its next poll. The trainer reads the failed job and records its end. A job that ended
    stays as it is. Return None when ``scenario`` has no job ``job_id``.
    """
    item = queue.read(JOBS_QUEUE, job_id)
    if item is None or item.scenario != scenario:
        return None
    cancelled = queue.cancel(JOBS_QUEUE, job_id, JOB_CANCELLED_ERROR)
    return job_json(item if cancelled is None else cancelled)


def cancel_request_jobs(queue: WorkQueue, scenario: str, request_id: str) -> dict[str, JsonValue] | None:
    """Cancel each training job of ``scenario`` that trains the request ``request_id``, as ``cancel_scenario_job``
    does; return the newest of them after the cancel, or None when the request has no job."""
    newest: dict[str, JsonValue] | None = None
    for item in queue.items(JOBS_QUEUE, scenario=scenario):
        request = request_summary(item)
        if request is not None and request["id"] == request_id:
            newest = cancel_scenario_job(queue, scenario, item.id)
    return newest


class JobQueueComponent(TrainingComponent):
    """The training component that control uses: it adds jobs to the jobs queue instead of running them.

    Values: ``facts`` are the facts of the backend whose component runs the jobs in the training service.
    ``queue`` is the work queue. ``answer_seconds`` is the longest time that ``check_remote_artifact`` waits for
    the training service, a positive number. A job goes to the queue under its scenario, with no serial key: the
    trainer decides which jobs of a scenario run at the same time (``JobLoop.accepts_job``).

    Each member, with ``component`` the backend's component of the same settings:

    - ``name``, ``artifact_format``, ``shares_scenarios``, ``concurrent_jobs`` and ``parameters_schema`` are the
      facts' members, which equal the component's.
    - ``root_start(root)`` raises as ``component.root_start(root)`` does (``facts.check_root_start``). Otherwise it
      returns the name of the root snapshot, ``SnapshotStart(revision, root.source).to_text()``.
      ``snapshot_start(artifact)`` raises as ``component.snapshot_start(artifact)`` does
      (``facts.check_import_start``), else returns ``SnapshotStart(revision).to_text()``. The training service
      makes the component's start of the snapshot when it takes a job that names it (``resolved_job``).
    - ``submit(job)`` adds ``job`` to the jobs queue after the training service accepted it
      (``component.check_job``). The view keeps no state of its own: the check of a job is the item
      ``job:<job ID>`` of the ``training_checks`` queue. A call adds the item when there is none, and raises
      ``JobCheckPendingError`` until the item has an answer that no call read. The call that reads the answer adds
      the job, or raises ``ValueError`` with the component's message for a refused job, or ``RuntimeError`` when
      the training service could not run the check; that error says so and is no refusal. A call after an answer
      was read, or after the check worker dropped the check as no longer asked for, queues the check again.
    - ``status(job_id)`` is the status of the job's queue item. An item that does not parse, such as one that
      another release wrote, is a failed job whose error names what does not parse; the view cancels the item, so
      no worker runs it further. ``cancel`` cancels the item.
    - ``check_remote_artifact(names)`` adds a check item and waits for its answer: it returns, raises
      ``ValueError`` with the component's message, raises ``RuntimeError`` when the training service could not run
      the check, or raises ``ArtifactError`` when it gave no answer within ``answer_seconds``.
    - ``changed_parts(start, checkpoint)`` is ``checkpoint.changed_parts``, which the training worker recorded.
    - ``import_start``, ``check_job`` and ``check_task_jobs`` raise ``InternalError``: control reaches the
      component only through the members above.

    Purpose: the dispatcher makes one for each backend whose facts its process holds, and control's trainers
    submit jobs and read their status through it (``JobLoop``). The training service runs the jobs with
    ``TrainingJobWorker`` and answers the checks with ``TrainingCheckWorker``.
    """

    def __init__(self, facts: TrainingBackendFacts, queue: WorkQueue, *, answer_seconds: float = 60.0) -> None:
        if not isinstance(facts, TrainingBackendFacts):
            raise TypeError(f"facts must be TrainingBackendFacts, got {type(facts).__name__}")
        if answer_seconds <= 0:
            raise ValueError("answer_seconds is positive")
        self.facts = facts
        self.queue = queue
        self.answer_seconds = answer_seconds

    @property
    def name(self) -> str:
        return self.facts.name

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return self.facts.artifact_format

    @property
    def shares_scenarios(self) -> bool:
        return self.facts.shares_scenarios

    @property
    def concurrent_jobs(self) -> bool:
        return self.facts.concurrent_jobs

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        return self.facts.parameters_schema(kind)

    def root_start(self, root: ScenarioRoot) -> str:
        self.facts.check_root_start(root)
        return SnapshotStart(root.artifact.ref.revision, root.source).to_text()

    def snapshot_start(self, artifact: Artifact) -> str:
        if artifact.local_path is None:
            raise ValueError(f"artifact {artifact.ref.revision!r} has no local files to train from")
        self.facts.check_import_start(artifact.local_path)
        return SnapshotStart(artifact.ref.revision).to_text()

    def import_start(self, artifact_dir: Path) -> str:
        raise InternalError("control names the snapshot of a start (snapshot_start); the training service imports it")

    def check_job(self, job: TrainingJob) -> None:
        raise InternalError("the training service checks a job; control asks it in submit")

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        raise InternalError("the training service checks the recipe's task jobs when it starts")

    def changed_parts(self, start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        return checkpoint.changed_parts

    def job_progress(self, job_id: str) -> Mapping[str, Any] | None:
        """The progress that the training worker last wrote into the job's item, or None."""
        item = self.queue.read(JOBS_QUEUE, job_id)
        value = None if item is None else item.metadata.get(PROGRESS_METADATA_KEY)
        return None if value is None else json_object(value, "a job's progress")

    def archive_scenario(self, scenario: str) -> None:
        """Ask the training service to move aside what its component keeps for the deleted ``scenario``. The
        request waits in the queue while no training service runs, and nobody reads its answer."""
        item_id = f"archive:{scenario}:{uuid.uuid4().hex}"
        self.queue.append(NewWork(CHECKS_QUEUE, item_id, scenario, check_kind(self.name), {"check": "archive"}))

    def read_answer(self, item: WorkItem, subject: str) -> None:
        """Read the answer of the final check ``item`` about ``subject``: return when the component accepted, raise
        ``ValueError`` with its message when it refused, and ``RuntimeError`` when the check did not run."""
        if item.result is None:
            raise RuntimeError(
                f"the {self.name} training service could not check {subject}, so nothing was decided about it: "
                f"{item.error}"
            )
        refusal = item.result.get("refusal")
        if refusal is not None:
            raise ValueError(str(refusal))

    def require_checked(self, job: TrainingJob) -> None:
        """Return when the training service accepted ``job``; raise as ``submit`` does otherwise."""
        item_id = f"job:{job.job_id}"
        pending = JobCheckPendingError(f"the {self.name} training service has not checked job {job.job_id} yet")
        item = self.queue.read(CHECKS_QUEUE, item_id)
        if item is None:
            payload: dict[str, JsonValue] = {"check": "job", "job_id": job.job_id, "job": job_payload(job)}
            self.queue.append(NewWork(CHECKS_QUEUE, item_id, job.scenario, check_kind(self.name), payload))
            raise pending
        self.queue.write_metadata(CHECKS_QUEUE, item_id, ASKED_AT_KEY, time.time())
        if not item.final:
            raise pending
        if item.metadata.get(ANSWER_READ_KEY) is True or item.error == UNASKED_CHECK_ERROR:
            # The answer was for an earlier submit, or the check was dropped: ask again.
            self.queue.retry(CHECKS_QUEUE, item_id)
            self.queue.write_metadata(CHECKS_QUEUE, item_id, ANSWER_READ_KEY, False)
            raise pending
        self.queue.write_metadata(CHECKS_QUEUE, item_id, ANSWER_READ_KEY, True)
        self.read_answer(item, f"job {job.job_id}")

    def check_remote_artifact(self, names: Mapping[str, JsonValue]) -> None:
        # The check belongs to no scenario's work, so the item carries the backend's name in that place.
        item = self.queue.append(
            NewWork(
                CHECKS_QUEUE,
                f"artifact:{uuid.uuid4().hex}",
                f"training:{self.name}",
                check_kind(self.name),
                {"check": "artifact", "names": dict(names)},
            )
        )
        deadline = time.monotonic() + self.answer_seconds
        while not item.final:
            if time.monotonic() >= deadline:
                self.queue.cancel(CHECKS_QUEUE, item.id, "control stopped waiting")
                raise ArtifactError(
                    f"the {self.name} training service did not check the artifact within "
                    f"{self.answer_seconds:g} seconds; try again when it runs"
                )
            time.sleep(0.05)
            held = self.queue.read(CHECKS_QUEUE, item.id)
            if held is None:
                raise RuntimeError(f"the check of the artifact left the {CHECKS_QUEUE} queue without an answer")
            item = held
        self.read_answer(item, "the artifact")

    def submit(self, job: TrainingJob) -> None:
        """Add ``job`` to the jobs queue once the training service accepted it (see the class), with the summary
        of its training request, which the queue of training requests still holds, for the job listing.

        A job of a request that failed meanwhile, such as one that a client cancelled while control built the job's
        data, is cancelled as soon as it is in the queue. A cancel marks the request first and cancels its jobs
        second, so one of the two finds the job."""
        if job.backend != self.name:
            raise ValueError(f"the job is for {job.backend!r}, not the {self.name!r} training component")
        if self.queue.read(JOBS_QUEUE, job.job_id) is not None:
            return
        self.require_checked(job)
        request: dict[str, JsonValue] | None = None
        if job.request_id is not None:
            held = self.queue.read(TRAINING_REQUESTS_QUEUE, job.request_id)
            given = {} if held is None else held.payload
            text, parameters = given.get("text", ""), given.get("parameters", {})
            request = {"id": job.request_id, "text": text, "parameters": parameters}
        payload = job_payload(job, request)
        self.queue.append(NewWork(JOBS_QUEUE, job.job_id, job.scenario, job_kind(self.name), payload))
        if job.request_id is not None:
            held = self.queue.read(TRAINING_REQUESTS_QUEUE, job.request_id)
            if held is not None and REQUEST_FAILURE_KEY in held.metadata:
                self.queue.cancel(JOBS_QUEUE, job.job_id, JOB_CANCELLED_ERROR)

    def status(self, job_id: str) -> JobStatus:
        item = self.queue.read(JOBS_QUEUE, job_id)
        if item is None:
            raise KeyError(job_id)
        try:
            return status_from_item(item)
        except ValueError as exc:
            error = f"the job's queue item cannot be read; another release of Reef may have written it: {exc}"
            # An item that is not final still has a worker or a place in the queue: end it.
            self.queue.cancel(JOBS_QUEUE, job_id, error)
            return JobStatus(job_id, "failed", error=error)

    def cancel(self, job_id: str) -> None:
        self.queue.cancel(JOBS_QUEUE, job_id, "cancelled")


class JobExperimentRecorder(ExperimentTracker):
    """Control's experiment tracker: ``tracker``, which also writes into each training job's item the run that
    records the job.

    Values: ``tracker`` is the experiment tracker that records the scenarios' training, and not a
    ``JobExperimentRecorder``; ``queue`` holds the jobs queue. Every method but ``record`` is ``tracker``'s.

    ``record(event)`` gives ``event`` to ``tracker``. Then, for an event of a training job, it sets
    ``metadata["experiment"]`` of the job's item to ``tracker.experiment_run(event.context).to_dict()``
    (``WorkQueue.write_metadata``). Nothing is written when the tracker has no run for the event, or the queue
    has no such item, such as for a job that its component runs in the trainer's own process. Each commit of a
    job (its submission, its checkpoints and its end) records the same run, so a write after the first changes
    nothing.

    Purpose: the dispatcher of control wraps its tracker in it when the recipe trains with a training component.
    The tracker reports from control's report thread (``ExperimentReportQueue``), after each commit lands. The
    run is open in this process, so its URL is the one that the run gives; the job listing of any service reads
    it from the item (``job_json``).
    """

    def __init__(self, tracker: ExperimentTracker, queue: WorkQueue) -> None:
        if isinstance(tracker, JobExperimentRecorder):
            raise TypeError("a job experiment recorder wraps the tracker that records the training")
        self.tracker = tracker
        self.queue = queue

    @property
    def operational_metrics_enabled(self) -> bool:
        return self.tracker.operational_metrics_enabled

    def bind_scenario(
        self, *, scenario: str, recipe: str, source_artifacts: ArtifactSelection, run_segment: int
    ) -> ExperimentLogger:
        return self.tracker.bind_scenario(
            scenario=scenario, recipe=recipe, source_artifacts=source_artifacts, run_segment=run_segment
        )

    def correlation_metrics(self, context: TrainingExperimentContext) -> Mapping[str, Any]:
        return self.tracker.correlation_metrics(context)

    def experiment_run(self, context: TrainingExperimentContext) -> ExperimentRun | None:
        return self.tracker.experiment_run(context)

    def record(self, event: TrainingExperimentEvent) -> None:
        self.tracker.record(event)
        if event.training_job_id is None:
            return
        run = self.tracker.experiment_run(event.context)
        if run is not None:
            self.queue.write_metadata(
                JOBS_QUEUE, event.training_job_id, EXPERIMENT_METADATA_KEY, cast(JsonValue, run.to_dict())
            )

    def record_train_from(self, event: TrainFromExperimentEvent) -> None:
        self.tracker.record_train_from(event)

    def close(self) -> None:
        self.tracker.close()


class TrainingJobWorker(QueueWorker):
    """One worker of the training service: it takes the jobs of one component from the jobs queue and runs them.

    Values: as ``QueueWorker``, on the jobs of ``component``, which runs them; ``data_store`` is the data store,
    which keeps the data directory of each job; ``backends`` opens the artifact store of a scenario, where the
    worker stores the files of each checkpoint and reads the snapshot that a job's start names;
    ``local_artifact_dir`` is where such a snapshot is made local (``resolved_job``). ``poll_seconds`` is also
    the wait between two looks at a running job; ``slots`` is the number of jobs that the worker runs at the same
    time.

    Purpose: the training service starts one per training component (``start``); ``stop`` stops taking jobs and
    closes the component, which the worker owns. Each job runs to its end on this worker unless the worker loses
    the lease; then the component cancels its local run, and the worker that takes the job next continues it from
    the job's checkpoints. A job that the component reports failed fails for good.
    """

    def __init__(
        self,
        component: TrainingComponent,
        queue: WorkQueue,
        data_store: DataStore,
        backends: RepositoryBackendFactory,
        *,
        worker_id: str,
        local_artifact_dir: Path | None = None,
        poll_seconds: float = 0.5,
        slots: int = 1,
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a training worker runs jobs on the component itself, not on a jobs-queue component")
        super().__init__(
            queue, JOBS_QUEUE, [job_kind(component.name)], worker_id=worker_id, poll_seconds=poll_seconds, slots=slots
        )
        self.component = component
        self.data_store = data_store
        self.backends = backends
        self.local_artifact_dir = local_artifact_dir

    def start(self) -> None:
        """Start the component's jobs in this process (``TrainingComponent.start_jobs``), then the worker."""
        if self.thread is None:
            self.component.start_jobs()
        super().start()

    def stop(self) -> None:
        """Stop taking jobs, give a job in progress back to the queue, wait for the thread, and close the
        component. Another worker continues the job."""
        super().stop()
        self.component.close()

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        """Run the job of ``item`` on the component, and store and record its checkpoints, by step.

        The component gets the job with its own start (``resolved_job``). For each checkpoint that the item does
        not hold yet, the worker writes the name of its content (``checkpoint_content``) into the item's
        metadata, then stores the files of its ``artifact_path`` under that name in the artifact store of the
        job's scenario, and then appends the progress entry, which names that content and the parts that the
        checkpoint changed from the start (``TrainingComponent.changed_parts``). A worker that stops after it
        stored and before it recorded leaves no progress entry: the worker that continues the job stores the same
        name again, and when nobody continues the job, the sweep finds the name in the metadata. The worker also copies the component's
        ``job_progress`` into the item's metadata when it changed, and it stores the job's record files
        (``job_records``) under ``job_records_content`` when the job ends. At each poll the worker
        reads the item: a job that control cancelled, or that another worker took, ends here, and the component
        cancels its local run. A worker that continues a job reports only the checkpoints
        whose step the item does not hold yet. When the worker stops, the job goes back to the queue, so another
        worker continues it at once: the component keeps running it, and that worker's ``submit`` continues from
        the job's latest checkpoint.
        """
        job = resolved_job(
            self.component, job_from_item(item, self.data_store), self.backends, self.local_artifact_dir
        )
        self.component.submit(job)
        reported = {checkpoint_from_json(entry).step for entry in item.progress}
        progress_count = len(item.progress)
        written_progress: Mapping[str, Any] | None = None
        stored_names = [str(name) for name in cast(list[JsonValue], item.metadata.get(STORED_CONTENT_KEY, []))]
        try:
            while True:
                job_progress = self.component.job_progress(job.job_id)
                status = self.component.status(job.job_id)
                if status.state != "running":
                    job_progress = None
                    records = self.component.job_records(job.job_id)
                    if records is not None:
                        self.backends(job.scenario).store_content(job_records_content(item.id), records)
                if job_progress != written_progress:
                    value = None if job_progress is None else cast(JsonValue, dict(job_progress))
                    self.queue.write_metadata(JOBS_QUEUE, item.id, PROGRESS_METADATA_KEY, value)
                    written_progress = job_progress
                for checkpoint in status.checkpoints:
                    if checkpoint.step in reported:
                        continue
                    if checkpoint.artifact_path is None:
                        raise InternalError(
                            f"the {self.component.name} training component reported a checkpoint with no directory"
                        )
                    content = checkpoint_content(item.id, checkpoint.step)
                    if content not in stored_names:
                        stored_names.append(content)
                        named = self.queue.write_metadata(
                            JOBS_QUEUE, item.id, STORED_CONTENT_KEY, cast(JsonValue, list(stored_names))
                        )
                        if named is None:
                            raise WorkLeaseLostError(f"job {item.id} left the queue")
                    self.backends(job.scenario).store_content(content, Path(checkpoint.artifact_path))
                    stored = replace(
                        checkpoint,
                        artifact_path=None,
                        content=content,
                        changed_parts=self.component.changed_parts(job.start, checkpoint),
                    )
                    self.queue.append_progress(
                        JOBS_QUEUE, item.id, self.worker_id, checkpoint_json(stored), number=progress_count
                    )
                    reported.add(checkpoint.step)
                    progress_count += 1
                if status.state == "finished":
                    return {"state": "finished", "metrics": cast(JsonValue, dict(status.metrics))}
                if status.state == "failed":
                    if status.metrics:
                        self.queue.write_metadata(
                            JOBS_QUEUE, item.id, FAILED_METRICS_KEY, cast(JsonValue, dict(status.metrics))
                        )
                    raise ItemFailedError(status.error or "the job failed")
                if self.stopped.wait(self.poll_seconds):
                    raise WorkerStoppedError
                # Control can cancel the job (a deleted scenario, a new training start): stop at the next poll.
                held = self.queue.read(JOBS_QUEUE, item.id)
                if lease_lost.is_set() or held is None or held.worker_id != self.worker_id or held.final:
                    raise WorkLeaseLostError(f"job {item.id} is no longer held by {self.worker_id}")
        except WorkLeaseLostError:
            logger.warning("training worker %s lost job %s; cancelling its local run", self.worker_id, item.id)
            self.component.cancel(item.id)
            raise


class TrainingCheckWorker(QueueWorker):
    """The worker of the training service that answers control's checks with one component.

    Values: as ``QueueWorker``, on the ``training_checks`` items of ``component``, one at a time; ``data_store``
    and ``backends`` with ``local_artifact_dir`` are those of ``TrainingJobWorker``, for the job of a job check.

    ``run_item`` answers ``{"refusal": None}`` when the component accepted and ``{"refusal": message}`` when it
    raised ``ValueError(message)``: ``check_job`` of the job with its own start (``resolved_job``) for a ``job``
    check, ``check_remote_artifact`` for an ``artifact`` check. An ``archive`` runs ``archive_scenario`` of the
    item's scenario and answers ``{"refusal": None}``, however old it is. Any other payload fails the item. So does
    a job or artifact check that control last asked for more than ``UNASKED_CHECK_SECONDS`` ago (``metadata["asked_at"]``, else the
    item's ``created_at``), with ``UNASKED_CHECK_ERROR``: nobody waits for it, so it imports no start.

    Purpose: the training service starts one beside each ``TrainingJobWorker``. It has no job slot, so a check is
    answered while jobs run. The job worker owns and closes the component.
    """

    def __init__(
        self,
        component: TrainingComponent,
        queue: WorkQueue,
        data_store: DataStore,
        backends: RepositoryBackendFactory | None,
        *,
        worker_id: str,
        local_artifact_dir: Path | None = None,
        poll_seconds: float = 0.2,
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a check worker asks the component itself, not a jobs-queue component")
        super().__init__(
            queue, CHECKS_QUEUE, [check_kind(component.name)], worker_id=worker_id, poll_seconds=poll_seconds
        )
        self.component = component
        self.data_store = data_store
        self.backends = backends
        self.local_artifact_dir = local_artifact_dir

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        payload = item.payload
        check = payload.get("check")
        if check == "archive":
            # Nobody waits for this answer, and the request stays valid however long the service was down.
            self.component.archive_scenario(item.scenario)
            return {"refusal": None}
        asked_at = item.metadata.get(ASKED_AT_KEY, item.created_at)
        if isinstance(asked_at, int | float) and time.time() - asked_at > UNASKED_CHECK_SECONDS:
            raise ItemFailedError(UNASKED_CHECK_ERROR)
        if check == "job":
            job = job_from_payload(
                json_text(payload.get("job_id"), "a job check's job_id"),
                item.scenario,
                json_object(payload.get("job"), "a job check's job"),
                self.data_store,
            )
            try:
                self.component.check_job(resolved_job(self.component, job, self.backends, self.local_artifact_dir))
            except ValueError as exc:
                return {"refusal": str(exc)}
        elif check == "artifact":
            names = json_object(payload.get("names"), "an artifact check's names")
            try:
                self.component.check_remote_artifact(names)
            except ValueError as exc:
                return {"refusal": str(exc)}
        else:
            raise ItemFailedError(f"a training check is of a job, of an artifact or an archive, not {check!r}")
        return {"refusal": None}


def discard_ended_content(
    queue: WorkQueue, backends: RepositoryBackendFactory, *, ended_before: float, give_up_before: float
) -> tuple[str, ...]:
    """Discard the stored content of every checkpoint of the job items that ended before ``ended_before``
    (seconds), and return the IDs of those items that must stay because their content could not be discarded.

    These are the items that ``WorkQueue.delete_final`` deletes next. The names are those that the training worker
    wrote into the item before it stored each checkpoint (``STORED_CONTENT_KEY``), so the content of a checkpoint
    that no commit took, or that no progress entry names, goes with its item. Discarding content that a commit
    discarded already changes nothing. A failure of one item, whatever it raises, is logged and leaves the other
    items alone. Such an item stays for the next sweep, unless it ended before ``give_up_before``: then it goes,
    and the log names the content that stays in the artifact store.
    """
    kept: list[str] = []
    for item in queue.items(JOBS_QUEUE):
        if not item.final or item.ended_at is None or item.ended_at >= ended_before:
            continue
        names = item.metadata.get(STORED_CONTENT_KEY, [])
        try:
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("a job's stored content is a list of names")
            backend = backends(item.scenario)
            for name in names:
                backend.discard_content(cast(str, name))
        except Exception:
            # A sweep reads every scenario's backend, also a plugin's: one failure must not stop the others.
            if item.ended_at < give_up_before:
                logger.exception("gave up on the stored checkpoints %s of job %s; they stay stored", names, item.id)
            else:
                logger.exception("could not discard the stored checkpoints of job %s; the next sweep tries", item.id)
                kept.append(item.id)
    return tuple(kept)


__all__ = [
    "ANSWER_READ_KEY",
    "ASKED_AT_KEY",
    "CHECKS_QUEUE",
    "EXPERIMENT_METADATA_KEY",
    "JOBS_QUEUE",
    "JOB_CANCELLED_ERROR",
    "PROGRESS_METADATA_KEY",
    "STORED_CONTENT_KEY",
    "UNASKED_CHECK_ERROR",
    "UNASKED_CHECK_SECONDS",
    "JobExperimentRecorder",
    "JobQueueComponent",
    "TrainingCheckWorker",
    "TrainingJobWorker",
    "cancel_request_jobs",
    "cancel_scenario_job",
    "check_kind",
    "checkpoint_content",
    "discard_ended_content",
    "job_from_item",
    "job_from_payload",
    "job_json",
    "job_kind",
    "job_payload",
    "job_records_content",
    "request_summary",
    "resolved_job",
    "scenario_jobs",
    "status_from_item",
]
