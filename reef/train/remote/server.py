"""The generic training backend server: one ``TrainingBackend`` behind the HTTP contract (RFC #1146).

The server runs each job in its own thread. Before the backend runs a job, the server opens the job's artifact and
data directory into local directories through ``files``, a segment store (``SegmentStore.open_directory``). Each
checkpoint that the backend saves goes to ``files`` as the directory ``checkpoints/<job id>/<step>``, and the
server reports its URI. A backend keeps no state between jobs: a job reads everything that it continues from in its
artifact, and writes it into its checkpoints.

The server keeps its jobs in memory. After a restart, a job that it does not know is a 404, and Reef fails it.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml
from aiohttp import web

from reef.core.plugins import Plugin
from reef.storage.segments import LocalSegmentStore, SegmentStore
from reef.train.remote.contract import BackendInfo, RemoteCheckpoint, RemoteJob, RemoteJobStatus, refusal

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """The job was cancelled. A backend raises it, or lets ``JobRun.save`` raise it, to end the job."""


@dataclass
class JobRun:
    """The state of one job on the server, and what a running backend writes to it.

    Values: ``job`` is the job. ``checkpoints`` are its saved checkpoints, in step order; ``progress`` is the
    backend's latest progress, or None. ``cancelled`` is set when Reef cancels the job.
    Purpose: the server makes one per job and gives it to ``TrainingBackend.run``.
    """

    job: RemoteJob
    files: SegmentStore
    state: str = "running"
    checkpoints: list[RemoteCheckpoint] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    progress: Mapping[str, Any] | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, step: int, directory: Path, metrics: Mapping[str, Any] | None = None) -> RemoteCheckpoint:
        """Keep ``directory``, an artifact directory with ``release/`` and ``state/``, as the checkpoint of ``step``.
        ``JobCancelled`` when the job was cancelled."""
        if self.cancelled.is_set():
            raise JobCancelled
        uri = self.files.put_directory(f"checkpoints/{self.job.job_id}/{step}", directory)
        checkpoint = RemoteCheckpoint(step, uri, dict(metrics or {}))
        with self.lock:
            if self.checkpoints and step <= self.checkpoints[-1].step:
                raise ValueError(f"a checkpoint's step must grow, got {step} after {self.checkpoints[-1].step}")
            self.checkpoints.append(checkpoint)
        return checkpoint

    def report(self, progress: Mapping[str, Any]) -> None:
        """Set the job's progress, which Reef shows while the job runs."""
        with self.lock:
            self.progress = dict(progress)

    def status(self) -> RemoteJobStatus:
        with self.lock:
            return RemoteJobStatus(
                self.job.job_id,
                self.state,  # type: ignore[arg-type]
                tuple(self.checkpoints),
                dict(self.metrics),
                self.error,
                self.progress,
            )


class TrainingBackend(Plugin, ABC):
    """A training backend that a backend server runs.

    Values: ``info`` describes it. A backend keeps no state between jobs.

    Operations:

    * ``from_config(config)`` builds the backend that the server's configuration names (``backend``).

    * ``check(job)`` raises ``ValueError`` for a job that the backend cannot train. The default accepts every job.
    * ``check_release(paths, adapter_config)`` raises ``ValueError`` unless an imported release with these file paths
      fits the backend. The default accepts every release.
    * ``run(job, artifact, data, run)`` trains the job. ``artifact`` is a local directory of the artifact that the
      job starts from, or None when the server cannot open its URI (such as ``hf://``, which the backend reads
      itself); ``data`` is the local job data directory. The backend saves each checkpoint with ``run.save`` and its
      progress with ``run.report``, and returns the job's metrics. A raised exception fails the job.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self: ...

    @property
    @abstractmethod
    def info(self) -> BackendInfo: ...

    def check(self, job: RemoteJob) -> None:
        return

    def check_release(self, paths: tuple[str, ...], adapter_config: Mapping[str, Any] | None) -> None:
        return

    @abstractmethod
    def run(self, job: RemoteJob, artifact: Path | None, data: Path, run: JobRun) -> Mapping[str, Any]: ...


def bearer_token(request: web.Request) -> str:
    header = request.headers.get("Authorization", "")
    return header.removeprefix("Bearer ") if header.startswith("Bearer ") else ""


class BackendServer:
    """The HTTP app of one backend (``app``), which runs its jobs.

    Values: ``backend`` trains the jobs; ``files`` opens the artifact and data URIs and keeps the checkpoints;
    ``token`` is the bearer token that each request must carry.
    """

    def __init__(self, backend: TrainingBackend, files: SegmentStore, *, token: str) -> None:
        if not token:
            raise ValueError("a backend server needs a token")
        self.backend = backend
        self.files = files
        self.token = token
        self.runs: dict[str, JobRun] = {}
        self.lock = threading.Lock()

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self.authenticate])
        app.router.add_get("/healthz", self.health)
        app.router.add_get("/v1/backend", self.describe)
        app.router.add_post("/v1/jobs/check", self.check)
        app.router.add_post("/v1/jobs", self.submit)
        app.router.add_get("/v1/jobs/{job_id}", self.status)
        app.router.add_post("/v1/jobs/{job_id}/cancel", self.cancel)
        app.router.add_post("/v1/releases/check", self.check_release)
        return app

    @web.middleware
    async def authenticate(self, request: web.Request, handler: Callable) -> web.StreamResponse:
        if request.path != "/healthz" and not hmac.compare_digest(bearer_token(request), self.token):
            return web.Response(status=401, body=refusal("a bearer token is needed"), content_type="application/json")
        return await handler(request)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def describe(self, request: web.Request) -> web.Response:
        return web.json_response(self.backend.info.to_dict())

    async def job_of(self, request: web.Request) -> RemoteJob:
        try:
            return RemoteJob.from_dict(await request.json())
        except ValueError as exc:
            raise web.HTTPUnprocessableEntity(body=refusal(str(exc)), content_type="application/json") from exc

    async def check(self, request: web.Request) -> web.Response:
        job = await self.job_of(request)
        try:
            self.backend.check(job)
        except ValueError as exc:
            return web.Response(status=422, body=refusal(str(exc)), content_type="application/json")
        return web.json_response({"ok": True})

    async def submit(self, request: web.Request) -> web.Response:
        job = await self.job_of(request)
        with self.lock:
            if job.job_id in self.runs:
                return web.json_response({"ok": True}, status=202)
            try:
                self.backend.check(job)
            except ValueError as exc:
                return web.Response(status=422, body=refusal(str(exc)), content_type="application/json")
            run = self.runs[job.job_id] = JobRun(job, self.files)
        threading.Thread(target=self.run, args=(run,), name=f"reef-backend-{job.job_id[:12]}", daemon=True).start()
        return web.json_response({"ok": True}, status=202)

    async def status(self, request: web.Request) -> web.Response:
        run = self.runs.get(request.match_info["job_id"])
        if run is None:
            return web.Response(status=404, body=refusal("no such job"), content_type="application/json")
        return web.json_response(run.status().to_dict())

    async def cancel(self, request: web.Request) -> web.Response:
        run = self.runs.get(request.match_info["job_id"])
        if run is not None:
            run.cancelled.set()
        return web.json_response({"ok": True}, status=202)

    async def check_release(self, request: web.Request) -> web.Response:
        body = await request.json()
        paths = body.get("paths") if isinstance(body, Mapping) else None
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            return web.Response(
                status=422, body=refusal("paths must be a list of strings"), content_type="application/json"
            )
        try:
            self.backend.check_release(tuple(paths), body.get("adapter_config"))
        except ValueError as exc:
            return web.Response(status=422, body=refusal(str(exc)), content_type="application/json")
        return web.json_response({"ok": True})

    def run(self, run: JobRun) -> None:
        """Run ``run``'s job to its end: open its directories, train, and record the result."""
        job = run.job
        try:
            data = self.files.open_directory(job.data["uri"])
            artifact = self.open_artifact(job.artifact_uri)
            metrics = self.backend.run(job, artifact, data, run)
            state, error = ("failed", "the job was cancelled") if run.cancelled.is_set() else ("finished", None)
        except JobCancelled:
            metrics, state, error = {}, "failed", "the job was cancelled"
        except Exception as exc:
            logger.exception("training job %s failed", job.job_id)
            metrics, state, error = {}, "failed", f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        with run.lock:
            run.metrics, run.state, run.error = dict(metrics), state, error

    def open_artifact(self, uri: str) -> Path | None:
        """A local directory of the artifact at ``uri``, or None for a URI that ``files`` does not open."""
        try:
            return self.files.open_directory(uri)
        except (ValueError, FileNotFoundError):
            if uri.startswith(("file:", "s3:")):
                raise
            return None


def main(argv: list[str] | None = None) -> None:
    """Serve one backend: ``python -m reef.train.remote.server --config backend.yaml``.

    The configuration holds ``backend`` (a ``TrainingBackend`` plugin reference with its config), ``files`` (a
    ``SegmentStore`` plugin reference, or a local directory), ``token_env`` (the variable of the bearer token,
    default ``REEF_TRAINING_BACKEND_TOKEN``), ``host`` (default ``0.0.0.0``) and ``port`` (default 8000).
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config = yaml.safe_load(parser.parse_args(argv).config.read_text(encoding="utf-8")) or {}
    reference = TrainingBackend.reference(config.get("backend"), setting="backend")
    backend = TrainingBackend.referenced_class(reference).from_config(reference.config)
    files_setting = config.get("files", ".reef/backend-files")
    if isinstance(files_setting, str) and ":" not in files_setting:
        files: SegmentStore = LocalSegmentStore(Path(files_setting))
    else:
        files_reference = SegmentStore.reference(files_setting, setting="files")
        directory = Path(config.get("cache_dir", ".reef/backend-cache"))
        files = SegmentStore.referenced_class(files_reference).from_config(files_reference.config, directory=directory)
    token_env = config.get("token_env", "REEF_TRAINING_BACKEND_TOKEN")
    server = BackendServer(backend, files, token=os.environ.get(token_env, ""))
    web.run_app(server.app(), host=config.get("host", "0.0.0.0"), port=int(config.get("port", 8000)))


if __name__ == "__main__":
    main()


__all__ = ["BackendServer", "JobCancelled", "JobRun", "TrainingBackend", "main"]
