"""The training component that calls a training backend service (RFC #1146).

``RemoteTrainingComponent`` maps each method of ``TrainingComponent`` to a route of the contract
(``reef.train.remote.contract``). A job's start is the URI of an artifact directory that the backend reads:

- ``hf://<repo>@<commit>``: a Hugging Face revision, such as the base model or a checkpoint that the backend
  committed. Its release in Reef is a reference to the revision (``REFERENCE_FILE``), so Reef copies nothing.
- ``file://<path>``: a directory on a volume that Reef and the backend share. Its release is ``<path>/release``, or
  ``<path>`` itself when it holds no ``release/``.

The component keeps the release of each checkpoint, and each start that ``import_start`` copies, under
``state_dir``, which control and the training service share.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.artifact.sources import REFERENCE_FILE, HuggingFaceRevision, read_reference
from reef.core.config import config_option
from reef.core.weight_layout import ADAPTER_DIR, ReleaseFormat
from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.train.remote.contract import RELEASE_DIR, BackendInfo, RemoteJob, RemoteJobStatus


class BackendRefused(ValueError):
    """The backend service refused a request: ``status`` is its HTTP 4xx status, and the message its reason."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RemoteConfig:
    """``training.options`` of a remote training backend.

    Values: ``url`` is the base URL of the backend service. ``token_env`` names the variable that holds its bearer
    token. ``base`` is the URI of the artifact that a new scenario starts from (``hf://`` or ``file://``).
    ``state_dir`` is a directory that control and the training service share. ``timeout_s`` bounds each call.
    """

    url: str = config_option(help="Base URL of the training backend service.")
    base: str = config_option(help="URI of the artifact that a new scenario starts from: hf:// or file://.")
    state_dir: str = config_option(help="Directory that control and the training service share.")
    token_env: str = config_option("REEF_TRAINING_BACKEND_TOKEN", help="Variable that holds the bearer token.")
    timeout_s: float = config_option(30.0, help="Seconds that one call to the backend service can take.")

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"url must be an http(s) URL, got {self.url!r}")
        if not self.base.startswith(("hf://", "file://")):
            raise ValueError(f"base must be an hf:// or file:// URI, got {self.base!r}")
        if not self.state_dir.strip():
            raise ValueError("state_dir must be a directory")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


class RemoteTrainingComponent(TrainingComponent):
    """A ``TrainingComponent`` whose jobs run on a training backend service."""

    def __init__(self, config: RemoteConfig, token: str) -> None:
        if not token:
            raise ValueError(f"the variable {config.token_env} holds no token for the training backend")
        self.config = config
        self.token = token
        self.root = Path(config.state_dir).expanduser()
        self._info: BackendInfo | None = None

    # -- HTTP

    def call(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        """The JSON answer of ``method path``; ``BackendRefused`` for a 4xx answer, and ``OSError`` for a failed
        call or a 5xx answer."""
        request = urllib.request.Request(
            self.config.url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise BackendRefused(exc.code, refusal_message(exc)) from None
            raise OSError(f"the training backend answered {exc.code}: {refusal_message(exc)}") from None

    @property
    def info(self) -> BackendInfo:
        if self._info is None:
            self._info = BackendInfo.from_dict(self.call("GET", "/v1/backend"))
        return self._info

    # -- TrainingComponent

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def release_format(self) -> ReleaseFormat | None:
        return self.info.release_format

    @property
    def shares_scenarios(self) -> bool:
        return self.info.shares_scenarios

    def default_start(self) -> DefaultStart:
        """A reference to ``base``, so a root release made from it starts its first job there."""
        if self.config.base.startswith("hf://"):
            reference = json.dumps(HuggingFaceRevision.parse(self.config.base).to_dict()) + "\n"
            return DefaultStart(files={REFERENCE_FILE: reference}, inputs={"model": self.config.base})
        return DefaultStart(files={START_FILE: self.config.base + "\n"}, inputs={"model": self.config.base})

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        """The backend checks each job (``check_job``); a recipe can name task jobs."""
        return

    def check_release(self, files: ArtifactFiles) -> None:
        paths = tuple(files.paths())
        adapter_config: Any = None
        if f"{ADAPTER_DIR}/adapter_config.json" in paths:
            adapter_config = json.loads(files.read_bytes(f"{ADAPTER_DIR}/adapter_config.json"))
        self.call("POST", "/v1/releases/check", {"paths": list(paths), "adapter_config": adapter_config})

    def import_start(self, release: Path) -> str:
        """The URI that ``release`` refers to: its Hugging Face revision, the start that it names, or a copy of it
        under ``state_dir``, which the backend reads through the shared volume."""
        revision = read_reference(release)
        if revision is not None:
            return revision.uri
        if (release / START_FILE).is_file():
            return (release / START_FILE).read_text(encoding="utf-8").strip()
        target = self.root / "starts" / tree_digest(release)
        if not target.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
            shutil.copytree(release, staging / RELEASE_DIR)
            staging.replace(target)
        return target.resolve().as_uri()

    def remote_job(self, job: TrainingJob) -> dict[str, Any]:
        data = {**job.data.to_dict(), "uri": job.data_uri}
        return RemoteJob(job.job_id, job.scenario, job.start, data, dict(job.parameters)).to_dict()

    def check_job(self, job: TrainingJob) -> None:
        self.call("POST", "/v1/jobs/check", self.remote_job(job))

    def submit(self, job: TrainingJob) -> None:
        self.call("POST", "/v1/jobs", self.remote_job(job))

    def status(self, job_id: str) -> JobStatus:
        try:
            status = RemoteJobStatus.from_dict(self.call("GET", f"/v1/jobs/{quote(job_id, safe='')}"))
        except BackendRefused as exc:
            if exc.status == 404:
                raise KeyError(job_id) from None
            raise
        checkpoints = tuple(
            SavedCheckpoint(
                checkpoint.step,
                checkpoint.artifact_uri,
                str(self.release(job_id, checkpoint.step, checkpoint.artifact_uri)),
                checkpoint.metrics,
            )
            for checkpoint in status.checkpoints
        )
        return JobStatus(job_id, status.state, checkpoints, status.metrics, status.error)

    def cancel(self, job_id: str) -> None:
        self.call("POST", f"/v1/jobs/{quote(job_id, safe='')}/cancel")

    def release(self, job_id: str, step: int, uri: str) -> Path:
        """The release of the checkpoint at ``uri``: a reference to its revision, or its ``release/`` on the
        shared volume."""
        if uri.startswith("hf://"):
            directory = self.root / "releases" / tree_name(job_id) / str(step)
            if not (directory / REFERENCE_FILE).is_file():
                directory.mkdir(parents=True, exist_ok=True)
                revision = HuggingFaceRevision.parse(uri)
                (directory / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()) + "\n", encoding="utf-8")
            return directory
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"a checkpoint of a remote backend is an hf:// or file:// URI, got {uri!r}")
        path = Path(unquote(parsed.path))
        return path / RELEASE_DIR if (path / RELEASE_DIR).is_dir() else path


#: The file of a root release that names its start by URI, when the start is no Hugging Face revision.
START_FILE = "reef-start.txt"


def refusal_message(error: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(error.read())["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return error.reason if isinstance(error.reason, str) else f"HTTP {error.code}"


def tree_name(job_id: str) -> str:
    """``job_id`` as one directory name."""
    return hashlib.sha256(job_id.encode()).hexdigest()[:32]


def tree_digest(directory: Path) -> str:
    """The SHA-256 of the file paths and contents under ``directory``."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def remote_component(config: Mapping[str, Any], environ: Mapping[str, str]) -> RemoteTrainingComponent:
    """The component that ``training.options`` describe, with the token from ``environ``."""
    settings = RemoteConfig(**{key: value for key, value in config.items() if key != "type"})
    return RemoteTrainingComponent(settings, environ.get(settings.token_env) or os.environ.get(settings.token_env, ""))


__all__ = ["START_FILE", "BackendRefused", "RemoteConfig", "RemoteTrainingComponent", "remote_component"]
