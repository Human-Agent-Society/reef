from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

import pytest
from aiohttp import web
from tests.support._trajectories import policy_trajectory

from reef.core.job_data import JobData
from reef.storage.data import MemoryDataStore
from reef.storage.job_data import read_job_data
from reef.storage.segments import LocalSegmentStore
from reef.storage.tasks import LocalTaskFileStore
from reef.train.component import TrainingJob
from reef.train.remote.client import BackendRefused, RemoteConfig, RemoteTrainingComponent
from reef.train.remote.contract import RELEASE_DIR, STATE_DIR, BackendInfo, RemoteJob
from reef.train.remote.server import BackendServer, JobRun, TrainingBackend

TOKEN = "backend-token"


class CountingBackend(TrainingBackend):
    """Saves one checkpoint per sample of a job. Its state is the number of samples trained so far, which each
    checkpoint keeps in ``state/``; ``release/`` holds the same number. A job with ``refuse`` is refused, and a job
    with ``wait`` runs until it is cancelled."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("counting", None)

    def check(self, job: RemoteJob) -> None:
        if job.parameters.get("refuse"):
            raise ValueError("the job asks to be refused")

    def run(self, job: RemoteJob, artifact: Path | None, data: Path, run: JobRun) -> Mapping[str, Any]:
        state = artifact / STATE_DIR / "trained.txt" if artifact is not None else None
        trained = int(state.read_text()) if state is not None and state.is_file() else 0
        samples = read_job_data(data).samples
        while job.parameters.get("wait") and not run.cancelled.is_set():
            run.report({"waiting": True})
            time.sleep(0.01)
        for step, _ in enumerate(samples):
            trained += 1
            directory = data.parent / f"{job.job_id}-{step}"
            (directory / RELEASE_DIR).mkdir(parents=True)
            (directory / STATE_DIR).mkdir()
            (directory / RELEASE_DIR / "model.txt").write_text(str(trained))
            (directory / STATE_DIR / "trained.txt").write_text(str(trained))
            run.save(step, directory, {"trained": trained})
        return {"trained": trained}


@pytest.fixture
def backend_url(tmp_path: Path) -> Iterator[str]:
    """The URL of a backend server with ``CountingBackend``, which keeps its checkpoints under ``tmp_path``."""
    server = BackendServer(CountingBackend(), LocalSegmentStore(tmp_path / "backend-files"), token=TOKEN)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(server.app())
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.run_until_complete(runner.cleanup())
    loop.close()


def component(url: str, tmp_path: Path, token: str = TOKEN) -> RemoteTrainingComponent:
    base = tmp_path / "base"
    (base / RELEASE_DIR).mkdir(parents=True, exist_ok=True)
    (base / RELEASE_DIR / "model.txt").write_text("0")
    config = RemoteConfig(url=url, base=base.as_uri(), state_dir=str(tmp_path / "reef"), timeout_s=5.0)
    return RemoteTrainingComponent(config, token)


def job(job_id: str, start: str, samples: int, **parameters: Any) -> TrainingJob:
    sample = policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0)
    data = MemoryDataStore(LocalTaskFileStore()).write_job_directory(JobData(f"s:data:{job_id}", (sample,) * samples))
    return TrainingJob(job_id, "counting", start, data, parameters, scenario="s")


def ended(remote: RemoteTrainingComponent, job_id: str):
    deadline = time.monotonic() + 10
    while (status := remote.status(job_id)).state == "running":
        assert time.monotonic() < deadline, "the job did not end"
        time.sleep(0.01)
    return status


def test_a_job_runs_on_the_service_and_the_next_job_continues_from_its_checkpoint(backend_url, tmp_path):
    remote = component(backend_url, tmp_path)
    assert (remote.name, remote.release_format, remote.shares_scenarios) == ("counting", None, False)
    # The root release that Reef writes from the default start names the base artifact.
    root = tmp_path / "root"
    root.mkdir()
    for path, text in remote.default_start().files.items():
        (root / path).write_text(text)
    start = remote.import_start(root)
    assert start == remote.config.base
    # A release without a reference, such as an imported one, is copied to the shared volume as a release.
    copied = Path(remote.import_start(tmp_path / "base" / RELEASE_DIR).removeprefix("file://"))
    assert (copied / RELEASE_DIR / "model.txt").read_text() == "0"

    first = job("j1", start, 2)
    remote.check_job(first)
    remote.submit(first)
    remote.submit(first)
    status = ended(remote, "j1")
    assert (status.state, status.metrics) == ("finished", {"trained": 2})
    assert [checkpoint.step for checkpoint in status.checkpoints] == [0, 1]
    last = status.checkpoints[-1]
    assert Path(last.release_path).name == RELEASE_DIR and (Path(last.release_path) / "model.txt").read_text() == "2"

    # The next job starts from the last checkpoint's artifact, so it continues the state that the artifact holds.
    remote.submit(job("j2", last.path, 1))
    status = ended(remote, "j2")
    assert status.metrics == {"trained": 3}


def test_a_refused_job_a_cancelled_job_and_an_unknown_job(backend_url, tmp_path):
    remote = component(backend_url, tmp_path)
    with pytest.raises(BackendRefused, match="asks to be refused"):
        remote.check_job(job("j1", remote.config.base, 1, refuse=True))
    with pytest.raises(BackendRefused, match="asks to be refused"):
        remote.submit(job("j1", remote.config.base, 1, refuse=True))
    with pytest.raises(KeyError):
        remote.status("j1")

    remote.submit(job("j2", remote.config.base, 1, wait=True))
    deadline = time.monotonic() + 10
    while remote.call("GET", "/v1/jobs/j2")["progress"] is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    remote.cancel("j2")
    status = ended(remote, "j2")
    assert (status.state, status.error, status.checkpoints) == ("failed", "the job was cancelled", ())


def test_a_call_without_the_token_is_refused(backend_url, tmp_path):
    with pytest.raises(BackendRefused) as refused:
        component(backend_url, tmp_path, token="wrong").call("GET", "/v1/backend")
    assert refused.value.status == 401
