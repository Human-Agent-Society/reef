"""A training backend server in a test, and the client that calls it (RFC #1146)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aiohttp import web

from reef.storage.segments import LocalSegmentStore
from reef.train.component import JobStatus
from reef.train.remote.client import RemoteConfig, RemoteTrainingComponent
from reef.train.remote.contract import RELEASE_DIR
from reef.train.remote.server import BackendServer, TrainingBackend

TOKEN = "backend-token"


@contextmanager
def serving(backend: TrainingBackend, tmp_path: Path) -> Iterator[str]:
    """The URL of a backend server with ``backend``, which keeps its checkpoints under ``tmp_path``."""
    server = BackendServer(backend, LocalSegmentStore(tmp_path / "backend-files"), token=TOKEN)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(server.app())
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.run_until_complete(runner.cleanup())
        loop.close()


def remote_component(url: str, tmp_path: Path, token: str = TOKEN) -> RemoteTrainingComponent:
    base = tmp_path / "base"
    (base / RELEASE_DIR).mkdir(parents=True, exist_ok=True)
    (base / RELEASE_DIR / "model.txt").write_text("0")
    config = RemoteConfig(url=url, base=base.as_uri(), state_dir=str(tmp_path / "reef"), timeout_s=5.0)
    return RemoteTrainingComponent(config, token)


def ended(remote: RemoteTrainingComponent, job_id: str) -> JobStatus:
    deadline = time.monotonic() + 10
    while (status := remote.status(job_id)).state == "running":
        assert time.monotonic() < deadline, "the job did not end"
        time.sleep(0.01)
    return status
