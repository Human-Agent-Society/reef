"""The data service's routes: imports and exports at a URI, item archives, artifact files, jobs and signed URLs;
and control's cancel of a job and the contract's capabilities."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import (
    TOKEN,
    UploadRecipe,
    atif,
    dispatcher_for,
    trajectory_lines,
    upload_trajectories,
)

from reef.artifact.artifact import ArtifactRef
from reef.artifact.files import DirectoryFiles
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.repository import FileDownload
from reef.core.job_data import JobData
from reef.core.tasks import write_harbor_task
from reef.dispatcher import Dispatcher
from reef.runtime.jobs import TrainingJob
from reef.service.app import create_app
from reef.storage.data_queue import DATA_QUEUE
from reef.storage.work_queue import NewWork
from reef.train.job_queue import JOB_CANCELLED_ERROR, JOBS_QUEUE, job_kind, job_payload

AUTH = {"Authorization": f"Bearer {TOKEN}"}
DATA = "/reef/namespaces/acme/data"


def queued_job(job_id: str, scenario: str) -> NewWork:
    """A Tinker training job of ``scenario`` with no samples, as control adds it to the jobs queue."""
    job = TrainingJob(job_id, "tinker", "tinker:base", JobData(f"{scenario}:data:1"), {"lr": 1e-5}, scenario=scenario)
    return NewWork(JOBS_QUEUE, job_id, scenario, job_kind("tinker"), job_payload(job), serial_key=scenario)


def served(dispatcher: Dispatcher, check) -> None:
    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await check(client)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


async def finished(client: TestClient, location: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = await (await client.get(location, headers=AUTH)).json()
        if status["state"] in ("finished", "failed"):
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"{location} did not end")


def test_an_import_from_a_uri_stores_tasks_and_trajectories_and_an_export_writes_them_back(tmp_path: Path) -> None:
    tasks = tmp_path / "source-tasks"
    for name in ("task-a", "task-b"):
        write_harbor_task(task_for(name, "record-1", 9000), tasks)
    trajectories = tmp_path / "source-trajectories"
    trajectories.mkdir()
    # One ATIF document a line; a file of another type is not read.
    (trajectories / "batch.jsonl").write_bytes(trajectory_lines("t1", "t2"))
    (trajectories / "t3.json").write_text(json.dumps(atif("t3")))
    exported = tmp_path / "out" / "export.tar.gz"
    exported.parent.mkdir()
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def check(client: TestClient) -> None:
        started = []
        for uri, kind in ((tasks.as_uri(), "task"), (trajectories.as_uri(), "trajectory")):
            response = await client.post(
                f"{DATA}/imports", headers=AUTH, json={"uri": uri, "metadata": {"kind": kind}}
            )
            assert response.status == 202, await response.text()
            started.append((await response.json())["id"])
        for transfer_id in started:
            status = await finished(client, f"{DATA}/imports/{transfer_id}")
            assert status["state"] == "finished", status
            assert status["result"] == {"items": 2}
        listing = await (await client.get(DATA, headers=AUTH)).json()
        assert sorted(item["id"] for item in listing["items"]) == ["t1", "t2", "task-a", "task-b"]
        with tarfile.open(
            fileobj=io.BytesIO(await (await client.get(f"{DATA}/task-a/archive", headers=AUTH)).read())
        ) as tar:
            assert "task-a/task.toml" in tar.getnames()

        response = await client.post(f"{DATA}/exports", headers=AUTH, json={"uri": exported.as_uri()})
        assert response.status == 202, await response.text()
        status = await finished(client, f"{DATA}/exports/{(await response.json())['id']}")
        assert status["state"] == "finished" and status["result"]["items"] == 4, status
        with tarfile.open(exported) as tar:
            manifest = [json.loads(line) for line in tar.extractfile("manifest.jsonl").read().splitlines()]
            by_id = {entry["id"]: entry for entry in manifest}
            assert json.loads(tar.extractfile(by_id["t1"]["path"]).read()) == atif("t1")
            assert f"{by_id['task-a']['path']}/task.toml" in tar.getnames()
        # An export never replaces a file, and an unknown transfer is not found.
        again = await client.post(f"{DATA}/exports", headers=AUTH, json={"uri": exported.as_uri()})
        failed = await finished(client, f"{DATA}/exports/{(await again.json())['id']}")
        assert failed["state"] == "failed" and "FileExistsError" in failed["error"]
        assert (await client.get(f"{DATA}/imports/data_import:nothing", headers=AUTH)).status == 404
        # An import that cannot succeed fails at its first attempt: a retry reads the same input.
        missing = await client.post(
            f"{DATA}/imports",
            headers=AUTH,
            json={"uri": (tmp_path / "missing").as_uri(), "metadata": {"kind": "trajectory"}},
        )
        missing_id = (await missing.json())["id"]
        assert (await finished(client, f"{DATA}/imports/{missing_id}"))["state"] == "failed"
        assert dispatcher.queue.get(DATA_QUEUE, missing_id).attempt_count == 1
        refused = await client.post(f"{DATA}/imports", headers=AUTH, json={"uri": "x", "metadata": {"kind": "task"}})
        assert refused.status == 400

    served(dispatcher, check)


def test_an_item_archive_downloads_with_the_token_or_with_a_signed_url_alone(tmp_path: Path) -> None:
    async def check(client: TestClient) -> None:
        assert (await upload_trajectories(client, AUTH, "t1", scenario=None, base=DATA)).status == 201
        archive = f"{DATA}/t1/archive"
        assert json.loads(await (await client.get(archive, headers=AUTH)).read()) == atif("t1")
        assert (await client.get(archive)).status == 401

        signed = await (await client.get(f"{archive}?signed=1&expires_in=60", headers=AUTH)).json()
        assert signed["expires_at"] > time.time()
        response = await client.get(signed["url"])
        assert response.status == 200 and json.loads(await response.read()) == atif("t1")
        # The signature opens that download alone, and only until it expires.
        assert (await client.get(signed["url"].replace("/t1/", "/t2/"))).status == 401
        assert (await client.get(signed["url"].replace("expires=", "expires=1"))).status == 401
        assert (await client.get(f"{archive}?signed=1&expires_in=0", headers=AUTH)).status == 400

    served(dispatcher_for(tmp_path, UploadRecipe()), check)


def test_an_artifacts_files_list_and_download_and_the_jobs_of_a_scenario_read_from_the_queue(tmp_path: Path) -> None:
    # The base artifact that each scenario starts from holds one file.
    (tmp_path / "initial").mkdir()
    (tmp_path / "initial" / "model.txt").write_text("w0")
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    dispatcher.queue.add(queued_job("job-1", "s"))

    async def check(client: TestClient) -> None:
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})).status == 201
        artifacts = await (await client.get("/reef/scenarios/s/artifacts", headers=AUTH)).json()
        artifact_id = artifacts["artifacts"][0]["artifact_id"]
        files_route = f"/reef/scenarios/s/artifacts/{artifact_id}/files"
        listing = await (await client.get(files_route, headers=AUTH)).json()
        paths = [entry["path"] for entry in listing["files"]]
        assert paths and all(entry["bytes"] >= 0 for entry in listing["files"])
        # The in-memory repository signs no URLs, so data serves the files itself.
        assert all(entry["url"] is None and entry["url_expires_at"] is None for entry in listing["files"])
        signed = await (await client.get(f"{files_route}?path={paths[0]}&signed=1", headers=AUTH)).json()
        downloaded = await client.get(signed["url"])
        assert downloaded.status == 200
        assert (await client.get(f"{files_route}?path=missing", headers=AUTH)).status == 404
        # The signature covers what the download reads, the component too.
        named = await (
            await client.get(f"{files_route}?path={paths[0]}&component=weights&signed=1", headers=AUTH)
        ).json()
        assert "component=weights" in named["url"]
        assert (await client.get(named["url"].replace("component=weights", "component=other"))).status == 401

        jobs = await (await client.get("/reef/scenarios/s/jobs", headers=AUTH)).json()
        assert [(job["id"], job["state"]) for job in jobs["jobs"]] == [("job-1", "queued")]
        (job,) = jobs["jobs"]
        assert (job["parameters"], job["sample_count"], job["sample_kind"], job["instruction"]) == (
            {"lr": 1e-5},
            0,
            None,
            None,
        )
        assert job["created_at"].endswith("+00:00") and (job["started_at"], job["ended_at"]) == (None, None)
        # The deployment has no experiment tracker, so no run records the job.
        assert job["experiment"] is None
        assert (await client.get("/reef/scenarios/nothing/jobs", headers=AUTH)).status == 404

    served(dispatcher, check)


class SigningBackend(InMemoryRepositoryBackend):
    """An in-memory repository that signs a URL for each file of a release, as an object store does."""

    def file_downloads(self, ref: ArtifactRef, expires_in_seconds: int) -> tuple[FileDownload, ...]:
        files = DirectoryFiles(self.materialize(ref).local_path)
        expires_at = int(time.time()) + expires_in_seconds
        return tuple(
            FileDownload(path, files.file(path).stat().st_size, f"https://store.example/{path}", expires_at)
            for path in files.paths()
        )


def test_an_artifacts_file_list_gives_the_repositorys_urls_without_materializing_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "initial").mkdir()
    (tmp_path / "initial" / "model.txt").write_text("w0")
    factory = SigningBackend.factory(tmp_path / "initial", root=tmp_path / "repository")
    dispatcher = dispatcher_for(tmp_path, UploadRecipe(), factory)

    def materialized(*args: object) -> None:
        raise AssertionError("the file list materialized the artifact")

    async def check(client: TestClient) -> None:
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})).status == 201
        artifacts = await (await client.get("/reef/scenarios/s/artifacts", headers=AUTH)).json()
        files_route = f"/reef/scenarios/s/artifacts/{artifacts['artifacts'][0]['artifact_id']}/files"
        monkeypatch.setattr("reef.service.scenario_files.artifact_directory", materialized)
        before = int(time.time())
        listing = await (await client.get(f"{files_route}?expires_in=600", headers=AUTH)).json()
        assert [(entry["path"], entry["bytes"], entry["url"]) for entry in listing["files"]] == [
            ("model.txt", 2, "https://store.example/model.txt")
        ]
        assert before + 600 <= listing["files"][0]["url_expires_at"] <= int(time.time()) + 600
        assert (await client.get(f"{files_route}?expires_in=0", headers=AUTH)).status == 400

    served(dispatcher, check)


def test_a_client_cancels_a_training_job_of_its_scenario_and_an_ended_job_stays_as_it_is(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    dispatcher.queue.add(queued_job("job-1", "s"))
    dispatcher.queue.add(queued_job("job-t", "t"))

    async def check(client: TestClient) -> None:
        for name in ("s", "t"):
            assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": name})).status == 201
        cancel = "/reef/scenarios/s/jobs/job-1/cancel"
        answer = await (await client.post(cancel, headers=AUTH)).json()
        assert answer["scenario"] == "s"
        assert (answer["job"]["state"], answer["job"]["error"]) == ("failed", JOB_CANCELLED_ERROR)
        # A job that never ran has no start; the cancel ended it.
        assert answer["job"]["started_at"] is None and answer["job"]["ended_at"] is not None
        # The cancel is idempotent: the job ended, so a second cancel changes nothing.
        again = await (await client.post(cancel, headers=AUTH)).json()
        assert again["job"] == answer["job"]
        jobs = await (await client.get("/reef/scenarios/s/jobs", headers=AUTH)).json()
        assert [job["state"] for job in jobs["jobs"]] == ["failed"]
        # A job of another scenario, an unknown job and an unknown scenario are not found.
        assert (await client.post("/reef/scenarios/s/jobs/job-t/cancel", headers=AUTH)).status == 404
        assert (await client.post("/reef/scenarios/s/jobs/missing/cancel", headers=AUTH)).status == 404
        assert (await client.post("/reef/scenarios/nothing/jobs/job-1/cancel", headers=AUTH)).status == 404
        assert dispatcher.queue.get(JOBS_QUEUE, "job-t").state == "queued"

    served(dispatcher, check)


def test_the_contract_reports_what_the_scenario_can_run(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())

    async def check(client: TestClient) -> None:
        assert (await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})).status == 201
        contract = await (await client.get("/reef/scenarios/s/contract", headers=AUTH)).json()
        # The trainer steps on a candidate backend, so it can start from an artifact; the scenario has no runtime to
        # evaluate on and none that loads its artifacts, so a served artifact needs only the durable history.
        assert contract["capabilities"] == {"serve_beside": True, "evaluate": [], "train_from": ["records"]}

    served(dispatcher, check)
