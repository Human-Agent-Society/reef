"""The generator service from the processor's side: one asynchronous method per step, jobs waited for by polling."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from reef.core.tasks import HarborTask
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset.designer import DesignerRequest
from reef.record2dataset.harbor import OracleResult
from reef.record2dataset.wire import (
    WireError,
    checked_object,
    checked_string,
    oracle_from_document,
    play_from_document,
    task_document,
    task_from_document,
)

DEFAULT_POLL_S = 2.0
DEFAULT_TIMEOUT_S = 60.0
# The longest a play or check may take: several episodes of a large model on one task run for hours, not minutes.
DEFAULT_JOB_TIMEOUT_S = 4 * 3600.0


class GeneratorError(RuntimeError):
    """The generator service refused a request, failed a job, or could not be reached."""


class DuplicateTask(GeneratorError):
    """A task with the same content is already under the tasks root."""


class TaskNameConflict(GeneratorError):
    """A different task already holds the name under the tasks root."""


@dataclass(frozen=True)
class ProposedTask:
    """What one designer call produced: the record its call left, and the task or the reason there is none."""

    record_id: str
    task: HarborTask | None
    refusal: str


@dataclass(frozen=True)
class WrittenTask:
    path: Path
    name: str
    digest: str


class Generator(ABC):
    """What a task generating processor asks of the generator service."""

    @abstractmethod
    async def propose(
        self,
        request: DesignerRequest,
        *,
        scenario: str,
        generation: int,
        index: int,
        tags: Mapping[str, str],
        model: str | None = None,
    ) -> ProposedTask: ...

    @abstractmethod
    async def report_proposal(
        self, record_id: str, *, scenario: str, score: float, metadata: Mapping[str, object]
    ) -> str: ...

    @abstractmethod
    async def write_task(self, task: HarborTask) -> WrittenTask: ...

    @abstractmethod
    async def delete_task(self, name: str) -> None: ...

    @abstractmethod
    async def check(self, task_path: Path) -> OracleResult: ...

    @abstractmethod
    async def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_files: Sequence[str],
        tags: Mapping[str, str],
        model: str | None = None,
    ) -> tuple[TaskPlay, ...]: ...

    @abstractmethod
    async def write_manifest(
        self, *, generation: int, names: Sequence[str], eval_fraction: float, seed: int
    ) -> Path: ...


class HttpGenerator(Generator):
    """The generator service over HTTP; a job is polled every ``poll_s`` seconds until it is done or failed.

    ``timeout_s`` bounds one HTTP exchange; ``job_timeout_s`` bounds a whole job, the longest a play or check
    may take, after which the job is given up and reported as an error.
    """

    def __init__(
        self,
        url: str,
        *,
        poll_s: float = DEFAULT_POLL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise GeneratorError("the generator url must be a non-empty string")
        self.url = url.rstrip("/")
        self.poll_s = poll_s
        self.job_timeout_s = job_timeout_s
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def request(
        self, method: str, path: str, *, body: Mapping[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        try:
            async with (
                aiohttp.ClientSession(timeout=self.timeout) as session,
                session.request(method, f"{self.url}{path}", json=body) as response,
            ):
                status = response.status
                text = await response.text()
        except (TimeoutError, aiohttp.ClientError) as exc:
            raise GeneratorError(f"{method} {path} did not reach the generator at {self.url}: {exc}") from exc
        if status == 204:
            return status, {}
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise GeneratorError(f"{method} {path} answered {status} without JSON: {text[:200]}") from exc
        if not isinstance(document, Mapping):
            raise GeneratorError(f"{method} {path} answered with something other than a JSON object")
        return status, {str(key): value for key, value in document.items()}

    async def call(self, method: str, path: str, *, body: Mapping[str, object] | None = None) -> dict[str, object]:
        status, document = await self.request(method, path, body=body)
        if status >= 400:
            raise GeneratorError(f"{method} {path} was refused ({status}): {document.get('error', document)}")
        return document

    async def job_result(self, submitted: Mapping[str, object]) -> dict[str, object]:
        job_id = checked_string(submitted, "job", label="a submitted job")
        started = time.monotonic()
        while True:
            job = await self.call("GET", f"/jobs/{job_id}")
            state = job.get("state")
            if state == "done":
                return checked_object(job.get("result"), "a job result")
            if state == "failed":
                raise GeneratorError(f"job {job_id} failed: {job.get('error')}")
            elapsed = time.monotonic() - started
            if elapsed >= self.job_timeout_s:
                raise GeneratorError(
                    f"{job.get('kind', '')} job {job_id} is still {state} after {elapsed:.1f} s "
                    f"(job_timeout_s is {self.job_timeout_s:g})"
                )
            await asyncio.sleep(self.poll_s)

    async def propose(
        self,
        request: DesignerRequest,
        *,
        scenario: str,
        generation: int,
        index: int,
        tags: Mapping[str, str],
        model: str | None = None,
    ) -> ProposedTask:
        body: dict[str, object] = {
            "scenario": scenario,
            "generation": generation,
            "index": index,
            "tags": dict(tags),
            "request": dataclasses.asdict(request),
        }
        if model is not None:
            body["model"] = model
        result = await self.job_result(await self.call("POST", "/proposals", body=body))
        try:
            task = task_from_document(result["task"]) if result.get("task") is not None else None
        except WireError as exc:
            raise GeneratorError(f"the proposal's task cannot be read: {exc}") from exc
        return ProposedTask(
            record_id=checked_string(result, "record_id", label="a proposal"),
            task=task,
            refusal=str(result.get("refusal", "")),
        )

    async def report_proposal(
        self, record_id: str, *, scenario: str, score: float, metadata: Mapping[str, object]
    ) -> str:
        answer = await self.call(
            "POST",
            f"/proposals/{record_id}/report",
            body={"scenario": scenario, "score": score, "metadata": dict(metadata)},
        )
        return str(answer.get("agent_record_id", ""))

    async def write_task(self, task: HarborTask) -> WrittenTask:
        status, document = await self.request("POST", "/tasks", body=task_document(task))
        if status == 409 and document.get("duplicate") is True:
            raise DuplicateTask(str(document.get("error")))
        if status == 409 and document.get("duplicate") is False:
            raise TaskNameConflict(str(document.get("error")))
        if status >= 400:
            raise GeneratorError(f"POST /tasks was refused ({status}): {document.get('error', document)}")
        return WrittenTask(
            path=Path(checked_string(document, "path", label="a written task")),
            name=checked_string(document, "name", label="a written task"),
            digest=checked_string(document, "digest", label="a written task"),
        )

    async def delete_task(self, name: str) -> None:
        await self.call("DELETE", f"/tasks/{name}")

    async def check(self, task_path: Path) -> OracleResult:
        result = await self.job_result(await self.call("POST", "/checks", body={"path": str(task_path)}))
        try:
            return oracle_from_document(result)
        except WireError as exc:
            raise GeneratorError(f"the check's result cannot be read: {exc}") from exc

    async def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_files: Sequence[str],
        tags: Mapping[str, str],
        model: str | None = None,
    ) -> tuple[TaskPlay, ...]:
        body: dict[str, object] = {
            "scenario": scenario,
            "path": str(task_path),
            "arm": arm,
            "plays": plays,
            "is_reporting": is_reporting,
            "extra_instruction_files": list(extra_instruction_files),
            "tags": dict(tags),
        }
        if model is not None:
            body["model"] = model
        result = await self.job_result(await self.call("POST", "/plays", body=body))
        rows = result.get("plays")
        if not isinstance(rows, list):
            raise GeneratorError("the play's result holds no episodes")
        try:
            return tuple(play_from_document(row) for row in rows)
        except WireError as exc:
            raise GeneratorError(f"the play's episodes cannot be read: {exc}") from exc

    async def write_manifest(self, *, generation: int, names: Sequence[str], eval_fraction: float, seed: int) -> Path:
        answer = await self.call(
            "POST",
            "/manifests",
            body={"generation": generation, "names": list(names), "eval_fraction": eval_fraction, "seed": seed},
        )
        return Path(checked_string(answer, "path", label="a manifest"))
