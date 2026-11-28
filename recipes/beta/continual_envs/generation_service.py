"""The generation service that builds tasks from traffic sessions: Reef Platform's generation API.

A generation run reads the record folders of its sources, mines environments from them, builds Harbor tasks on those
environments, and publishes each task that passed its checks and its review as a task item named after the task.
The Platform keeps the versions of each spec and relays each call to the generation Cell. Each route takes a
workspace access token with the scope ``generators:write``:

    POST /api/v1/generation/runs {spec, generator, sources, params, targets?}   start a run: the queued run
    GET  /api/v1/generation/runs?generator=&limit=                               the runs of a generator, newest first
    GET  /api/v1/generation/runs/{id}                                            one run
    GET  /api/v1/generation/runs/{id}/items?stage=&state=&limit=&after=          the items of the run's generator
    POST /api/v1/generation/runs/{id}/cancel                                     cancel a run
    GET  /api/v1/generation/tasks/{name}/archive                                 302 to the signed tar.gz of a task

A read that fails for a reason that can pass (a 5xx answer, a lost connection, a timeout) is tried again, at most
``ATTEMPTS`` times with a growing, jittered wait. A start is not tried again: the caller finds the run that a lost
answer may have queued (``runs``).
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiohttp

from reef.core.values import JsonValue

#: The states of a run, and those in which it ended.
RUN_STATES = frozenset({"queued", "running", "finished", "failed", "cancelled"})
ENDED_STATES = frozenset({"finished", "failed", "cancelled"})
#: The tries of one read, the wait before the second try, and the bound of one call.
ATTEMPTS = 4
RETRY_SECONDS = 2.0
CALL_TIMEOUT_SECONDS = 60.0
#: The most items of one page of a run's items.
PAGE_ITEMS = 500


class GenerationServiceError(RuntimeError):
    """A call to the generation service failed, or its answer does not have the form of the API."""


@dataclass(frozen=True)
class GenerationRun:
    """One run of the generation service.

    Values: ``run_id`` is the run's non-empty ID, ``state`` one of ``RUN_STATES``, ``error`` the run's error text
    ("" for none), and ``created_at`` the time the run was queued, in seconds since the epoch.
    Purpose: the service makes it from the API's answer; the generator waits until ``ended`` and then reads the tasks
    of a ``finished`` run.
    """

    run_id: str
    state: str
    error: str
    created_at: float

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("a generation run has a non-empty ID")
        if self.state not in RUN_STATES:
            raise ValueError(f"a generation run's state is one of {', '.join(sorted(RUN_STATES))}")

    @property
    def ended(self) -> bool:
        return self.state in ENDED_STATES


class GenerationService(ABC):
    """The generation runs of one workspace: start one, read them, cancel one, and download the tasks of one."""

    @abstractmethod
    async def start(
        self,
        *,
        spec: str,
        generator: str,
        sources: Sequence[Mapping[str, JsonValue]],
        params: Mapping[str, JsonValue],
        targets: Mapping[str, JsonValue],
    ) -> GenerationRun:
        """Queue a run of ``spec`` (``<name>`` or ``<name>@<version>``) for ``generator``, and return it."""

    @abstractmethod
    async def run(self, run_id: str) -> GenerationRun:
        """The run ``run_id``."""

    @abstractmethod
    async def runs(self, generator: str) -> tuple[GenerationRun, ...]:
        """The newest runs of ``generator``, newest first."""

    @abstractmethod
    async def cancel(self, run_id: str) -> None:
        """Cancel the run ``run_id``. A run that ended does not change."""

    @abstractmethod
    async def published_tasks(self, run_id: str, stage: str) -> tuple[str, ...]:
        """The names of the tasks that the run ``run_id`` published from its build stage ``stage``, by name."""

    @abstractmethod
    async def download_task(self, name: str, directory: Path) -> Path:
        """Write the files of the published task ``name`` to ``directory / name`` and return that path."""


class PlatformGenerationService(GenerationService):
    """The generation service of Reef Platform at ``base_url``, with the workspace access token ``token``."""

    def __init__(self, base_url: str, token: str) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("the generation service's URL must be an http or https URL")
        if not token:
            raise ValueError("the generation service needs a workspace access token")
        self.base_url = base_url.rstrip("/") + "/api/v1/generation"
        self.headers = {"Authorization": f"Bearer {token}"}

    async def call(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, JsonValue] | None = None,
        retries: bool = True,
    ) -> dict[str, JsonValue]:
        """One call of the API and its JSON object. ``retries`` tries a call again that failed for a reason that
        can pass."""
        timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT_SECONDS)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                async with (
                    aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session,
                    session.request(method, self.base_url + path, params=query, json=body) as response,
                ):
                    text = await response.text()
                    if response.status < 500:
                        return answer(method, path, response.status, text)
                    failure = f"{method} {path} answered {response.status}: {text[:300]}"
            except (TimeoutError, aiohttp.ClientConnectionError) as error:
                failure = f"{method} {path} did not reach the generation service: {error!r}"
            if not retries or attempt == ATTEMPTS:
                raise GenerationServiceError(failure)
            await asyncio.sleep(RETRY_SECONDS * 2 ** (attempt - 1) * random.uniform(0.5, 1.5))
        raise GenerationServiceError(f"{method} {path} made no call")

    async def start(
        self,
        *,
        spec: str,
        generator: str,
        sources: Sequence[Mapping[str, JsonValue]],
        params: Mapping[str, JsonValue],
        targets: Mapping[str, JsonValue],
    ) -> GenerationRun:
        body: dict[str, JsonValue] = {
            "spec": spec,
            "generator": generator,
            "sources": [dict(source) for source in sources],
            "params": dict(params),
        }
        if targets:
            body["targets"] = dict(targets)
        return generation_run(await self.call("POST", "/runs", body=body, retries=False))

    async def run(self, run_id: str) -> GenerationRun:
        return generation_run(await self.call("GET", f"/runs/{quote(run_id, safe='')}"))

    async def runs(self, generator: str) -> tuple[GenerationRun, ...]:
        listed = (await self.call("GET", "/runs", query={"generator": generator, "limit": "20"})).get("runs")
        if not isinstance(listed, list):
            raise GenerationServiceError("the list of runs has no runs")
        return tuple(generation_run(run) for run in listed)

    async def cancel(self, run_id: str) -> None:
        await self.call("POST", f"/runs/{quote(run_id, safe='')}/cancel")

    async def published_tasks(self, run_id: str, stage: str) -> tuple[str, ...]:
        names: list[str] = []
        query = {"stage": stage, "state": "verified", "limit": str(PAGE_ITEMS)}
        while True:
            page = await self.call("GET", f"/runs/{quote(run_id, safe='')}/items", query=query)
            items, following = page.get("items"), page.get("next")
            if not isinstance(items, list):
                raise GenerationServiceError("a page of items has no items")
            for item in items:
                if not isinstance(item, dict):
                    raise GenerationServiceError("an item is a JSON object")
                fields = item.get("fields")
                task = fields.get("task") if isinstance(fields, dict) else None
                # The items of every run of the generator are listed; a published task's item names its run.
                if item.get("run_id") == run_id and isinstance(task, str):
                    names.append(task)
            if not isinstance(following, str):
                return tuple(sorted(names))
            query = {**query, "after": following}

    async def download_task(self, name: str, directory: Path) -> Path:
        timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT_SECONDS)
        url = f"{self.base_url}/tasks/{quote(name, safe='')}/archive"
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # The signed address needs no token: the token goes to the Platform only.
                async with session.get(url, headers=self.headers, allow_redirects=False) as response:
                    if response.status != 302:
                        raise GenerationServiceError(f"the task {name} has no download: {response.status}")
                    signed = response.headers["Location"]
                async with session.get(signed) as response:
                    if response.status != 200:
                        raise GenerationServiceError(f"the download of the task {name} answered {response.status}")
                    archive = await response.read()
        except (TimeoutError, aiohttp.ClientConnectionError) as error:
            raise GenerationServiceError(f"the task {name} did not download: {error!r}") from error
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            if {member.name.split("/", 1)[0] for member in tar.getmembers()} != {name}:
                raise GenerationServiceError(f"the archive of the task {name} holds another directory")
            tar.extractall(directory, filter="data")
        return directory / name


def answer(method: str, path: str, status: int, text: str) -> dict[str, JsonValue]:
    """The JSON object of an answer below 500. A refusal (4xx) raises ``GenerationServiceError`` with its reason."""
    if status >= 400:
        raise GenerationServiceError(f"{method} {path} was refused ({status}): {text[:500]}")
    try:
        document = json.loads(text) if text else {}
    except ValueError as error:
        raise GenerationServiceError(f"{method} {path} answered no JSON") from error
    if not isinstance(document, dict):
        raise GenerationServiceError(f"{method} {path} answered no JSON object")
    return document


def generation_run(document: JsonValue) -> GenerationRun:
    """The run of an answer of the API."""
    if not isinstance(document, dict):
        raise GenerationServiceError("a run is a JSON object")
    created_at = document.get("created_at")
    try:
        return GenerationRun(
            run_id=str(document.get("id") or ""),
            state=str(document.get("state") or ""),
            error=str(document.get("error") or ""),
            created_at=float(created_at) if isinstance(created_at, (int, float)) else 0.0,
        )
    except ValueError as error:
        raise GenerationServiceError(f"the answer is not a run: {error}") from error
