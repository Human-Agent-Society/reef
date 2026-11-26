"""``reef.execution.sandbox.agentenv.images``: the prepared images of a task, their lookup by task path and directory,
and the native start of a sandbox from one image on a local stand-in for the AgentENV API."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import aiohttp
import pytest
from tests.support.agentenv import ColdAnswer, FakeAgentEnvApi, serving

from reef.execution.sandbox.agentenv.images import (
    TASK_IMAGES_VARIABLE,
    ColdSandbox,
    ColdSandboxError,
    PreparedImage,
    TaskImages,
    is_missing_snapshot,
    prepared_images_from_json,
    select_prepared_image,
)
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

#: A synthetic image of a task's environment, by digest.
IMAGE = "registry.example:5000/tasks/add@sha256:" + "a" * 64
#: The JSON object of a prepared image with each optional field set.
ENTRY: dict[str, str | None] = {
    "directory": "environment",
    "source_image": None,
    "image": IMAGE,
    "template_id": "template-1",
    "start": "/usr/local/bin/serve",
    "ready": "test -f /run/ready",
}


@pytest.mark.unit
def test_a_list_of_prepared_images_reads_back_as_it_was_written() -> None:
    verifier = {**ENTRY, "directory": "steps/build/tests", "template_id": None, "start": None, "ready": None}
    images = prepared_images_from_json([ENTRY, verifier])
    assert [image.to_json() for image in images] == [ENTRY, verifier]
    assert images[1] == PreparedImage("steps/build/tests", None, IMAGE, None, None, None)
    # Two configurations of one directory differ by their source image.
    other = {**ENTRY, "source_image": "registry.example/tasks/add:1"}
    assert len(prepared_images_from_json([ENTRY, other])) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        {**ENTRY, "image": "registry.example/tasks/add:latest"},
        {**ENTRY, "image": "registry.example/tasks/add:1@sha256:" + "a" * 64},
        {**ENTRY, "image": "registry.example/tasks/add@sha256:" + "A" * 64},
        {**ENTRY, "image": "add@sha256:" + "a" * 64},
        {**ENTRY, "directory": "../outside"},
        {**ENTRY, "directory": "/environment"},
        {**ENTRY, "directory": "steps//tests"},
        {**ENTRY, "directory": "./environment"},
        {**ENTRY, "directory": ""},
        {**ENTRY, "start": ""},
        {**ENTRY, "template_id": 7},
        {key: value for key, value in ENTRY.items() if key != "ready"},
        {**ENTRY, "user": "root"},
    ],
    ids=[
        "tag",
        "tag-and-digest",
        "uppercase-digest",
        "no-registry",
        "parent",
        "absolute",
        "empty-part",
        "dot-part",
        "empty-directory",
        "empty-command",
        "template-id-type",
        "missing-field",
        "unknown-field",
    ],
)
def test_a_forged_prepared_image_is_refused(entry: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="prepared image"):
        prepared_images_from_json([entry])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("image", "valid"),
    [
        ("registry.example/tasks/add@sha256:" + "a" * 64, True),
        ("registry:5000/add@sha256:" + "a" * 64, True),
        ("localhost/tasks/add@sha256:" + "a" * 64, True),
        ("localhost:5000/add@sha256:" + "a" * 64, True),
        # Without a dot, a port or localhost, AgentENV reads the first part as its own shared store: aenv/ is its
        # local prefix, and team/ is not a registry either.
        ("aenv/example@sha256:" + "a" * 64, False),
        ("team/example@sha256:" + "a" * 64, False),
        ("sha256:" + "a" * 64, False),
    ],
)
def test_a_prepared_image_names_an_external_registry_host(image: str, valid: bool) -> None:
    entry = {**ENTRY, "image": image}
    if valid:
        assert prepared_images_from_json([entry])[0].image == image
        return
    with pytest.raises(ValueError) as refused:
        prepared_images_from_json([entry])
    assert str(refused.value) == (
        "a prepared image must be host/repository@sha256:<digest>, whose registry host has a dot or a port or is "
        f"localhost, not {image!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "message", "missing"),
    [
        (400, "400: template checked-0 not found", True),
        (400, "400: worker error: template checked-0 not found", True),
        (500, "500: worker error: snapshot 'checked-0' is not ready", True),
        # Another template, a longer ID, or the same text with another status is not this template's answer.
        (400, "400: template checked-1 not found", False),
        (400, "400: template checked-0-copy not found", False),
        (400, "400: subtemplate checked-0 not found", False),
        (500, "500: template checked-0 not found", False),
        (400, "400: snapshot 'checked-0' is not ready", False),
        (404, "404: template checked-0 not found", False),
        (503, "503: no nodes available", False),
        (400, "400: cpuCount must be at most 16", False),
    ],
)
def test_only_agentenvs_answers_of_a_missing_snapshot_of_the_template_count(
    status: int, message: str, missing: bool
) -> None:
    assert is_missing_snapshot(status, message, "checked-0") is missing


@pytest.mark.unit
def test_a_list_with_one_environment_twice_or_no_list_is_refused() -> None:
    with pytest.raises(ValueError, match="same directory and source image"):
        prepared_images_from_json([ENTRY, {**ENTRY, "image": IMAGE.replace("add", "sort")}])
    with pytest.raises(ValueError, match="JSON list"):
        prepared_images_from_json({"environment": ENTRY})


@pytest.mark.unit
def test_an_environment_selects_the_image_of_its_own_directory_and_source_image(tmp_path: Path) -> None:
    task = tmp_path / "task"
    for directory in ("environment", "tests", "steps/build/tests"):
        (task / directory).mkdir(parents=True)
    images = prepared_images_from_json(
        [
            ENTRY,
            {**ENTRY, "source_image": "registry.example/tasks/add:1", "image": IMAGE.replace("add", "pinned")},
            {**ENTRY, "directory": "tests", "image": IMAGE.replace("add", "tests")},
            {**ENTRY, "directory": "steps/build/tests", "image": IMAGE.replace("add", "step")},
        ]
    )
    assert select_prepared_image(images, task, task / "environment", None) == images[0]
    assert select_prepared_image(images, task, task / "environment", "registry.example/tasks/add:1") == images[1]
    assert select_prepared_image(images, task, task / "environment", images[1].image) == images[1]
    assert select_prepared_image(images, task, task / "tests", images[1].image) is None
    assert select_prepared_image(images, task, task / "tests", None) == images[2]
    assert select_prepared_image(images, task, task / "steps" / "build" / "tests", None) == images[3]
    # Another source image, a directory without an image, and a directory outside the task have none.
    assert select_prepared_image(images, task, task / "environment", "registry.example/tasks/add:2") is None
    assert select_prepared_image(images, task, task / "steps" / "build", None) is None
    assert select_prepared_image(images, task, tmp_path / "environment", None) is None
    # A task reached through a link finds the same directories.
    (tmp_path / "linked").symlink_to(task)
    assert select_prepared_image(images, tmp_path / "linked", task / "tests", None) == images[2]


@pytest.mark.unit
def test_a_training_process_reads_the_images_of_its_tasks_by_their_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images_file = tmp_path / "private" / "task-images.json"
    images_file.parent.mkdir()
    job = tmp_path / "job" / "tasks"
    images_file.write_text(json.dumps({f"{job}/./add": [ENTRY], f"{job}/sort/": []}))
    images = TaskImages.from_environ({TASK_IMAGES_VARIABLE: str(images_file)})
    assert images is not None
    # The lookup normalizes both paths lexically, and keeps a link of the job's mount as it is.
    assert images.for_task(job / "add") == prepared_images_from_json([ENTRY])
    assert images.for_task(job / "sort") == ()
    assert (images.has_images(job / "add"), images.has_images(job / "sort")) == (True, False)
    # A task is never found by its name alone: a relative path, or the same name in another directory, has no entry.
    monkeypatch.chdir(tmp_path)
    for path in (Path("add"), tmp_path / "other" / "add"):
        with pytest.raises(ValueError, match="list no task at"):
            images.for_task(path)
        assert images.has_images(path) is False
    # Without the variable, the process has no prepared images.
    assert TaskImages.from_environ({}) is None and TaskImages.from_environ({TASK_IMAGES_VARIABLE: " "}) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "error", "match"),
    [
        (None, OSError, None),
        ("{not json", ValueError, "invalid JSON"),
        ('["add"]', ValueError, "not an object"),
        ('{"tasks/add": []}', ValueError, "not absolute"),
        ('{"/job/add": [], "/job/./add": []}', ValueError, "twice"),
        ('{"/job/add": [], "/job/add": []}', ValueError, "repeats a key"),
        (json.dumps({"/job/add": [{**ENTRY, "image": "registry.example/tasks/add:latest"}]}), ValueError, "digest"),
    ],
    ids=["missing", "not-json", "not-object", "relative", "twice", "duplicate-key", "forged"],
)
def test_a_training_process_fails_closed_on_a_bad_file_of_task_images(
    tmp_path: Path, content: str | None, error: type[Exception], match: str | None
) -> None:
    images_file = tmp_path / "task-images.json"
    if content is not None:
        images_file.write_text(content)
    with pytest.raises(error, match=match):
        TaskImages.from_environ({TASK_IMAGES_VARIABLE: str(images_file)})


@pytest.mark.unit
def test_a_task_of_one_evaluation_has_its_images_under_its_absolute_path(tmp_path: Path) -> None:
    images = prepared_images_from_json([ENTRY])
    task_images = TaskImages.of_task(tmp_path / "tasks" / ".." / "tasks" / "add", images)
    assert task_images.by_task == {os.path.join(str(tmp_path), "tasks", "add"): images}
    with pytest.raises(ValueError, match="absolute normalized path"):
        TaskImages({"tasks/add": images})


def cold_sandbox(**changes: object) -> ColdSandbox:
    """The cold start of ``IMAGE`` for a trial of an allowlisted task with 2 CPUs and 4096 MB, with ``changes``."""
    fields: dict[str, object] = {
        "image": IMAGE,
        "timeout_seconds": 4920,
        "allow_internet_access": True,
        "network": {"allow_out": ["pypi.org"], "deny_out": ["0.0.0.0/0"]},
        "metadata": {"environment_name": "add", "session_id": "t__env"},
        "env_vars": {"TASK_SETTING": "from-the-task"},
        "cpu_count": 2,
        "memory_mb": 4096,
    }
    return ColdSandbox(**{**fields, **changes})


def cold_start(api: FakeAgentEnvApi, wait_seconds: float = 30.0) -> str:
    """``cold_sandbox().create`` on ``api``, served on a free local port."""

    async def run() -> str:
        async with serving(api) as templates:
            return await cold_sandbox().create(templates, wait_seconds)

    return asyncio.run(run())


@pytest.mark.unit
def test_a_cold_start_sends_the_values_of_the_template_start_and_returns_the_sandbox() -> None:
    api = FakeAgentEnvApi([(201, {"sandboxID": "sandbox-7", "templateID": "cold"})])
    assert cold_start(api) == "sandbox-7"
    assert api.cold_bodies == [
        {
            "image": IMAGE,
            "timeout": 4920,
            # Deleted at the end of its lifetime, as a template's sandbox of a trial is.
            "autoPause": False,
            "secure": True,
            "allowInternetAccess": True,
            "network": {"allowOut": ["pypi.org"], "denyOut": ["0.0.0.0/0"]},
            "metadata": {"environment_name": "add", "session_id": "t__env"},
            "envVars": {"TASK_SETTING": "from-the-task"},
            "cpuCount": 2,
            "memoryMB": 4096,
        }
    ]
    # No internet at creation, and AgentENV's default resources, are sent as they are.
    plain = cold_sandbox(allow_internet_access=False, network=None, cpu_count=None, memory_mb=None).body()
    assert plain["allowInternetAccess"] is False
    assert not {"network", "cpuCount", "memoryMB"} & set(plain)


@pytest.mark.unit
def test_a_cold_start_refuses_a_network_option_that_it_cannot_send() -> None:
    with pytest.raises(ValueError, match="cannot apply the network options block_dns"):
        cold_sandbox(network={"deny_out": ["0.0.0.0/0"], "block_dns": True})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("answer", "raised", "match"),
    [
        # No worker has room now, and AgentENV made no sandbox: the rollout waits for capacity, as for a template's
        # start.
        ((503, {"message": "no nodes available"}), SandboxCapacityError, "HTTP 503"),
        ((429, {"message": "too many requests"}), SandboxCapacityError, "HTTP 429"),
        ((400, {"message": "cpuCount must be at most 16"}), SandboxRequestError, "at most 16"),
        # A gateway or a proxy stopped waiting, or no answer came: AgentENV can still start the sandbox, so the
        # caller does not ask for another one.
        ((504, "<html>Gateway Time-out</html>"), ColdSandboxError, "HTTP 504"),
        ((500, "<html>Internal Server Error</html>"), ColdSandboxError, "HTTP 500"),
        ("slow", ColdSandboxError, "in time"),
        ("drop", ColdSandboxError, "lost its answer"),
        ((201, {"templateID": "cold"}), ColdSandboxError, "named no sandbox ID"),
        # A redirect is not followed: the create is not sent again, and the API key goes to no other address.
        ("redirect", ColdSandboxError, "HTTP 307"),
        # A credential or service error keeps its meaning.
        ((401, {"message": "invalid API key"}), ColdSandboxError, "HTTP 401"),
        ((403, {"message": "image not allowed"}), ColdSandboxError, "HTTP 403"),
        ((500, {"message": "image pull failed"}), ColdSandboxError, "image pull failed"),
    ],
    ids=[
        "busy",
        "rate",
        "bad-request",
        "gateway",
        "proxy",
        "slow",
        "dropped",
        "no-id",
        "redirect",
        "unauthorized",
        "forbidden",
        "service",
    ],
)
def test_a_cold_start_that_does_not_start_raises_by_its_answer_and_is_sent_once(
    answer: ColdAnswer, raised: type[Exception], match: str
) -> None:
    api = FakeAgentEnvApi([answer])
    with pytest.raises(raised, match=match):
        cold_start(api, wait_seconds=0.2 if answer == "slow" else 30.0)
    assert (len(api.cold_bodies), api.followed) == (1, [])


@pytest.mark.unit
def test_only_a_ready_template_is_ready() -> None:
    statuses: dict[str, str | int] = {"ready-1": "ready", "building-1": "building", "failed-1": "error"}
    api = FakeAgentEnvApi(statuses={**statuses, "denied-1": 403, "moved-1": "redirect", "slow-1": "slow"})

    async def read(*templates: str, wait_seconds: float = 30.0) -> list[bool]:
        async with serving(api) as deployment:
            return [await deployment.is_ready(template, wait_seconds) for template in templates]

    assert asyncio.run(read(*statuses, "gone-1")) == [True, False, False, False]
    # A refused read, and a redirect, which the read does not follow with the API key, raise.
    with pytest.raises(aiohttp.ClientResponseError, match="403"):
        asyncio.run(read("denied-1"))
    with pytest.raises(aiohttp.ClientResponseError, match="redirected"):
        asyncio.run(read("moved-1"))
    assert api.followed == []
    # A read that gets no answer in its time ends.
    with pytest.raises(TimeoutError):
        asyncio.run(read("slow-1", wait_seconds=0.2))
