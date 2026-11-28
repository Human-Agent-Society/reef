"""``reef.execution.sandbox.prepared_images``: the prepared images of a task and their lookup by task path and
directory, and the manifest that a training process's deployment names (``task_images_from_environ``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reef.execution.sandbox.prepared_images import (
    PreparedImage,
    TaskImages,
    prepared_images_from_json,
    select_prepared_image,
)
from reef.execution.sandbox.task_environment import TASK_IMAGES_VARIABLE, task_images_from_environ

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
        # Without a dot, a port or localhost, the first part names no external registry: aenv/ is a sandbox
        # service's own local prefix, and team/ is not a registry either.
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
    # Deployments set the variable by this name.
    assert TASK_IMAGES_VARIABLE == "REEF_AGENTENV_TASK_IMAGES"
    images = task_images_from_environ("e2b", {TASK_IMAGES_VARIABLE: str(images_file)})
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
    # Without the variable, the process has no prepared images; nor has one whose trials start no prepared image.
    assert task_images_from_environ("e2b", {}) is None
    assert task_images_from_environ("e2b", {TASK_IMAGES_VARIABLE: " "}) is None
    for environment in ("docker", "modal"):
        assert task_images_from_environ(environment, {TASK_IMAGES_VARIABLE: str(images_file)}) is None


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
    with pytest.raises(error, match=match) as raised:
        task_images_from_environ("e2b", {TASK_IMAGES_VARIABLE: str(images_file)})
    # An invalid manifest names the variable that named it.
    assert error is OSError or str(raised.value).startswith(f"{TASK_IMAGES_VARIABLE}: ")


@pytest.mark.unit
def test_a_task_of_one_evaluation_has_its_images_under_its_absolute_path(tmp_path: Path) -> None:
    images = prepared_images_from_json([ENTRY])
    task_images = TaskImages.of_task(tmp_path / "tasks" / ".." / "tasks" / "add", images)
    assert task_images.by_task == {os.path.join(str(tmp_path), "tasks", "add"): images}
    with pytest.raises(ValueError, match="absolute normalized path"):
        TaskImages({"tasks/add": images})
