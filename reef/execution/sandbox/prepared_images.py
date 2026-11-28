"""Images a deployment has verified for a specific task version.

Evaluation receives image records from its template builder. Training reads them from a deployment-supplied
manifest. Each trial uses the task directory and source image to select the record for its environment.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from reef.core.values import strict_json_object

#: The fields of the JSON object of one ``PreparedImage``: each one is required, and no other field is allowed.
PREPARED_IMAGE_FIELDS = ("directory", "source_image", "image", "template_id", "start", "ready")
#: One label of a registry host name.
HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
#: A registry host and repository pinned by SHA-256 digest. The hostname must contain a dot, include a port,
#: or be ``localhost`` to identify an explicit registry in a Docker image reference.
IMAGE_REFERENCE = re.compile(
    rf"(?:(?:{HOST_LABEL}(?:\.{HOST_LABEL})+|localhost)(?::[0-9]+)?|{HOST_LABEL}:[0-9]+)"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}"
)

#: The JSON object of one ``PreparedImage`` (``PreparedImage.to_json``).
PreparedImageJson = dict[str, str | None]


@dataclass(frozen=True)
class PreparedImage:
    """One environment image of a task, which a deployment verified for one version of the task.

    Values: ``directory`` is the environment's directory in the task, a relative POSIX path whose parts are not
    empty, ``.`` or ``..``: ``environment``, ``tests``, or the directory of a step's verifier. ``source_image`` is
    the ``docker_image`` of that environment's config, or None when the config has none; with ``directory``, it
    tells apart two environments in one directory. ``image`` is the image by digest (``IMAGE_REFERENCE``).
    ``template_id`` is a template of the image in the sandbox service that was ready when the deployment checked it,
    or None. ``start`` is the command that the template started before its snapshot, and ``ready`` the command whose
    success ended that start; each is a non-empty string, or None when the template has none.
    Purpose: ``prepared_images_from_json`` reads a list of them, and a trial environment starts from the one of its
    directory and its ``docker_image`` (``select_prepared_image``).
    """

    directory: str
    source_image: str | None
    image: str
    template_id: str | None
    start: str | None
    ready: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str) or any(part in ("", ".", "..") for part in self.directory.split("/")):
            raise ValueError(f"a prepared image's directory must be a normalized task path, not {self.directory!r}")
        if not isinstance(self.image, str) or IMAGE_REFERENCE.fullmatch(self.image) is None:
            raise ValueError(
                "a prepared image must be host/repository@sha256:<digest>, whose registry host has a dot or a port "
                f"or is localhost, not {self.image!r}"
            )
        for name, value in (
            ("source_image", self.source_image),
            ("template_id", self.template_id),
            ("start", self.start),
            ("ready", self.ready),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"a prepared image's {name} must be a non-empty string or null")

    @classmethod
    def from_json(cls, value: object) -> Self:
        """The descriptor of the JSON object ``value``. Raise ``ValueError`` unless the object has exactly the
        fields of ``PREPARED_IMAGE_FIELDS``, with valid values."""
        if not isinstance(value, Mapping) or set(value) != set(PREPARED_IMAGE_FIELDS):
            fields = ", ".join(PREPARED_IMAGE_FIELDS)
            raise ValueError(f"a prepared image is a JSON object with exactly the fields {fields}")
        return cls(
            directory=value["directory"],
            source_image=value["source_image"],
            image=value["image"],
            template_id=value["template_id"],
            start=value["start"],
            ready=value["ready"],
        )

    def to_json(self) -> PreparedImageJson:
        """The JSON object of this descriptor, with each field, null too."""
        return {
            "directory": self.directory,
            "source_image": self.source_image,
            "image": self.image,
            "template_id": self.template_id,
            "start": self.start,
            "ready": self.ready,
        }


def prepared_images_from_json(value: object) -> tuple[PreparedImage, ...]:
    """The descriptors of the JSON list ``value``, in order. Raise ``ValueError`` unless it is a list of valid
    descriptors (``PreparedImage.from_json``) in which no two have the same directory and source image."""
    if not isinstance(value, list):
        raise ValueError("the prepared images of a task are a JSON list")
    images = tuple(PreparedImage.from_json(entry) for entry in value)
    keys = [(image.directory, image.source_image) for image in images]
    if len(set(keys)) != len(keys):
        raise ValueError("two prepared images of a task have the same directory and source image")
    return images


def select_prepared_image(
    images: Sequence[PreparedImage], task_directory: Path, environment_dir: Path, source_image: str | None
) -> PreparedImage | None:
    """The descriptor of ``images`` for the environment of the task at ``task_directory`` whose directory is
    ``environment_dir`` and whose config's ``docker_image`` is ``source_image``. A deployment may have rewritten
    the config to the verified digest. That digest must identify one descriptor in this directory.

    The environment's directory in the task is its path from the task's directory after both resolve their links, as
    a build reads them. An environment directory outside the task has no descriptor."""
    try:
        directory = Path(os.path.realpath(environment_dir)).relative_to(os.path.realpath(task_directory)).as_posix()
    except ValueError:
        return None
    candidates = [image for image in images if image.directory == directory]
    for image in candidates:
        if image.source_image == source_image:
            return image
    pinned = [image for image in candidates if image.image == source_image]
    return pinned[0] if len(pinned) == 1 else None


@dataclass(frozen=True)
class TaskImages:
    """The prepared images of the tasks of one run, by task directory.

    Values: ``by_task`` maps task directories to their descriptors. Each key is an absolute path in the form of
    ``os.path.abspath``: lexically normalized, with its links as they are. A task with an empty tuple has no prepared
    image.
    Purpose: a trusted caller makes one, ``from_manifest`` in a training process or ``of_task`` in an evaluation.
    ``TaskRolloutRunner`` gives each trial the descriptors of its task (``for_task``).
    """

    by_task: Mapping[str, tuple[PreparedImage, ...]]

    def __post_init__(self) -> None:
        for task, images in self.by_task.items():
            if not isinstance(task, str) or os.path.abspath(task) != task:
                raise ValueError(f"a task of prepared images is an absolute normalized path, not {task!r}")
            if not isinstance(images, tuple) or not all(isinstance(image, PreparedImage) for image in images):
                raise TypeError("the prepared images of a task are a tuple of PreparedImage")

    @classmethod
    def of_task(cls, task_path: Path, images: Sequence[PreparedImage]) -> Self:
        """The prepared images ``images`` of the one task at ``task_path``."""
        return cls({os.path.abspath(task_path): tuple(images)})

    def for_task(self, task_path: Path) -> tuple[PreparedImage, ...]:
        """The descriptors of the task at ``task_path``, found by the ``os.path.abspath`` of the path, never by the
        task's name alone. Raise ``ValueError`` for a task without an entry: a run with prepared images lists each of
        its tasks."""
        key = os.path.abspath(task_path)
        if key not in self.by_task:
            raise ValueError(f"the prepared task images list no task at {key}")
        return self.by_task[key]

    def has_images(self, task_path: Path) -> bool:
        """Whether the task at ``task_path`` has an entry with one or more descriptors (``for_task``)."""
        return bool(self.by_task.get(os.path.abspath(task_path)))

    @classmethod
    def from_manifest(cls, manifest: Path) -> Self:
        """The prepared images of the manifest file ``manifest``: a JSON object that maps each absolute task path of
        the run to its JSON list of descriptors (``prepared_images_from_json``). Raise ``OSError`` for a file that
        cannot be read, and ``ValueError`` for one that is not such an object or that names one task twice."""
        try:
            value = strict_json_object(manifest.read_bytes())
        except ValueError as error:
            raise ValueError(f"invalid JSON task image manifest: {error}") from error
        by_task: dict[str, tuple[PreparedImage, ...]] = {}
        for task, images in value.items():
            if not os.path.isabs(task):
                raise ValueError(f"the task image manifest names the task path {task!r}, which is not absolute")
            key = os.path.normpath(task)
            if key in by_task:
                raise ValueError(f"the task image manifest names the task {key} twice")
            by_task[key] = prepared_images_from_json(images)
        return cls(by_task)


__all__ = [
    "IMAGE_REFERENCE",
    "PREPARED_IMAGE_FIELDS",
    "PreparedImage",
    "PreparedImageJson",
    "TaskImages",
    "prepared_images_from_json",
    "select_prepared_image",
]
