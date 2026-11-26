"""Prepared task images on AgentENV: the images that a deployment verified for a task, and a sandbox start from one.

A deployment that keeps each task's environment images in a registry can verify them for one task version. It then
gives Reef one descriptor for each environment of the task (``PreparedImage``). Reef's E2B trial class starts such an
environment from the descriptor's template when the template is ready. Otherwise it starts the image itself with
AgentENV's ``POST /sandboxes-cold`` (``ColdSandbox``). So a template that is missing or not ready never sends a task
with a verified image back through a Dockerfile build.

Only a trusted caller gives the descriptors, never a task's own files. An evaluation reads them from the result of the
task's template item (``TaskTemplateBuilder.build_for_version``). A training process reads them from the file that
``TASK_IMAGES_VARIABLE`` names (``TaskImages.from_environ``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import aiohttp

from reef.core.errors import InternalError
from reef.core.values import strict_json_object
from reef.execution.sandbox.agentenv.build import BUILD_OUTPUT_TAIL_CHARS, AgentEnvTemplates
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

#: The fields of the JSON object of one ``PreparedImage``: each one is required, and no other field is allowed.
PREPARED_IMAGE_FIELDS = ("directory", "source_image", "image", "template_id", "start", "ready")
#: One label of a registry host name.
HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
#: An image reference by digest in an external registry: a registry host, one or more repository path components,
#: and a SHA-256 digest of 64 lowercase hex digits, without a tag. AgentENV reads the first component as a registry
#: host only when it has a dot or a port, or is ``localhost``. Any other first component, such as ``aenv``, names
#: AgentENV's own shared store, which a prepared image does not use.
IMAGE_REFERENCE = re.compile(
    rf"(?:(?:{HOST_LABEL}(?:\.{HOST_LABEL})+|localhost)(?::[0-9]+)?|{HOST_LABEL}:[0-9]+)"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}"
)
#: The variable of a training process that names the JSON file of the prepared images of its tasks.
TASK_IMAGES_VARIABLE = "REEF_AGENTENV_TASK_IMAGES"
#: The statuses with which AgentENV refuses a cold start for want of a free worker now (429, 503): it made no
#: sandbox, and a later start can pass. Any other failure can leave a sandbox that still starts, such as a gateway's
#: 502 or 504, or has a lasting cause, so the start is not sent again.
CAPACITY_STATUSES = (429, 503)
#: The names of E2B's network options (``SandboxNetworkOpts`` of the E2B SDK) in AgentENV's sandbox request.
NETWORK_FIELDS = {
    "allow_out": "allowOut",
    "deny_out": "denyOut",
    "allow_public_traffic": "allowPublicTraffic",
    "mask_request_host": "maskRequestHost",
}

#: The JSON object of one ``PreparedImage`` (``PreparedImage.to_json``).
PreparedImageJson = dict[str, str | None]


@dataclass(frozen=True)
class PreparedImage:
    """One environment image of a task, which a deployment verified for one version of the task.

    Values: ``directory`` is the environment's directory in the task, a relative POSIX path whose parts are not
    empty, ``.`` or ``..``: ``environment``, ``tests``, or the directory of a step's verifier. ``source_image`` is
    the ``docker_image`` of that environment's config, or None when the config has none; with ``directory``, it
    tells apart two environments in one directory. ``image`` is the image by digest (``IMAGE_REFERENCE``).
    ``template_id`` is an AgentENV template of the image that was ready when the deployment checked it, or None.
    ``start`` is the command that the template started before its snapshot, and ``ready`` the command whose success
    ended that start; each is a non-empty string, or None when the template has none.
    Purpose: ``prepared_images_from_json`` reads a list of them. ``KillOnTimeoutE2BEnvironment`` starts its
    environment from the one of its directory and its ``docker_image`` (``select_prepared_image``).
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


def is_missing_snapshot(status: int | None, message: str, template_id: str) -> bool:
    """Whether an answer of AgentENV's sandbox create, with the HTTP status ``status`` and the message ``message``,
    refused the template ``template_id`` because the worker has no snapshot of it. AgentENV gives these answers before
    it allocates a sandbox: HTTP 400 ``template <ID> not found``, and HTTP 500 ``snapshot '<ID>' is not ready``. The
    message can have text before the answer, but it must name ``template_id`` whole: the answer of another template,
    or any other answer with these statuses, is not one."""
    identifier = re.escape(template_id)
    if status == 400:
        pattern = rf"(?<![^\s:])template {identifier} not found\b"
    elif status == 500:
        pattern = rf"(?<![^\s:])snapshot '{identifier}' is not ready\b"
    else:
        return False
    return re.search(pattern, message) is not None


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
    Purpose: a trusted caller makes one, ``from_environ`` in a training process or ``of_task`` in an evaluation.
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
    def from_environ(cls, environ: Mapping[str, str]) -> Self | None:
        """The prepared images of the file that ``TASK_IMAGES_VARIABLE`` names in ``environ``, or None when it is unset
        or empty. The file is a JSON object that maps each absolute task path of the run to its JSON list of
        descriptors (``prepared_images_from_json``). Raise ``OSError`` for a file that cannot be read, and
        ``ValueError`` for one that is not such an object or that names one task twice."""
        path = environ.get(TASK_IMAGES_VARIABLE, "").strip()
        if not path:
            return None
        try:
            value = strict_json_object(Path(path).read_bytes())
        except ValueError as error:
            raise ValueError(f"invalid JSON task image manifest in {TASK_IMAGES_VARIABLE}: {error}") from error
        by_task: dict[str, tuple[PreparedImage, ...]] = {}
        for task, images in value.items():
            if not os.path.isabs(task):
                raise ValueError(f"{TASK_IMAGES_VARIABLE} names the task path {task!r}, which is not absolute")
            key = os.path.normpath(task)
            if key in by_task:
                raise ValueError(f"{TASK_IMAGES_VARIABLE} names the task {key} twice")
            by_task[key] = prepared_images_from_json(images)
        return cls(by_task)


class ColdSandboxError(InternalError):
    """AgentENV did not start a sandbox from a prepared image, or the sandbox did not pass its start. A start does not
    send its request again after this error, so it never makes a second sandbox for an answer that it could not
    read."""


@dataclass(frozen=True)
class ColdSandbox:
    """One request to AgentENV's ``POST /sandboxes-cold``: a sandbox started from an image, without a template.

    Values: ``image`` is the image by digest. ``timeout_seconds`` is the sandbox's lifetime, at least 1; AgentENV
    deletes the sandbox at its end and does not pause it. ``allow_internet_access``, ``network``, ``metadata`` and
    ``env_vars`` are the values of E2B's create call: ``network`` has the names of the E2B SDK's network options
    (``NETWORK_FIELDS``), or is None for no options. ``cpu_count`` and ``memory_mb`` are the sandbox's resources,
    None for AgentENV's default.
    Purpose: ``KillOnTimeoutE2BEnvironment`` makes one for an environment whose prepared image has no ready template,
    and ``create`` sends it.
    """

    image: str
    timeout_seconds: int
    allow_internet_access: bool
    network: Mapping[str, object] | None
    metadata: Mapping[str, str]
    env_vars: Mapping[str, str]
    cpu_count: int | None
    memory_mb: int | None

    def __post_init__(self) -> None:
        if IMAGE_REFERENCE.fullmatch(self.image) is None:
            raise ValueError(f"a cold sandbox starts an image by digest, not {self.image!r}")
        if self.timeout_seconds < 1:
            raise ValueError("a cold sandbox needs a lifetime of at least 1 s")
        unknown = set(self.network or {}) - set(NETWORK_FIELDS)
        if unknown:
            # A network option that the request cannot carry would leave the sandbox less isolated than asked.
            raise ValueError(f"a cold sandbox cannot apply the network options {', '.join(sorted(unknown))}")

    def body(self) -> dict[str, object]:
        """The JSON body of the request."""
        body: dict[str, object] = {
            "image": self.image,
            "timeout": self.timeout_seconds,
            "autoPause": False,
            "secure": True,
            "allowInternetAccess": self.allow_internet_access,
            "metadata": dict(self.metadata),
            "envVars": dict(self.env_vars),
        }
        if self.network is not None:
            body["network"] = {NETWORK_FIELDS[name]: value for name, value in self.network.items()}
        if self.cpu_count is not None:
            body["cpuCount"] = self.cpu_count
        if self.memory_mb is not None:
            body["memoryMB"] = self.memory_mb
        return body

    async def create(self, templates: AgentEnvTemplates, wait_seconds: float) -> str:
        """Send the request once to ``templates``' deployment, wait at most ``wait_seconds``, a positive number, for
        the answer, and return the ID of the new sandbox (HTTP 201).

        Raise ``SandboxCapacityError`` for a status of ``CAPACITY_STATUSES``, and ``SandboxRequestError`` for HTTP
        400. Raise ``ColdSandboxError`` for any other answer, such as 401, 403, 500, or a gateway's 502 or 504, and
        when no answer came in time or the connection was lost. After those, AgentENV can still start the sandbox,
        whose ID the caller does not know; AgentENV deletes it at the end of its lifetime. A redirect is such an
        answer too: the request is not sent again, to the same host or another one, with the API key."""
        try:
            async with (
                aiohttp.ClientSession(
                    headers={"X-API-Key": templates.api_key}, timeout=aiohttp.ClientTimeout(total=wait_seconds)
                ) as session,
                session.post(
                    f"{templates.api_url}/sandboxes-cold", json=self.body(), allow_redirects=False
                ) as response,
            ):
                if response.status == 201:
                    answer = await response.json()
                    sandbox_id = answer.get("sandboxID") if isinstance(answer, dict) else None
                    if not isinstance(sandbox_id, str) or not sandbox_id:
                        raise ColdSandboxError(f"AgentENV started a sandbox of {self.image} and named no sandbox ID")
                    return sandbox_id
                status = response.status
                text = (await response.text())[-BUILD_OUTPUT_TAIL_CHARS:]
        except TimeoutError as error:
            raise ColdSandboxError(f"AgentENV did not answer the cold start of {self.image} in time") from error
        except (aiohttp.ClientError, json.JSONDecodeError) as error:
            raise ColdSandboxError(f"the cold start of {self.image} lost its answer: {error!r}") from error
        message = f"AgentENV refused a sandbox of {self.image} (HTTP {status}): {text}"
        if status in CAPACITY_STATUSES:
            raise SandboxCapacityError(message)
        if status == 400:
            raise SandboxRequestError(message)
        raise ColdSandboxError(message)


__all__ = [
    "CAPACITY_STATUSES",
    "IMAGE_REFERENCE",
    "NETWORK_FIELDS",
    "PREPARED_IMAGE_FIELDS",
    "TASK_IMAGES_VARIABLE",
    "ColdSandbox",
    "ColdSandboxError",
    "PreparedImage",
    "PreparedImageJson",
    "TaskImages",
    "is_missing_snapshot",
    "prepared_images_from_json",
    "select_prepared_image",
]
