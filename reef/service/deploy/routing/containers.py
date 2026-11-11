"""Docker boundary for the one Semantic Router container of a managed routing deployment."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import queue
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from reef.service.deploy.routing.registry import DeploymentOwnership, RoutingOwnershipError
from reef.storage.durable_files import fsync_dir

# Supported upstream source revision, shipped in wheels without requiring a Git checkout.
SEMANTIC_ROUTER_REVISION = "c89d5d4c15117b0a7b1e09c87c6c863dd1efcb59"
#: The build context of the router image: Docker fetches this exact commit itself, so no checkout is needed.
SEMANTIC_ROUTER_BUILD_CONTEXT = f"https://github.com/vllm-project/semantic-router.git#{SEMANTIC_ROUTER_REVISION}"
SEMANTIC_ROUTER_DOCKERFILE = "src/vllm-sr/Dockerfile"
REVISION_LABEL = "org.opencontainers.image.revision"
#: The label of an image that Reef built: the build recipe it was built with.
BUILD_RECIPE_LABEL = "io.reef.semantic-router.build"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
#: One image inspection returns the image's ID and labels together, as one JSON object.
IMAGE_IDENTITY_FORMAT = '{"id": {{json .Id}}, "labels": {{json .Config.Labels}}}'
#: The build metadata's digests of the built image: Docker's classic image store names a local image by its config
#: digest, the containerd image store by its image (manifest) digest.
BUILD_CONFIG_DIGEST_KEY = "containerimage.config.digest"
BUILD_IMAGE_DIGEST_KEY = "containerimage.digest"


def image_build_arguments(metadata_file: str, recipe: str, tag: str) -> list[str]:
    """The ``docker`` arguments that build the router image from the pinned commit and load it into the daemon.

    The build writes its result metadata, with the image's config and image digests, to ``metadata_file`` and
    labels the image with the pinned revision and ``recipe``; ``tag`` names it. No build argument, secret or
    registry push is part of it.
    """
    return [
        "buildx",
        "build",
        "--load",
        "--progress",
        "plain",
        "--metadata-file",
        metadata_file,
        "-f",
        SEMANTIC_ROUTER_DOCKERFILE,
        "--label",
        f"{REVISION_LABEL}={SEMANTIC_ROUTER_REVISION}",
        "--label",
        f"{BUILD_RECIPE_LABEL}={recipe}",
        "-t",
        tag,
        SEMANTIC_ROUTER_BUILD_CONTEXT,
    ]


#: Names how Reef builds the image: the hash of the build's arguments, source included, with placeholders for the
#: per-build metadata file, the recipe and the tag. A changed build gets another recipe and cache tag.
IMAGE_BUILD_RECIPE = hashlib.sha256(
    json.dumps(image_build_arguments("{metadata_file}", "{recipe}", "{tag}")).encode()
).hexdigest()[:16]
#: The tag of the image that Reef built with this recipe, which later startups reuse.
IMAGE_CACHE_TAG = f"reef-semantic-router:{SEMANTIC_ROUTER_REVISION[:12]}-{IMAGE_BUILD_RECIPE}"


class ContainerCommandError(RuntimeError):
    """The CLI failed; a submitted mutation may still have reached the daemon.

    ``outcome_unknown`` is a bool. True means a mutation may have reached the daemon although its result is
    unknown, so callers must not assume a rollback or start a replacement; false makes no claim about the
    resource's health. The message never contains CLI output, which may hold secrets.
    """

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        if type(outcome_unknown) is not bool:
            raise ValueError("container command outcome_unknown must be a boolean")
        super().__init__(message)
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class ContainerState:
    """One owned container as Docker reported it, whether or not it is usable.

    Values: ``resource_id`` and ``service`` are strings as Docker reported them (the Compose service label);
    ``running`` and ``paused`` are bools; ``health`` is None (no health check) or Docker's health status,
    including values Reef does not know.
    Purpose: ``ContainerRuntime.inspect`` makes it; the controller's ``inspect_ready`` alone decides whether
    it is the one healthy Semantic Router.
    """

    resource_id: str
    service: str
    running: bool
    health: str | None
    paused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not isinstance(self.service, str):
            raise ValueError("container identity and service must be strings")
        if type(self.running) is not bool or type(self.paused) is not bool:
            raise ValueError("container running and paused states must be booleans")
        if self.health is not None and not isinstance(self.health, str):
            raise ValueError("container health must be None or a string")


@dataclass(frozen=True)
class ContainerInventory:
    """Every owned container and network one ownership-checked inspection found.

    Values: ``containers`` is a tuple of ``ContainerState`` and ``network_ids`` a tuple of strings, in the order
    found; either may be empty. Nothing is removed or merged, so several, duplicate or failed observations
    stay visible to the reader. Order has no meaning.
    Purpose: ``ContainerRuntime.inspect`` makes it; the controller checks readiness with it, and cleanup
    checks that it is empty. It describes resources and owns none of them.
    """

    containers: tuple[ContainerState, ...] = ()
    network_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.containers) is not tuple or not all(isinstance(item, ContainerState) for item in self.containers):
            raise ValueError("container inventory containers must be a tuple of container states")
        if type(self.network_ids) is not tuple or not all(isinstance(item, str) for item in self.network_ids):
            raise ValueError("container inventory network IDs must be a tuple of strings")


@dataclass(frozen=True)
class ContainerEvent:
    """One Docker event, or the end of the observation.

    Values: ``container_id`` and ``action`` are strings as Docker reported them; any action, including one
    Reef does not interpret, is a valid event. The action ``stream_disconnected`` is not a Docker action:
    the event stream adds it, with an empty ``container_id``, when its reader stops, overflows or reads
    malformed input, so the observation is incomplete.
    Purpose: a ``ContainerEventStream`` makes it; the controller reacts to the disconnect and to failure
    actions of its own containers and ignores the rest.
    """

    container_id: str
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.container_id, str) or not isinstance(self.action, str):
            raise ValueError("container event identity and action must be strings")


class ContainerEventStream(ABC):
    """Docker events for one deployment, owned by the controller that subscribed.

    ``poll(timeout_seconds)`` waits up to ``timeout_seconds`` and returns the oldest queued event, removing
    it, or None if none arrived; None is silence, not the end. The end of the observation is the
    ``stream_disconnected`` event. ``close`` stops and reaps the reader; the stream is not used after it.
    """

    @abstractmethod
    def poll(self, timeout_seconds: float = 0) -> ContainerEvent | None:
        """Return an event; EOF is a stream_disconnected event, not silence."""

    @abstractmethod
    def close(self) -> None:
        """Stop and reap the CLI reader within a bounded time."""


class ContainerRuntime(ABC):
    """Local container operations for one deployment owner, each limited to resources it owns.

    ``compose_environment`` holds the private values later Compose commands receive; ``use_environment``
    replaces it with a copy. Its values never appear in arguments or files, and an image build never gets
    them. ``inspect`` returns every resource the ownership labels match and refuses a foreign one; ``start``
    and ``remove`` change only owned resources; ``events`` returns a new stream that the caller owns.
    ``resolve_image`` creates no container or network. One runtime serves one owner, because its environment
    is mutable.
    """

    #: Variables Compose substitutes into containers, such as the router management token.
    compose_environment: dict[str, str]

    def use_environment(self, values: Mapping[str, str]) -> None:
        """Pass private values to later Compose commands through their environment only."""
        self.compose_environment = dict(values)

    @abstractmethod
    def ensure_available(self) -> None:
        """Validate local daemon access and Compose availability."""

    @abstractmethod
    def resolve_image(self, build_log: Path, timeout_seconds: float, stop_requested: Callable[[], bool]) -> str:
        """The local ID (``sha256:<hex>``) of the router image that this Reef version builds.

        A cached image is reused when its labels name the pinned revision and ``IMAGE_BUILD_RECIPE``. Without
        one, the image is built, for at most ``timeout_seconds``, with its progress in the private file
        ``build_log``; ``stop_requested`` is polled meanwhile and stops the build when it returns true.

        Raises:
            ContainerCommandError: Docker failed, the cached image has other labels, or the build failed or
                timed out. The message never holds build output.
            InterruptedError: ``stop_requested`` stopped the build.
        """

    @abstractmethod
    def start(self, compose_file: Path, ownership: DeploymentOwnership, timeout_seconds: float) -> None:
        """Start the frozen configuration after checking resource ownership."""

    @abstractmethod
    def inspect(self, ownership: DeploymentOwnership, timeout_seconds: float = 5) -> ContainerInventory:
        """Inspect resources, rejecting every conflicting project/deployment label."""

    @abstractmethod
    def remove(self, ownership: DeploymentOwnership, stop_timeout_seconds: float) -> None:
        """Remove only resources with verified ownership, with their anonymous volumes; never a named volume."""

    @abstractmethod
    def events(self, ownership: DeploymentOwnership) -> ContainerEventStream:
        """Subscribe before inspecting; inspect again on every reconnect."""


class GuardedCommand:
    """An isolated CLI group whose lifetime is bounded by its caller's pipe.

    Values: ``command`` is a non-empty sequence of strings, not itself a string or bytes, whose first item
    (the executable) is non-empty; later arguments may be empty. ``timeout_seconds`` is a finite positive
    number. Both are checked before any pipe or process exists. If the guard cannot be spawned, both ends
    of the owner pipe are closed before the error propagates. ``output`` is None, or an open file descriptor
    that receives the CLI's stdout and stderr instead of pipes; the caller keeps and closes it.
    State: ``process`` is the guard (``command_runner.py``), leader of a new process group that also holds
    the CLI and its descendants. ``owner_descriptor`` is the write end of the guard's owner pipe while this
    object holds it, and None once released. ``release_owner`` closes it at most once, from any thread, and
    makes the guard stop the group; ``close`` also waits for the guard and kills the group if the guard does
    not exit. Without ``output``, the caller reads and closes the guard's stdout and stderr.
    """

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        discard_stderr: bool = False,
        environment: Mapping[str, str] | None = None,
        output: int | None = None,
    ) -> None:
        if (
            isinstance(command, (str, bytes))
            or not isinstance(command, Sequence)
            or not command
            or any(not isinstance(argument, str) for argument in command)
            or not command[0]
        ):
            raise ValueError("a guarded command needs a non-empty executable and string arguments")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("a guarded command needs a finite positive timeout")
        if output is not None and (isinstance(output, bool) or not isinstance(output, int) or output < 0):
            raise ValueError("a guarded command's output must be an open file descriptor")
        if output is None:
            stdout: int = subprocess.PIPE
            stderr: int = subprocess.DEVNULL if discard_stderr else subprocess.PIPE
        else:
            stdout, stderr = output, subprocess.STDOUT
        self.owner_lock = threading.Lock()
        read_descriptor, write_descriptor = os.pipe()
        self.owner_descriptor: int | None = write_descriptor
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("command_runner.py")),
                    str(read_descriptor),
                    str(timeout_seconds),
                    *command,
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                pass_fds=(read_descriptor,),
                start_new_session=True,
                # Secret values reach Compose through its environment, never its arguments.
                env=None if environment is None else {**os.environ, **environment},
            )
        except (OSError, ValueError, TypeError):
            # Spawning failed, or Popen refused an argument or environment value it cannot encode.
            os.close(write_descriptor)
            self.owner_descriptor = None
            raise
        finally:
            os.close(read_descriptor)

    def release_owner(self) -> None:
        with self.owner_lock:
            if self.owner_descriptor is not None:
                os.close(self.owner_descriptor)
                self.owner_descriptor = None

    def close(self) -> None:
        self.release_owner()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=2)


class DockerEventStream(ContainerEventStream):
    """``docker events`` read by one thread into a queue of at most 1024 events.

    The reader queues each JSON line as one event, in arrival order. A malformed line or the end of the
    CLI's output queues a ``stream_disconnected`` event after the earlier ones; a full queue instead
    discards what is queued and keeps only that event. The reader then stops, and ``poll`` returns what is
    left, then None. ``closed`` is set when the observation must stop; ``close`` sets it, ends the guarded
    command, joins the reader and closes the command's pipes.
    """

    def __init__(self, command: Sequence[str]) -> None:
        # Events intentionally outlive individual CLI timeouts. The owner pipe
        # and explicit close bound their lifetime to this controller instance.
        self.command = GuardedCommand(command, timeout_seconds=365 * 24 * 3600, discard_stderr=True)
        self.pending: queue.Queue[ContainerEvent] = queue.Queue(maxsize=1024)
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self.read_events, name="reef-routing-docker-events", daemon=True)
        self.reader.start()

    def enqueue(self, event: ContainerEvent) -> None:
        try:
            self.pending.put_nowait(event)
        except queue.Full:
            # Losing event ordering requires a fresh inspect. Do not silently
            # treat overflow as a healthy, fully observed deployment.
            while not self.pending.empty():
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    break
            self.pending.put_nowait(ContainerEvent("", "stream_disconnected"))
            self.closed.set()
            self.command.release_owner()

    def read_events(self) -> None:
        stdout = self.command.process.stdout
        if stdout is None:
            self.enqueue(ContainerEvent("", "stream_disconnected"))
            return
        try:
            for line in stdout:
                if self.closed.is_set():
                    break
                try:
                    data = json.loads(line)
                    container_id = data["Actor"]["ID"]
                    action = data["Action"]
                    if not isinstance(container_id, str) or not isinstance(action, str):
                        raise ValueError("invalid event fields")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self.enqueue(ContainerEvent("", "stream_disconnected"))
                    self.command.release_owner()
                    break
                self.enqueue(ContainerEvent(container_id, action))
        finally:
            if not self.closed.is_set():
                self.enqueue(ContainerEvent("", "stream_disconnected"))

    def poll(self, timeout_seconds: float = 0) -> ContainerEvent | None:
        try:
            return self.pending.get(timeout=max(0, timeout_seconds))
        except queue.Empty:
            return None

    def close(self) -> None:
        self.closed.set()
        self.command.close()
        self.reader.join(timeout=2)
        if self.reader.is_alive():
            raise ContainerCommandError("Docker events reader did not stop")
        for stream in (self.command.process.stdout, self.command.process.stderr):
            if stream is not None:
                stream.close()


class DockerComposeRuntime(ContainerRuntime):
    """The local Docker CLI and its Compose plugin, run through ``GuardedCommand``.

    ``docker_command`` is a non-empty executable name or path. Making the runtime starts nothing.
    ``run(arguments)`` returns the CLI's UTF-8 stdout or raises ``ContainerCommandError``, whose outcome is
    unknown for a mutation. ``mutate`` writes a marker file before a mutation and removes it only after the
    mutation succeeds; while it exists, ``start`` refuses, until ``acknowledge_unknown`` clears it. An image
    build is no mutation of owned resources: it writes no marker, and stopping it can leave at most a built,
    labeled image that a later startup checks before it reuses it.
    """

    def __init__(self, docker_command: str = "docker") -> None:
        if not isinstance(docker_command, str) or not docker_command:
            raise ValueError("the Docker command must be a non-empty executable name or path")
        self.docker_command = docker_command
        self.compose_environment = {}

    def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
        if timeout_seconds <= 0:
            raise ContainerCommandError("Docker operation exceeded its deadline", outcome_unknown=mutation)
        try:
            command = GuardedCommand(
                [self.docker_command, *arguments], timeout_seconds, environment=self.compose_environment
            )
        except OSError as exc:
            raise ContainerCommandError("could not start Docker CLI") from exc
        try:
            try:
                stdout = command.process.communicate(timeout=timeout_seconds + 5)[0]
            except subprocess.TimeoutExpired as exc:
                raise ContainerCommandError("Docker CLI deadline expired", outcome_unknown=mutation) from exc
            if command.process.returncode != 0:
                # Docker output may contain expanded configuration or secrets.
                # The operation and status suffice; never interpolate stderr.
                raise ContainerCommandError(
                    f"Docker {arguments[0]} failed with status {command.process.returncode}", outcome_unknown=mutation
                )
            try:
                return stdout.decode("utf-8")
            except UnicodeDecodeError:
                raise ContainerCommandError("Docker CLI returned invalid UTF-8", outcome_unknown=mutation) from None
        finally:
            command.close()

    def ensure_available(self) -> None:
        if sys.platform != "linux":
            raise ContainerCommandError("managed routing requires Linux and a local Docker daemon")
        host = os.environ.get("DOCKER_HOST")
        if os.environ.get("DOCKER_CONTEXT") or not host:
            try:
                host = json.loads(self.run(["context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"]))
            except json.JSONDecodeError as exc:
                raise ContainerCommandError("invalid Docker context response") from exc
        if not isinstance(host, str) or not host.startswith("unix://"):
            raise ContainerCommandError("managed routing requires a local unix:// Docker daemon")
        self.run(["info", "--format", "{{.ServerVersion}}"])
        self.run(["compose", "version", "--short"])

    def resolve_image(self, build_log: Path, timeout_seconds: float, stop_requested: Callable[[], bool]) -> str:
        # A successful listing, not a failed inspection, shows that no image has the tag: an inspection fails
        # for daemon and permission errors too.
        listed = self.run(["image", "ls", "--quiet", "--no-trunc", "--filter", f"reference={IMAGE_CACHE_TAG}"])
        if listed.strip():
            return self.checked_image(IMAGE_CACHE_TAG, None)
        try:
            self.run(["buildx", "version"])
        except ContainerCommandError as exc:
            raise ContainerCommandError(
                "building the Semantic Router image needs Docker Buildx, and `docker buildx version` failed"
            ) from exc
        image_id = self.built_image_id(self.build_image(build_log, timeout_seconds, stop_requested), build_log)
        return self.checked_image(image_id, image_id)

    def built_image_id(self, digests: frozenset[str], build_log: Path) -> str:
        """The local ID of the image that a build loaded: the one of its ``digests`` that names a local image.

        A complete listing of local image IDs separates an absent image from a failed daemon. The shared cache tag
        is never read, since another startup's build may have moved it.

        Raises:
            ContainerCommandError: The listing failed or is malformed, or none or more than one of ``digests``
                names a local image.
        """
        local_ids = set(self.run(["image", "ls", "--all", "--quiet", "--no-trunc"]).split())
        if not all(IMAGE_ID.fullmatch(local_id) for local_id in local_ids):
            raise ContainerCommandError("invalid Docker image listing")
        found = sorted(digests & local_ids)
        if not found:
            raise ContainerCommandError(
                f"the Semantic Router image build succeeded, but no local image has its digests; see {build_log}"
            )
        if len(found) > 1:
            raise ContainerCommandError(
                "both the config digest and the image digest of the Semantic Router image build name local "
                f"images, so Reef cannot tell which one the build loaded; see {build_log}"
            )
        return found[0]

    def checked_image(self, reference: str, built_id: str | None) -> str:
        """The ID of image ``reference`` once one inspection shows the pinned revision and this build recipe.

        ``built_id`` is None for a cached image, else the local ID of this build's image, which it must have.

        Raises:
            ContainerCommandError: The inspection failed or is malformed, or the image has another ID or labels.
        """
        try:
            identity = json.loads(self.run(["image", "inspect", "--format", IMAGE_IDENTITY_FORMAT, reference]))
        except json.JSONDecodeError:
            raise ContainerCommandError("invalid Docker image identity") from None
        image_id = identity.get("id") if isinstance(identity, dict) else None
        labels = identity.get("labels") if isinstance(identity, dict) else None
        if not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id) or not isinstance(labels, dict | None):
            raise ContainerCommandError("invalid Docker image identity")
        if built_id is not None and image_id != built_id:
            raise ContainerCommandError("the built Semantic Router image does not have the ID that its build reported")
        labels = labels or {}
        if (
            labels.get(REVISION_LABEL) != SEMANTIC_ROUTER_REVISION
            or labels.get(BUILD_RECIPE_LABEL) != IMAGE_BUILD_RECIPE
        ):
            raise ContainerCommandError(
                f"image {reference} does not carry this Reef version's source revision and build recipe labels; "
                f"Reef does not replace it. Remove it with `docker image rm {reference}` and start again"
            )
        return image_id

    def build_image(
        self, build_log: Path, timeout_seconds: float, stop_requested: Callable[[], bool]
    ) -> frozenset[str]:
        """Build the router image and return the digests that its private metadata file holds.

        The image digest is required; the config digest is optional for older Docker exporters. Each present
        digest is ``sha256:<64 hex>``; one is the image's local ID (``built_image_id``). The build gets the owner's
        environment, never ``compose_environment``. Its plain progress goes to ``build_log``, created with mode
        0600 (it can still show Docker or proxy details); errors name the file.

        Raises:
            ContainerCommandError: The build could not start, failed, timed out or wrote no valid digests.
            InterruptedError: ``stop_requested`` returned true; the build's process group was stopped.
        """
        # A private directory (mode 0700) for the metadata file, which the build itself writes.
        with tempfile.TemporaryDirectory(prefix="reef-sr-build-") as temporary:
            metadata_file = Path(temporary) / "metadata.json"
            arguments = image_build_arguments(str(metadata_file), IMAGE_BUILD_RECIPE, IMAGE_CACHE_TAG)
            try:
                log = os.open(build_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            except OSError as exc:
                raise ContainerCommandError(f"cannot write the image build log {build_log}: {exc.strerror}") from None
            try:
                os.fchmod(log, 0o600)
                command = GuardedCommand([self.docker_command, *arguments], timeout_seconds, output=log)
            except OSError as exc:
                raise ContainerCommandError("could not start the Docker image build") from exc
            finally:
                os.close(log)
            started = time.monotonic()
            try:
                deadline = started + timeout_seconds + 5
                while command.process.poll() is None:
                    if stop_requested():
                        raise InterruptedError("the Semantic Router image build was stopped")
                    if time.monotonic() >= deadline:
                        raise ContainerCommandError(f"the Semantic Router image build did not end; see {build_log}")
                    time.sleep(0.2)
            finally:
                # Ends the build's whole process group unless it already exited.
                command.close()
            if command.process.returncode != 0:
                # At its timeout the guard kills the build's process group, itself included.
                if time.monotonic() - started >= timeout_seconds:
                    raise ContainerCommandError(
                        f"the Semantic Router image build did not finish within {timeout_seconds} seconds "
                        f"(reef.routing_deployment.image_build_timeout_seconds); see {build_log}"
                    )
                raise ContainerCommandError(
                    f"the Semantic Router image build failed with status {command.process.returncode}; see {build_log}"
                )
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                metadata = None
        values = [metadata.get(BUILD_IMAGE_DIGEST_KEY)] if isinstance(metadata, dict) else [None]
        if isinstance(metadata, dict) and BUILD_CONFIG_DIGEST_KEY in metadata:
            values.append(metadata[BUILD_CONFIG_DIGEST_KEY])
        digests: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not IMAGE_ID.fullmatch(value):
                raise ContainerCommandError(
                    f"the Semantic Router image build wrote no valid image digests to its metadata; see {build_log}"
                )
            digests.add(value)
        return frozenset(digests)

    def inspect(self, ownership: DeploymentOwnership, timeout_seconds: float = 5) -> ContainerInventory:
        deadline = time.monotonic() + timeout_seconds
        resources: dict[str, list[dict[str, object]]] = {}
        for kind in ("container", "network"):
            resource_ids: set[str] = set()
            for label in (
                f"com.docker.compose.project={ownership.project_name}",
                f"io.reef.routing.deployment={ownership.deployment_id}",
            ):
                arguments = [kind, "ls", "--quiet", "--no-trunc", "--filter", f"label={label}"]
                if kind == "container":
                    arguments.append("--all")
                resource_ids.update(self.run(arguments, deadline - time.monotonic()).split())
            if not resource_ids:
                resources[kind] = []
                continue
            try:
                data = json.loads(self.run([kind, "inspect", *sorted(resource_ids)], deadline - time.monotonic()))
            except json.JSONDecodeError:
                raise ContainerCommandError("invalid Docker inspect JSON") from None
            if not isinstance(data, list):
                raise ContainerCommandError("invalid Docker inspect response")
            inspected: list[dict[str, object]] = []
            for resource in data:
                if not isinstance(resource, dict):
                    raise ContainerCommandError("invalid Docker resource response")
                if kind == "container":
                    config = resource.get("Config")
                    if not isinstance(config, dict):
                        raise ContainerCommandError("invalid Docker container configuration")
                    labels = config.get("Labels")
                else:
                    labels = resource.get("Labels")
                expected = {**ownership.labels, "com.docker.compose.project": ownership.project_name}
                if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected.items()):
                    raise RoutingOwnershipError(f"refusing to adopt or change a foreign routing {kind}")
                inspected.append(resource)
            resources[kind] = inspected
        containers: list[ContainerState] = []
        for resource in resources["container"]:
            state = resource.get("State")
            config = resource.get("Config")
            if not isinstance(state, dict) or not isinstance(config, dict):
                raise ContainerCommandError("invalid Docker container state")
            labels = config.get("Labels")
            if not isinstance(labels, dict):
                raise ContainerCommandError("invalid Docker container labels")
            health = state.get("Health", {})
            if not isinstance(health, dict):
                raise ContainerCommandError("invalid Docker health state")
            resource_id = resource.get("Id")
            service = labels.get("com.docker.compose.service")
            running = state.get("Running")
            paused = state.get("Paused", False)
            health_status = health.get("Status")
            if (
                not isinstance(resource_id, str)
                or service != "semantic-router"
                or not isinstance(running, bool)
                or not isinstance(paused, bool)
                or (health_status is not None and not isinstance(health_status, str))
            ):
                raise ContainerCommandError("unexpected managed routing container")
            containers.append(ContainerState(resource_id, service, running, health_status, paused))
        networks: list[str] = []
        for network in resources["network"]:
            resource_id = network.get("Id")
            if not isinstance(resource_id, str):
                raise ContainerCommandError("invalid Docker network identity")
            networks.append(resource_id)
        return ContainerInventory(tuple(containers), tuple(networks))

    def pending_operation_id(self, ownership: DeploymentOwnership) -> str | None:
        path = ownership.configuration_directory / "docker-mutation.json"
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise RoutingOwnershipError("Docker mutation state must be private and owned by the current user")
            try:
                record = json.load(stream)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RoutingOwnershipError(
                    "invalid Docker mutation state; refusing to forget the unknown result"
                ) from None
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in ownership.labels.items()):
            raise RoutingOwnershipError("Docker mutation state does not match routing ownership")
        operation_id = record.get("operation_id")
        if not isinstance(operation_id, str) or not re.fullmatch(r"[a-f0-9]{32}", operation_id):
            raise RoutingOwnershipError("invalid Docker mutation identity")
        return operation_id

    def mutation_pending(self, ownership: DeploymentOwnership) -> bool:
        return self.pending_operation_id(ownership) is not None

    def mutate(self, arguments: Sequence[str], ownership: DeploymentOwnership, timeout_seconds: float = 5) -> None:
        """Preserve uncertainty across parent death, including a lost CLI reply.

        A previous marker is never cleared by cleanup. An empty inspect is not
        proof that an earlier, interrupted create cannot still reach the daemon.
        """
        already_pending = self.mutation_pending(ownership)
        path = ownership.configuration_directory / "docker-mutation.json"
        operation_id = uuid.uuid4().hex
        if not already_pending:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({**ownership.labels, "operation_id": operation_id, "operation": arguments[0]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_dir(ownership.configuration_directory)
        self.run(arguments, timeout_seconds, mutation=True)
        if not already_pending:
            # Successful return belongs to this serialized command. Failed and
            # interrupted commands leave their marker for explicit reconciliation.
            path.unlink()
            fsync_dir(ownership.configuration_directory)

    def acknowledge_unknown(self, ownership: DeploymentOwnership, operation_id: str) -> None:
        """Clear an unknown operation only after an explicit operator decision.

        Callers must first establish that the daemon has finished prior requests
        (for example, during controlled daemon maintenance). Resource absence
        alone is insufficient; this method must never be an automatic retry.
        """
        if self.pending_operation_id(ownership) != operation_id:
            raise ContainerCommandError("unknown Docker operation changed; inspect and confirm its current identity")
        inventory = self.inspect(ownership)
        if inventory.containers or inventory.network_ids:
            raise ContainerCommandError("remove observed owned resources before acknowledging an unknown operation")
        if self.pending_operation_id(ownership) != operation_id:
            raise ContainerCommandError("unknown Docker operation changed during acknowledgement")
        (ownership.configuration_directory / "docker-mutation.json").unlink()
        fsync_dir(ownership.configuration_directory)

    def start(self, compose_file: Path, ownership: DeploymentOwnership, timeout_seconds: float) -> None:
        if self.mutation_pending(ownership):
            raise ContainerCommandError("prior Docker mutation is unresolved; refusing to start", outcome_unknown=True)
        inventory = self.inspect(ownership)
        if inventory.containers or inventory.network_ids:
            raise ContainerCommandError("routing resources already exist; reconcile before creating a new group")
        # The private Compose file comes from our renderer and already has a validated digest reference.
        image = yaml.safe_load(compose_file.read_text())["services"]["semantic-router"]["image"]
        try:
            labels = json.loads(self.run(["image", "inspect", "--format", "{{json .Config.Labels}}", image]))
        except json.JSONDecodeError:
            raise ContainerCommandError("invalid Docker image revision metadata") from None
        if not isinstance(labels, dict) or labels.get("org.opencontainers.image.revision") != SEMANTIC_ROUTER_REVISION:
            raise ContainerCommandError(
                f"SR image revision must match the pinned upstream commit {SEMANTIC_ROUTER_REVISION}"
            )
        self.mutate(
            [
                "compose",
                "--project-name",
                ownership.project_name,
                "--file",
                str(compose_file),
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
            ],
            ownership,
            timeout_seconds,
        )
        self.inspect(ownership)

    def remove(self, ownership: DeploymentOwnership, stop_timeout_seconds: float) -> None:
        # Verify the entire group before changing any resource. Re-inspect on
        # each iteration so cleanup is repeatable after partial failure.
        inventory = self.inspect(ownership)
        for container in inventory.containers:
            current = self.inspect(ownership)
            if container.resource_id not in {item.resource_id for item in current.containers}:
                continue
            if container.paused:
                self.mutate(["unpause", container.resource_id], ownership)
            if container.running:
                self.mutate(
                    ["stop", "--time", str(math.ceil(stop_timeout_seconds)), container.resource_id],
                    ownership,
                    stop_timeout_seconds + 5,
                )
            # The router image declares an anonymous /app/models volume; --volumes removes this container's
            # anonymous volumes only, so restarts do not accumulate them. Named volumes are never removed.
            self.mutate(["container", "rm", "--force", "--volumes", container.resource_id], ownership)
        current = self.inspect(ownership)
        for resource_id in current.network_ids:
            self.mutate(["network", "rm", resource_id], ownership)
        remaining = self.inspect(ownership)
        if remaining.containers or remaining.network_ids:
            raise ContainerCommandError("routing cleanup incomplete; ownership state retained", outcome_unknown=True)
        if self.mutation_pending(ownership):
            raise ContainerCommandError(
                "observed resources removed, but a prior Docker mutation remains unresolved; "
                "operator must establish daemon quiescence before acknowledging cleanup",
                outcome_unknown=True,
            )

    def events(self, ownership: DeploymentOwnership) -> ContainerEventStream:
        return DockerEventStream(
            [
                self.docker_command,
                "events",
                "--format",
                "{{json .}}",
                "--filter",
                "type=container",
                "--filter",
                f"label=com.docker.compose.project={ownership.project_name}",
                "--filter",
                f"label=io.reef.routing.owner={ownership.owner_id}",
                # A healthcheck creates several exec events per second. Keep
                # the startup queue for lifecycle transitions, not probe noise.
                "--filter",
                "event=die",
                "--filter",
                "event=destroy",
                "--filter",
                "event=pause",
                "--filter",
                "event=health_status",
            ]
        )
