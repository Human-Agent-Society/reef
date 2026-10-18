"""Docker boundary for the one Semantic Router container of a managed routing deployment."""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.runtime.recovery import fsync_dir
from reef.service.deploy.routing_registry import DeploymentOwnership, RoutingOwnershipError


class ContainerCommandError(RuntimeError):
    """The CLI failed; a submitted mutation may still have reached the daemon."""

    def __init__(self, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class ContainerState:
    resource_id: str
    service: str
    running: bool
    health: str | None
    paused: bool = False


@dataclass(frozen=True)
class ContainerInventory:
    containers: tuple[ContainerState, ...] = ()
    network_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContainerEvent:
    container_id: str
    action: str


class ContainerEventStream(ABC):
    @abstractmethod
    def poll(self, timeout_seconds: float = 0) -> ContainerEvent | None:
        """Return an event; EOF is a stream_disconnected event, not silence."""

    @abstractmethod
    def close(self) -> None:
        """Stop and reap the CLI reader within a bounded time."""


class ContainerRuntime(ABC):
    #: Variables Compose substitutes into containers, such as the router management token.
    compose_environment: dict[str, str]

    def use_environment(self, values: Mapping[str, str]) -> None:
        """Pass private values to later Compose commands through their environment only."""
        self.compose_environment = dict(values)

    @abstractmethod
    def ensure_available(self) -> None:
        """Validate local daemon access and Compose availability."""

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
    """An isolated CLI group whose lifetime is bounded by its caller's pipe."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        discard_stderr: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.owner_lock = threading.Lock()
        read_descriptor, write_descriptor = os.pipe()
        self.owner_descriptor: int | None = write_descriptor
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("routing_command.py")),
                    str(read_descriptor),
                    str(timeout_seconds),
                    *command,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if discard_stderr else subprocess.PIPE,
                pass_fds=(read_descriptor,),
                start_new_session=True,
                # Secret values reach Compose through its environment, never its arguments.
                env=None if environment is None else {**os.environ, **environment},
            )
        except OSError:
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
    def __init__(self, docker_command: str = "docker") -> None:
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
