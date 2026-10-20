"""Private, host-local ownership and locking for a managed routing deployment."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from reef.runtime.recovery import write_json


class RoutingOwnershipError(RuntimeError):
    """A resource or registry does not belong to this managed deployment."""


@dataclass(frozen=True)
class DeploymentOwnership:
    deployment_id: str
    owner_id: str
    generation: str
    configuration_directory: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", self.deployment_id):
            raise ValueError("routing deployment_id must be 1-48 lowercase letters, digits or hyphens")
        if not re.fullmatch(r"[a-f0-9]{32}", self.owner_id):
            raise ValueError("routing owner_id must be a UUID hex string")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", self.generation):
            raise ValueError("routing generation must be 1-64 lowercase letters, digits or hyphens")
        if not self.configuration_directory.is_absolute():
            raise ValueError("routing configuration_directory must be absolute")

    @property
    def project_name(self) -> str:
        return f"reef-{self.deployment_id}-{self.generation}"

    @property
    def labels(self) -> dict[str, str]:
        return {
            "io.reef.routing.deployment": self.deployment_id,
            "io.reef.routing.owner": self.owner_id,
            "io.reef.routing.generation": self.generation,
        }


class DeploymentRegistry:
    """Hold one flock in the orchestrator and save ownership before Docker changes.

    The registry path is independent of the run directory. The lock descriptor
    is non-inheritable, and mutation methods reject calls from a forked child.
    """

    def __init__(self, deployment_id: str, root: Path | None = None) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", deployment_id):
            raise ValueError("invalid routing deployment_id")
        if root is None:
            state_home = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
            if not state_home.is_absolute():
                raise ValueError("XDG_STATE_HOME must be an absolute path")
            root = state_home / "reef/routing"
        self.deployment_id = deployment_id
        self.directory = root / deployment_id
        self.lock_descriptor: int | None = None
        self.lock_owner_pid: int | None = None

    def acquire(self) -> None:
        # Delay the POSIX-only import so this module can be imported on other platforms.
        import fcntl

        if self.lock_descriptor is not None:
            self.require_lock()
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = self.directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise RoutingOwnershipError("routing registry must be a real directory owned by the current user")
        self.directory.chmod(0o700)
        descriptor = os.open(self.directory / "lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise RoutingOwnershipError("routing registry lock must be a regular file owned by the current user")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RoutingOwnershipError(f"routing deployment {self.deployment_id!r} is already controlled") from exc
        except (OSError, RoutingOwnershipError):
            os.close(descriptor)
            raise
        self.lock_descriptor = descriptor
        self.lock_owner_pid = os.getpid()

    def require_lock(self) -> None:
        if self.lock_descriptor is None or self.lock_owner_pid != os.getpid():
            raise RoutingOwnershipError("routing registry mutation requires the orchestrator's deployment lock")

    def load(self) -> DeploymentOwnership | None:
        self.require_lock()
        return self.read_owner()

    def read_owner(self) -> DeploymentOwnership | None:
        """Read an atomic ownership snapshot without taking the controller lock.

        This is suitable for status only. Mutation and acknowledgement still
        require a locked registry and fresh Docker ownership checks.
        """
        try:
            directory_metadata = self.directory.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(directory_metadata.st_mode) or directory_metadata.st_uid != os.getuid():
            raise RoutingOwnershipError("routing registry must be a real directory owned by the current user")
        try:
            descriptor = os.open(self.directory / "owner.json", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise RoutingOwnershipError("routing owner state must be private and owned by the current user")
            try:
                data = json.load(stream)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RoutingOwnershipError("invalid routing ownership JSON; refusing to adopt resources") from None
        if not isinstance(data, dict) or set(data) != {
            "deployment_id",
            "owner_id",
            "generation",
            "configuration_directory",
        }:
            raise RoutingOwnershipError("invalid routing ownership state; refusing to adopt resources")
        if not all(isinstance(value, str) for value in data.values()) or data["deployment_id"] != self.deployment_id:
            raise RoutingOwnershipError("routing ownership state does not match this deployment")
        try:
            return DeploymentOwnership(
                deployment_id=data["deployment_id"],
                owner_id=data["owner_id"],
                generation=data["generation"],
                configuration_directory=Path(data["configuration_directory"]),
            )
        except ValueError:
            raise RoutingOwnershipError("invalid routing ownership fields; refusing to adopt resources") from None

    def prepare(self, generation: str, configuration_directory: Path) -> DeploymentOwnership:
        previous = self.load()
        owner_id = uuid.uuid4().hex if previous is None else previous.owner_id
        ownership = DeploymentOwnership(self.deployment_id, owner_id, generation, configuration_directory.resolve())
        self.save(ownership)
        return ownership

    def save(self, ownership: DeploymentOwnership) -> None:
        self.require_lock()
        if ownership.deployment_id != self.deployment_id:
            raise RoutingOwnershipError("cannot save another routing deployment's ownership")
        previous = self.load()
        if previous is not None and previous.owner_id != ownership.owner_id:
            raise RoutingOwnershipError("cannot replace the stable routing owner_id")
        data = asdict(ownership)
        data["configuration_directory"] = str(ownership.configuration_directory)
        # An owner-only temporary file replaces owner.json; the file and the directory entry are synced.
        write_json(self.directory / "owner.json", data)

    def release(self) -> None:
        if self.lock_descriptor is None:
            return
        import fcntl

        self.require_lock()
        fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
        os.close(self.lock_descriptor)
        self.lock_descriptor = None
        self.lock_owner_pid = None

    def __enter__(self) -> DeploymentRegistry:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
