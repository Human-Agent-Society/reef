"""Canonical cumulative witness-ledger helpers."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .submission import (
    MAX_SECRET_ABS,
    MAX_SUBMISSION_BYTES,
    MAX_SUBMISSION_NODES,
    MAX_SUBMISSION_RECORDS,
    MAX_SUBMISSION_SECRET_COMPONENTS,
)
from .strict_json import loads_object


MAX_LEDGER_RECORDS = MAX_SUBMISSION_RECORDS
MAX_LEDGER_BYTES = MAX_SUBMISSION_BYTES
MAX_LEDGER_NODES = MAX_SUBMISSION_NODES
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class LedgerContractError(ValueError):
    """A decoded cumulative ledger violates its public contract."""


@dataclass(frozen=True, slots=True)
class Ledger:
    schema_version: int
    solutions: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        copied = {
            instance_id: tuple(secret)
            for instance_id, secret in self.solutions.items()
        }
        object.__setattr__(
            self,
            "solutions",
            MappingProxyType(dict(sorted(copied.items()))),
        )


@contextmanager
def _ledger_lock_at(parent_fd: int, ledger_name: str) -> Iterator[None]:
    """Hold the sibling lock relative to an already pinned parent."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("platform cannot safely open the ledger lock")
    descriptor = os.open(
        f"{ledger_name}.lock",
        os.O_RDWR
        | os.O_CREAT
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
        dir_fd=parent_fd,
    )
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LedgerContractError("ledger lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def ledger_lock(path: str | Path) -> Iterator[None]:
    """Hold the stable sibling lock for a complete ledger transaction."""

    ledger_path = Path(path)
    parent_fd = _open_parent_directory(ledger_path)
    try:
        with _ledger_lock_at(parent_fd, ledger_path.name):
            yield
    finally:
        os.close(parent_fd)


def _validate_instance_id(instance_id: object) -> str:
    if (
        not isinstance(instance_id, str)
        or _INSTANCE_ID.fullmatch(instance_id) is None
    ):
        raise ValueError("invalid instance ID")
    return instance_id


def _normalize_secret(secret: Sequence[int]) -> tuple[int, ...]:
    try:
        candidate = tuple(secret)
    except TypeError as exc:
        raise ValueError("invalid secret") from exc
    if len(candidate) > MAX_SUBMISSION_SECRET_COMPONENTS or any(
        type(value) is not int or abs(value) > MAX_SECRET_ABS
        for value in candidate
    ):
        raise ValueError("invalid secret")
    return candidate


def _open_regular_readonly(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("platform cannot safely open a ledger")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LedgerContractError("ledger must be a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_regular_readonly_at(parent_fd: int, name: str) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("platform cannot safely open a ledger")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LedgerContractError("ledger must be a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_parent_directory(path: Path) -> int:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise OSError("platform cannot safely open the ledger directory")
    descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise LedgerContractError("ledger parent must be a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _load_ledger_descriptor(descriptor: int) -> Ledger:
    with os.fdopen(descriptor, "rb") as handle:
        data = handle.read(MAX_LEDGER_BYTES + 1)
    document = loads_object(
        data,
        max_bytes=MAX_LEDGER_BYTES,
        max_depth=4,
        max_nodes=MAX_LEDGER_NODES,
    )
    if set(document) != {"schema_version", "solutions"}:
        raise LedgerContractError("invalid ledger fields")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise LedgerContractError("ledger schema_version must be integer 1")
    raw_solutions = document["solutions"]
    if not isinstance(raw_solutions, list):
        raise LedgerContractError("ledger solutions must be a JSON array")
    if len(raw_solutions) > MAX_LEDGER_RECORDS:
        raise LedgerContractError("ledger has too many solution records")

    solutions: dict[str, tuple[int, ...]] = {}
    for raw_record in raw_solutions:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "instance_id",
            "secret",
        }:
            raise LedgerContractError("invalid ledger solution record")
        try:
            instance_id = _validate_instance_id(raw_record["instance_id"])
        except ValueError as exc:
            raise LedgerContractError("invalid ledger instance ID") from exc
        if instance_id in solutions:
            raise LedgerContractError("repeated ledger instance ID")
        raw_secret = raw_record["secret"]
        if not isinstance(raw_secret, list):
            raise LedgerContractError("invalid ledger secret")
        try:
            solutions[instance_id] = _normalize_secret(raw_secret)
        except ValueError as exc:
            raise LedgerContractError("invalid ledger secret") from exc

    return Ledger(schema_version=1, solutions=solutions)


def _load_ledger_at(parent_fd: int, name: str) -> Ledger:
    return _load_ledger_descriptor(_open_regular_readonly_at(parent_fd, name))


def load_ledger(path: str | Path) -> Ledger:
    return _load_ledger_descriptor(_open_regular_readonly(Path(path)))


def _canonical_bytes(ledger: Ledger) -> bytes:
    if type(ledger.schema_version) is not int or ledger.schema_version != 1:
        raise LedgerContractError("ledger schema_version must be integer 1")
    if not isinstance(ledger.solutions, Mapping):
        raise LedgerContractError("invalid ledger solutions")
    if len(ledger.solutions) > MAX_LEDGER_RECORDS:
        raise LedgerContractError("ledger has too many solution records")

    normalized: list[tuple[str, tuple[int, ...]]] = []
    for raw_instance_id, raw_secret in ledger.solutions.items():
        try:
            instance_id = _validate_instance_id(raw_instance_id)
            secret = _normalize_secret(raw_secret)
        except (TypeError, ValueError) as exc:
            raise LedgerContractError("invalid ledger solution record") from exc
        normalized.append((instance_id, secret))
    normalized.sort(key=lambda item: item[0])
    document = {
        "schema_version": 1,
        "solutions": [
            {"instance_id": instance_id, "secret": list(secret)}
            for instance_id, secret in normalized
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_LEDGER_BYTES:
        raise LedgerContractError("ledger exceeds byte limit")
    return encoded


@dataclass(slots=True)
class _Quarantine:
    directory: Path
    name: str
    parent_fd: int
    outer_fd: int
    inner_fd: int
    traversal_available: bool = True


def _open_prepared_quarantine(
    destination: Path,
    parent_fd: int,
    name: str,
) -> _Quarantine:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise OSError("platform cannot safely prepare ledger cleanup")
    quarantine_directory = destination.parent / name
    outer_fd: int | None = None
    inner_fd: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        outer_fd = os.open(
            name,
            directory_flags,
            dir_fd=parent_fd,
        )
        outer = os.fstat(outer_fd)
        if (
            not stat.S_ISDIR(outer.st_mode)
            or stat.S_IMODE(outer.st_mode) != 0o700
        ):
            raise OSError("ledger quarantine outer directory is not private")
        os.mkdir("private", mode=0o700, dir_fd=outer_fd)
        inner_fd = os.open("private", directory_flags, dir_fd=outer_fd)
        inner = os.fstat(inner_fd)
        if (
            not stat.S_ISDIR(inner.st_mode)
            or stat.S_IMODE(inner.st_mode) != 0o700
        ):
            raise OSError("ledger quarantine inner directory is not private")
        quarantine = _Quarantine(
            directory=quarantine_directory,
            name=name,
            parent_fd=parent_fd,
            outer_fd=outer_fd,
            inner_fd=inner_fd,
        )
        outer_fd = None
        inner_fd = None
        return quarantine
    except BaseException:
        if inner_fd is not None:
            try:
                os.close(inner_fd)
            except OSError:
                pass
        if outer_fd is not None:
            try:
                os.rmdir("private", dir_fd=outer_fd)
            except OSError:
                pass
            try:
                os.close(outer_fd)
            except OSError:
                pass
        try:
            os.rmdir(name, dir_fd=parent_fd)
        except OSError:
            pass
        raise


def _prepare_quarantine(destination: Path, parent_fd: int) -> _Quarantine:
    """Create the standalone writer's private cleanup namespace."""

    quarantine_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.quarantine.",
            dir=destination.parent,
        )
    )
    return _open_prepared_quarantine(
        destination,
        parent_fd,
        quarantine_directory.name,
    )


def _prepare_quarantine_at(
    destination: Path, parent_fd: int
) -> _Quarantine:
    """Create the transaction cleanup namespace without resolving its parent."""

    prefix = f".{destination.name}.quarantine."
    for _attempt in range(128):
        name = f"{prefix}{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return _open_prepared_quarantine(destination, parent_fd, name)
    raise FileExistsError("unable to allocate ledger quarantine")


def _close_quarantine(quarantine: _Quarantine) -> None:
    if not quarantine.traversal_available:
        try:
            os.fchmod(quarantine.outer_fd, 0o700)
            quarantine.traversal_available = True
        except OSError:
            pass
    try:
        os.close(quarantine.inner_fd)
    except OSError:
        pass
    private_removed = False
    if quarantine.traversal_available:
        try:
            os.rmdir("private", dir_fd=quarantine.outer_fd)
        except OSError:
            pass
        else:
            private_removed = True
    try:
        os.close(quarantine.outer_fd)
    except OSError:
        pass
    if private_removed:
        try:
            os.rmdir(quarantine.name, dir_fd=quarantine.parent_fd)
        except OSError:
            pass


def _quarantine_owned_temp(
    quarantine: _Quarantine,
    identity: tuple[int, int] | None,
) -> None:
    """Remove only the writer-owned inode from the private namespace.

    The temporary entry is created directly inside the descriptor-held inner
    directory. Pathname traversal through its outer directory is revoked before
    the entry is inspected or written. Cleanup opens the entry with
    ``O_NOFOLLOW`` and compares its descriptor identity before unlinking it
    relative to the inner directory descriptor. Any missing, unverifiable, or
    foreign entry is retained in the quarantine for recovery.

    This portable boundary excludes a process that already holds the freshly
    created inner directory descriptor or can change quarantine permissions.
    It prevents ordinary pathname races on both macOS and the Linux judge
    without relying on platform-specific conditional-unlink operations.
    """

    if identity is None:
        return
    entry_fd: int | None = None
    try:
        if quarantine.traversal_available:
            try:
                os.fchmod(quarantine.outer_fd, 0)
            except OSError:
                return
            quarantine.traversal_available = False
        try:
            entry_flags = (
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            entry_fd = os.open(
                "entry", entry_flags, dir_fd=quarantine.inner_fd
            )
            quarantined = os.fstat(entry_fd)
        except OSError:
            return
        if (
            (quarantined.st_dev, quarantined.st_ino) == identity
            and stat.S_ISREG(quarantined.st_mode)
        ):
            try:
                os.unlink("entry", dir_fd=quarantine.inner_fd)
            except OSError:
                return
    finally:
        if entry_fd is not None:
            try:
                os.close(entry_fd)
            except OSError:
                pass


def _verify_owned_temp(
    quarantine: _Quarantine, identity: tuple[int, int]
) -> int:
    """Open and pin the verified private source through replacement."""

    entry_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    entry_fd = os.open("entry", entry_flags, dir_fd=quarantine.inner_fd)
    try:
        metadata = os.fstat(entry_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise OSError("ledger temporary file identity changed")
    except BaseException:
        try:
            os.close(entry_fd)
        except OSError:
            pass
        raise
    return entry_fd


def _write_ledger_bytes_at(
    destination: Path,
    parent_fd: int,
    data: bytes,
    *,
    descriptor_quarantine: bool,
) -> None:
    quarantine: _Quarantine | None = None
    temporary_name: str | None = None
    temporary_fd: int | None = None
    writer_fd: int | None = None
    verified_fd: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        try:
            destination_fd = os.open(
                destination.name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            pass
        else:
            try:
                if not stat.S_ISREG(os.fstat(destination_fd).st_mode):
                    raise LedgerContractError(
                        "ledger must be a regular file"
                    )
            finally:
                os.close(destination_fd)
        prepare_quarantine = (
            _prepare_quarantine_at
            if descriptor_quarantine
            else _prepare_quarantine
        )
        quarantine = prepare_quarantine(destination, parent_fd)
        os.fchmod(quarantine.outer_fd, 0)
        quarantine.traversal_available = False
        temporary_fd = os.open(
            "entry",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=quarantine.inner_fd,
        )
        temporary_name = "entry"
        created = os.fstat(temporary_fd)
        if not stat.S_ISREG(created.st_mode):
            raise OSError("ledger temporary file is not regular")
        temporary_identity = (created.st_dev, created.st_ino)
        writer_fd = os.dup(temporary_fd)
        with os.fdopen(writer_fd, "wb") as handle:
            writer_fd = None
            data_view = memoryview(data)
            offset = 0
            while offset < len(data_view):
                written = handle.write(data_view[offset:])
                remaining = len(data_view) - offset
                if (
                    type(written) is not int
                    or written <= 0
                    or written > remaining
                ):
                    raise OSError("unable to write complete ledger")
                offset += written
            handle.flush()
            os.fsync(handle.fileno())
        verified_fd = _verify_owned_temp(quarantine, temporary_identity)
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=quarantine.inner_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    except BaseException:
        if writer_fd is not None:
            try:
                os.close(writer_fd)
            except OSError:
                pass
        if temporary_name is not None and quarantine is not None:
            _quarantine_owned_temp(
                quarantine,
                temporary_identity,
            )
        raise
    finally:
        if verified_fd is not None:
            try:
                os.close(verified_fd)
            except OSError:
                pass
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if quarantine is not None:
            _close_quarantine(quarantine)


def _write_ledger_atomic_at(
    destination: Path, parent_fd: int, ledger: Ledger
) -> None:
    _write_ledger_bytes_at(
        destination,
        parent_fd,
        _canonical_bytes(ledger),
        descriptor_quarantine=True,
    )


def write_ledger_atomic(path: str | Path, ledger: Ledger) -> None:
    destination = Path(path)
    data = _canonical_bytes(ledger)
    parent_fd = _open_parent_directory(destination)
    try:
        _write_ledger_bytes_at(
            destination,
            parent_fd,
            data,
            descriptor_quarantine=False,
        )
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass


def merge_witness(
    ledger: Ledger,
    *,
    instance_id: str,
    secret: Sequence[int],
    replace: bool = False,
) -> Ledger:
    instance_id = _validate_instance_id(instance_id)
    candidate = _normalize_secret(secret)
    if not replace and ledger.solutions.get(instance_id) == candidate:
        return ledger
    if not replace and instance_id in ledger.solutions:
        raise ValueError("instance already has a different witness")
    if (
        instance_id not in ledger.solutions
        and len(ledger.solutions) >= MAX_LEDGER_RECORDS
    ):
        raise ValueError("ledger may contain at most 200 witnesses")
    solutions = dict(ledger.solutions)
    solutions[instance_id] = candidate
    return Ledger(schema_version=ledger.schema_version, solutions=solutions)


def merge_witness_transaction(
    path: str | Path,
    *,
    instance_id: str,
    secret: Sequence[int],
    replace: bool = False,
) -> Ledger:
    """Merge and persist while pinning one parent namespace throughout."""

    destination = Path(path)
    parent_fd = _open_parent_directory(destination)
    try:
        with _ledger_lock_at(parent_fd, destination.name):
            current = _load_ledger_at(parent_fd, destination.name)
            updated = merge_witness(
                current,
                instance_id=instance_id,
                secret=secret,
                replace=replace,
            )
            _write_ledger_atomic_at(destination, parent_fd, updated)
            return updated
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass
