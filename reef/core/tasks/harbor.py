"""One task as a Harbor directory: written whole or not at all, read back and checked against its digest.

A task is the directory Harbor loads (``harbor.models.task.task.Task``). Reef
writes ``task.toml``, with a ``[metadata.reef]`` table that holds the task's
digest and the agent record ids it was made from. Every other file is the
task's own: ``instruction.md``, ``environment/``, ``tests/``, ``solution/``,
``steps/`` or any other file Harbor reads, as text or binary data. The digest
covers every file, so a replay that writes the same task again is a no-op and
a directory edited by hand, or given an extra entry of any kind, is refused.
A task is staged under ``<root>/.staging/`` and renamed into place; that
directory stays, and nothing under it is a task.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import tomllib
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import tomli_w
from harbor.models.task.task import Task as HarborTaskLoader

from reef.core.errors import ReefError

#: The ``version`` every task.toml under ``recipes/`` declares; Harbor reads it as ``schema_version``.
TASK_CONFIG_VERSION = "1.0"
#: The task.toml keys that reef writes itself, so ``HarborTask.config`` cannot set them.
RESERVED_CONFIG_KEYS = ("version", "schema_version", "metadata")
#: In ``HarborTask.document()``, a binary file is ``{BINARY_FILE_KEY: <its bytes, base64 encoded>}``.
BINARY_FILE_KEY = "base64"
STAGING_DIRECTORY = ".staging"
TASK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
REEF_TABLE_KEYS = ("digest", "source_agent_record_ids")
MAX_NAME_COMPONENT_BYTES = 255
MAX_NAME_BYTES = 1024


def primary_reward(rewards: Mapping[str, object]) -> object:
    """Return the ``reward`` entry, the sole entry, or None when no primary reward is named.

    Preserve the selected value; callers validate its type and range.
    """
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


class HarborTaskError(ReefError):
    """A task spec or task directory that Harbor could not run."""


class HarborTaskConflictError(HarborTaskError):
    """The target directory already holds a different task under the same name."""


@dataclass(frozen=True)
class HarborTask:
    """A Harbor task before it is written: every file of its directory and every task.toml value.

    Generators make it. :func:`write_harbor_task` writes it, :func:`read_harbor_task` reads it back, and the
    generator service sends it as :meth:`document`.

    Values, checked at construction:

    * ``name`` matches :data:`TASK_NAME_PATTERN` and does not contain ``..``.
    * ``files`` maps the path of each file of the directory other than ``task.toml``, relative to the directory and
      written as a plain posix path, to its content: its text (``str``) when its bytes are UTF-8, its bytes
      otherwise. Bytes given for UTF-8 content become text, so each content has one form. No path leaves the
      directory, and no two paths fold together on a case folding or normalizing volume.
    * ``config`` holds the task.toml keys other than :data:`RESERVED_CONFIG_KEYS`.
    * ``metadata`` holds the ``[metadata]`` keys other than ``reef``.
    * ``config`` and ``metadata`` read back from task.toml with the same JSON form, so they hold no TOML dates.
    * ``source_agent_record_ids`` is a tuple of distinct non-empty strings.
    * Harbor's own task loader loads the directory that :meth:`directory_files` lays out. Reef keeps no rule of its
      own about what a Harbor task needs (an instruction, a verifier script, an image, config values).
    """

    name: str
    files: Mapping[str, str | bytes]
    config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    source_agent_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TASK_NAME_PATTERN.fullmatch(self.name) or ".." in self.name:
            raise HarborTaskError(
                f"task name {self.name!r} must match {TASK_NAME_PATTERN.pattern} and never contain '..'"
            )
        object.__setattr__(self, "files", checked_files(self.files))
        object.__setattr__(self, "config", checked_table("config", self.config))
        reserved = sorted(key for key in self.config if key in RESERVED_CONFIG_KEYS)
        if reserved:
            raise HarborTaskError(f"config must not set {', '.join(reserved)}: reef writes them")
        object.__setattr__(self, "metadata", checked_table("metadata", self.metadata))
        if "reef" in self.metadata:
            raise HarborTaskError("metadata.reef is written by reef; put your own keys elsewhere")
        record_ids = self.source_agent_record_ids
        if (
            not isinstance(record_ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise HarborTaskError("source_agent_record_ids must be a tuple of distinct non-empty strings")
        try:
            self.task_toml().encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"task.toml is not valid Unicode text: {exc.reason}") from exc
        # Harbor decides what a task is: its own loader reads the directory this value lays out.
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / self.name
            directory.mkdir()
            write_task_files(self, directory)
            try:
                HarborTaskLoader(directory)
            except (OSError, ValueError, RuntimeError) as exc:
                raise HarborTaskError(f"Harbor cannot load the task {self.name!r}: {exc}") from exc

    def document(self) -> dict[str, object]:
        """Return the fields of the task as JSON values.

        A text file is its string. A binary file is ``{BINARY_FILE_KEY: <its bytes, base64 encoded>}``.
        """
        return {
            "name": self.name,
            "files": {
                path: content if isinstance(content, str) else {BINARY_FILE_KEY: base64.b64encode(content).decode()}
                for path, content in self.files.items()
            },
            "config": dict(self.config),
            "metadata": dict(self.metadata),
            "source_agent_record_ids": list(self.source_agent_record_ids),
        }

    @property
    def digest(self) -> str:
        """sha256 over the task's content and its source record ids, the same for the same task however it was built."""
        canonical = json.dumps(self.document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def directory_files(self) -> dict[str, str | bytes]:
        """Every file the directory holds, ``task.toml`` included, as {relative posix path: content}."""
        return {"task.toml": self.task_toml(), **self.files}

    def task_toml(self) -> str:
        """The task.toml text: the version, the metadata with reef's digest and source ids, then the config keys."""
        document: dict[str, object] = {
            "version": TASK_CONFIG_VERSION,
            "metadata": {
                **self.metadata,
                "reef": {"digest": self.digest, "source_agent_record_ids": list(self.source_agent_record_ids)},
            },
            **self.config,
        }
        return tomli_w.dumps(document)


def write_harbor_task(task: HarborTask, root: Path) -> Path:
    """Write ``task`` under ``root/<name>`` atomically; the same task again is a no-op, a different one a conflict."""
    root = Path(root)
    target = root / task.name
    if os.path.lexists(target):
        require_same_task(task, target)
        return target
    # A plain mkdir, not mkdtemp: the directory keeps the umask mode it will be published with. The staging
    # parent is never removed, so two writers cannot pull it out from under each other.
    staging = root / STAGING_DIRECTORY / f"{task.name}.{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        write_task_files(task, staging)
        # What the filesystem kept must be what was written: a case folding or normalizing volume merges names.
        staged_files, _ = read_all_entries(staging)
        if staged_files != task.directory_files():
            raise HarborTaskError(f"{staging} does not hold the files written to it; is the volume case folding?")
        try:
            os.rename(staging, target)
        except OSError:
            if not os.path.lexists(target):
                raise
            # A concurrent writer won the rename: accept its directory only if it holds the same task.
            require_same_task(task, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def task_instruction(path: Path) -> str:
    """The text of ``instruction.md`` in any Harbor task directory, written by Reef or not; ``HarborTaskError``
    when ``path`` holds no ``task.toml`` or no ``instruction.md``."""
    path = Path(path)
    for name in ("task.toml", "instruction.md"):
        if not (path / name).is_file():
            raise HarborTaskError(f"{path / name} is missing")
    try:
        return (path / "instruction.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"cannot read {path / 'instruction.md'}: {exc}") from exc


def read_harbor_task(path: Path) -> HarborTask:
    """Read a directory written by :func:`write_harbor_task` back, refusing one whose content no longer matches its digest."""
    path = Path(path)
    if not path.is_dir():
        raise HarborTaskError(f"{path} is not a task directory")
    # The parent resolved physically, the last component kept as written: a symlink alias keeps its own name,
    # "." gains one, and a path ending in ".." names the directory it lands in.
    name = Path(os.path.normpath(os.path.join(os.path.realpath(path.parent), path.name))).name
    files, directories = read_all_entries(path)
    task_toml = files.get("task.toml")
    if task_toml is None:
        raise HarborTaskError(f"{path / 'task.toml'} is missing")
    if not isinstance(task_toml, str):
        raise HarborTaskError(f"{path / 'task.toml'} is not UTF-8 text")
    try:
        document = tomllib.loads(task_toml)
    except tomllib.TOMLDecodeError as exc:
        raise HarborTaskError(f"{path / 'task.toml'} is not valid TOML: {exc}") from exc
    if document.get("version") != TASK_CONFIG_VERSION:
        raise HarborTaskError(f"{path / 'task.toml'} must declare version = {TASK_CONFIG_VERSION!r}")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("reef"), dict):
        raise HarborTaskError(f"{path / 'task.toml'} carries no [metadata.reef] table; reef did not write it")
    reef_table = metadata["reef"]
    record_ids = reef_table.get("source_agent_record_ids")
    if sorted(reef_table) != sorted(REEF_TABLE_KEYS) or not isinstance(record_ids, list):
        raise HarborTaskError(f"{path / 'task.toml'} metadata.reef must hold exactly {' and '.join(REEF_TABLE_KEYS)}")
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise HarborTaskError(f"{path / 'task.toml'} metadata.reef.source_agent_record_ids must hold strings")
    # A directory is reef's only when it holds one of the hashed files below it.
    parents_of_files = {str(parent) for relative in files for parent in PurePosixPath(relative).parents} - {"."}
    empty = sorted(directory for directory in directories if directory not in parents_of_files)
    if empty:
        raise HarborTaskError(f"{path} holds entries reef did not write: {', '.join(empty)}")
    task = HarborTask(
        name=name,
        files={relative: content for relative, content in files.items() if relative != "task.toml"},
        config={key: value for key, value in document.items() if key not in ("version", "metadata")},
        metadata={key: value for key, value in metadata.items() if key != "reef"},
        source_agent_record_ids=tuple(record_ids),
    )
    if reef_table["digest"] != task.digest:
        raise HarborTaskError(
            f"{path} does not match its digest: it was edited, renamed or read through another name after it was written"
        )
    return task


def read_all_entries(root: Path) -> tuple[dict[str, str | bytes], set[str]]:
    """Every regular file under ``root`` as {relative posix path: content} and every directory seen; any other kind of entry is refused."""
    files: dict[str, str | bytes] = {}
    directories: set[str] = set()
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda scanned: scanned.name):
                    mode = entry.stat(follow_symlinks=False).st_mode
                    relative = Path(entry.path).relative_to(root).as_posix()
                    if stat.S_ISDIR(mode):
                        directories.add(relative)
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(mode):
                        files[relative] = file_content(Path(entry.path).read_bytes())
                    else:
                        raise HarborTaskError(
                            f"{root / relative} is not a regular file or directory; reef wrote neither"
                        )
    except OSError as exc:
        raise HarborTaskError(f"{root} cannot be read: {exc}") from exc
    return files, directories


def file_content(data: bytes) -> str | bytes:
    """The one form of a file's content: its text when ``data`` is UTF-8, else ``data`` itself."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data


def checked_files(files: object) -> dict[str, str | bytes]:
    """The task's files as {relative posix path: content in its one form (:func:`file_content`)}; the paths must
    stay inside the task directory, leave ``task.toml`` to reef, and never collide."""
    if not isinstance(files, Mapping):
        raise HarborTaskError("files must map relative file paths to text or bytes")
    checked: dict[str, str | bytes] = {}
    folded_files: set[str] = set()
    folded_directories: dict[str, str] = {}
    for name, content in files.items():
        if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
            raise HarborTaskError(f"file name {name!r} must be a relative posix path without control characters")
        pure = PurePosixPath(name)
        if pure.is_absolute() or not pure.parts or any(part in (".", "..") for part in pure.parts):
            raise HarborTaskError(f"file name {name!r} must stay inside the task directory")
        if str(pure) != name:
            raise HarborTaskError(f"file name {name!r} must be written plainly, as {str(pure)!r}")
        try:
            encoded_name = name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"file name {name!r} is not valid Unicode text: {exc.reason}") from exc
        if len(encoded_name) > MAX_NAME_BYTES or any(
            len(part.encode("utf-8")) > MAX_NAME_COMPONENT_BYTES for part in pure.parts
        ):
            raise HarborTaskError(f"file name {name!r} is longer than a filesystem allows")
        if isinstance(content, bytes):
            content = file_content(content)
        elif not isinstance(content, str):
            raise HarborTaskError(f"{name} must be text or bytes")
        else:
            try:
                content.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise HarborTaskError(f"{name} is not valid Unicode text: {exc.reason}") from exc
        # Two spellings one filesystem may merge (case, Unicode normalization) would leave one of them unwritten,
        # whether they name two files, a file and a directory, or two directories. task.toml is reef's.
        folded_name = folded(name)
        clash = folded_name in folded_files or folded_name in folded_directories or folded_name == "task.toml"
        for parent in pure.parents:
            if str(parent) == ".":
                continue
            folded_parent = folded(str(parent))
            clash = (
                clash
                or folded_parent in folded_files
                or folded_parent == "task.toml"
                or folded_directories.get(folded_parent, str(parent)) != str(parent)
            )
            folded_directories.setdefault(folded_parent, str(parent))
        if clash:
            raise HarborTaskError(
                f"files name {name!r} twice, under a spelling a filesystem may fold together, or as reef's task.toml"
            )
        checked[name] = content
        folded_files.add(folded_name)
    return checked


def folded(name: str) -> str:
    """The spelling a case folding, normalizing filesystem would keep for ``name``."""
    return unicodedata.normalize("NFC", name).casefold()


def checked_table(label: str, table: object) -> dict[str, object]:
    """``table`` as a dict, when its keys are non-empty strings and it reads back from task.toml with the same JSON
    form, the form that the digest hashes."""
    if not isinstance(table, Mapping) or any(not isinstance(key, str) or not key for key in table):
        raise HarborTaskError(f"{label} must be a table with string keys")
    checked = dict(table)
    try:
        written = json.dumps(checked, sort_keys=True)
        read_back = json.dumps(tomllib.loads(tomli_w.dumps({label: checked}))[label], sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HarborTaskError(f"{label} must hold strings, numbers, booleans, lists and tables only: {exc}") from exc
    if read_back != written:
        raise HarborTaskError(f"{label} does not read back from task.toml as it was written")
    return checked


def write_task_files(task: HarborTask, root: Path) -> None:
    """Lay the task's files out under the existing directory ``root``."""
    for relative, content in task.directory_files().items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not text mode: the bytes on disk are the bytes hashed, carriage returns included.
        target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)


def require_same_task(task: HarborTask, target: Path) -> None:
    """Accept an existing ``target`` only when it holds exactly ``task``; anything else is a conflict."""
    try:
        if not target.is_dir() or target.is_symlink():
            raise HarborTaskError(f"{target} is not a directory")
        existing = read_harbor_task(target)
    except HarborTaskError as exc:
        raise HarborTaskConflictError(f"{target} exists and is not a task reef wrote: {exc}") from exc
    except OSError as exc:
        raise HarborTaskConflictError(f"{target} exists and cannot be read: {exc}") from exc
    if existing.digest != task.digest:
        raise HarborTaskConflictError(f"{target} already holds a different task with the same name")
