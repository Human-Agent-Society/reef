"""One task as a Harbor directory: written whole or not at all, read back and checked against its digest.

The layout is the one Harbor 0.23 loads (``harbor.models.task.paths.TaskPaths``)
and every example under ``recipes/`` ships::

    <name>/
      task.toml            version, [metadata], [verifier], [agent], [environment]
      instruction.md       the prompt shown to the model
      environment/         Dockerfile and whatever the image needs
      tests/test.sh        the verifier, with any helper it calls beside it
      solution/            optional reference files

``task.toml`` carries a ``[metadata.reef]`` table with the task's digest and
the agent record ids it was made from, and when set, the digests of its
parents and ``is_imported``. The digest covers every file, so a replay that
writes the same task again is a no-op and a directory edited by hand, or
given an extra entry of any kind, is refused. A task is staged under
``<root>/.staging/`` and renamed into place; that directory stays, and nothing
under it is a task. :func:`import_harbor_task` stamps a Harbor task reef did
not write; its ``task.toml`` may carry every key Harbor 0.20.0 reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tomllib
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network
from pathlib import Path, PurePosixPath

import tomli_w

from reef.core.errors import ReefError

#: The ``version`` every task.toml under ``recipes/`` declares; Harbor reads it as ``schema_version``.
TASK_CONFIG_VERSION = "1.0"
NETWORK_MODES = ("no-network", "public", "allowlist")
STAGING_DIRECTORY = ".staging"
TASK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
SIZE_KEYS = ("cpus", "memory_mb", "storage_mb", "gpus")
#: The task.toml tables reef writes and, per table, the keys Harbor 0.23 reads (``harbor.models.task.config``).
KNOWN_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "verifier": ("timeout_sec", "env", "user"),
    "agent": ("timeout_sec", "user"),
    "environment": (
        "build_timeout_sec",
        "docker_image",
        "network_mode",
        "allowed_hosts",
        "env",
        "workdir",
        *SIZE_KEYS,
    ),
}
TOP_LEVEL_KEYS = ("version", "metadata", *KNOWN_CONFIG_KEYS)
#: The task.toml tables and keys Harbor 0.20.0 reads (``harbor.models.task.config.TaskConfig``), for an imported task.
HARBOR_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "task": ("name", "description", "authors", "keywords"),
    "verifier": (
        "timeout_sec",
        "env",
        "user",
        "network_mode",
        "allowed_hosts",
        "environment_mode",
        "environment",
        "collect",
    ),
    "agent": ("timeout_sec", "user", "network_mode", "allowed_hosts"),
    "environment": (
        "build_timeout_sec",
        "docker_image",
        "os",
        *SIZE_KEYS,
        "gpu_types",
        "tpu",
        "mcp_servers",
        "env",
        "skills_dir",
        "healthcheck",
        "workdir",
        "allow_internet",
        "network_mode",
        "allowed_hosts",
    ),
    "solution": ("env",),
}
HARBOR_TOP_LEVEL_KEYS = ("source", "artifacts", "multi_step_reward_strategy", "steps")
IMPORTED_TOP_LEVEL_KEYS = ("version", "metadata", *HARBOR_CONFIG_KEYS, *HARBOR_TOP_LEVEL_KEYS)
#: Values whose shape Harbor's own model checks; reef keeps them when both TOML and JSON can hold them.
STRUCTURED_KEYS = (
    "authors",
    "keywords",
    "gpu_types",
    "tpu",
    "mcp_servers",
    "healthcheck",
    "collect",
    "artifacts",
    "steps",
)
KEY_CHOICES: dict[str, tuple[str, ...]] = {
    "network_mode": NETWORK_MODES,
    "environment_mode": ("shared", "separate"),
    "os": ("linux", "windows"),
    "multi_step_reward_strategy": ("mean", "final"),
}
REEF_TABLE_KEYS = ("digest", "source_agent_record_ids")
#: The [metadata.reef] keys written only when set, so the digest and the task.toml of a task without them hold.
OPTIONAL_REEF_TABLE_KEYS = ("parents", "is_imported")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TREE_DIRECTORIES = ("tests", "environment", "solution")
ROOT_FILES = ("task.toml", "instruction.md")
MAX_NAME_COMPONENT_BYTES = 255
MAX_NAME_BYTES = 1024


class HarborTaskError(ReefError):
    """A task spec or task directory that Harbor could not run."""


class HarborTaskConflict(HarborTaskError):
    """The target directory already holds a different task under the same name."""


@dataclass(frozen=True)
class HarborTask:
    """A task before it is written: everything the directory will hold, validated at construction."""

    name: str
    instruction: str
    tests: Mapping[str, str]
    environment: Mapping[str, str]
    config: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    solution: Mapping[str, str] = field(default_factory=dict)
    source_agent_record_ids: tuple[str, ...] = ()
    #: Digests of the tasks a Designer prompt showed, or of the benchmark item this task came from.
    parents: tuple[str, ...] = ()
    #: Copied by :func:`import_harbor_task`: its task.toml may carry every key Harbor 0.20.0 reads.
    is_imported: bool = False
    #: Harbor's top level task.toml keys outside any table; an imported task's only.
    top_level_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TASK_NAME_PATTERN.fullmatch(self.name) or ".." in self.name:
            raise HarborTaskError(
                f"task name {self.name!r} must match {TASK_NAME_PATTERN.pattern} and never contain '..'"
            )
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise HarborTaskError("instruction must be non-empty text")
        object.__setattr__(self, "tests", checked_files("tests", self.tests))
        if not self.tests.get("test.sh", "").strip():
            raise HarborTaskError("tests/test.sh must be non-empty text: it is the verifier Harbor runs")
        object.__setattr__(self, "environment", checked_files("environment", self.environment))
        object.__setattr__(self, "solution", checked_files("solution", self.solution))
        if not isinstance(self.is_imported, bool):
            raise HarborTaskError("is_imported must be a boolean")
        object.__setattr__(self, "config", checked_config(self.config, is_imported=self.is_imported))
        if "Dockerfile" not in self.environment and "docker_image" not in self.config.get("environment", {}):
            raise HarborTaskError("environment needs a Dockerfile or config environment.docker_image")
        object.__setattr__(
            self, "top_level_config", checked_top_level_config(self.top_level_config, is_imported=self.is_imported)
        )
        object.__setattr__(self, "metadata", checked_metadata(self.metadata))
        record_ids = self.source_agent_record_ids
        if (
            not isinstance(record_ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise HarborTaskError("source_agent_record_ids must be a tuple of distinct non-empty strings")
        parents = self.parents
        if (
            not isinstance(parents, tuple)
            or any(not isinstance(parent, str) or not DIGEST_PATTERN.fullmatch(parent) for parent in parents)
            or len(set(parents)) != len(parents)
        ):
            raise HarborTaskError("parents must be a tuple of distinct task digests (64 lowercase hex digits)")
        # One encode of everything the writer will put on disk: a lone surrogate anywhere fails here, not mid write.
        try:
            self.task_toml().encode("utf-8")
            self.instruction.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"the task is not valid Unicode text: {exc.reason}") from exc

    @property
    def digest(self) -> str:
        """sha256 over the task's content, source record ids and stamp; the same however the task was built."""
        content: dict[str, object] = {
            "name": self.name,
            "instruction": self.instruction,
            "tests": dict(self.tests),
            "environment": dict(self.environment),
            "solution": dict(self.solution),
            "config": {table: dict(values) for table, values in self.config.items()},
            "metadata": dict(self.metadata),
            "source_agent_record_ids": list(self.source_agent_record_ids),
        }
        # Only when set, so the digest of a task written before these fields existed holds.
        if self.parents:
            content["parents"] = list(self.parents)
        if self.is_imported:
            content["is_imported"] = True
        if self.top_level_config:
            content["top_level_config"] = dict(self.top_level_config)
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def files(self) -> dict[str, str]:
        """Every file the directory will hold, as {relative posix path: text}."""
        files = {"task.toml": self.task_toml(), "instruction.md": self.instruction}
        for directory, tree in (("tests", self.tests), ("environment", self.environment), ("solution", self.solution)):
            for name, text in tree.items():
                files[f"{directory}/{name}"] = text
        return files

    def task_toml(self) -> str:
        """The task.toml text: the version, any top level keys, the tables in Harbor's order and reef's stamp."""
        reef_table: dict[str, object] = {
            "digest": self.digest,
            "source_agent_record_ids": list(self.source_agent_record_ids),
        }
        if self.parents:
            reef_table["parents"] = list(self.parents)
        if self.is_imported:
            reef_table["is_imported"] = True
        document: dict[str, object] = {"version": TASK_CONFIG_VERSION, **self.top_level_config}
        document["metadata"] = {**self.metadata, "reef": reef_table}
        for table in HARBOR_CONFIG_KEYS:
            if table in self.config:
                document[table] = dict(self.config[table])
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
        if staged_files != task.files:
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


def read_harbor_task(path: Path) -> HarborTask:
    """Read a directory written by :func:`write_harbor_task` back, refusing one whose content no longer matches its digest."""
    path = Path(path)
    if not path.is_dir():
        raise HarborTaskError(f"{path} is not a task directory")
    files, directories = read_all_entries(path)
    if "task.toml" not in files:
        raise HarborTaskError(f"{path / 'task.toml'} is missing")
    try:
        document = tomllib.loads(files["task.toml"])
    except tomllib.TOMLDecodeError as exc:
        raise HarborTaskError(f"{path / 'task.toml'} is not valid TOML: {exc}") from exc
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("reef"), dict):
        raise HarborTaskError(
            f"{path / 'task.toml'} carries no [metadata.reef] table; reef did not write it; "
            "import it with reef.core.tasks.import_harbor_task"
        )
    reef_table = metadata["reef"]
    is_imported = reef_table.get("is_imported") is True
    if document.get("version") != TASK_CONFIG_VERSION:
        raise HarborTaskError(f"{path / 'task.toml'} must declare version = {TASK_CONFIG_VERSION!r}")
    unknown_tables = sorted(
        key for key in document if key not in (IMPORTED_TOP_LEVEL_KEYS if is_imported else TOP_LEVEL_KEYS)
    )
    if unknown_tables:
        raise HarborTaskError(f"{path / 'task.toml'} carries tables reef did not write: {', '.join(unknown_tables)}")
    record_ids = reef_table.get("source_agent_record_ids")
    parents = reef_table.get("parents", [])
    if (
        not set(REEF_TABLE_KEYS) <= set(reef_table) <= {*REEF_TABLE_KEYS, *OPTIONAL_REEF_TABLE_KEYS}
        or not isinstance(record_ids, list)
        or not isinstance(parents, list)
        or ("is_imported" in reef_table and not is_imported)
    ):
        raise HarborTaskError(
            f"{path / 'task.toml'} metadata.reef must hold exactly {' and '.join(REEF_TABLE_KEYS)}, "
            "and may hold a list of parents and is_imported = true"
        )
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise HarborTaskError(f"{path / 'task.toml'} metadata.reef.source_agent_record_ids must hold strings")
    trees = task_trees(path, files, directories)
    task = HarborTask(
        name=task_directory_name(path),
        instruction=files["instruction.md"],
        tests=trees["tests"],
        environment=trees["environment"],
        config={table: document[table] for table in HARBOR_CONFIG_KEYS if table in document},
        metadata={key: value for key, value in metadata.items() if key != "reef"},
        solution=trees["solution"],
        source_agent_record_ids=tuple(record_ids),
        parents=tuple(parents),
        is_imported=is_imported,
        top_level_config={key: document[key] for key in HARBOR_TOP_LEVEL_KEYS if key in document},
    )
    if reef_table["digest"] != task.digest:
        raise HarborTaskError(
            f"{path} does not match its digest: it was edited, renamed or read through another name after it was written"
        )
    return task


def import_harbor_task(source_path: Path, root: Path, *, parents: tuple[str, ...] = ()) -> Path:
    """Copy the Harbor task at ``source_path`` under ``root`` as a reef task: every file kept, a digest stamped.

    Every task.toml key Harbor 0.20.0 reads is kept, except the schema version: the copy declares reef's. Only text
    files are copied, and only ``tests/test.sh`` is made executable. The copy reads back with :func:`read_harbor_task`
    and is refused there once edited.
    """
    source_path = Path(source_path)
    if not source_path.is_dir():
        raise HarborTaskError(f"{source_path} is not a task directory")
    files, directories = read_all_entries(source_path)
    if "task.toml" not in files:
        raise HarborTaskError(f"{source_path / 'task.toml'} is missing")
    try:
        document = tomllib.loads(files["task.toml"])
    except tomllib.TOMLDecodeError as exc:
        raise HarborTaskError(f"{source_path / 'task.toml'} is not valid TOML: {exc}") from exc
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HarborTaskError(f"{source_path / 'task.toml'} [metadata] must be a table")
    if "reef" in metadata:
        raise HarborTaskError(
            f"{source_path / 'task.toml'} carries a [metadata.reef] table: it is already a reef task"
        )
    unknown_keys = sorted(key for key in document if key not in (*IMPORTED_TOP_LEVEL_KEYS, "schema_version"))
    if unknown_keys:
        raise HarborTaskError(
            f"{source_path / 'task.toml'} carries keys Harbor 0.20.0 does not read: {', '.join(unknown_keys)}"
        )
    trees = task_trees(source_path, files, directories)
    task = HarborTask(
        name=task_directory_name(source_path),
        instruction=files["instruction.md"],
        tests=trees["tests"],
        environment=trees["environment"],
        config={table: document[table] for table in HARBOR_CONFIG_KEYS if table in document},
        metadata=metadata,
        solution=trees["solution"],
        parents=parents,
        is_imported=True,
        top_level_config={key: document[key] for key in HARBOR_TOP_LEVEL_KEYS if key in document},
    )
    return write_harbor_task(task, root)


def task_directory_name(path: Path) -> str:
    """The name of the task directory at ``path``, however the path spells it."""
    # The parent resolved physically, the last component kept as written: a symlink alias keeps its own name,
    # "." gains one, and a path ending in ".." names the directory it lands in.
    return Path(os.path.normpath(os.path.join(os.path.realpath(path.parent), path.name))).name


def task_trees(path: Path, files: Mapping[str, str], directories: set[str]) -> dict[str, dict[str, str]]:
    """The files of each tree directory of the task at ``path``; any other entry but the root files is refused."""
    trees: dict[str, dict[str, str]] = {directory: {} for directory in TREE_DIRECTORIES}
    foreign: list[str] = []
    for relative, text in files.items():
        parts = relative.split("/")
        if len(parts) == 1 and parts[0] in ROOT_FILES:
            continue
        if len(parts) > 1 and parts[0] in TREE_DIRECTORIES:
            trees[parts[0]]["/".join(parts[1:])] = text
        else:
            foreign.append(relative)
    # A directory is reef's only when it is a tree root or holds one of the hashed files below it.
    parents_of_files = {str(parent) for relative in files for parent in PurePosixPath(relative).parents} - {"."}
    foreign.extend(sorted(d for d in directories if d not in TREE_DIRECTORIES and d not in parents_of_files))
    if foreign:
        raise HarborTaskError(f"{path} holds entries reef did not write: {', '.join(sorted(foreign))}")
    if "instruction.md" not in files:
        raise HarborTaskError(f"{path / 'instruction.md'} is missing")
    for directory in ("tests", "environment"):
        if directory not in directories:
            raise HarborTaskError(f"{path / directory} is missing")
    return trees


def read_all_entries(root: Path) -> tuple[dict[str, str], set[str]]:
    """Every regular file under ``root`` as {relative posix path: text} and every directory seen; any other kind of entry is refused."""
    files: dict[str, str] = {}
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
                        files[relative] = read_text_file(Path(entry.path))
                    else:
                        raise HarborTaskError(
                            f"{root / relative} is not a regular file or directory; reef wrote neither"
                        )
    except OSError as exc:
        raise HarborTaskError(f"{root} cannot be read: {exc}") from exc
    return files, directories


def read_text_file(path: Path) -> str:
    """The file's text, read with ``newline=""`` so the bytes hashed are the bytes on disk."""
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError as exc:
        raise HarborTaskError(f"{path} is not UTF-8 text; reef writes text files only") from exc
    except OSError as exc:
        raise HarborTaskError(f"{path} cannot be read: {exc}") from exc


def checked_files(directory: str, files: object) -> dict[str, str]:
    """A directory's files as {relative posix path: text}; the paths must stay inside the directory and never collide."""
    if not isinstance(files, Mapping):
        raise HarborTaskError(f"{directory} must map relative file paths to text")
    checked: dict[str, str] = {}
    folded_files: set[str] = set()
    folded_directories: dict[str, str] = {}
    for name, text in files.items():
        if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
            raise HarborTaskError(
                f"{directory} file name {name!r} must be a relative posix path without control characters"
            )
        pure = PurePosixPath(name)
        if pure.is_absolute() or not pure.parts or any(part in (".", "..") for part in pure.parts):
            raise HarborTaskError(f"{directory} file name {name!r} must stay inside the {directory} directory")
        if str(pure) != name:
            raise HarborTaskError(f"{directory} file name {name!r} must be written plainly, as {str(pure)!r}")
        try:
            encoded_name = name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"{directory} file name {name!r} is not valid Unicode text: {exc.reason}") from exc
        if len(encoded_name) > MAX_NAME_BYTES or any(
            len(part.encode("utf-8")) > MAX_NAME_COMPONENT_BYTES for part in pure.parts
        ):
            raise HarborTaskError(f"{directory} file name {name!r} is longer than a filesystem allows")
        if not isinstance(text, str):
            raise HarborTaskError(f"{directory}/{name} must be text")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise HarborTaskError(f"{directory}/{name} is not valid Unicode text: {exc.reason}") from exc
        # Two spellings one filesystem may merge (case, Unicode normalization) would leave one of them unwritten,
        # whether they name two files, a file and a directory, or two directories.
        folded_name = folded(name)
        clash = folded_name in folded_files or folded_name in folded_directories
        for parent in pure.parents:
            if str(parent) == ".":
                continue
            folded_parent = folded(str(parent))
            clash = (
                clash
                or folded_parent in folded_files
                or folded_directories.get(folded_parent, str(parent)) != str(parent)
            )
            folded_directories.setdefault(folded_parent, str(parent))
        if clash:
            raise HarborTaskError(
                f"{directory} names {name!r} twice, or under a spelling a filesystem may fold together"
            )
        checked[name] = text
        folded_files.add(folded_name)
    return checked


def folded(name: str) -> str:
    """The spelling a case folding, normalizing filesystem would keep for ``name``."""
    return unicodedata.normalize("NFC", name).casefold()


def checked_config(config: object, *, is_imported: bool = False) -> dict[str, dict[str, object]]:
    """The task.toml tables, checked against the keys Harbor reads: reef's set, or Harbor 0.20.0's for an import."""
    if not isinstance(config, Mapping):
        raise HarborTaskError("config must map task.toml table names to their keys")
    known_keys = HARBOR_CONFIG_KEYS if is_imported else KNOWN_CONFIG_KEYS
    checked: dict[str, dict[str, object]] = {}
    for table, values in config.items():
        if table not in known_keys:
            raise HarborTaskError(f"config table {table!r} is not one reef writes ({', '.join(known_keys)})")
        checked[table] = checked_table(f"config.{table}", known_keys[table], values)
    return checked


def checked_table(key_path: str, keys: tuple[str, ...], values: object) -> dict[str, object]:
    """One task.toml table that holds only ``keys``, each value in the form Harbor's own model would accept."""
    if not isinstance(values, Mapping):
        raise HarborTaskError(f"{key_path} must be a table")
    checked: dict[str, object] = {}
    for key, value in values.items():
        if key not in keys:
            raise HarborTaskError(f"{key_path}.{key} is not a key Harbor reads")
        checked[key] = checked_config_value(f"{key_path}.{key}", key, value)
    if checked.get("allowed_hosts") and checked.get("network_mode") != "allowlist":
        raise HarborTaskError(f"{key_path}.allowed_hosts needs network_mode = 'allowlist'")
    return checked


def checked_top_level_config(top_level_config: object, *, is_imported: bool) -> dict[str, object]:
    """Harbor's top level task.toml keys outside any table, which only an imported task carries."""
    if not isinstance(top_level_config, Mapping):
        raise HarborTaskError("top_level_config must map top level task.toml keys to their values")
    if top_level_config and not is_imported:
        raise HarborTaskError("top_level_config belongs to an imported task only; reef writes no top level keys")
    checked: dict[str, object] = {}
    for key, value in top_level_config.items():
        if key not in HARBOR_TOP_LEVEL_KEYS:
            raise HarborTaskError(
                f"top level key {key!r} is not one Harbor reads ({', '.join(HARBOR_TOP_LEVEL_KEYS)})"
            )
        checked[key] = checked_config_value(key, key, value)
    return checked


def checked_config_value(key_path: str, key: str, value: object) -> object:
    """One task.toml value in the form Harbor's own model would accept for ``key``."""
    if key.endswith("timeout_sec"):
        try:
            seconds = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
        except OverflowError:
            seconds = math.inf
        if not math.isfinite(seconds) or seconds <= 0:
            raise HarborTaskError(f"{key_path} must be a positive number of seconds")
        return seconds
    if key in SIZE_KEYS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HarborTaskError(f"{key_path} must be a non-negative integer")
        return value
    if key in KEY_CHOICES:
        if value not in KEY_CHOICES[key]:
            raise HarborTaskError(f"{key_path} must be one of {', '.join(KEY_CHOICES[key])}")
        return value
    if key == "allow_internet":
        if not isinstance(value, bool):
            raise HarborTaskError(f"{key_path} must be a boolean")
        return value
    if key == "environment":
        # The verifier's own container, as Harbor reads it: the same keys as the task's [environment].
        return checked_table(key_path, HARBOR_CONFIG_KEYS["environment"], value)
    if key in STRUCTURED_KEYS:
        try:
            tomli_w.dumps({key: value})
            json.dumps(value, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise HarborTaskError(
                f"{key_path} must hold strings, numbers, booleans, lists and tables only: {exc}"
            ) from exc
        return value
    if key == "allowed_hosts":
        if not isinstance(value, list):
            raise HarborTaskError(f"{key_path} must be a list of host names")
        return [checked_allowed_host(key_path, host) for host in value]
    if key == "env":
        if not isinstance(value, Mapping) or any(
            not isinstance(variable, str) or not variable or not isinstance(setting, str)
            for variable, setting in value.items()
        ):
            raise HarborTaskError(f"{key_path} must map variable names to strings")
        return dict(value)
    if key == "user":
        if isinstance(value, bool) or not (
            (isinstance(value, str) and value) or (isinstance(value, int) and value >= 0)
        ):
            raise HarborTaskError(f"{key_path} must be a user name or a non-negative uid")
        return value
    if not isinstance(value, str) or not value:
        raise HarborTaskError(f"{key_path} must be a non-empty string")
    return value


def checked_allowed_host(key_path: str, host: object) -> str:
    """One allowlist entry in the normalized form Harbor 0.23 accepts: a host name, a leading wildcard, an address or a CIDR range."""
    if not isinstance(host, str) or not host.strip():
        raise HarborTaskError(f"{key_path} entries must be non-empty host names")
    host = host.strip().lower().rstrip(".")
    reject = HarborTaskError(
        f"{key_path} entry {host!r} must be a host name, an IP address or a CIDR range, not a URL, port or path"
    )
    if "%" in host or "[" in host or "]" in host:
        raise reject
    if "/" in host:
        try:
            return ip_network(host, strict=True).compressed
        except ValueError:
            raise reject from None
    if ":" in host:
        try:
            address = ip_address(host)
        except ValueError:
            raise reject from None
        if address.version != 6:
            raise reject
        return address.compressed
    labels = host[2:] if host.startswith("*.") else host
    if not labels or "*" in labels:
        raise reject
    if host.startswith("*."):
        try:
            ip_address(labels)
        except ValueError:
            pass
        else:
            raise reject
    if not all(HOST_LABEL_PATTERN.fullmatch(label) for label in labels.split(".")):
        raise reject
    return host


def checked_metadata(metadata: object) -> dict[str, object]:
    """The user's metadata table, as long as both task.toml and the digest can hold it."""
    if not isinstance(metadata, Mapping) or any(not isinstance(key, str) or not key for key in metadata):
        raise HarborTaskError("metadata must be a table with string keys")
    if "reef" in metadata:
        raise HarborTaskError("metadata.reef is written by reef; put your own keys elsewhere")
    try:
        # Both writers must accept it: tomli_w for task.toml, json for the digest (so no TOML dates).
        tomli_w.dumps({"metadata": dict(metadata)})
        json.dumps(dict(metadata), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HarborTaskError(f"metadata must hold strings, numbers, booleans, lists and tables only: {exc}") from exc
    return dict(metadata)


def write_task_files(task: HarborTask, root: Path) -> None:
    """Lay the task out under ``root``; tests/ and environment/ exist even when empty, Harbor refuses a task without them."""
    for directory in ("tests", "environment"):
        (root / directory).mkdir()
    for relative, text in task.files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # newline="" on both sides: the bytes hashed are the bytes on disk, carriage returns included.
        target.write_text(text, encoding="utf-8", newline="")
    (root / "tests" / "test.sh").chmod(0o755)


def require_same_task(task: HarborTask, target: Path) -> None:
    """Accept an existing ``target`` only when it holds exactly ``task``; anything else is a conflict."""
    try:
        if not target.is_dir() or target.is_symlink():
            raise HarborTaskError(f"{target} is not a directory")
        existing = read_harbor_task(target)
    except HarborTaskError as exc:
        raise HarborTaskConflict(f"{target} exists and is not a task reef wrote: {exc}") from exc
    except OSError as exc:
        raise HarborTaskConflict(f"{target} exists and cannot be read: {exc}") from exc
    if existing.digest != task.digest:
        raise HarborTaskConflict(f"{target} already holds a different task with the same name")
