from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Self
from urllib.parse import urlsplit

from reef.artifact.artifact import (
    ARTIFACT_ID_METADATA_KEY,
    Artifact,
    ArtifactConflictError,
    ArtifactMaterializationError,
    ArtifactNotFoundError,
    ArtifactPublicationError,
    ArtifactSourceError,
    SnapshotRef,
    artifact_components,
    content_metadata,
)
from reef.artifact.composite import link_or_copy
from reef.artifact.git_client import GitClient
from reef.artifact.repository import (
    CachedRepositoryBackendFactory,
    ConfigurableRepositoryBackendFactory,
    RepositoryBackendContext,
    StagedSnapshotRepositoryBackend,
    StoredFiles,
)
from reef.core.config import config_option
from reef.core.plugins import PluginSettings

_MANIFEST = "reef-artifact.json"
#: The directory of a content commit that holds the content's files, beside the backend's own LFS attributes.
CONTENT_DIRECTORY = "content"
#: How many commits one ``git fetch`` names: 40 hexadecimal digits each stay far below the limit of a command line.
FETCH_BATCH_COMMITS = 1000
_LFS_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
_LFS_ATTRIBUTES = "".join(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n" for pattern in _LFS_PATTERNS)
# Backend instances in one process can share a working directory.
WORKSPACE_LOCKS: dict[Path, RLock] = {}
WORKSPACE_LOCKS_LOCK = Lock()


def _check_tools() -> None:
    git = GitClient(Path())
    git.run(("git", "--version"), source_error=True)
    try:
        git.run(("git", "lfs", "version"), source_error=True)
    except ArtifactSourceError as exc:
        raise ArtifactSourceError(f"{exc}; install the Git LFS system package before starting Reef") from exc


def _initialize_local_repository(repository: Path, git: GitClient) -> None:
    repository.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{repository.name}-", dir=repository.parent))
    try:
        git.run(("git", "init", "--bare", str(temporary)), source_error=True)
        try:
            temporary.rename(repository)
        except OSError:
            if not repository.is_dir():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


#: The remote tracking ref naming the snapshot a workspace fetched last, a commit the remote already holds.
FETCHED_SNAPSHOT_REF = "refs/remotes/origin/reef-fetched"


@dataclass(frozen=True)
class RefLayout:
    """Where the backend keeps its refs in the repository (RFC #1002).

    Values: ``moving`` is the prefix of the refs that move (each scenario's registration and head), and ``fixed``
    the prefix of the refs that never move once written (each snapshot, and each
    archived scenario head); both are full ref prefixes that end in ``/``. ``REEF_REFS`` keeps both under
    ``refs/reef/``. ``HUB_REFS`` puts the moving refs on branches and the fixed refs on tags, the only refs that the
    Hugging Face Hub accepts. A snapshot's ID is its commit under both, so on the Hub it is a revision.
    Purpose: ``GitLFSSettings.refs`` selects one; the backend and its factory name every ref through it.
    """

    moving: str
    fixed: str

    def __post_init__(self) -> None:
        if not self.moving.startswith("refs/") or not self.moving.endswith("/") or not self.fixed.endswith("/"):
            raise ValueError("a ref layout's prefixes are full ref prefixes that end in /")

    def scenario(self, encoded: str) -> str:
        """The ref of the scenario whose base64url name is ``encoded``: its registration and its head."""
        return f"{self.moving}scenarios/{encoded}"

    def snapshot(self, commit: str) -> str:
        return f"{self.fixed}releases/{commit}"

    def contents(self, encoded: str) -> str:
        """The prefix of the refs of the stored content of the scenario ``encoded``. A content's ref is replaced
        and deleted, so it is a moving ref."""
        return f"{self.moving}contents/{encoded}/"

    def archived(self, encoded: str, stamp: str) -> str:
        return f"{self.fixed}archived/{encoded}/{stamp}"

    @property
    def patterns(self) -> tuple[str, ...]:
        """The ``ls-remote`` patterns of every ref of the layout."""
        return tuple(dict.fromkeys((f"{self.moving}*", f"{self.fixed}*")))


REEF_REFS = RefLayout("refs/reef/", "refs/reef/")
HUB_REFS = RefLayout("refs/heads/reef/", "refs/tags/reef/")
#: ``GitLFSSettings.refs`` values.
REF_LAYOUTS = {"reef": REEF_REFS, "branches-and-tags": HUB_REFS}


def token_config(repository: str, token: str) -> dict[str, str]:
    """Git config that sends ``token`` as Basic authorization to the host of the ``https://`` ``repository`` only.

    The Hugging Face Hub takes a token as the password of Basic authorization for Git, and refuses a Bearer token.
    ``ValueError`` for a repository that is not an ``https://`` URL.
    """
    parts = urlsplit(repository)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"a token authorizes an https:// repository, not {repository!r}")
    credential = base64.b64encode(f"reef:{token}".encode()).decode()
    return {f"http.https://{parts.netloc}/.extraHeader": f"Authorization: Basic {credential}"}


class _GitWorkspace:
    """Manage one git working tree shared by scenario backend instances.

    Encapsulates git clone/checkout/commit/push so the backend does not
    call git directly. Holds the work_dir, clone_dir, and a per-process
    lock shared by backends that use this working directory.
    """

    def __init__(
        self,
        *,
        repository: str,
        work_dir: Path,
        git_client: GitClient,
        refs: RefLayout,
    ) -> None:
        self.repository = repository
        self.refs = refs
        # Git resolves GIT_INDEX_FILE against the subprocess's working directory.
        self.work_dir = Path(work_dir).resolve()
        self.clone_dir = self.work_dir / "repository"
        self._git_client = git_client
        #: The LFS object directory of a repository that is a local bare directory, else None.
        self.local_objects = Path(repository) / "lfs" / "objects" if Path(repository).is_dir() else None
        with WORKSPACE_LOCKS_LOCK:
            self.lock = WORKSPACE_LOCKS.setdefault(self.work_dir, RLock())

    def open_repository(self) -> None:
        if not (self.clone_dir / ".git").exists():
            if self.clone_dir.exists():
                shutil.rmtree(self.clone_dir)
            self._run(("git", "clone", "--no-checkout", self.repository, str(self.clone_dir)))
        self.git("config", "user.name", "Reef Repository Backend")
        self.git("config", "user.email", "reef-artifacts@localhost")
        self._install_lfs(self.clone_dir)
        # A publish indexes a staged tree that has no .gitattributes of Reef's own, so the LFS
        # patterns also live in the repository's attributes, which take precedence over the tree's.
        repository_attributes = self.clone_dir / ".git" / "info" / "attributes"
        repository_attributes.parent.mkdir(exist_ok=True)
        write_fresh(repository_attributes, _LFS_ATTRIBUTES)

    def _install_lfs(self, repository: Path) -> None:
        # Keep LFS hooks separate from global hooks and hooks copied by Git templates.
        hooks = repository.resolve() / ".git" / "reef-hooks"
        self._run(("git", "config", "--local", "core.hooksPath", str(hooks)), cwd=repository)
        self._run(("git", "lfs", "install", "--local", "--skip-smudge"), cwd=repository)

    def checkout(self, version: str) -> None:
        self.fetch_version(version)
        self.git("checkout", "--detach", "FETCH_HEAD")
        self.git("reset", "--hard", "FETCH_HEAD")
        self.git("clean", "-fdx")

    def orphan_checkout(self) -> None:
        self.git("checkout", "--orphan", f"reef-initial-{uuid.uuid4().hex}")
        self.git("rm", "-rf", "--ignore-unmatch", ".")
        self.git("clean", "-fdx")

    def replace_tree(self, source: Path, *, subdirectory: str | None = None) -> None:
        """Replace the work tree with ``source``, placed under ``subdirectory`` when one is named.

        The files are hard links to ``source``'s, so nothing may write through
        them afterwards: a work tree file Reef rewrites goes through
        ``write_fresh``, which unlinks it first.
        """
        for child in self.clone_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        target = self.clone_dir if subdirectory is None else self.clone_dir / subdirectory
        target.mkdir(parents=True, exist_ok=True)
        # Stored files are immutable, so the work tree links them rather than copying a checkpoint per step.
        for child in source.iterdir():
            destination = target / child.name
            if child.is_dir():
                try:
                    shutil.copytree(child, destination, symlinks=False, copy_function=link_or_copy)
                except shutil.Error as exc:
                    # copytree collects (source, destination, why) per file; the first names the fault.
                    failures = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
                    first = (
                        failures[0] if failures and isinstance(failures[0], tuple) and len(failures[0]) == 3 else None
                    )
                    if first is None:
                        raise ArtifactSourceError(
                            f"artifact contains a file that cannot be read under {child}: {exc}"
                        ) from exc
                    source_path, _, why = first
                    if "No such file" in str(why):
                        raise ArtifactSourceError(f"artifact contains a broken symlink: {source_path}") from exc
                    raise ArtifactSourceError(
                        f"artifact contains a file that cannot be read: {source_path}: {why}"
                    ) from exc
            else:
                try:
                    link_or_copy(str(child), str(destination))
                except FileNotFoundError as exc:
                    raise ArtifactSourceError(f"artifact contains a broken symlink: {child}") from exc

    def write_lfs_attributes(self) -> None:
        write_fresh(self.clone_dir / ".gitattributes", _LFS_ATTRIBUTES)

    def source_files(self, source: Path) -> tuple[str, ...]:
        """The files ``git add -A`` would take from ``source``, relative to it; nothing is hashed."""
        index = self.work_dir / "source-index"
        index.unlink(missing_ok=True)
        output = self._run(
            ("git", f"--work-tree={source.resolve()}", "ls-files", "-z", "--others", "--exclude-standard"),
            cwd=self.clone_dir,
            environment={"GIT_INDEX_FILE": str(index)},
        )
        return tuple(path for path in output.split("\0") if path)

    def commit_files(
        self,
        source: Path,
        paths: Sequence[str],
        *,
        parent: str | None,
        carried_from: Path | None,
        files: Mapping[str, str],
        message: str,
    ) -> str:
        """Commit ``paths`` from ``source`` plus ``files`` as the child of ``parent`` without checking it out; with
        no ``parent``, as a commit with no parent.

        The index starts from the parent's tree, so a file that is a hard link
        to the same path in ``carried_from`` (the parent's materialized tree)
        keeps the parent's blob instead of being read and hashed again; only
        the other paths go through the LFS clean filter. The work tree is left
        untouched.
        """
        index = self.work_dir / "publish-index"
        index.unlink(missing_ok=True)
        environment = {"GIT_INDEX_FILE": str(index)}
        try:
            if parent is None:
                self._run(("git", "read-tree", "--empty"), cwd=self.clone_dir, environment=environment)
            else:
                self.fetch_version(parent)
                self._run(("git", "read-tree", parent), cwd=self.clone_dir, environment=environment)
            parent_paths = {
                path
                for path in self._run(("git", "ls-files", "-z"), cwd=self.clone_dir, environment=environment).split(
                    "\0"
                )
                if path
            }
            removed = parent_paths.difference(paths, files)
            if removed:
                self._run(
                    ("git", "update-index", "-z", "--force-remove", "--stdin"),
                    cwd=self.clone_dir,
                    environment=environment,
                    input_text="\0".join(sorted(removed)),
                )
            changed = [
                path
                for path in paths
                if path not in parent_paths
                or carried_from is None
                or not same_file(source / path, carried_from / path)
            ]
            if changed:
                self._run(
                    ("git", f"--work-tree={source.resolve()}", "update-index", "-z", "--add", "--stdin"),
                    cwd=self.clone_dir,
                    environment=environment,
                    input_text="\0".join(changed),
                )
            return self.commit_index(environment, files, parent=parent, message=message)
        finally:
            index.unlink(missing_ok=True)

    def commit_index(
        self, environment: Mapping[str, str], files: Mapping[str, str], *, parent: str | None, message: str
    ) -> str:
        """Add ``files`` (text by path) to the index of ``environment`` and commit it as the child of ``parent``."""
        entries = []
        for path, text in files.items():
            blob = self._run(("git", "hash-object", "-w", "--stdin"), cwd=self.clone_dir, input_text=text)
            entries.append(f"100644 {blob}\t{path}\n")
        self._run(
            ("git", "update-index", "--index-info"),
            cwd=self.clone_dir,
            environment=environment,
            input_text="".join(entries),
        )
        tree = self._run(("git", "write-tree"), cwd=self.clone_dir, environment=environment)
        parents = () if parent is None else ("-p", parent)
        return self.git("commit-tree", tree, *parents, "-m", message)

    def commit_trees(
        self,
        trees: Mapping[str | None, str | None],
        *,
        parent: str | None,
        files: Mapping[str, str],
        removed: Sequence[str] = (),
        message: str,
    ) -> str:
        """Commit fetched trees plus ``files`` as the child of ``parent``, or with no parent; no file is read,
        hashed or downloaded.

        ``trees`` maps a directory name to the tree that the commit holds there, byte for byte, or to None for no
        file. The one key None puts its tree at the root, where ``files`` replace the paths of their names.
        ``removed`` are paths that the commit drops after it read the trees.
        """
        index = self.work_dir / "publish-index"
        index.unlink(missing_ok=True)
        environment = {"GIT_INDEX_FILE": str(index)}
        try:
            if parent is not None:
                self.fetch_version(parent)
            self._run(("git", "read-tree", "--empty"), cwd=self.clone_dir, environment=environment)
            for directory, tree in trees.items():
                if tree is not None:
                    selector = (tree,) if directory is None else (f"--prefix={directory}/", tree)
                    self._run(("git", "read-tree", *selector), cwd=self.clone_dir, environment=environment)
            if removed:
                self._run(
                    ("git", "update-index", "--force-remove", *removed), cwd=self.clone_dir, environment=environment
                )
            return self.commit_index(environment, files, parent=parent, message=message)
        finally:
            index.unlink(missing_ok=True)

    def write_tree(self, source: Path, paths: Sequence[str]) -> str:
        """The Git tree of ``paths`` from ``source``, written to the clone: LFS files go through the clean filter."""
        index = self.work_dir / "content-index"
        index.unlink(missing_ok=True)
        environment = {"GIT_INDEX_FILE": str(index)}
        try:
            self._run(("git", "read-tree", "--empty"), cwd=self.clone_dir, environment=environment)
            if paths:
                self._run(
                    ("git", f"--work-tree={source.resolve()}", "update-index", "-z", "--add", "--stdin"),
                    cwd=self.clone_dir,
                    environment=environment,
                    input_text="\0".join(paths),
                )
            return self._run(("git", "write-tree"), cwd=self.clone_dir, environment=environment)
        finally:
            index.unlink(missing_ok=True)

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def push(self, *refs: str, uploads_lfs: bool = True) -> None:
        """Push ``refs``. ``uploads_lfs=False`` skips the LFS pre push hook (``--no-verify``): the hook refuses a
        commit whose LFS objects this clone does not hold, although the remote holds them. A caller passes it only
        for a commit that points at LFS objects that an earlier push uploaded."""
        self.git("push", *(() if uploads_lfs else ("--no-verify",)), "origin", *refs)

    def force_push_with_lease(self, lease: str, *refs: str, uploads_lfs: bool = True) -> None:
        """``push`` with the lease ``lease``."""
        self.git("push", *(() if uploads_lfs else ("--no-verify",)), lease, "origin", *refs)

    def fetch_version(self, version: str) -> str:
        """Fetch ``version`` from the remote and return its commit id.

        Every publish path fetches the snapshot it builds on or pushes through
        here. The remote holds that commit and its history, and a remote
        tracking ref records it, so the LFS pre push check scans only the
        commits a later push adds, not the older snapshots whose objects this
        clone never fetched. A push to a new ref, such as a snapshot saved for
        review without moving the head, has no remote commit of its own to stop
        the scan.
        """
        with self.lock:
            self.git("fetch", "origin", version)
            self.git("update-ref", FETCHED_SNAPSHOT_REF, "FETCH_HEAD")
            return self.git("rev-parse", "FETCH_HEAD")

    def show_file(self, version: str, path: str) -> str:
        with self.lock:
            self.fetch_version(version)
            return self.git("show", f"FETCH_HEAD:{path}")

    def ls_remote(self, ref_name: str) -> str | None:
        output = self.git("ls-remote", "origin", ref_name)
        if not output:
            return None
        return output.split()[0]

    def ls_remote_prefix(self, prefix: str) -> dict[str, str]:
        """The commit of each remote ref whose name starts with ``prefix``, by ref name."""
        output = self.git("ls-remote", "origin", f"{prefix}*")
        return {name: commit for commit, name in (line.split() for line in output.splitlines())}

    def fetch_commits(self, commits: Collection[str]) -> None:
        """Fetch ``commits``, in groups that fit on a command line."""
        ordered = sorted(commits)
        for start in range(0, len(ordered), FETCH_BATCH_COMMITS):
            self.git("fetch", "origin", *ordered[start : start + FETCH_BATCH_COMMITS])

    def clone_for_materialize(self, destination: Path, version: str, directory: str | None = None) -> None:
        """Check out ``version`` in a new clone at ``destination``, with the bytes of its LFS files; with
        ``directory``, only the files under that directory of the snapshot, so no other file is fetched.

        The checkout leaves an LFS file as its pointer. When the repository is a local directory, each such file
        becomes a hard link to its object in the repository's own LFS storage, or a copy across file systems, so
        a snapshot that a process of this host stored is made local without a second copy of its weights. A
        linked file is the repository's own object, so the object is made read-only first: a write through the
        materialized path is refused, and a reader that needs a writable tree copies it. An object whose size is
        not the size that the pointer names is not linked. ``git lfs pull``, which checks each object's digest,
        fetches what is still a pointer: every LFS file of a remote repository, and none when every object was
        linked.
        """
        self._run(("git", "clone", "--no-checkout", self.repository, str(destination)))
        self._install_lfs(destination)
        self._run(("git", "fetch", "origin", version), cwd=destination)
        if directory is not None:
            self._run(("git", "sparse-checkout", "set", "--no-cone", f"/{directory}/"), cwd=destination)
        self._run(("git", "checkout", "--detach", "FETCH_HEAD"), cwd=destination)
        pointers = [
            line.split(" ", 2)
            for line in self._run(("git", "lfs", "ls-files", "--long"), cwd=destination).splitlines()
            if line
        ]
        if directory is not None:
            pointers = [pointer for pointer in pointers if pointer[2].startswith(f"{directory}/")]
        missing = bool(pointers)
        if self.local_objects is not None:
            missing = False
            for oid, _, path in pointers:
                stored = self.local_objects / oid[:2] / oid[2:4] / oid
                pointer = destination / path
                sizes = [
                    line.split(" ", 1)[1] for line in pointer.read_text().splitlines() if line.startswith("size ")
                ]
                if stored.is_file() and sizes == [str(stored.stat().st_size)]:
                    stored.chmod(0o444)
                    pointer.unlink()
                    link_or_copy(str(stored), str(pointer))
                else:
                    missing = True
        if missing:
            only = () if directory is None else ("--include", f"{directory}/**")
            self._run(("git", "lfs", "pull", *only), cwd=destination)

    def git(self, *args: str) -> str:
        return self._git_client.git(*args)

    def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        source_error: bool = False,
        input_text: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        return self._git_client.run(
            command, cwd=cwd, source_error=source_error, input_text=input_text, environment=environment
        )


def same_file(first: Path, second: Path) -> bool:
    """Whether both paths name one file; a missing path names none."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def write_fresh(path: Path, text: str) -> None:
    """Write ``text`` to a new inode: a work tree file may be a link to a stored or cached file."""
    path.unlink(missing_ok=True)
    path.write_text(text, encoding="utf-8")


def materialized_metadata(checkout: Path) -> Mapping[str, object]:
    """The snapshot metadata recorded in a materialized checkout's manifest; a bootstrap tree has none."""
    manifest_path = checkout / _MANIFEST
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactMaterializationError(f"invalid artifact manifest in {checkout}") from exc
    metadata = manifest.get("metadata", {}) if isinstance(manifest, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ArtifactMaterializationError(f"invalid artifact metadata in {checkout}")
    return dict(metadata)


class _ArtifactManifest:
    """Read and write the reef-artifact.json manifest inside a git work tree."""

    def __init__(self, workspace: _GitWorkspace) -> None:
        self._workspace = workspace

    @staticmethod
    def text(
        *,
        parent_revision: str | None,
        source: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> str:
        manifest = {
            "parent_revision": parent_revision,
            "source": dict(source),
            "metadata": dict(metadata),
        }
        return json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    def write(
        self,
        *,
        parent_revision: str | None,
        source: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> None:
        write_fresh(
            self._workspace.clone_dir / _MANIFEST,
            self.text(parent_revision=parent_revision, source=source, metadata=metadata),
        )

    def read(self, version: str, *, local: bool = False) -> Mapping[str, object]:
        raw = (
            self._workspace.git("show", f"{version}:{_MANIFEST}")
            if local
            else self._workspace.show_file(version, _MANIFEST)
        )
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArtifactSourceError(f"invalid artifact manifest at {version}") from exc
        if not isinstance(manifest, Mapping):
            raise ArtifactSourceError(f"invalid artifact manifest at {version}")
        return manifest

    def snapshot(self, version: str, *, local: bool = False) -> SnapshotRef:
        manifest = self.read(version, local=local)
        try:
            parent_revision = manifest.get("parent_revision")
            if parent_revision is not None and not isinstance(parent_revision, str):
                raise TypeError("parent_revision must be a string or null")
            return SnapshotRef(
                revision=(
                    self._workspace.git("rev-parse", version) if local else self._workspace.fetch_version(version)
                ),
                parent_revision=parent_revision,
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactSourceError(f"invalid artifact manifest at {version}") from exc


class GitLFSRepositoryBackend(StagedSnapshotRepositoryBackend):
    """The snapshots of one scenario in a Git LFS repository.

    The scenario's ref names its head. ``register`` pushes the scenario's root snapshot, a commit with no parent, to
    that ref; each later snapshot is a descendant of the root. Scenarios of one repository share no snapshot.
    """

    def __init__(
        self,
        scenario: str,
        repository: str | Path,
        *,
        work_dir: Path,
        cache_dir: Path,
        refs: RefLayout = REEF_REFS,
        git_config: Mapping[str, str] | None = None,
    ) -> None:
        if not scenario:
            raise ValueError("scenario must be non-empty")
        _check_tools()
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        self.refs = refs
        self.artifact_ref_prefix = f"{refs.fixed}artifacts/{encoded}/"
        self.content_ref_prefix = refs.contents(encoded)
        self.ref_name = refs.scenario(encoded)
        self.work_dir = Path(work_dir)
        self.cache_dir = Path(cache_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._git_client = GitClient(self.work_dir / "repository", git_config)
        local_repository = repository if isinstance(repository, Path) else None
        if local_repository is not None and not local_repository.exists():
            _initialize_local_repository(local_repository, self._git_client)
        self.repository = str(repository)
        self._workspace = _GitWorkspace(
            repository=self.repository,
            work_dir=self.work_dir,
            git_client=self._git_client,
            refs=refs,
        )
        self._manifest = _ArtifactManifest(self._workspace)
        self._open_repository()
        other = HUB_REFS if refs == REEF_REFS else REEF_REFS
        if self._workspace.git("ls-remote", "origin", *other.patterns):
            raise ArtifactSourceError(
                f"artifact repository {self.repository} keeps its refs under {other.moving} and {other.fixed}; "
                "configure the storage.artifact_backend refs that match it"
            )

    @classmethod
    def factory(
        cls,
        repository: str | Path,
        *,
        work_dir: Path,
        cache_dir: Path,
        refs: RefLayout = REEF_REFS,
        git_config: Mapping[str, str] | None = None,
    ) -> GitLFSRepositoryBackendFactory:
        """A factory of scenario backends over one repository."""
        _check_tools()
        return GitLFSRepositoryBackendFactory(
            cls,
            repository,
            work_dir=work_dir,
            cache_dir=cache_dir,
            refs=refs,
            git_config=git_config,
        )

    def resolve_snapshot(self, revision: str) -> SnapshotRef:
        selector = revision.removeprefix("git+lfs://")
        resolved_snapshot = self._workspace.ls_remote(self.refs.snapshot(selector))
        if resolved_snapshot is None:
            try:
                resolved_snapshot = self._workspace.fetch_version(selector)
            except ArtifactPublicationError as exc:
                raise ArtifactNotFoundError(f"snapshot does not exist: {revision}") from exc
        return self._manifest.snapshot(resolved_snapshot)

    def artifact_ref_name(self, artifact_id: str) -> str:
        """The Git ref that binds ``artifact_id`` to its snapshot commit in this scenario."""
        return self.artifact_ref_prefix + hashlib.sha256(artifact_id.encode()).hexdigest()

    def stored_snapshot(self, commit: str) -> Artifact:
        """The snapshot ``commit``, fetched already, with its metadata and without its files."""
        manifest = self._manifest.read(commit, local=True)
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ArtifactSourceError(f"invalid artifact metadata at {commit}")
        return Artifact(self._manifest.snapshot(commit, local=True), None, metadata=metadata)

    def artifact_content(self, ref: SnapshotRef, artifact_id: str) -> tuple[object, dict[str, object]]:
        """The Git file objects and the metadata of the artifact, read without fetching weight bytes."""
        commit = ref.revision
        snapshot = self.stored_snapshot(commit)
        components = artifact_components(snapshot)
        if artifact_id not in components:
            raise ArtifactNotFoundError(f"storage snapshot does not contain artifact {artifact_id!r}")
        component = components[artifact_id]
        manifest = snapshot.components
        prefix = "" if component is None or manifest is None or manifest.single else f"{component}/"
        files: dict[str, str] = {}
        for entry in self._workspace.git("ls-tree", "-r", "-z", commit).split("\0"):
            if not entry:
                continue
            descriptor, path = entry.split("\t", 1)
            if path in (_MANIFEST, ".gitattributes") or not path.startswith(prefix):
                continue
            files[path.removeprefix(prefix)] = descriptor
        return files, content_metadata(snapshot, component)

    def bind_artifact_id(self, artifact_id: str, ref: SnapshotRef) -> SnapshotRef | None:
        return self.push_binding(artifact_id, ref, uploads_lfs=True)

    def push_binding(self, artifact_id: str, ref: SnapshotRef, *, uploads_lfs: bool) -> SnapshotRef | None:
        """Push the immutable binding ref of ``artifact_id`` unless it exists; fetch an existing one for comparison.
        ``uploads_lfs`` is as for ``_GitWorkspace.push``."""
        ref_name = self.artifact_ref_name(artifact_id)
        existing = self._workspace.ls_remote(ref_name)
        if existing is None:
            try:
                self._workspace.force_push_with_lease(
                    f"--force-with-lease={ref_name}:", f"{ref.revision}:{ref_name}", uploads_lfs=uploads_lfs
                )
                return None
            except ArtifactPublicationError:
                existing = self._workspace.ls_remote(ref_name)
                if existing is None:
                    raise
        if existing != ref.revision:
            self._workspace.fetch_version(existing)
        return self._manifest.snapshot(existing, local=True)

    def artifact_snapshot(self, artifact_id: str) -> SnapshotRef:
        with self._workspace.lock:
            commit = self._workspace.ls_remote(self.artifact_ref_name(artifact_id))
            if commit is None:
                raise ArtifactNotFoundError(f"repository has no artifact {artifact_id!r}")
            self._workspace.fetch_version(commit)
            return self._manifest.snapshot(commit, local=True)

    def artifact_snapshots(self, artifact_ids: Collection[str]) -> dict[str, SnapshotRef]:
        """Read every binding of this scenario with one ``ls-remote``, and fetch their commits with one ``fetch``."""
        with self._workspace.lock:
            bound = self._workspace.ls_remote_prefix(self.artifact_ref_prefix)
            commits = {
                artifact_id: bound[self.artifact_ref_name(artifact_id)]
                for artifact_id in artifact_ids
                if self.artifact_ref_name(artifact_id) in bound
            }
            self._workspace.fetch_commits(set(commits.values()))
            snapshots = {commit: self._manifest.snapshot(commit, local=True) for commit in set(commits.values())}
            return {artifact_id: snapshots[commit] for artifact_id, commit in commits.items()}

    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> SnapshotRef:
        """Commit the files of ``root`` with no parent and push the commit to the scenario's ref.

        A ref that exists already refuses the unrelated commit, so the first registration wins and a later one
        returns the winner's head. The registration also fills the root's cache entry, so the first
        ``materialize`` links the files and pulls nothing.
        """
        if root.local_path is None or not root.local_path.is_dir():
            raise ArtifactPublicationError("a root snapshot needs an existing local artifact directory")
        with self._workspace.lock:
            existing = self._workspace.ls_remote(self.ref_name)
            if existing is not None:
                self._workspace.fetch_version(existing)
                self.bind_artifact_ids(self.stored_snapshot(existing))
                return self._manifest.snapshot(existing)
            self._workspace.orphan_checkout()
            self._workspace.replace_tree(root.local_path)
            self._workspace.write_lfs_attributes()
            self._manifest.write(
                parent_revision=None,
                source={"kind": "root"},
                metadata={**root.storage_metadata, **dict(metadata)},
            )
            # Identical roots registered in the same second still need distinct snapshot revisions.
            commit = self._workspace.commit(f"register scenario {uuid.uuid4().hex}")
            self.bind_artifact_ids(self.stored_snapshot(commit))
            try:
                self._workspace.push(f"{commit}:{self.ref_name}")
            except ArtifactPublicationError:
                existing = self._workspace.ls_remote(self.ref_name)
                if existing is None:
                    raise
                return self._manifest.snapshot(existing)
            paths = tuple(
                path
                for path in self._workspace.git("ls-files", "-z").split("\0")
                if path and path != _MANIFEST and path != ".gitattributes"
            )
            self._cache_snapshot(
                commit,
                self._workspace.clone_dir,
                paths,
                {".gitattributes": _LFS_ATTRIBUTES, _MANIFEST: (self._workspace.clone_dir / _MANIFEST).read_text()},
            )
            return self._manifest.snapshot(commit)

    def metadata(self) -> Mapping[str, object] | None:
        version = self._workspace.ls_remote(self.ref_name)
        if version is None:
            return None
        manifest = self._manifest.read(version)
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ArtifactSourceError(f"invalid artifact metadata at {version}")
        return {key: value for key, value in metadata.items() if key != ARTIFACT_ID_METADATA_KEY}

    def current(self) -> SnapshotRef:
        version = self._workspace.ls_remote(self.ref_name)
        if version is None:
            raise ArtifactNotFoundError("repository has no current artifact")
        return self._manifest.snapshot(version)

    def materialize(self, ref: SnapshotRef) -> Artifact:
        destination = self.cache_dir / ref.revision
        if destination.is_dir():
            return Artifact(ref, None, local_path=destination, metadata=materialized_metadata(destination))
        temporary = Path(tempfile.mkdtemp(prefix=f".{ref.revision}-", dir=self.cache_dir))
        checkout = temporary / "artifact"
        try:
            self._workspace.clone_for_materialize(checkout, ref.revision)
            git_metadata = checkout / ".git"
            if git_metadata.is_dir():
                shutil.rmtree(git_metadata)
            else:
                git_metadata.unlink(missing_ok=True)
            try:
                checkout.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        except ArtifactPublicationError as exc:
            raise ArtifactMaterializationError(f"failed to materialize artifact {ref.revision}: {exc}") from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if not destination.is_dir():
            raise ArtifactMaterializationError(f"artifact cache was not created: {ref.revision}")
        return Artifact(ref, None, local_path=destination, metadata=materialized_metadata(destination))

    def materialize_component(self, ref: SnapshotRef, name: str) -> Artifact:
        """The component ``name`` of the snapshot ``ref`` from a sparse checkout of its directory: the LFS objects
        of the other components are not fetched. A snapshot that is local as a whole, or that holds one component,
        answers from its whole checkout."""
        with self._workspace.lock:
            self._workspace.fetch_version(ref.revision)
            snapshot = self.stored_snapshot(ref.revision)
        manifest = snapshot.components
        if (self.cache_dir / ref.revision).is_dir() or manifest is None or manifest.single:
            return self.materialize(ref).component(name)
        if name not in manifest.entries:
            raise ArtifactNotFoundError(f"snapshot {ref.revision!r} binds no component {name!r}")
        destination = self.cache_dir / f"{ref.revision}.{name}"
        if not destination.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix=f".{ref.revision}-", dir=self.cache_dir))
            try:
                self._workspace.clone_for_materialize(temporary / "artifact", ref.revision, name)
                # Git keeps no directory for a component with no file.
                (temporary / "artifact" / name).mkdir(exist_ok=True)
                try:
                    (temporary / "artifact" / name).rename(destination)
                except OSError:
                    if not destination.is_dir():
                        raise
            except ArtifactPublicationError as exc:
                raise ArtifactMaterializationError(
                    f"failed to materialize component {name!r} of {ref.revision}: {exc}"
                ) from exc
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return Artifact(
            snapshot.ref,
            None,
            local_path=destination,
            metadata=manifest.entries[name].metadata,
            artifact_id=manifest.artifact_ids(ref.revision)[name],
        )

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: SnapshotRef,
        advance_head: bool = True,
    ) -> SnapshotRef:
        if artifact.local_path is None or not artifact.local_path.is_dir():
            raise ArtifactPublicationError("artifact ref must contain an existing local artifact directory")
        with self._workspace.lock:
            current = self._workspace.ls_remote(self.ref_name)
            if current != expected_parent.revision:
                raise ArtifactConflictError(
                    f"repository is at {current}, not expected parent {expected_parent.revision}"
                )
            files = {
                ".gitattributes": _LFS_ATTRIBUTES,
                _MANIFEST: _ArtifactManifest.text(
                    parent_revision=expected_parent.revision,
                    source={"kind": "training"},
                    metadata=artifact.storage_metadata,
                ),
            }
            paths = tuple(path for path in self._workspace.source_files(artifact.local_path) if path not in files)
            linked = any((artifact.local_path / path).is_symlink() for path in paths)
            if linked:
                # A symlinked tree is published by its targets' bytes, which only a copy into the work tree resolves.
                self._workspace.checkout(expected_parent.revision)
                self._workspace.replace_tree(artifact.local_path)
                for path, text in files.items():
                    write_fresh(self._workspace.clone_dir / path, text)
                commit = self._workspace.commit("publish artifact")
            else:
                commit = self._workspace.commit_files(
                    artifact.local_path,
                    paths,
                    parent=expected_parent.revision,
                    carried_from=self.cache_dir / expected_parent.revision,
                    files=files,
                    message="publish artifact",
                )
            self.push_snapshot(commit, expected_parent, advance_head)
            if not linked:
                self._cache_snapshot(commit, artifact.local_path, paths, files)
            return self._manifest.snapshot(commit)

    def push_snapshot(
        self, commit: str, expected_parent: SnapshotRef, advance_head: bool, *, uploads_lfs: bool = True
    ) -> None:
        """Bind the artifact IDs of the local snapshot ``commit`` and push it: to the scenario's ref when
        ``advance_head``, else under the snapshots namespace. The caller holds the workspace lock, and
        ``uploads_lfs`` is as for ``_GitWorkspace.push``."""
        snapshot = self.stored_snapshot(commit)
        for artifact_id in artifact_components(snapshot):
            bound = self.push_binding(artifact_id, snapshot.ref, uploads_lfs=uploads_lfs)
            self.require_same_content(artifact_id, bound, snapshot.ref)
        if not advance_head:
            # A pending snapshot lives under the snapshots namespace; the scenario's ref and the head stay put.
            self._workspace.push(f"+{commit}:{self.refs.snapshot(commit)}", uploads_lfs=uploads_lfs)
            return
        try:
            self._workspace.force_push_with_lease(
                f"--force-with-lease={self.ref_name}:{expected_parent.revision}",
                f"{commit}:{self.ref_name}",
                uploads_lfs=uploads_lfs,
            )
        except ArtifactPublicationError as exc:
            current = self._workspace.ls_remote(self.ref_name)
            if current != expected_parent.revision:
                raise ArtifactConflictError(f"repository advanced from {expected_parent.revision}") from exc
            raise

    def content_ref(self, name: str) -> str:
        """The Git ref of the stored content ``name`` of this scenario."""
        if not isinstance(name, str) or not name:
            raise ValueError("a content's name is a non-empty string")
        return self.content_ref_prefix + hashlib.sha256(name.encode()).hexdigest()

    def store_content(self, name: str, directory: Path) -> None:
        """Commit the files of ``directory`` under ``CONTENT_DIRECTORY``, with no parent, and push the commit to
        the content's ref.

        The files keep every name, also ``.gitattributes`` and the manifest's: the backend's own LFS attributes
        are at the commit's root, outside the content. Weight files go through the LFS clean filter, so Git LFS
        keeps one object for equal bytes, whatever content or snapshot holds them. The push uploads the LFS
        objects before the ref moves, so content that exists has its objects on the remote.
        """
        ref_name = self.content_ref(name)
        if not directory.is_dir():
            raise ArtifactPublicationError(f"content {name!r} needs an existing directory, not {directory}")
        with self._workspace.lock:
            paths = self._workspace.source_files(directory)
            if any((directory / path).is_symlink() for path in paths):
                # A symlinked tree is stored by its targets' bytes, which only a copy into the work tree resolves.
                self._workspace.orphan_checkout()
                self._workspace.replace_tree(directory, subdirectory=CONTENT_DIRECTORY)
                self._workspace.write_lfs_attributes()
                commit = self._workspace.commit("store content")
            else:
                commit = self._workspace.commit_trees(
                    {CONTENT_DIRECTORY: self._workspace.write_tree(directory, paths)},
                    parent=None,
                    files={".gitattributes": _LFS_ATTRIBUTES},
                    message="store content",
                )
            self._workspace.push(f"+{commit}:{ref_name}")

    def content_commit(self, name: str) -> str:
        """The commit of the stored content ``name``; ``ArtifactNotFoundError`` when the repository has none."""
        commit = self._workspace.ls_remote(self.content_ref(name))
        if commit is None:
            raise ArtifactNotFoundError(f"repository has no content {name!r}")
        return commit

    def materialize_content(self, name: str) -> Path:
        with self._workspace.lock:
            commit = self.content_commit(name)
        path = self.materialize(SnapshotRef(commit, None)).local_path
        if path is None:
            raise ArtifactMaterializationError(f"content {name!r} was not made local")
        # Git keeps no directory for content with no file.
        (path / CONTENT_DIRECTORY).mkdir(exist_ok=True)
        return path / CONTENT_DIRECTORY

    def discard_content(self, name: str) -> None:
        ref_name = self.content_ref(name)
        with self._workspace.lock:
            if self._workspace.ls_remote(ref_name) is None:
                return
            try:
                self._workspace.push(f":{ref_name}")
            except ArtifactPublicationError:
                # Another process can remove the ref between the two calls.
                if self._workspace.ls_remote(ref_name) is not None:
                    raise

    def discard_all_content(self) -> None:
        """Delete every content ref of the scenario, in groups that fit on a command line."""
        with self._workspace.lock:
            names = sorted(self._workspace.ls_remote_prefix(self.content_ref_prefix))
            for start in range(0, len(names), FETCH_BATCH_COMMITS):
                self._workspace.push(*(f":{name}" for name in names[start : start + FETCH_BATCH_COMMITS]))

    def publish_parts(
        self,
        parts: Mapping[str | None, StoredFiles],
        *,
        metadata: Mapping[str, object],
        expected_parent: SnapshotRef,
        advance_head: bool,
    ) -> SnapshotRef:
        """Commit the trees of the named files as the child of ``expected_parent`` and push the commit.

        The commit points at the Git trees and LFS pointers that the remote holds already. This clone fetches the
        commits of those trees, not their LFS objects, and it uploads none: the push of the content
        (``store_content``), or of the snapshot that bound an artifact, uploaded them before its ref moved. A
        component's tree is carried byte for byte. At the root of a flat snapshot, the backend's own two files
        replace files of their names, as ``publish`` does.
        """
        with self._workspace.lock:
            trees: dict[str | None, str | None] = {}
            removed: list[str] = []
            for directory, part in parts.items():
                trees[directory], flat_snapshot = self.stored_tree(part)
                if flat_snapshot and directory is not None:
                    # The root of a flat snapshot holds the backend's own two files, which are no files of it.
                    removed.extend(f"{directory}/{path}" for path in (".gitattributes", _MANIFEST))
            commit = self._workspace.commit_trees(
                trees,
                parent=expected_parent.revision,
                files={
                    ".gitattributes": _LFS_ATTRIBUTES,
                    _MANIFEST: _ArtifactManifest.text(
                        parent_revision=expected_parent.revision,
                        source={"kind": "training"},
                        metadata=metadata,
                    ),
                },
                removed=removed,
                message="publish artifact",
            )
            self.push_snapshot(commit, expected_parent, advance_head, uploads_lfs=False)
            return self._manifest.snapshot(commit)

    def stored_tree(self, files: StoredFiles) -> tuple[str | None, bool]:
        """The fetched Git tree of ``files``, or None for no file, and whether it is the root of a flat snapshot,
        which also holds the backend's own two files."""
        if files.content is not None:
            content = self.content_commit(files.content)
            self._workspace.git("fetch", "origin", content)
            holds_files = bool(self._workspace.git("ls-tree", content, CONTENT_DIRECTORY))
            return (f"{content}:{CONTENT_DIRECTORY}" if holds_files else None), False
        artifact_id = str(files.artifact_id)
        commit = self._workspace.ls_remote(self.artifact_ref_name(artifact_id))
        if commit is None:
            raise ArtifactNotFoundError(f"repository has no artifact {artifact_id!r}")
        self._workspace.git("fetch", "origin", commit)
        snapshot = self.stored_snapshot(commit)
        components = artifact_components(snapshot)
        if artifact_id not in components:
            raise ArtifactNotFoundError(f"storage snapshot does not contain artifact {artifact_id!r}")
        component = components[artifact_id]
        manifest = snapshot.components
        if component is None or manifest is None or manifest.single:
            return f"{commit}^{{tree}}", True
        # Git keeps no directory for a component with no file.
        if not self._workspace.git("ls-tree", commit, component):
            return None, False
        return f"{commit}:{component}", False

    def snapshot_metadata(self, ref: SnapshotRef) -> Mapping[str, object]:
        """The metadata in the snapshot's manifest, read from its commit without its LFS objects."""
        with self._workspace.lock:
            self._workspace.fetch_version(ref.revision)
            return self.stored_snapshot(ref.revision).storage_metadata

    def _cache_snapshot(self, revision: str, source: Path, paths: Sequence[str], files: Mapping[str, str]) -> None:
        """Fill a just-published snapshot's cache entry from the staged tree it was committed from.

        The staged files hold the committed bytes, so the entry links them
        instead of cloning the snapshot and pulling its LFS objects back. An
        entry that cannot be filled is cloned by the first ``materialize``.
        """
        destination = self.cache_dir / revision
        if destination.is_dir():
            return
        temporary = Path(tempfile.mkdtemp(prefix=f".{revision}-", dir=self.cache_dir))
        tree = temporary / "artifact"
        try:
            tree.mkdir()
            for path in paths:
                (tree / path).parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(str(source / path), str(tree / path))
            for path, text in files.items():
                (tree / path).write_text(text, encoding="utf-8")
            tree.rename(destination)
        except OSError:
            pass
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def commit_snapshot(self, ref: SnapshotRef, *, expected_parent: SnapshotRef) -> None:
        with self._workspace.lock:
            staged = self.resolve_snapshot(ref.revision)
            if staged != ref or ref.parent_revision != expected_parent.revision:
                raise ArtifactPublicationError("committed snapshot differs from its staged identity or parent")
            current = self._workspace.ls_remote(self.ref_name)
            if current == ref.revision:
                return
            if current != expected_parent.revision:
                raise ArtifactConflictError("repository head changed before the staged snapshot was committed")
            try:
                self._workspace.force_push_with_lease(
                    f"--force-with-lease={self.ref_name}:{expected_parent.revision}",
                    f"{ref.revision}:{self.ref_name}",
                )
            except ArtifactPublicationError as exc:
                current = self._workspace.ls_remote(self.ref_name)
                if current == ref.revision:
                    return
                if current != expected_parent.revision:
                    raise ArtifactConflictError("repository head changed while committing a staged snapshot") from exc
                raise

    def _open_repository(self) -> None:
        with self._workspace.lock:
            self._workspace.open_repository()


@dataclass(frozen=True)
class GitLFSSettings(PluginSettings):
    """The config of ``GitLFSRepositoryBackendFactory``."""

    repository: str = config_option(help="The Git repository: a local path, created on first use, or a remote URL.")
    refs: str = config_option(
        "reef",
        help=(
            "Where the refs live: reef keeps them under refs/reef/; branches-and-tags keeps them on branches and "
            "tags under reef/, which a Hugging Face repo needs."
        ),
    )
    token_env: str | None = config_option(
        None, help="The environment variable of a token that authorizes an https:// repository, such as HF_TOKEN."
    )

    def __post_init__(self) -> None:
        if not self.repository.strip():
            raise ValueError("repository must be non-empty")
        if self.refs not in REF_LAYOUTS:
            raise ValueError(f"refs must be one of {sorted(REF_LAYOUTS)}, got {self.refs!r}")


class GitLFSRepositoryBackendFactory(CachedRepositoryBackendFactory, ConfigurableRepositoryBackendFactory):
    """Scenario backends over one Git LFS repository; ``storage.artifact_backend: git_lfs`` selects it."""

    def __init__(
        self,
        backend_type: type[GitLFSRepositoryBackend],
        repository: str | Path,
        *,
        work_dir: Path,
        cache_dir: Path,
        refs: RefLayout = REEF_REFS,
        git_config: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._repository = repository
        self._work_dir = Path(work_dir)
        self._cache_dir = Path(cache_dir)
        self._refs = refs
        self._git_config = dict(git_config or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: RepositoryBackendContext) -> Self:
        settings = GitLFSSettings.parse(config, plugin_class=cls)
        repository = settings.repository
        git_config: dict[str, str] = {}
        if settings.token_env is not None:
            token = os.environ.get(settings.token_env, "").strip()
            if not token:
                raise ArtifactSourceError(f"the artifact repository's token needs the variable {settings.token_env}")
            git_config = token_config(repository, token)
        _check_tools()
        return cls(
            GitLFSRepositoryBackend,
            repository if "://" in repository or repository.startswith("git@") else Path(repository),
            work_dir=context.work_dir,
            cache_dir=context.cache_dir,
            refs=REF_LAYOUTS[settings.refs],
            git_config=git_config,
        )

    def _build_backend(self, scenario: str) -> GitLFSRepositoryBackend:
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        return self._backend_type(
            scenario,
            self._repository,
            work_dir=self._work_dir / encoded,
            cache_dir=self._cache_dir,
            refs=self._refs,
            git_config=self._git_config,
        )

    def _has_persisted_registration(self, scenario: str) -> bool:
        if isinstance(self._repository, Path) and not self._repository.exists():
            return False
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        ref_name = self._refs.scenario(encoded)
        output = GitClient(self._work_dir / ".registration-check", self._git_config).run(
            ("git", "ls-remote", str(self._repository), ref_name),
            source_error=True,
        )
        return bool(output)

    def _archive_persisted_registration(self, scenario: str) -> tuple[str, ...]:
        """Rename the scenario's ref into the layout's ``archived/<scenario>/<time>`` and drop its work clone.

        The commits stay reachable under the archived ref, so nothing the
        scenario published is lost; its pending snapshots stay under the
        snapshots namespace. Only a local bare
        repository can be renamed in place.
        """
        if not isinstance(self._repository, Path):
            raise NotImplementedError("archiving a scenario needs a local artifact repository")
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        archived: list[str] = []
        ref_name = self._refs.scenario(encoded)
        if self._repository.exists():
            client = GitClient(self._work_dir / ".registration-check")
            git_dir = ("git", "--git-dir", str(self._repository))
            sha = client.run((*git_dir, "rev-parse", "--verify", "--quiet", ref_name), source_error=True)
            if sha:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                archived_ref = self._refs.archived(encoded, stamp)
                client.run((*git_dir, "update-ref", archived_ref, sha), source_error=True)
                client.run((*git_dir, "update-ref", "-d", ref_name, sha), source_error=True)
                archived.append(archived_ref)
        work_dir = self._work_dir / encoded
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            archived.append(str(work_dir))
        return tuple(archived)

    def _list_persisted_registrations(self) -> tuple[str, ...]:
        if isinstance(self._repository, Path) and not self._repository.exists():
            return ()
        output = GitClient(self._work_dir / ".registration-check", self._git_config).run(
            ("git", "ls-remote", str(self._repository), self._refs.scenario("*")),
            source_error=True,
        )
        names = []
        for line in output.splitlines():
            encoded = line.split(self._refs.scenario(""))[-1].strip()
            if not encoded:
                continue
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                names.append(base64.urlsafe_b64decode(padded).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
        return tuple(names)
