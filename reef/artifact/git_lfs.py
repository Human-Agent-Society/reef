from __future__ import annotations

import base64
import json
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Lock

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactMaterializationError,
    ArtifactNotFound,
    ArtifactPublicationError,
    ArtifactRef,
    ArtifactSourceError,
)
from reef.artifact.git_client import GitClient
from reef.artifact.repository import CachedRepositoryBackendFactory, RepositoryBackend
from reef.artifact.sources import GitVersionSource, download_huggingface_snapshot, parse_artifact_source

_MANIFEST = "reef-artifact.json"
_LFS_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")


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


class _GitWorkspace:
    """Manage one git working tree used by a single scenario backend.

    Encapsulates git clone/checkout/commit/push so the backend does not
    call git directly. Holds the work_dir, clone_dir, and a per-process
    lock serializing mutations.
    """

    def __init__(
        self,
        *,
        repository: str,
        work_dir: Path,
        git_client: GitClient,
    ) -> None:
        self.repository = repository
        self.work_dir = Path(work_dir)
        self.clone_dir = self.work_dir / "repository"
        self._git_client = git_client
        self.lock = Lock()

    def open_repository(self) -> None:
        if not (self.clone_dir / ".git").exists():
            if self.clone_dir.exists():
                shutil.rmtree(self.clone_dir)
            self._run(("git", "clone", "--no-checkout", self.repository, str(self.clone_dir)))
        self.git("config", "user.name", "Reef Repository Backend")
        self.git("config", "user.email", "reef-artifacts@localhost")
        self.git("config", "core.hooksPath", str(self.clone_dir / ".git" / "hooks"))
        self.git("lfs", "install", "--local", "--skip-smudge")

    def checkout(self, version: str) -> None:
        self.git("fetch", "origin", version)
        self.git("checkout", "--detach", "FETCH_HEAD")
        self.git("reset", "--hard", "FETCH_HEAD")
        self.git("clean", "-fdx")

    def orphan_checkout(self) -> None:
        self.git("checkout", "--orphan", f"reef-initial-{uuid.uuid4().hex}")
        self.git("rm", "-rf", "--ignore-unmatch", ".")
        self.git("clean", "-fdx")

    def replace_tree(self, source: Path) -> None:
        for child in self.clone_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in source.iterdir():
            destination = self.clone_dir / child.name
            if child.is_dir():
                shutil.copytree(child, destination, symlinks=False)
            else:
                try:
                    shutil.copy2(child, destination, follow_symlinks=True)
                except FileNotFoundError as exc:
                    raise ArtifactSourceError(f"artifact contains a broken symlink: {child}") from exc

    def write_lfs_attributes(self) -> None:
        attributes = "".join(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n" for pattern in _LFS_PATTERNS)
        (self.clone_dir / ".gitattributes").write_text(attributes)

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def push(self, *refs: str) -> None:
        self.git("push", "origin", *refs)

    def force_push_with_lease(self, lease: str, *refs: str) -> None:
        self.git("push", lease, "origin", *refs)

    def fetch_version(self, version: str) -> str:
        self.git("fetch", "origin", version)
        return self.git("rev-parse", "FETCH_HEAD")

    def fetch_lfs_objects(self, version: str) -> None:
        self.git("lfs", "fetch", "origin", version)

    def show_file(self, version: str, path: str) -> str:
        self.git("fetch", "origin", version)
        return self.git("show", f"FETCH_HEAD:{path}")

    def ls_remote(self, ref_name: str) -> str | None:
        output = self.git("ls-remote", "origin", ref_name)
        if not output:
            return None
        return output.split()[0]

    def has_refs(self) -> bool:
        return bool(self.git("ls-remote", "origin"))

    def clone_for_materialize(self, destination: Path, version: str) -> None:
        self._run(("git", "clone", "--no-checkout", self.repository, str(destination)))
        self._run(("git", "lfs", "install", "--local", "--skip-smudge"), cwd=destination)
        self._run(("git", "fetch", "origin", version), cwd=destination)
        self._run(("git", "checkout", "--detach", "FETCH_HEAD"), cwd=destination)
        self._run(("git", "lfs", "pull"), cwd=destination)

    def git(self, *args: str) -> str:
        return self._git_client.git(*args)

    def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        source_error: bool = False,
    ) -> str:
        return self._git_client.run(command, cwd=cwd, source_error=source_error)


class _ArtifactManifest:
    """Read and write the reef-artifact.json manifest inside a git work tree."""

    def __init__(self, workspace: _GitWorkspace) -> None:
        self._workspace = workspace

    def write(
        self,
        *,
        content_id: str,
        parent_release_id: str | None,
        source: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> None:
        manifest = {
            "content_id": content_id,
            "parent_release_id": parent_release_id,
            "source": dict(source),
            "metadata": dict(metadata),
        }
        (self._workspace.clone_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def read(self, version: str) -> Mapping[str, object]:
        raw = self._workspace.show_file(version, _MANIFEST)
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArtifactSourceError(f"invalid artifact manifest at {version}") from exc
        if not isinstance(manifest, Mapping):
            raise ArtifactSourceError(f"invalid artifact manifest at {version}")
        return manifest

    def artifact_ref(self, version: str) -> ArtifactRef:
        manifest = self.read(version)
        try:
            content_id = manifest["content_id"]
            if not isinstance(content_id, str):
                raise TypeError("content_id must be a string")
            parent_release_id = manifest.get("parent_release_id")
            if parent_release_id is not None and not isinstance(parent_release_id, str):
                raise TypeError("parent_release_id must be a string or null")
            return ArtifactRef(
                content_id=content_id,
                release_id=self._workspace.fetch_version(version),
                parent_release_id=parent_release_id,
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactSourceError(f"invalid artifact manifest at {version}") from exc


class GitLFSRepositoryBackend(RepositoryBackend):
    def __init__(
        self,
        scenario: str,
        repository: str | Path,
        bootstrap_artifact: str | None = None,
        *,
        work_dir: Path,
        cache_dir: Path,
        snapshot_download: Callable[..., str] | None = None,
        bootstrap_files: Mapping[str, str] | None = None,
    ) -> None:
        if not scenario:
            raise ValueError("scenario must be non-empty")
        _check_tools()
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        self.ref_name = f"refs/reef/scenarios/{encoded}"
        self.work_dir = Path(work_dir)
        self.cache_dir = Path(cache_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._git_client = GitClient(self.work_dir / "repository")
        local_repository = repository if isinstance(repository, Path) else None
        if local_repository is not None and not local_repository.exists():
            _initialize_local_repository(local_repository, self._git_client)
        self.repository = str(repository)
        self._workspace = _GitWorkspace(
            repository=self.repository,
            work_dir=self.work_dir,
            git_client=self._git_client,
        )
        self._manifest = _ArtifactManifest(self._workspace)
        self._open_repository()
        if bootstrap_artifact is not None:
            self._bootstrap(bootstrap_artifact, snapshot_download=snapshot_download)
        elif local_repository is not None:
            self._bootstrap_empty(bootstrap_files or {})

    @classmethod
    def factory(
        cls,
        repository: str | Path,
        bootstrap_artifact: str | None = None,
        *,
        work_dir: Path,
        cache_dir: Path,
        snapshot_download: Callable[..., str] | None = None,
        bootstrap_files: Mapping[str, str] | None = None,
    ) -> CachedRepositoryBackendFactory:
        _check_tools()
        return _GitLFSRepositoryBackendFactory(
            cls,
            repository,
            bootstrap_artifact,
            work_dir=work_dir,
            cache_dir=cache_dir,
            snapshot_download=snapshot_download,
            bootstrap_files=bootstrap_files,
        )

    def resolve_release(self, release_id: str | None = None) -> ArtifactRef:
        if release_id is None or release_id == "head":
            resolved_release = self._workspace.ls_remote("refs/reef/head") or self._workspace.ls_remote(
                "refs/reef/base"
            )
            if resolved_release is None:
                raise ArtifactNotFound("artifact repository has no head release")
            return self._manifest.artifact_ref(resolved_release)
        selector = release_id.removeprefix("git+lfs://")
        resolved_release = self._workspace.ls_remote(f"refs/reef/releases/{selector}")
        if resolved_release is None:
            try:
                resolved_release = self._workspace.fetch_version(selector)
            except ArtifactPublicationError as exc:
                raise ArtifactNotFound(f"release does not exist: {release_id}") from exc
        return self._manifest.artifact_ref(resolved_release)

    def fork(
        self,
        release_id: str | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        with self._workspace.lock:
            existing = self._workspace.ls_remote(self.ref_name)
            if existing is not None:
                return self._manifest.artifact_ref(existing)
            selected = self.resolve_release(release_id)
            self._workspace.checkout(selected.release_id)
            self._workspace.fetch_lfs_objects(selected.release_id)
            self._manifest.write(
                content_id=selected.content_id,
                parent_release_id=selected.release_id,
                source={"kind": "fork", "release_id": selected.release_id},
                metadata=metadata or {},
            )
            commit = self._workspace.commit("fork scenario repository")
            try:
                self._workspace.push(f"{commit}:{self.ref_name}")
            except ArtifactPublicationError:
                existing = self._workspace.ls_remote(self.ref_name)
                if existing is None:
                    raise
                return self._manifest.artifact_ref(existing)
            return self._manifest.artifact_ref(commit)

    def metadata(self) -> Mapping[str, object] | None:
        version = self._workspace.ls_remote(self.ref_name)
        if version is None:
            return None
        manifest = self._manifest.read(version)
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ArtifactSourceError(f"invalid artifact metadata at {version}")
        return dict(metadata)

    def current(self) -> ArtifactRef:
        version = self._workspace.ls_remote(self.ref_name)
        if version is None:
            raise ArtifactNotFound("repository has no current artifact")
        return self._manifest.artifact_ref(version)

    def materialize(self, ref: ArtifactRef) -> Artifact:
        destination = self.cache_dir / ref.release_id
        if destination.is_dir():
            return Artifact(ref, None, local_path=destination)
        temporary = Path(tempfile.mkdtemp(prefix=f".{ref.release_id}-", dir=self.cache_dir))
        checkout = temporary / "artifact"
        try:
            self._workspace.clone_for_materialize(checkout, ref.release_id)
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
            raise ArtifactMaterializationError(f"failed to materialize artifact {ref.release_id}: {exc}") from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if not destination.is_dir():
            raise ArtifactMaterializationError(f"artifact cache was not created: {ref.release_id}")
        return Artifact(ref, None, local_path=destination)

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
        advance_head: bool = True,
    ) -> ArtifactRef:
        if artifact.local_path is None or not artifact.local_path.is_dir():
            raise ArtifactPublicationError("artifact ref must contain an existing local artifact directory")
        with self._workspace.lock:
            current = self._workspace.ls_remote(self.ref_name)
            if current != expected_parent.release_id:
                raise ArtifactConflict(f"repository is at {current}, not expected parent {expected_parent.release_id}")
            self._workspace.checkout(expected_parent.release_id)
            self._workspace.replace_tree(artifact.local_path)
            self._workspace.write_lfs_attributes()
            self._manifest.write(
                content_id=artifact.ref.content_id,
                parent_release_id=expected_parent.release_id,
                source={"kind": "training"},
                metadata=artifact.metadata,
            )
            commit = self._workspace.commit("publish artifact")
            if not advance_head:
                # A pending release lives under the releases namespace; the branch and refs/reef/head stay put.
                self._workspace.push(f"+{commit}:refs/reef/releases/{commit}")
                return self._manifest.artifact_ref(commit)
            try:
                self._workspace.force_push_with_lease(
                    f"--force-with-lease={self.ref_name}:{expected_parent.release_id}",
                    f"{commit}:{self.ref_name}",
                    f"+{commit}:refs/reef/head",
                )
            except ArtifactPublicationError as exc:
                current = self._workspace.ls_remote(self.ref_name)
                if current != expected_parent.release_id:
                    raise ArtifactConflict(f"repository advanced from {expected_parent.release_id}") from exc
                raise
            return self._manifest.artifact_ref(commit)

    def _bootstrap(
        self,
        artifact_source: str,
        *,
        snapshot_download: Callable[..., str] | None,
    ) -> ArtifactRef:
        source = parse_artifact_source(artifact_source)
        if isinstance(source, GitVersionSource):
            version = self._workspace.fetch_version(source.version)
            self._workspace.push(f"+{version}:refs/reef/head")
            return self._manifest.artifact_ref(version)
        existing = self._workspace.ls_remote("refs/reef/base")
        if existing is not None:
            if self._workspace.ls_remote("refs/reef/head") is None:
                self._workspace.push(f"{existing}:refs/reef/head")
            return self._manifest.artifact_ref(existing)
        downloaded = download_huggingface_snapshot(
            source,
            snapshot_download=snapshot_download,
        )
        with self._workspace.lock:
            self._workspace.orphan_checkout()
            self._workspace.replace_tree(downloaded.local_path)
            self._workspace.write_lfs_attributes()
            self._manifest.write(
                content_id=f"content:{uuid.uuid4().hex}",
                parent_release_id=None,
                source={
                    "kind": "huggingface",
                    "name": source.model_name,
                    "version": downloaded.version,
                },
                metadata={},
            )
            commit = self._workspace.commit("import bootstrap artifact")
            self._workspace.push(
                f"{commit}:refs/reef/base",
                f"{commit}:refs/reef/head",
            )
            return self._manifest.artifact_ref(commit)

    def _bootstrap_empty(self, files: Mapping[str, str] = {}) -> ArtifactRef:
        """The base artifact of a local repository: ``files`` when the recipe seeds one, else empty.

        An existing base always wins, so a redeploy with a changed seed keeps
        the base its scenarios forked from."""
        existing = self._bootstrap_ref()
        if existing is not None:
            return existing
        if self._workspace.has_refs():
            existing = self._bootstrap_ref()
            if existing is not None:
                return existing
            raise ArtifactSourceError("artifact repository has refs but no base or head release")
        with self._workspace.lock:
            self._workspace.orphan_checkout()
            for relative, text in files.items():
                target = self._workspace.clone_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            self._manifest.write(
                content_id=f"content:{uuid.uuid4().hex}",
                parent_release_id=None,
                source={"kind": "seed" if files else "empty"},
                metadata={},
            )
            commit = self._workspace.commit("initialize artifact repository")
            try:
                self._workspace.push(f"{commit}:refs/reef/base")
                base = commit
            except ArtifactPublicationError:
                winner = self._workspace.ls_remote("refs/reef/base")
                if winner is None:
                    raise
                base = winner
            return self._manifest.artifact_ref(self._restore_head(base))

    def _bootstrap_ref(self) -> ArtifactRef | None:
        base = self._workspace.ls_remote("refs/reef/base")
        head = self._workspace.ls_remote("refs/reef/head")
        if base is not None:
            return self._manifest.artifact_ref(head or self._restore_head(base))
        if head is not None:
            return self._manifest.artifact_ref(head)
        return None

    def _restore_head(self, base: str) -> str:
        head = self._workspace.ls_remote("refs/reef/head")
        if head is not None:
            return head
        self._workspace.fetch_version(base)
        try:
            self._workspace.push(f"{base}:refs/reef/head")
            return base
        except ArtifactPublicationError:
            head = self._workspace.ls_remote("refs/reef/head")
            if head is None:
                raise
            return head

    def _open_repository(self) -> None:
        self._workspace.open_repository()


class _GitLFSRepositoryBackendFactory(CachedRepositoryBackendFactory):
    def __init__(
        self,
        backend_type: type[GitLFSRepositoryBackend],
        repository: str | Path,
        bootstrap_artifact: str | None,
        *,
        work_dir: Path,
        cache_dir: Path,
        snapshot_download: Callable[..., str] | None,
        bootstrap_files: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._repository = repository
        self._bootstrap_artifact = bootstrap_artifact
        self._work_dir = Path(work_dir)
        self._cache_dir = Path(cache_dir)
        self._snapshot_download = snapshot_download
        self._bootstrap_files = None if bootstrap_files is None else dict(bootstrap_files)

    def _build_backend(self, scenario: str) -> GitLFSRepositoryBackend:
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        return self._backend_type(
            scenario,
            self._repository,
            self._bootstrap_artifact,
            work_dir=self._work_dir / encoded,
            cache_dir=self._cache_dir,
            snapshot_download=self._snapshot_download,
            bootstrap_files=self._bootstrap_files,
        )

    def _has_persisted_registration(self, scenario: str) -> bool:
        if isinstance(self._repository, Path) and not self._repository.exists():
            return False
        encoded = base64.urlsafe_b64encode(scenario.encode()).decode().rstrip("=")
        ref_name = f"refs/reef/scenarios/{encoded}"
        output = GitClient(self._work_dir / ".registration-check").run(
            ("git", "ls-remote", str(self._repository), ref_name),
            source_error=True,
        )
        return bool(output)

    def _list_persisted_registrations(self) -> tuple[str, ...]:
        if isinstance(self._repository, Path) and not self._repository.exists():
            return ()
        output = GitClient(self._work_dir / ".registration-check").run(
            ("git", "ls-remote", str(self._repository), "refs/reef/scenarios/*"),
            source_error=True,
        )
        names = []
        for line in output.splitlines():
            encoded = line.split("refs/reef/scenarios/")[-1].strip()
            if not encoded:
                continue
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                names.append(base64.urlsafe_b64decode(padded).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
        return tuple(names)
