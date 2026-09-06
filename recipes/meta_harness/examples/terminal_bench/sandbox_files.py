"""Bounded, path-safe episode transport; no host paths are mounted remotely."""

import io
import tarfile
from pathlib import Path, PurePosixPath

from reef.harness.executor import EpisodeLaunchError


def episode_archive(root, writable_paths, readonly_paths=()):
    root = Path(root).resolve()
    writable = {Path(path).resolve().relative_to(root).as_posix() for path in writable_paths}
    writable.add("workspace")
    readonly = {Path(path).resolve().relative_to(root).as_posix() for path in readonly_paths}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        paths = [root, *sorted(root.rglob("*"))]
        for path in paths:
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise EpisodeLaunchError("episode inputs must be regular files and directories")
            name = path.relative_to(root).as_posix()
            info = archive.gettarinfo(str(path), arcname=name)
            can_write = any(name == item or name.startswith(item + "/") for item in writable)
            can_write = can_write and name not in readonly
            info.uid = info.gid = 1001 if can_write else 0
            info.uname = info.gname = ""
            info.mode = (0o700 if can_write else 0o555) if path.is_dir() else 0o444
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    return stream.getvalue()


def collect_outputs(stream, root, writable_paths, *, max_bytes=512 * 1024 * 1024):
    """Reject links, traversal and special files before restoring any output."""
    root = Path(root).resolve()
    prefixes = {Path(path).resolve().relative_to(root).as_posix() for path in writable_paths}
    prefixes.add("workspace")
    with tarfile.open(fileobj=stream, mode="r:*") as archive:
        members = archive.getmembers()
        if sum(item.size for item in members) > max_bytes:
            raise EpisodeLaunchError("sandbox output exceeds the evidence-size limit")
        selected = []
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise EpisodeLaunchError("unsafe path or link in sandbox output")
            if not (member.isdir() or member.isfile()):
                raise EpisodeLaunchError("special file in sandbox output")
            normalized = name.as_posix()
            if normalized in seen:
                raise EpisodeLaunchError("duplicate path in sandbox output")
            seen.add(normalized)
            if not any(normalized == item or normalized.startswith(item + "/") for item in prefixes):
                continue  # rendered inputs remain the host's frozen copy
            selected.append((member, name))
        for member, name in selected:
            destination = root.joinpath(*name.parts)
            for parent in (destination, *destination.parents):
                if parent == root:
                    break
                if parent.is_symlink():
                    raise EpisodeLaunchError("output would cross a host symlink")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("wb") as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
