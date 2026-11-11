"""The Node.js that Reef gives an npm agent in a glibc task container.

Reef downloads one Node build on its own host, checks it, and uploads it into
the container. The container then needs no download tool and no package
manager for Node: a task can replace ``curl`` or hold an old glibc, and the
agent still installs. A musl image takes the Node of its own packages
(``reef.harness.harbor.agent``).
"""

from __future__ import annotations

import gzip
import hashlib
import lzma
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from reef.core.errors import ReefError

#: The Node release that every build below is of.
NODE_VERSION = "v22.23.4"
#: The oldest glibc that the official Node builds of ``NODE_VERSION`` run on.
OFFICIAL_GLIBC = (2, 28)
#: The oldest glibc that the build for old x64 systems runs on.
OLD_GLIBC = (2, 17)
#: The shell command that prints a container's machine and, on its last line, its glibc version.
PLATFORM_COMMAND = "uname -m; getconf GNU_LIBC_VERSION 2>/dev/null || ldd --version 2>&1 | head -n 1"
#: How long one read of the download may wait.
DOWNLOAD_TIMEOUT_SECONDS = 120.0


class NodeRuntimeError(ReefError):
    """No Node build fits the container, or the build cannot be downloaded or fails its checksum."""


@dataclass(frozen=True)
class NodeBuild:
    """One published build of Node ``NODE_VERSION`` for Linux.

    Values: ``url`` is the address of its ``.tar.xz`` archive, and ``sha256`` is the SHA-256 of that archive as
    the publisher's ``SHASUMS256.txt`` gives it, in lowercase hexadecimal.
    Purpose: ``NODE_BUILDS`` holds the builds; ``node_archive`` downloads and checks one.
    """

    url: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.url.endswith(".tar.xz"):
            raise ValueError(f"a Node build is a .tar.xz archive, not {self.url!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal digits")


#: The builds by name. ``x64`` and ``arm64`` are nodejs.org's own. ``x64-glibc-217`` is the Node project's
#: unofficial build for x64 systems with a glibc from 2.17, which the official x64 build does not run on.
NODE_BUILDS = {
    "x64": NodeBuild(
        f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz",
        "1805bcd35e6815559bd2eb97db438d3fdb0eb3a5e6a977b7c1badca76337e9ce",
    ),
    "arm64": NodeBuild(
        f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-arm64.tar.xz",
        "6287aa46b6c375d816e63ae66d024fbbfd70404825aabdfdf898a970871d2f4b",
    ),
    "x64-glibc-217": NodeBuild(
        f"https://unofficial-builds.nodejs.org/download/release/{NODE_VERSION}/"
        f"node-{NODE_VERSION}-linux-x64-glibc-217.tar.xz",
        "021b9ef3861926d32d0323c24a554c29f26d95895bfc56f5fdc3c2be7d062aa5",
    ),
}
#: The names of ``uname -m`` by the architecture of a build.
MACHINES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
#: One download at a time: the rollouts of an evaluation start together and ask for the same build.
download_lock = threading.Lock()


def node_build_name(platform: str) -> str:
    """The name in ``NODE_BUILDS`` of the build for a container whose ``PLATFORM_COMMAND`` printed ``platform``.

    The official build of the machine for a glibc from ``OFFICIAL_GLIBC``; ``x64-glibc-217`` for an x64 machine
    with a glibc from ``OLD_GLIBC``. ``NodeRuntimeError`` for another machine, an older glibc, or an output with no
    glibc version.
    """
    lines = [line.strip() for line in platform.splitlines() if line.strip()]
    version = re.search(r"(\d+)\.(\d+)\s*$", lines[-1]) if len(lines) >= 2 else None
    architecture = MACHINES.get(lines[0]) if lines else None
    if version is None or architecture is None:
        raise NodeRuntimeError(f"Reef has no Node build for a container that reports {platform.strip()!r}")
    glibc = (int(version.group(1)), int(version.group(2)))
    if glibc >= OFFICIAL_GLIBC:
        return architecture
    if architecture == "x64" and glibc >= OLD_GLIBC:
        return "x64-glibc-217"
    raise NodeRuntimeError(f"Reef has no Node build for {architecture} with glibc {glibc[0]}.{glibc[1]}")


def node_cache_directory() -> Path:
    """The directory on this host that keeps the downloaded builds: ``reef/node`` under the user's cache."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reef" / "node"


def node_archive(name: str, cache: Path) -> Path:
    """The ``.tar.gz`` archive of the build ``name`` in ``cache``, downloaded on the first call.

    The archive holds the files of the published ``.tar.xz``, whose SHA-256 is the build's. It is gzip because a
    task image has gzip more often than xz. ``NodeRuntimeError`` when the download fails or the checksum differs;
    no archive is kept then.
    """
    build = NODE_BUILDS[name]
    archive = cache / f"node-{NODE_VERSION}-linux-{name}.tar.gz"
    with download_lock:
        if archive.is_file():
            return archive
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="node-download-", dir=cache) as work:
            published, repacked = Path(work) / "node.tar.xz", Path(work) / "node.tar.gz"
            digest = hashlib.sha256()
            try:
                with (
                    urllib.request.urlopen(build.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
                    published.open("wb") as file,
                ):
                    while chunk := response.read(1 << 20):
                        digest.update(chunk)
                        file.write(chunk)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise NodeRuntimeError(f"cannot download {build.url}: {exc}") from exc
            if digest.hexdigest() != build.sha256:
                raise NodeRuntimeError(f"{build.url} does not have the SHA-256 {build.sha256}")
            with lzma.open(published) as source, gzip.open(repacked, "wb", compresslevel=3) as target:
                shutil.copyfileobj(source, target, 1 << 20)
            repacked.replace(archive)
    return archive


__all__ = [
    "NODE_BUILDS",
    "NODE_VERSION",
    "PLATFORM_COMMAND",
    "NodeBuild",
    "NodeRuntimeError",
    "node_archive",
    "node_build_name",
    "node_cache_directory",
]
