"""The uv that Reef gives a task container.

Reef downloads one uv build on its own host, checks it, and uploads it into
the container. The container then needs no download tool for uv: a task image
without ``curl`` still gets the bridge's relay and a uv or git agent. The
builds are static, so one build runs on a glibc image and on a musl image.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from reef.core.errors import ReefError

#: The uv release that every build below is of. The python adapter's ``install.version`` is this release.
UV_VERSION = "0.13.4"
#: How long one read of the download may wait.
DOWNLOAD_TIMEOUT_SECONDS = 120.0


class UvRuntimeError(ReefError):
    """No uv build fits the container or the version, or the build cannot be downloaded or fails its checksum."""


@dataclass(frozen=True)
class UvBuild:
    """One published static build of uv ``UV_VERSION`` for Linux.

    Values: ``url`` is the address of its ``.tar.gz`` archive, which holds ``uv`` and ``uvx`` in one directory, and
    ``sha256`` is the SHA-256 of that archive as the publisher's ``.sha256`` file gives it, in lowercase
    hexadecimal.
    Purpose: ``UV_BUILDS`` holds the builds; ``uv_archive`` downloads and checks one.
    """

    url: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.url.endswith(".tar.gz"):
            raise ValueError(f"a uv build is a .tar.gz archive, not {self.url!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal digits")


RELEASE = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}"
#: Astral's static (musl) builds by architecture.
UV_BUILDS = {
    "x86_64": UvBuild(
        f"{RELEASE}/uv-x86_64-unknown-linux-musl.tar.gz",
        "28d4385f9f9d1d1c7654d85b3855e0e2615b2f6cc13136a2edff858bbdc16ff1",
    ),
    "aarch64": UvBuild(
        f"{RELEASE}/uv-aarch64-unknown-linux-musl.tar.gz",
        "253a17cadcb659ae77a0fe3b9a2f1c14297528d480c421899e61e64d64f24e8f",
    ),
}
#: The architecture of a build by the name that ``uname -m`` prints.
MACHINES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
#: One download at a time: the rollouts of an evaluation start together and ask for the same build.
download_lock = threading.Lock()


def uv_build_name(version: str, machine: str) -> str:
    """The name in ``UV_BUILDS`` of the build of uv ``version`` for a container whose ``uname -m`` printed
    ``machine``. ``UvRuntimeError`` for a version other than ``UV_VERSION`` or for another machine."""
    if version != UV_VERSION:
        raise UvRuntimeError(f"Reef has the checksums of uv {UV_VERSION}, not of uv {version}")
    name = MACHINES.get(machine.strip())
    if name is None:
        raise UvRuntimeError(f"Reef has no uv build for a container that reports {machine.strip()!r}")
    return name


def uv_cache_directory() -> Path:
    """The directory on this host that keeps the downloaded builds: ``reef/uv`` under the user's cache."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reef" / "uv"


def uv_archive(name: str, cache: Path) -> Path:
    """The published ``.tar.gz`` archive of the build ``name`` in ``cache``, downloaded on the first call.

    ``UvRuntimeError`` when the download fails or the checksum differs; no archive is kept then.
    """
    build = UV_BUILDS[name]
    archive = cache / f"uv-{UV_VERSION}-linux-{name}.tar.gz"
    with download_lock:
        if archive.is_file():
            return archive
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="uv-download-", dir=cache) as work:
            published = Path(work) / "uv.tar.gz"
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
                raise UvRuntimeError(f"cannot download {build.url}: {exc}") from exc
            if digest.hexdigest() != build.sha256:
                raise UvRuntimeError(f"{build.url} does not have the SHA-256 {build.sha256}")
            published.replace(archive)
    return archive


__all__ = [
    "UV_BUILDS",
    "UV_VERSION",
    "UvBuild",
    "UvRuntimeError",
    "uv_archive",
    "uv_build_name",
    "uv_cache_directory",
]
