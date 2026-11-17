"""An npm agent that Reef installs on its own host for a task container's platform.

npm on the host installs the agent's package for Linux, for the container's
CPU and C library, without running a script of the package. Reef keeps the
result as one archive in its cache and uploads it into each container, so the
container needs no package registry and no name resolution: a task can break
its network or leave little room in the user's home. In the container,
``npm rebuild`` runs the package's install scripts offline
(``reef.harness.harbor.agent``).
"""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Literal

from reef.core.errors import ReefError
from reef.harness.harbor.node_runtime import host_node, node_cache_directory

#: The C library of a container, as npm names it.
ContainerLibc = Literal["glibc", "musl"]
#: The directory of the archive that holds the npm prefix: ``npm/bin`` and ``npm/lib/node_modules``.
BUNDLE_PREFIX = "npm"
#: How long npm may take to install one agent on the host.
INSTALL_TIMEOUT_SECONDS = 900.0
#: One install at a time: the rollouts of an evaluation start together and ask for the same bundle.
build_lock = threading.Lock()


class AgentBundleError(ReefError):
    """npm on the host cannot install the agent for the container's platform."""


def agent_cache_directory() -> Path:
    """The directory on this host that keeps the agent bundles and npm's download cache: ``reef/agents`` under the
    user's cache."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reef" / "agents"


def root_owned(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """``member`` owned by root, so that the container's root unpacks files that no host user owns."""
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def agent_bundle(package: str, version: str, cpu: str, libc: ContainerLibc, cache: Path) -> Path:
    """The ``.tar.gz`` archive in ``cache`` of the npm package ``package`` at ``version``, installed for Linux on
    ``cpu`` (``x64`` or ``arm64``) with ``libc``; built on the first call for these four values.

    The archive holds the npm global prefix as ``BUNDLE_PREFIX``: the package with its dependencies for that
    platform, and its commands under ``bin``. No install script ran. ``AgentBundleError`` when npm fails or runs past
    ``INSTALL_TIMEOUT_SECONDS``; no archive is kept then. For ``glibc``, npm also fails when a package needs a Node
    that the pinned release does not satisfy: the host and a glibc container run that release. ``NodeRuntimeError``
    when the host has no Node build.
    """
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", package).strip("-")
    archive = cache / f"{name}@{version}-linux-{cpu}-{libc}.tar.gz"
    with build_lock:
        if archive.is_file():
            return archive
        node = host_node(node_cache_directory())
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agent-build-", dir=cache) as work:
            prefix = Path(work) / BUNDLE_PREFIX
            command = [
                str(node / "bin" / "node"),
                str(node / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"),
                "install",
                "--global",
                f"--prefix={prefix}",
                "--os=linux",
                f"--cpu={cpu}",
                f"--libc={libc}",
                # A script of the package would run for the host, not for the container: the container runs them.
                "--ignore-scripts",
                # This npm runs on the Node release that a glibc container gets, so the check of the packages'
                # engines is that container's. A musl container has the Node of its image, which the host cannot check.
                *(["--engine-strict"] if libc == "glibc" else []),
                "--no-audit",
                "--no-fund",
                f"--cache={cache / 'npm-cache'}",
                f"{package}@{version}",
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_SECONDS)
            except subprocess.CalledProcessError as exc:
                raise AgentBundleError(
                    f"npm cannot install {package}@{version} for linux {cpu} {libc}: {exc.stderr.strip()[-500:]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise AgentBundleError(
                    f"npm did not install {package}@{version} in {INSTALL_TIMEOUT_SECONDS:.0f} s"
                ) from exc
            packed = Path(work) / "bundle.tar.gz"
            with tarfile.open(packed, "w:gz", compresslevel=3) as tar:
                tar.add(prefix, arcname=BUNDLE_PREFIX, filter=root_owned)
            packed.replace(archive)
    return archive


__all__ = ["BUNDLE_PREFIX", "AgentBundleError", "ContainerLibc", "agent_bundle", "agent_cache_directory"]
