"""Server-side resolution of the harness binary through the vendor's channel.

Reef runs evaluation episodes with the same pinned binary the served install
script hands a client, and gets it the same way: the adapter descriptor's
``install`` section names the vendor's channel, the pinned version, and where
the installed binary lands under the install prefix. This module is that
step's Python twin - the same default prefix, the same "is the pin already
installed" gate, the same vendor command - so a harness-evolving deployment
does not ask its operator to install the agent by hand before ``reef serve``.

Reef still never hosts or proxies binary bytes: the install runs the vendor's
own command. An adapter that declares no ``install`` section (reef's native
loop, terminus) resolves to its bare binary name and is found on ``PATH`` as
before, and an explicit ``evolution.binary`` skips this module entirely.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.core.errors import ReefError
from reef.harness.adapters.descriptor import AdapterDescriptor, InstallSpec

#: Where per-adapter install prefixes live, matching the ``PREFIX`` default the
#: served install script bakes in; ``REEF_HARNESS_PREFIX`` moves the root, and
#: an adapter's own prefix is that root plus the adapter name. A server and a
#: client on one machine therefore share the installed binary.
DEFAULT_PREFIX_ROOT = "~/.local/share/reef-harness"

#: Environment override for :data:`DEFAULT_PREFIX_ROOT`.
PREFIX_ENV = "REEF_HARNESS_PREFIX"

#: Beside the prefix, the ``git`` kind's installed ``repository@ref``; the
#: install script writes the same file, because ``--version`` reports the
#: package version, which a ref moves independently of.
PIN_FILE = ".reef-install-pin"

#: A vendor install clones, builds and downloads; a version probe only execs.
INSTALL_TIMEOUT_S = 900.0
VERSION_TIMEOUT_S = 60.0


class VendorInstallError(ReefError):
    """The pinned harness binary is neither installed nor installable here."""


def install_prefix(descriptor: AdapterDescriptor) -> Path:
    """The install prefix this adapter's binary lives under."""
    root = os.environ.get(PREFIX_ENV, "").strip() or DEFAULT_PREFIX_ROOT
    return Path(root).expanduser().absolute() / descriptor.name


def resolve_binary(descriptor: AdapterDescriptor, *, prefix: Path | None = None) -> str:
    """The binary an evaluation episode launches, installing it if needed.

    An adapter without an ``install`` section resolves to its descriptor's
    bare binary name, left for ``PATH`` to answer. Otherwise the pinned
    binary under ``prefix`` answers when it is already current, and the
    vendor's own install command runs when it is absent or off the pin.
    """
    install = descriptor.install
    if install is None:
        return descriptor.binary
    root = prefix if prefix is not None else install_prefix(descriptor)
    binary = root / install.binary_path
    # Use the adapter's offline/update guards without its episode-root paths.
    probe_env = {key: value for key, value in descriptor.env.items() if "{root}" not in value}
    if _is_pinned(install, binary, root, env=probe_env):
        return str(binary)
    root.mkdir(parents=True, exist_ok=True)
    _install(install, root)
    if not os.access(binary, os.X_OK):
        raise VendorInstallError(
            f"installing {_pin(install)} left no executable at {binary}; "
            f"install the harness binary yourself and set evolution.binary to its path"
        )
    return str(binary)


def _pin(install: InstallSpec) -> str:
    """The pin as an operator reads it: a package at a version, a repository at a ref."""
    if install.kind == "git":
        return f"{install.repository}@{install.ref}"
    return f"{install.package}@{install.version}"


def _is_pinned(install: InstallSpec, binary: Path, prefix: Path, *, env: Mapping[str, str]) -> bool:
    """Whether the installed binary already answers the descriptor's pin.

    The same gate the install script applies: an executable at the expected
    path whose ``--version`` reports the pinned version, and, for the ``git``
    kind, whose recorded ref matches too.
    """
    if not os.access(binary, os.X_OK):
        return False
    if install.kind == "git" and _installed_ref(prefix) != _pin(install):
        return False
    reported = f" {_reported_version(binary, env=env)} "
    if install.kind == "git":
        # A Python CLI prints its version inside a label ("Hermes Agent
        # v0.21.0"), so the match is a substring, as in the install script.
        return install.version in reported
    return f" {install.version} " in reported


def _installed_ref(prefix: Path) -> str:
    try:
        return prefix.joinpath(PIN_FILE).read_text(encoding="utf-8").rstrip("\n")
    except OSError:
        return ""


def _reported_version(binary: Path, *, env: Mapping[str, str]) -> str:
    """What ``--version`` prints, or ``""`` when the binary cannot answer."""
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.rstrip("\n")


def _install(install: InstallSpec, prefix: Path) -> None:
    if install.kind == "git":
        _install_git(install, prefix)
        return
    _run(["npm", "install", "--prefix", str(prefix), f"{install.package}@{install.version}"])


def _install_git(install: InstallSpec, prefix: Path) -> None:
    """A checkout at the pinned ref, installed editable into its own venv.

    The recorded ref goes first, so an interrupted install never reads back
    as current, and the checkout and venv are cleared before the clone, which
    refuses a non-empty target. The checkout's ``.git`` goes so the agent's
    own startup update check has nothing to fetch. Unlike the install script,
    which can assume no interpreter but ``python3``, the venv is built by the
    interpreter running Reef.
    """
    prefix.joinpath(PIN_FILE).unlink(missing_ok=True)
    source = prefix / "src"
    venv = prefix / "venv"
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(venv, ignore_errors=True)
    _run(["git", "clone", "--quiet", "--depth", "1", "--branch", install.ref, install.repository, str(source)])
    shutil.rmtree(source / ".git", ignore_errors=True)
    _run([sys.executable, "-m", "venv", str(venv)])
    _run([str(venv / "bin" / "python"), "-m", "pip", "install", "--quiet", "-e", str(source)])
    prefix.joinpath(PIN_FILE).write_text(f"{_pin(install)}\n", encoding="utf-8")


def _run(command: Sequence[str]) -> None:
    """Run one install step, raising with the vendor's own last words on failure."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as error:
        raise VendorInstallError(
            f"{command[0]} is required to install the pinned harness binary but is not on PATH"
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise VendorInstallError(f"{command[0]} install step failed: {error}") from error
    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        detail = lines[-1] if lines else ""
        raise VendorInstallError(f"{command[0]} exited {completed.returncode}: {detail}".rstrip(": "))


__all__ = [
    "DEFAULT_PREFIX_ROOT",
    "PIN_FILE",
    "PREFIX_ENV",
    "VendorInstallError",
    "install_prefix",
    "resolve_binary",
]
