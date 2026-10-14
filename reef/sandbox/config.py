"""Build the executor a deployment selected; the only place that names a sandbox kind."""

from __future__ import annotations

import os
from collections.abc import Mapping

from reef.core.errors import ReefError
from reef.sandbox.e2b import E2BExecutor
from reef.sandbox.process import EpisodeExecutor, LocalExecutor, SandboxExecutor, SandboxLimits


def build_executor(
    config: Mapping[str, object] | None, *, environ: Mapping[str, str] | None = None
) -> EpisodeExecutor:
    """Build the executor a deployment selected.

    ``executor`` is ``local`` (default), ``sandbox`` or ``e2b``; a sandbox reads its
    ``egress_hosts``, ``limits``, and an explicit ``env_from`` list of host
    variable names from the same section. The executor is
    preflighted at build so a hosted deployment that requires the sandbox
    fails to start, not at the first episode. The caller binds the result to
    the program it runs with :meth:`EpisodeExecutor.for_program`.
    """
    config = config or {}
    kind = config.get("executor", "local")
    values = os.environ if environ is None else environ
    if kind == "local":
        return LocalExecutor()
    if kind == "e2b":
        section = config.get("sandbox") or {}
        if not isinstance(section, Mapping):
            raise ReefError("evolution.sandbox must be a mapping")
        remote = E2BExecutor.from_config(section, values)
        remote.preflight()
        return remote
    if kind != "sandbox":
        raise ReefError(f"unknown episode executor {kind!r}; use 'local', 'sandbox' or 'e2b'")
    sandbox_config = config.get("sandbox") or {}
    if not isinstance(sandbox_config, Mapping):
        raise ReefError("evolution.sandbox must be a mapping")
    egress = tuple(str(host) for host in (sandbox_config.get("egress_hosts") or ()))
    limits_config = sandbox_config.get("limits") or {}
    if not isinstance(limits_config, Mapping):
        raise ReefError("evolution.sandbox.limits must be a mapping")
    limits = SandboxLimits(
        cpu_seconds=int(limits_config.get("cpu_seconds", 0)),
        memory_bytes=int(limits_config.get("memory_bytes", 0)),
        processes=int(limits_config.get("processes", 0)),
        file_bytes=int(limits_config.get("file_bytes", 0)),
    )
    names = sandbox_config.get("env_from", [])
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in names
    ):
        raise ReefError("evolution.sandbox.env_from must be a list of environment variable names")
    env = {}
    for name in names:
        if name not in values or not values[name]:
            raise ReefError(f"evolution.sandbox.env_from requires environment variable {name!r}")
        env[name] = values[name]
    executor = SandboxExecutor(egress_hosts=egress, limits=limits, env=env)
    executor.preflight()
    return executor
