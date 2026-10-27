"""Build the executor a deployment selected; the only place that names a sandbox kind."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from reef.core.errors import ReefError
from reef.sandbox.e2b import E2BExecutor
from reef.sandbox.modal import ModalExecutor
from reef.sandbox.process import EpisodeExecutor, LocalExecutor, SandboxExecutor, SandboxLimits, SandboxUnavailable
from reef.sandbox.remote import RemoteExecutor, deployment_owner

#: The cloud sandbox providers, by the name a deployment selects.
REMOTE_EXECUTORS: Mapping[str, type[RemoteExecutor]] = {"e2b": E2BExecutor, "modal": ModalExecutor}


def remote_executor(
    kind: str,
    section: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    timeout_s: float | None = None,
    owner_anchor: Path | None = None,
) -> RemoteExecutor:
    """The cloud sandbox executor of provider ``kind``, from the provider's options in ``section``.

    ``section`` may hold other settings of its caller: only ``kind``'s own options
    (``RemoteExecutor.provider_options``) and the shared ``forward_ports`` and ``env_from`` are read. ``timeout_s``
    sets how long one run may take; ``owner_anchor``, a deployment's state directory, names the deployment on its
    sandboxes so a restart stops the ones a previous run left. Raise ``SandboxUnavailable`` for an unknown ``kind``
    or a malformed option; the caller preflights the result.
    """
    executor_class = REMOTE_EXECUTORS.get(kind)
    if executor_class is None:
        raise SandboxUnavailable(f"unknown cloud sandbox {kind!r}; use one of {', '.join(sorted(REMOTE_EXECUTORS))}")
    shared = executor_class.provider_options | {"forward_ports", "env_from"}
    executor = executor_class.from_config({key: value for key, value in section.items() if key in shared}, environ)
    if timeout_s is not None:
        executor = replace(executor, timeout_s=timeout_s)
    if owner_anchor is not None:
        executor = replace(executor, owner=deployment_owner(owner_anchor))
    return executor


def build_executor(
    config: Mapping[str, object] | None, *, environ: Mapping[str, str] | None = None
) -> EpisodeExecutor:
    """Build the executor a deployment selected.

    ``executor`` is ``local`` (default), ``sandbox``, or a cloud sandbox (``e2b``, ``modal``); a sandbox reads its
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
    if isinstance(kind, str) and kind in REMOTE_EXECUTORS:
        section = config.get("sandbox") or {}
        if not isinstance(section, Mapping):
            raise ReefError("evolution.sandbox must be a mapping")
        remote = REMOTE_EXECUTORS[kind].from_config(section, values)
        remote.preflight()
        return remote
    if kind != "sandbox":
        kinds = ", ".join(f"'{name}'" for name in ("local", "sandbox", *sorted(REMOTE_EXECUTORS)))
        raise ReefError(f"unknown episode executor {kind!r}; use one of {kinds}")
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
