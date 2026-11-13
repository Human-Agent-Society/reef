"""Build the executor a deployment selected; the only place that names an executor kind."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError, resolve_plugin_class
from reef.execution.sandbox.e2b import E2BExecutor
from reef.execution.sandbox.modal import ModalExecutor
from reef.execution.sandbox.process import (
    LocalExecutor,
    LocalSandboxExecutor,
    ProgramExecutor,
    SandboxExecutor,
    SandboxUnavailableError,
)
from reef.execution.sandbox.remote import RemoteExecutor, deployment_owner
from reef.execution.sandbox.seatbelt import SeatbeltExecutor

#: The cloud sandbox providers, by the name a deployment selects.
REMOTE_EXECUTORS: Mapping[str, type[RemoteExecutor]] = {"e2b": E2BExecutor, "modal": ModalExecutor}


def host_sandbox() -> type[LocalSandboxExecutor]:
    """The local sandbox this host runs: Seatbelt on macOS, else bubblewrap."""
    return SeatbeltExecutor if sys.platform == "darwin" else SandboxExecutor


def remote_executor(
    kind: str,
    section: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    timeout_s: float | None = None,
    owner_anchor: Path | None = None,
) -> RemoteExecutor:
    """The cloud sandbox executor of provider ``kind``, from the provider's options in ``section``.

    ``kind`` is a key of ``REMOTE_EXECUTORS``, or the ``package.module:ClassName`` reference of another
    ``RemoteExecutor`` subclass.

    ``section`` may hold other settings of its caller: only ``kind``'s own options
    (``RemoteExecutor.provider_options``) and the shared ``forward_ports`` and ``env_from`` are read. ``timeout_s``
    sets how long one run may take; ``owner_anchor``, a deployment's state directory, names the deployment on its
    sandboxes so a restart stops the ones a previous run left. Raise ``SandboxUnavailableError`` for an unknown
    ``kind`` or a malformed option; the caller preflights the result.
    """
    try:
        executor_class = resolve_plugin_class(
            kind,
            builtins=REMOTE_EXECUTORS,
            base=RemoteExecutor,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
            setting="cloud sandbox",
        )
    except PluginConfigError as exc:
        raise SandboxUnavailableError(str(exc)) from exc
    shared = executor_class.provider_options | {"forward_ports", "env_from"}
    executor = executor_class.from_config({key: value for key, value in section.items() if key in shared}, environ)
    if timeout_s is not None:
        executor = replace(executor, timeout_s=timeout_s)
    if owner_anchor is not None:
        executor = replace(executor, owner=deployment_owner(owner_anchor))
    return executor


def build_executor(
    config: Mapping[str, object] | None, *, environ: Mapping[str, str] | None = None
) -> ProgramExecutor:
    """Build the executor a deployment selected.

    ``executor`` is ``local`` (default), ``sandbox``, or a cloud sandbox (``e2b``, ``modal``). ``sandbox`` is the
    host's local sandbox (``host_sandbox``): bubblewrap on Linux, Seatbelt on macOS. Each executor reads its
    options from the ``sandbox`` section with its ``from_config``. The executor is
    preflighted at build so a hosted deployment that requires the sandbox
    fails to start, not at the first run. The caller binds the result to
    the program it runs with :meth:`ProgramExecutor.for_program`.
    """
    config = config or {}
    kind = config.get("executor", "local")
    values = os.environ if environ is None else environ
    if kind == "local":
        return LocalExecutor()
    section = config.get("sandbox") or {}
    if not isinstance(section, Mapping):
        raise ReefError("evolution.sandbox must be a mapping")
    executor: ProgramExecutor
    if isinstance(kind, str) and kind in REMOTE_EXECUTORS:
        executor = REMOTE_EXECUTORS[kind].from_config(section, values)
    elif kind == "sandbox":
        executor = host_sandbox().from_config(section, values)
    else:
        kinds = ", ".join(f"'{name}'" for name in ("local", "sandbox", *sorted(REMOTE_EXECUTORS)))
        raise ReefError(f"unknown run executor {kind!r}; use one of {kinds}")
    executor.preflight()
    return executor
