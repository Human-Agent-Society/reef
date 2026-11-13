"""Harbor task environments: where Harbor builds a task's ``environment/Dockerfile`` and runs its agent and verifier.

A Harbor task environment is the ``-e`` type of ``harbor run``, or the ``environment.type`` of a trial. Harbor starts
it, builds the task's image in it, runs the agent and the verifier there, and stops it. This module names the
environment types, checks what the host needs for each, and gives the Harbor environment config of a task's trial in
each (``environment_config``); it starts nothing.

An executor is a different thing: a ``ProgramExecutor`` (:mod:`reef.execution.sandbox.process`) launches one program, such as
a harness's pinned binary, and ``ExecutorEnvironment`` (:mod:`reef.execution.sandbox.environment`) adapts one to Harbor's
``BaseEnvironment`` without building a task image, so neither builds or runs a Harbor task. A caller that runs
Harbor tasks selects an environment by name and reads its needs here, instead of naming a provider's package or
credentials.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

from reef.execution.sandbox.remote import REMOTE_TASK_ENVIRONMENTS

#: Every Harbor task environment type Reef passes to Harbor: local Docker, and each cloud type in
#: ``REMOTE_TASK_ENVIRONMENTS``.
HARBOR_TASK_ENVIRONMENTS: tuple[str, ...] = ("docker", *REMOTE_TASK_ENVIRONMENTS)
#: The top level names that Harbor's e2b environment imports from the e2b package (harbor 0.25.0,
#: ``harbor/environments/e2b.py`` lines 32 to 39); without one of them Harbor marks the environment missing.
HARBOR_E2B_NAMES = ("ALL_TRAFFIC", "AsyncSandbox", "AsyncTemplate", "FileType", "SandboxNetworkOpts", "Template")
DOCKER_PROBE_TIMEOUT_S = 5.0
#: Harbor's time limits for a task whose ``task.toml`` names none, in seconds.
HARBOR_BUILD_TIMEOUT_S = 600
HARBOR_VERIFIER_TIMEOUT_S = 600
#: What a cloud sandbox of a trial lives beyond the time limits that ``environment_config`` adds up: the agent's setup
#: within Harbor's default limit, and the file transfers.
SANDBOX_LIFETIME_MARGIN_S = 3600
#: Harbor's default lifetime of an E2B and of a Modal sandbox, in seconds; a lifetime Reef computes is never longer.
SANDBOX_LIFETIME_MAX_S = 24 * 60 * 60
#: The lifetime of the Modal sandbox of a trial whose agent has no time limit, in seconds.
MODAL_SANDBOX_DEFAULT_S = 2 * 60 * 60
#: The Harbor environment class of an ``e2b`` trial: Harbor's own, with its sandbox deleted at its timeout.
HARBOR_E2B_ENVIRONMENT = "reef.execution.sandbox.e2b:KillOnTimeoutE2BEnvironment"


class ReadinessProbe(ABC):
    """One thing a host needs before it can run work, such as a Harbor task environment's daemon or package.

    Values: ``name`` is the key that a readiness report names the need by; each probe here sets a non-empty one.
    ``missing`` returns "" if and only if the need is met; otherwise it returns a reason, which never holds a secret
    value.
    Purpose: ``task_environment_probes`` makes the probes of a task environment; a service that waits for its host,
    such as the generator's ``/healthz``, reads ``name`` and ``missing``.
    """

    name: str = ""

    @abstractmethod
    def missing(self) -> str:
        """Why the need is not met, or an empty string when it is."""


class ModuleProbe(ReadinessProbe):
    """A module imports under this interpreter; ``find_spec`` looks without importing it.

    Values: ``module`` is a non-empty module name, and ``name`` is the same name. ``hint`` is text appended to a
    reason, empty for none. ``missing`` returns "" if and only if ``module`` can be found; otherwise it returns a
    reason that names the module and the interpreter, followed by ``hint``.
    Purpose: a package that a Harbor environment imports, such as ``dockerfile_parse`` for ``e2b``, or one that the
    caller itself needs.
    """

    def __init__(self, module: str, *, hint: str = "") -> None:
        if not isinstance(module, str) or not module:
            raise ValueError("module must be a non-empty module name")
        self.name = module
        self.module = module
        self.hint = hint

    def missing(self) -> str:
        try:
            if importlib.util.find_spec(self.module) is not None:
                return ""
        except ImportError:  # a dotted name whose parent package is missing
            pass
        reason = f"{self.module} does not import under {sys.executable}"
        return f"{reason}; {self.hint}" if self.hint else reason


class ModuleNamesProbe(ReadinessProbe):
    """A module imports under this interpreter and defines the names a caller imports from it.

    Values: ``module`` is a non-empty module name, and ``name`` is the same name. ``names`` is a tuple of non-empty
    attribute names. ``hint`` is text appended to a reason, empty for none. ``missing`` returns "" if and only if
    ``module`` imports and defines every name in ``names``; otherwise it returns a reason that names the module, the
    interpreter and the import error or the absent names, followed by ``hint``.
    Purpose: the e2b package with the names Harbor's e2b environment imports, so an e2b too old for Harbor fails
    readiness instead of every task.
    """

    def __init__(self, module: str, names: tuple[str, ...], *, hint: str = "") -> None:
        if not isinstance(module, str) or not module:
            raise ValueError("module must be a non-empty module name")
        if not isinstance(names, tuple) or not all(isinstance(name, str) and name for name in names):
            raise ValueError("names must be a tuple of non-empty attribute names")
        self.name = module
        self.module = module
        self.names = names
        self.hint = hint

    def missing(self) -> str:
        try:
            namespace = vars(importlib.import_module(self.module))
        except ImportError as exc:
            reason = f"{self.module} does not import under {sys.executable}: {exc}"
        else:
            absent = [name for name in self.names if name not in namespace]
            if not absent:
                return ""
            reason = f"{self.module} under {sys.executable} lacks {', '.join(absent)}"
        return f"{reason}; {self.hint}" if self.hint else reason


class DockerProbe(ReadinessProbe):
    """The Docker daemon answers ``docker version``, so Harbor can build and run a task's container.

    Values: ``name`` is ``docker``. ``docker`` is a non-empty command, and ``timeout_s`` a positive number of
    seconds the command may take. ``missing`` returns "" if and only if the command resolves and exits 0 in time;
    otherwise it returns a reason with the command's exit code or its error.
    Purpose: the probe of the ``docker`` task environment.
    """

    name = "docker"

    def __init__(self, *, docker: str = "docker", timeout_s: float = DOCKER_PROBE_TIMEOUT_S) -> None:
        if not isinstance(docker, str) or not docker:
            raise ValueError("docker must be a non-empty command")
        if (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive number of seconds")
        self.docker = docker
        self.timeout_s = float(timeout_s)

    def missing(self) -> str:
        if shutil.which(self.docker) is None:
            return f"the {self.docker} command line is not on PATH"
        try:
            completed = subprocess.run(
                [self.docker, "version"], capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"{self.docker} version did not answer: {exc}"
        if completed.returncode != 0:
            return f"{self.docker} version exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        return ""


class EnvironmentVariableProbe(ReadinessProbe):
    """A variable of this process's environment is set and not empty.

    Values: ``variable`` is a non-empty variable name, and ``name`` is the same name. ``hint`` is text appended to
    a reason, empty for none. ``missing`` returns "" if and only if ``os.environ[variable]`` is set and not empty;
    otherwise it returns "<variable> is not set", followed by ``hint``. It names the variable and never its value,
    since the value can be a secret.
    Purpose: one probe for each credential variable of a cloud task environment (``REMOTE_TASK_ENVIRONMENTS``).
    """

    def __init__(self, variable: str, *, hint: str = "") -> None:
        if not isinstance(variable, str) or not variable:
            raise ValueError("variable must be a non-empty environment variable name")
        self.name = variable
        self.variable = variable
        self.hint = hint

    def missing(self) -> str:
        if os.environ.get(self.variable):
            return ""
        reason = f"{self.variable} is not set"
        return f"{reason}; {self.hint}" if self.hint else reason


class ProbedTaskEnvironment(ABC):
    """A Harbor task environment whose host needs this module can check.

    Values: ``name`` is the environment type, a member of ``HARBOR_TASK_ENVIRONMENTS``. ``probes(credential_hint)``
    returns one readiness probe per need of the host; ``credential_hint`` follows the reason of a missing credential.
    Purpose: ``TASK_ENVIRONMENT_PROBES`` holds one for each environment ``task_environment_probes`` checks, so a new
    environment needs its own probes and never reads another's.
    """

    name: str = ""

    @abstractmethod
    def probes(self, credential_hint: str) -> tuple[ReadinessProbe, ...]:
        """One probe per need of a host that runs Harbor tasks in this environment."""


class DockerTaskEnvironment(ProbedTaskEnvironment):
    """``docker``: a Docker daemon that answers; it has no credentials."""

    name = "docker"

    def probes(self, credential_hint: str) -> tuple[ReadinessProbe, ...]:
        return (DockerProbe(),)


class E2BTaskEnvironment(ProbedTaskEnvironment):
    """``e2b``: its credential variables (``REMOTE_TASK_ENVIRONMENTS``), the e2b package with the names Harbor imports,
    and dockerfile-parse, which Harbor's e2b environment reads each task's Dockerfile with."""

    name = "e2b"

    def probes(self, credential_hint: str) -> tuple[ReadinessProbe, ...]:
        return (
            *(
                EnvironmentVariableProbe(variable, hint=credential_hint)
                for variable in REMOTE_TASK_ENVIRONMENTS["e2b"]
            ),
            ModuleNamesProbe(
                "e2b", HARBOR_E2B_NAMES, hint="Harbor's e2b environment needs it; install reef-infra[e2b]"
            ),
            ModuleProbe(
                "dockerfile_parse",
                hint="Harbor's e2b environment reads each task's Dockerfile with it; install reef-infra[e2b]",
            ),
        )


class ModalTaskEnvironment(ProbedTaskEnvironment):
    """``modal``: its credential variables (``REMOTE_TASK_ENVIRONMENTS``) and the modal package, which Harbor's
    modal environment imports."""

    name = "modal"

    def probes(self, credential_hint: str) -> tuple[ReadinessProbe, ...]:
        return (
            *(
                EnvironmentVariableProbe(variable, hint=credential_hint)
                for variable in REMOTE_TASK_ENVIRONMENTS["modal"]
            ),
            ModuleProbe("modal", hint="Harbor's modal environment needs it; install reef-infra[modal]"),
        )


#: Each task environment whose host needs can be checked, by name: every member of ``HARBOR_TASK_ENVIRONMENTS``.
TASK_ENVIRONMENT_PROBES: Mapping[str, ProbedTaskEnvironment] = {
    environment.name: environment
    for environment in (DockerTaskEnvironment(), E2BTaskEnvironment(), ModalTaskEnvironment())
}
#: The task environments ``task_environment_probes`` can check: ``HARBOR_TASK_ENVIRONMENTS``, in the same order.
PROBED_TASK_ENVIRONMENTS: tuple[str, ...] = tuple(TASK_ENVIRONMENT_PROBES)


def checked_task_environment(environment: object, label: str = "the Harbor environment") -> str:
    """``environment`` as a task environment that ``task_environment_probes`` can check.

    Raise ``ValueError`` that names ``label`` for a value outside ``PROBED_TASK_ENVIRONMENTS``, or one that is not a
    ``str``.
    """
    if not isinstance(environment, str) or environment not in PROBED_TASK_ENVIRONMENTS:
        raise ValueError(f"{label} must be {' or '.join(PROBED_TASK_ENVIRONMENTS)}, not {environment!r}")
    return environment


def task_environment_probes(
    environment: str, *, credential_hint: str = "export it before this process starts"
) -> tuple[ReadinessProbe, ...]:
    """What a host needs to run Harbor tasks in ``environment``, one probe per need (``TASK_ENVIRONMENT_PROBES``).

    ``credential_hint`` follows the reason of a missing credential, so a caller can say which process needs it.
    Raise ``ValueError`` as ``checked_task_environment`` does.
    """
    return TASK_ENVIRONMENT_PROBES[checked_task_environment(environment)].probes(credential_hint)


class HarborEnvironmentConfig(TypedDict):
    """The ``environment`` section of a Harbor trial config, as ``environment_config`` makes it.

    Values: ``type`` is the Harbor environment type that the caller selected. ``import_path`` is present if and only
    if ``type`` is ``e2b``, and then it is ``HARBOR_E2B_ENVIRONMENT``. ``kwargs`` is present if and only if it is not
    empty. Its keys are ``sandbox_timeout_secs``, on ``e2b`` and ``modal``, an integer number of seconds at most
    ``SANDBOX_LIFETIME_MAX_S``; and ``modal_vm_runtime``, on ``modal`` only, True. A ``modal`` value always has
    ``sandbox_timeout_secs``. An ``e2b`` value has it if and only if the trial's agent has a time limit.
    Purpose: ``environment_config`` makes one, and two callers read it. The task rollout runner gives the whole value
    to an in-process Harbor trial as its environment config, through reef-eval's overrides. Harbor then makes the
    class of ``import_path`` when the value has one, and the class of ``type`` otherwise. The generator's check reads
    ``import_path`` and ``kwargs`` for the ``-e`` and ``--ek`` options of its ``harbor run`` child. The config that
    Harbor's command line makes from those options has the import path and no ``type``.
    """

    type: str
    import_path: NotRequired[str]
    kwargs: NotRequired[dict[str, bool | int]]


def environment_config(
    environment: str, task_path: Path, agent: Mapping[str, object] | None = None
) -> HarborEnvironmentConfig:
    """The Harbor environment config of a trial of the task ``task_path`` in the Harbor environment type
    ``environment``, with the Harbor agent config ``agent``.

    ``agent`` has the field names of Harbor's ``AgentConfig``. None is an agent that sets no time limit of its own,
    such as the ``oracle`` and ``nop`` agents of the generator's checks.

    It is ``{"type": environment}`` for any type but ``e2b`` and ``modal``. For those two it adds:

    - ``import_path`` on ``e2b``: ``HARBOR_E2B_ENVIRONMENT``, Harbor's E2B environment with its sandbox deleted at its
      timeout (:mod:`reef.execution.sandbox.e2b`). Harbor's own class leaves the timeout to the service, and a
      self-hosted service can pause the sandbox instead. Harbor makes the class of ``import_path`` and does not read
      ``type`` then. That class is an ``e2b`` environment too.
    - ``kwargs.modal_vm_runtime`` on ``modal``: True when the task has ``environment/docker-compose.yaml`` and its
      ``task.toml`` asks for no GPU. Modal's default runtime puts each Compose service on the host network, and
      Compose refuses that for a service that names its networks. The VM runtime keeps the task's networks and their
      DNS aliases, but it has no GPUs.
    - ``kwargs.sandbox_timeout_secs`` on both, when the agent has a time limit: the lifetime of the sandbox. As in
      Harbor's trial, the agent's time limit is ``agent["override_timeout_sec"]``, or the task's when that override
      is absent, None or 0. The task's is the ``timeout_sec`` of the ``[agent]`` table of ``task.toml``. The lifetime
      is the sum of the task's build time limit, ``agent["override_setup_timeout_sec"]`` when the agent config sets
      it, the agent's time limit and the task's verifier time limit, plus ``SANDBOX_LIFETIME_MARGIN_S``. Harbor stops
      the sandbox when the trial ends. This lifetime ends the sandbox of a trial whose process was killed, which no
      one stops.
    - When the agent has no time limit, no other time limit counts. On ``modal``, ``kwargs.sandbox_timeout_secs`` is
      ``MODAL_SANDBOX_DEFAULT_S``. On ``e2b``, the config holds no lifetime, and the sandbox keeps Harbor's default.

    The lifetime has two limits. First, it is at most ``SANDBOX_LIFETIME_MAX_S``, Harbor's default. A trial whose time
    limits add up to more than that loses its sandbox at one day, as it does with Harbor's default. Second, the
    lifetime reads no other trial setting. It ignores ``agent["max_timeout_sec"]``, which only shortens the trial. It
    does not read Harbor's timeout multipliers or a verifier override, because no caller of this function passes one
    to Harbor.

    The config holds no task resource: Harbor reads the task's CPU, memory and time limits from its ``task.toml``.

    On ``e2b`` and ``modal``, Harbor's ``AgentConfig`` reads the two agent time limits, as it does for the trial. So
    a value that Harbor accepts counts here, such as a numeric string, and a value that Harbor refuses raises that
    model's ``ValueError``. No other field of ``agent`` is read or checked, and ``agent`` is not changed.
    """
    if environment not in ("e2b", "modal"):
        return {"type": environment}
    # Imported here: only a cloud trial needs Harbor's trial models. Only the two time limits go to the model. The
    # trial validates the other fields itself, after the caller has filled their placeholders.
    from harbor.models.trial.config import AgentConfig

    agent_fields = agent or {}
    agent_limits = AgentConfig.model_validate(
        {
            name: agent_fields[name]
            for name in ("override_timeout_sec", "override_setup_timeout_sec")
            if name in agent_fields
        }
    )
    task = tomllib.loads((task_path / "task.toml").read_text())
    task_environment = task.get("environment", {})
    config: HarborEnvironmentConfig = {"type": environment}
    kwargs: dict[str, bool | int] = {}
    if environment == "e2b":
        config["import_path"] = HARBOR_E2B_ENVIRONMENT
    elif (task_path / "environment" / "docker-compose.yaml").is_file() and task_environment.get("gpus", 0) == 0:
        kwargs["modal_vm_runtime"] = True
    # Harbor 0.25.0 (``Trial._compute_agent_timeout_sec``): the override, or the task's limit; None is no limit.
    agent_timeout_seconds = agent_limits.override_timeout_sec or task.get("agent", {}).get("timeout_sec")
    if agent_timeout_seconds is None:
        # E2B sets no lifetime here, so its sandbox keeps Harbor's default.
        if environment == "modal":
            kwargs["sandbox_timeout_secs"] = MODAL_SANDBOX_DEFAULT_S
    else:
        trial_seconds = (
            task_environment.get("build_timeout_sec", HARBOR_BUILD_TIMEOUT_S)
            # Without a setup time limit of the agent config, the margin covers Harbor's default.
            + (agent_limits.override_setup_timeout_sec or 0)
            + agent_timeout_seconds
            + task.get("verifier", {}).get("timeout_sec", HARBOR_VERIFIER_TIMEOUT_S)
        )
        # The one-day limit applies before the rounding: Harbor accepts an agent time limit of infinity, and
        # math.ceil refuses infinity.
        kwargs["sandbox_timeout_secs"] = math.ceil(
            min(trial_seconds + SANDBOX_LIFETIME_MARGIN_S, SANDBOX_LIFETIME_MAX_S)
        )
    if kwargs:
        config["kwargs"] = kwargs
    return config


__all__ = [
    "DOCKER_PROBE_TIMEOUT_S",
    "HARBOR_BUILD_TIMEOUT_S",
    "HARBOR_E2B_ENVIRONMENT",
    "HARBOR_E2B_NAMES",
    "HARBOR_TASK_ENVIRONMENTS",
    "HARBOR_VERIFIER_TIMEOUT_S",
    "MODAL_SANDBOX_DEFAULT_S",
    "PROBED_TASK_ENVIRONMENTS",
    "SANDBOX_LIFETIME_MARGIN_S",
    "SANDBOX_LIFETIME_MAX_S",
    "TASK_ENVIRONMENT_PROBES",
    "DockerProbe",
    "DockerTaskEnvironment",
    "E2BTaskEnvironment",
    "EnvironmentVariableProbe",
    "HarborEnvironmentConfig",
    "ModalTaskEnvironment",
    "ModuleNamesProbe",
    "ModuleProbe",
    "ProbedTaskEnvironment",
    "ReadinessProbe",
    "checked_task_environment",
    "environment_config",
    "task_environment_probes",
]
