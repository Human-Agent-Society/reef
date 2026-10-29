"""Harbor task environments: where Harbor builds a task's ``environment/Dockerfile`` and runs its agent and verifier.

A Harbor task environment is the ``-e`` type of ``harbor run``, or the ``environment.type`` of a trial. Harbor starts
it, builds the task's image in it, runs the agent and the verifier there, and stops it. This module names the
environment types and checks what the host needs for each; it starts nothing.

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
from abc import ABC, abstractmethod
from collections.abc import Mapping

from reef.execution.sandbox.remote import REMOTE_TASK_ENVIRONMENTS

#: Every Harbor task environment type Reef passes to Harbor: local Docker, and each cloud type in
#: ``REMOTE_TASK_ENVIRONMENTS``.
HARBOR_TASK_ENVIRONMENTS: tuple[str, ...] = ("docker", *REMOTE_TASK_ENVIRONMENTS)
#: The top level names that Harbor's e2b environment imports from the e2b package (harbor 0.20.0,
#: ``harbor/environments/e2b.py`` lines 30 to 60); without one of them Harbor marks the environment missing.
HARBOR_E2B_NAMES = ("ALL_TRAFFIC", "AsyncSandbox", "AsyncTemplate", "FileType", "SandboxNetworkOpts", "Template")
DOCKER_PROBE_TIMEOUT_S = 5.0


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


#: Each task environment whose host needs can be checked, by name. modal has credentials in
#: ``REMOTE_TASK_ENVIRONMENTS`` but no tested probes yet, so a caller that needs readiness refuses it.
TASK_ENVIRONMENT_PROBES: Mapping[str, ProbedTaskEnvironment] = {
    environment.name: environment for environment in (DockerTaskEnvironment(), E2BTaskEnvironment())
}
#: The task environments ``task_environment_probes`` can check; a subset of ``HARBOR_TASK_ENVIRONMENTS``.
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


__all__ = [
    "DOCKER_PROBE_TIMEOUT_S",
    "HARBOR_E2B_NAMES",
    "HARBOR_TASK_ENVIRONMENTS",
    "PROBED_TASK_ENVIRONMENTS",
    "TASK_ENVIRONMENT_PROBES",
    "DockerProbe",
    "DockerTaskEnvironment",
    "E2BTaskEnvironment",
    "EnvironmentVariableProbe",
    "ModuleNamesProbe",
    "ModuleProbe",
    "ProbedTaskEnvironment",
    "ReadinessProbe",
    "checked_task_environment",
    "task_environment_probes",
]
