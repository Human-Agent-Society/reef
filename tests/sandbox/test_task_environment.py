"""The Harbor task environments and the readiness probes that ``reef.execution.sandbox.task_environment`` owns."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from reef.execution.sandbox.remote import REMOTE_TASK_ENVIRONMENTS
from reef.execution.sandbox.task_environment import (
    HARBOR_TASK_ENVIRONMENTS,
    PROBED_TASK_ENVIRONMENTS,
    TASK_ENVIRONMENT_PROBES,
    DockerProbe,
    EnvironmentVariableProbe,
    ModuleNamesProbe,
    ModuleProbe,
    checked_task_environment,
    task_environment_probes,
)


@pytest.mark.unit
def test_every_probed_environment_is_a_harbor_task_environment() -> None:
    assert ("docker", *REMOTE_TASK_ENVIRONMENTS) == HARBOR_TASK_ENVIRONMENTS
    assert set(PROBED_TASK_ENVIRONMENTS) <= set(HARBOR_TASK_ENVIRONMENTS)
    assert tuple(TASK_ENVIRONMENT_PROBES) == PROBED_TASK_ENVIRONMENTS


@pytest.mark.unit
def test_only_docker_and_e2b_have_probes() -> None:
    assert checked_task_environment("docker") == "docker"
    assert checked_task_environment("e2b") == "e2b"
    for value in ("modal", "Docker", "", None, 3):
        with pytest.raises(ValueError, match=r"^the Harbor environment must be docker or e2b, not "):
            checked_task_environment(value)

    class EqualsDocker:
        """Not a str, but equal to one of the names."""

        def __eq__(self, other: object) -> bool:
            return other == "docker"

        def __hash__(self) -> int:
            return hash("docker")

    with pytest.raises(ValueError, match="must be docker or e2b"):
        checked_task_environment(EqualsDocker())
    with pytest.raises(ValueError, match=r"^generator.environment must be docker or e2b, not 'modal'$"):
        checked_task_environment("modal", "generator.environment")
    with pytest.raises(ValueError, match="must be docker or e2b, not 'modal'"):
        task_environment_probes("modal")


@pytest.mark.unit
def test_each_environment_names_what_its_host_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [probe.name for probe in task_environment_probes("docker")] == ["docker"]
    e2b = task_environment_probes("e2b", credential_hint="export it before reef serve starts")
    assert [probe.name for probe in e2b] == [*REMOTE_TASK_ENVIRONMENTS["e2b"], "e2b", "dockerfile_parse"]
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    assert e2b[0].missing() == "E2B_API_KEY is not set; export it before reef serve starts"
    assert (
        task_environment_probes("e2b")[0].missing() == "E2B_API_KEY is not set; export it before this process starts"
    )
    assert EnvironmentVariableProbe("E2B_API_KEY").missing() == "E2B_API_KEY is not set"


@pytest.mark.unit
def test_a_probe_refuses_values_it_cannot_check() -> None:
    with pytest.raises(ValueError, match="non-empty module name"):
        ModuleProbe("")
    with pytest.raises(ValueError, match="non-empty command"):
        DockerProbe(docker="")
    for timeout_s in (0, -1.0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive number of seconds"):
            DockerProbe(timeout_s=timeout_s)
    with pytest.raises(ValueError, match="non-empty environment variable name"):
        EnvironmentVariableProbe("")
    with pytest.raises(ValueError, match="non-empty module name"):
        ModuleNamesProbe("", ("A",))
    with pytest.raises(ValueError, match="tuple of non-empty attribute names"):
        ModuleNamesProbe("e2b", ["A"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuple of non-empty attribute names"):
        ModuleNamesProbe("e2b", ("A", ""))


@pytest.mark.unit
def test_a_probe_answers_with_a_reason_and_never_raises(tmp_path: Path) -> None:
    reason = ModuleProbe("no_such_package_anywhere.sub").missing()
    assert reason.startswith(f"no_such_package_anywhere.sub does not import under {sys.executable}")
    if os.name != "posix":
        return
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexec sleep 5\n")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    reason = DockerProbe(docker=str(docker), timeout_s=0.2).missing()
    assert reason.startswith(f"{docker} version did not answer:"), reason
