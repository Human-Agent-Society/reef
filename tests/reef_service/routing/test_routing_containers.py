"""Ownership, command lifetime and event contracts for managed routing."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from tests.reef_service.routing.routing_fakes import routing_manifest

from reef.service.deploy.routing.routing_config import RoutingDeploymentSettings
from reef.service.deploy.routing.routing_containers import (
    SEMANTIC_ROUTER_REVISION,
    ContainerCommandError,
    DockerComposeRuntime,
    DockerEventStream,
    GuardedCommand,
)
from reef.service.deploy.routing.routing_registry import DeploymentOwnership, DeploymentRegistry, RoutingOwnershipError
from reef.service.deploy.routing.routing_renderer import ManagedRoutingRender, generate

pytestmark = pytest.mark.unit


@pytest.fixture
def ownership(tmp_path: Path) -> DeploymentOwnership:
    return DeploymentOwnership("test-routing", "a" * 32, "first-generation", tmp_path)


class FakeDocker(DockerComposeRuntime):
    def __init__(self, ownership: DeploymentOwnership) -> None:
        super().__init__()
        self.commands: list[list[str]] = []
        self.container_labels = {**ownership.labels, "com.docker.compose.project": ownership.project_name}
        self.containers: dict[str, str] = {"student-router": "semantic-router"}
        self.networks = {"routing-network"}
        self.foreign = False
        self.paused = False
        self.image_labels = json.dumps({"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION})

    def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
        self.commands.append(list(arguments))
        if arguments[:2] == ["image", "inspect"]:
            return self.image_labels
        if arguments[:2] == ["container", "ls"]:
            return "\n".join(self.containers)
        if arguments[:2] == ["network", "ls"]:
            return "\n".join(self.networks)
        if arguments[:2] == ["container", "inspect"]:
            records = []
            for resource_id in arguments[2:]:
                labels = {**self.container_labels, "com.docker.compose.service": self.containers[resource_id]}
                if self.foreign:
                    labels["io.reef.routing.owner"] = "foreign"
                records.append(
                    {
                        "Id": resource_id,
                        "Config": {"Labels": labels},
                        "State": {"Running": True, "Paused": self.paused, "Health": {"Status": "healthy"}},
                    }
                )
            return json.dumps(records)
        if arguments[:2] == ["network", "inspect"]:
            return json.dumps([{"Id": name, "Labels": self.container_labels} for name in arguments[2:]])
        if arguments[:2] == ["container", "rm"]:
            self.containers.pop(arguments[-1])
        if arguments[:2] == ["network", "rm"]:
            self.networks.remove(arguments[-1])
        return ""


def test_registry_lock_is_stable_across_run_directories(tmp_path: Path) -> None:
    first = DeploymentRegistry("shared", root=tmp_path)
    second = DeploymentRegistry("shared", root=tmp_path)
    with first:
        generation = first.prepare("one", tmp_path / "first-run")
        with pytest.raises(RoutingOwnershipError, match="already controlled"):
            second.acquire()
        assert first.lock_descriptor is not None
        assert not os.get_inheritable(first.lock_descriptor)
        assert stat.S_IMODE(first.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE((first.directory / "owner.json").stat().st_mode) == 0o600
    with second:
        resumed = second.prepare("two", tmp_path / "different-run")
        assert resumed.owner_id == generation.owner_id
        assert resumed.project_name != generation.project_name
        assert second.load() == resumed


def test_registry_rejects_symlink_and_foreign_owner_state(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(RoutingOwnershipError):
        DeploymentRegistry("linked", root=tmp_path).acquire()
    with DeploymentRegistry("safe", root=tmp_path) as registry:
        registry.prepare("one", tmp_path / "generation")
        path = registry.directory / "owner.json"
        path.chmod(0o644)
        with pytest.raises(RoutingOwnershipError, match="private"):
            registry.load()


def test_registry_cannot_mutate_without_lock_or_adopt_new_owner(tmp_path: Path) -> None:
    registry = DeploymentRegistry("safe", root=tmp_path)
    with pytest.raises(RoutingOwnershipError, match="lock"):
        registry.prepare("one", tmp_path)
    with registry:
        registry.prepare("one", tmp_path)
        with pytest.raises(RoutingOwnershipError, match="stable"):
            registry.save(DeploymentOwnership("safe", "b" * 32, "two", tmp_path))


@pytest.mark.parametrize("deployment_id", ["../outside", "", "a/b", "Uppercase", "-bad", "x" * 49])
def test_registry_rejects_invalid_identity(deployment_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DeploymentRegistry(deployment_id, root=tmp_path)


def test_cleanup_validates_before_any_mutation_and_is_idempotent(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.foreign = True
    with pytest.raises(RoutingOwnershipError):
        runtime.remove(ownership, 0.1)
    assert not any(command[0] in ("stop", "unpause") or "rm" in command for command in runtime.commands)
    runtime.foreign = False
    runtime.remove(ownership, 0.1)
    runtime.remove(ownership, 0.1)
    removals = [command for command in runtime.commands if command[:2] == ["container", "rm"]]
    # The verified container goes with its anonymous volumes (the image's /app/models); nothing else does.
    assert removals == [["container", "rm", "--force", "--volumes", "student-router"]]
    assert not runtime.containers and not runtime.networks
    assert all("down" not in command and command[:1] != ["volume"] for command in runtime.commands)


def test_a_second_container_in_the_owned_group_is_refused(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.containers["ingress"] = "envoy"
    with pytest.raises(ContainerCommandError, match="unexpected managed routing container"):
        runtime.inspect(ownership)


def test_paused_owned_containers_unpause_before_stop(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.paused = True
    runtime.remove(ownership, 1)
    mutation_commands = [command for command in runtime.commands if command[0] in ("unpause", "stop")]
    assert [command[0] for command in mutation_commands] == ["unpause", "stop"]


def test_start_refuses_unreconciled_existing_resources(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    with pytest.raises(ContainerCommandError, match="reconcile"):
        runtime.start(ownership.configuration_directory / "compose.yaml", ownership, 5)
    assert all(command[0] != "compose" for command in runtime.commands)


def test_start_inspects_the_image_from_the_generated_compose(ownership: DeploymentOwnership) -> None:
    manifest_path = ownership.configuration_directory / "input.yaml"
    manifest = routing_manifest("http://127.0.0.1:8101")
    manifest_path.write_text(yaml.safe_dump(manifest))
    output = ownership.configuration_directory / "generated"
    generate(
        manifest_path,
        output,
        ManagedRoutingRender(
            RoutingDeploymentSettings(deployment_id=ownership.deployment_id),
            ownership.deployment_id,
            ownership.owner_id,
            ownership.generation,
        ),
    )
    runtime = FakeDocker(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    runtime.start(output / "compose.yaml", ownership, 5)
    image = yaml.safe_load(manifest_path.read_text())["deployment"]["semantic_router_image"]
    assert ["image", "inspect", "--format", "{{json .Config.Labels}}", image] in runtime.commands
    assert any(command[0] == "compose" for command in runtime.commands)


@pytest.mark.parametrize(
    "labels",
    [
        {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION},
        {},
        {"org.opencontainers.image.revision": "f" * 40},
        {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION[:12]},
        {"org.opencontainers.image.revision": None},
        None,
        [],
    ],
)
def test_start_checks_the_digest_image_revision_before_creating_resources(
    ownership: DeploymentOwnership, labels: dict[str, str | None] | list[object] | None
) -> None:
    runtime = FakeDocker(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    runtime.image_labels = json.dumps(labels)
    image = "example/router@sha256:" + "a" * 64
    compose = ownership.configuration_directory / "compose.yaml"
    compose.write_text(yaml.safe_dump({"services": {"semantic-router": {"image": image}}}))
    if labels == {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION}:
        runtime.start(compose, ownership, 5)
        create = next(command for command in runtime.commands if command[0] == "compose")
        assert create[-5:] == ["up", "--detach", "--no-build", "--pull", "never"]
        assert runtime.commands.index(create) > next(
            index for index, command in enumerate(runtime.commands) if command[:2] == ["image", "inspect"]
        )
    else:
        with pytest.raises(ContainerCommandError, match="revision") as error:
            runtime.start(compose, ownership, 5)
        assert not error.value.outcome_unknown
        assert not any(command[0] == "compose" for command in runtime.commands)
    assert [command for command in runtime.commands if command[:2] == ["image", "inspect"]] == [
        ["image", "inspect", "--format", "{{json .Config.Labels}}", image]
    ]
    assert not runtime.mutation_pending(ownership)


@pytest.mark.parametrize("failure", ["missing-image", "malformed-labels", "timeout"])
def test_image_inspection_failure_never_submits_container_creation(
    ownership: DeploymentOwnership, failure: str
) -> None:
    class UnavailableImage(FakeDocker):
        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            if arguments[:2] == ["image", "inspect"]:
                if failure == "malformed-labels":
                    return "secret-marker not JSON"
                raise ContainerCommandError(failure)
            return super().run(arguments, timeout_seconds, mutation=mutation)

    runtime = UnavailableImage(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    compose = ownership.configuration_directory / "compose.yaml"
    compose.write_text(
        yaml.safe_dump({"services": {"semantic-router": {"image": "example/router@sha256:" + "a" * 64}}})
    )
    with pytest.raises(ContainerCommandError) as error:
        runtime.start(compose, ownership, 5)
    assert not error.value.outcome_unknown
    assert "secret-marker" not in str(error.value)
    assert not any(command[0] == "compose" for command in runtime.commands)
    assert not runtime.mutation_pending(ownership)


def test_unknown_mutation_never_echoes_command_output() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    with pytest.raises(ContainerCommandError) as caught:
        runtime.run(["-c", "print('secret-value'); raise SystemExit(7)"], mutation=True)
    assert caught.value.outcome_unknown
    assert "secret-value" not in str(caught.value)


def test_command_timeout_is_bounded() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    started = time.monotonic()
    with pytest.raises(ContainerCommandError) as caught:
        runtime.run(["-c", "import time; time.sleep(30)"], timeout_seconds=0.1, mutation=True)
    assert caught.value.outcome_unknown
    assert time.monotonic() - started < 5


def test_command_loses_owner_and_reaps_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    command = GuardedCommand(
        [
            sys.executable,
            "-c",
            f"import os,time; open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(30)",
        ],
        30,
    )
    deadline = time.monotonic() + 5
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()
    child_pid = int(pid_path.read_text())
    command.close()
    assert command.process.returncode != 0
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize("ending", ["timeout", "owner_loss"])
def test_a_failed_command_retires_a_descendant_that_ignores_sigterm(tmp_path: Path, ending: str) -> None:
    """The CLI exits on SIGTERM, but the descendant it started ignores it and must still be gone afterwards.

    The descendant holds a FIFO open for writing. End of file there shows that it exited, even before
    anything reaps it.
    """
    fifo = tmp_path / "descendant.fifo"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    descendant = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"held = os.open({str(fifo)!r}, os.O_WRONLY); os.write(held, str(os.getpid()).encode()); time.sleep(30)"
    )
    cli = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {descendant!r}]); time.sleep(30)"
    command = GuardedCommand([sys.executable, "-c", cli], 2 if ending == "timeout" else 30)
    descendant_pid = None
    try:
        deadline = time.monotonic() + 5
        while descendant_pid is None and time.monotonic() < deadline:
            try:
                written = os.read(reader, 32)
            except BlockingIOError:
                written = b""
            if written:
                descendant_pid = int(written)
            else:
                time.sleep(0.01)
        assert descendant_pid is not None
        if ending == "timeout":
            command.process.wait(timeout=10)
        command.close()
        assert command.process.returncode != 0
        exited = False
        deadline = time.monotonic() + 5
        while not exited and time.monotonic() < deadline:
            try:
                exited = os.read(reader, 1) == b""
            except BlockingIOError:
                time.sleep(0.02)
        assert exited, "a descendant outlived its failed command"
        descendant_pid = None
    finally:
        if descendant_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(descendant_pid, signal.SIGKILL)
        os.close(reader)
        command.close()
        for stream in (command.process.stdout, command.process.stderr):
            if stream is not None:
                stream.close()


def test_owner_sigkill_closes_lease_without_inheriting_lock(tmp_path: Path) -> None:
    command_pid = tmp_path / "command.pid"
    controller_pid = tmp_path / "controller.pid"
    program = f"""
import os, sys, time
from reef.service.deploy.routing.routing_containers import GuardedCommand
from reef.service.deploy.routing.routing_registry import DeploymentRegistry
from pathlib import Path
registry = DeploymentRegistry("killed", root=Path({str(tmp_path)!r}))
registry.acquire()
command = GuardedCommand([sys.executable, "-c", "import os,time; open({str(command_pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"], 30)
Path({str(controller_pid)!r}).write_text(str(os.getpid()))
time.sleep(30)
"""
    controller = subprocess.Popen([sys.executable, "-c", program])
    try:
        deadline = time.monotonic() + 5
        while not command_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert command_pid.exists()
        controller.kill()
        controller.wait(timeout=3)
        with DeploymentRegistry("killed", root=tmp_path):
            pass
        child_pid = int(command_pid.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Docker command survived loss of its owner")
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)


def test_events_surface_disconnect_and_keep_container_identity() -> None:
    event = {"Actor": {"ID": "old-container"}, "Action": "die"}
    stream = DockerEventStream([sys.executable, "-c", f"print({json.dumps(event)!r})"])
    try:
        first = stream.poll(timeout_seconds=5)
        assert first is not None and first.container_id == "old-container" and first.action == "die"
        disconnected = stream.poll(timeout_seconds=5)
        assert disconnected is not None and disconnected.action == "stream_disconnected"
    finally:
        stream.close()


def test_interrupted_mutation_requires_matching_explicit_acknowledgement(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    marker = ownership.configuration_directory / "docker-mutation.json"
    marker.write_text(json.dumps({**ownership.labels, "operation_id": "c" * 32, "operation": "compose"}))
    marker.chmod(0o600)
    # Safe cleanup removes observed owned resources, but an empty snapshot is
    # not proof that an interrupted create can no longer reach the daemon.
    with pytest.raises(ContainerCommandError, match="remains unresolved"):
        runtime.remove(ownership, 0)
    assert not runtime.containers and not runtime.networks
    assert runtime.pending_operation_id(ownership) == "c" * 32
    with pytest.raises(ContainerCommandError, match="unresolved"):
        runtime.start(ownership.configuration_directory / "compose.yaml", ownership, 5)
    with pytest.raises(ContainerCommandError, match="changed"):
        runtime.acknowledge_unknown(ownership, "d" * 32)
    assert marker.exists()
    runtime.acknowledge_unknown(ownership, "c" * 32)
    assert not marker.exists()


def test_mutation_marker_precedes_cli_and_survives_failure(ownership: DeploymentOwnership) -> None:
    class FailingDocker(FakeDocker):
        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            if mutation:
                assert self.mutation_pending(ownership)
                raise ContainerCommandError("unknown result", outcome_unknown=True)
            return super().run(arguments, timeout_seconds, mutation=mutation)

    runtime = FailingDocker(ownership)
    with pytest.raises(ContainerCommandError, match="unknown result"):
        runtime.mutate(["compose", "up"], ownership)
    assert runtime.pending_operation_id(ownership) is not None


def test_successful_mutation_clears_its_own_marker(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.mutate(["compose", "up"], ownership)
    assert not runtime.mutation_pending(ownership)


def test_status_can_read_owner_without_acquiring_running_controller_lock(tmp_path: Path) -> None:
    registry = DeploymentRegistry("shared", root=tmp_path)
    with registry:
        ownership = registry.prepare("one", tmp_path)
        observer = DeploymentRegistry("shared", root=tmp_path)
        assert observer.read_owner() == ownership
        with pytest.raises(RoutingOwnershipError):
            observer.prepare("two", tmp_path)


def test_sigkill_during_mutation_blocks_immediate_replacement(ownership: DeploymentOwnership, tmp_path: Path) -> None:
    command_pid = tmp_path / "mutation-child.pid"
    program = f"""
import sys
from pathlib import Path
from reef.service.deploy.routing.routing_containers import DockerComposeRuntime
from reef.service.deploy.routing.routing_registry import DeploymentOwnership
ownership =DeploymentOwnership({ownership.deployment_id!r}, {ownership.owner_id!r}, {ownership.generation!r}, Path({str(ownership.configuration_directory)!r}))
DockerComposeRuntime(sys.executable).mutate(["-c", "import os,time; open({str(command_pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"], ownership, 30)
"""
    controller = subprocess.Popen([sys.executable, "-c", program])
    try:
        deadline = time.monotonic() + 5
        while not command_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert command_pid.exists()
        controller.kill()
        controller.wait(timeout=3)
        runtime = FakeDocker(ownership)
        runtime.containers.clear()
        runtime.networks.clear()
        # No polling delay or empty Docker snapshot can override the saved
        # unknown outcome. The next controller fails before attempting create.
        with pytest.raises(ContainerCommandError, match="unresolved"):
            runtime.start(tmp_path / "compose.yaml", ownership, 5)
        assert runtime.pending_operation_id(ownership) is not None
        assert not any(command[0] == "compose" for command in runtime.commands)
        child_pid = int(command_pid.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("mutation command survived its owner")
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)


def test_docker_event_filters_prevent_healthcheck_exec_noise_overflow(
    ownership: DeploymentOwnership,
    tmp_path: Path,
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        f"""#!{sys.executable}
import json, sys
filters = [item[6:] for item in sys.argv if item.startswith("event=")]
# Simulate a long startup's per-probe exec events while nobody polls.
for index in range(3000):
    action = ("exec_create", "exec_start", "exec_die")[index % 3]
    if not filters or action in filters:
        print(json.dumps({{"Actor": {{"ID": "router"}}, "Action": action}}))
for action in ("health_status: unhealthy", "die"):
    if not filters or action.split(":")[0] in filters:
        print(json.dumps({{"Actor": {{"ID": "router"}}, "Action": action}}))
"""
    )
    fake_docker.chmod(0o700)
    stream = DockerComposeRuntime(str(fake_docker)).events(ownership)
    try:
        assert isinstance(stream, DockerEventStream)
        stream.command.process.wait(timeout=5)
        first = stream.poll(timeout_seconds=5)
        second = stream.poll(timeout_seconds=5)
        assert first is not None and first.action == "health_status: unhealthy"
        assert second is not None and second.action == "die"
        final = stream.poll(timeout_seconds=5)
        assert final is not None and final.action == "stream_disconnected"
    finally:
        stream.close()


@pytest.mark.parametrize("malformation", ["bad-json", "missing-state", "missing-id", "bad-config"])
def test_invalid_inspect_responses_have_typed_sanitized_errors(
    ownership: DeploymentOwnership,
    malformation: str,
) -> None:
    class MalformedDocker(FakeDocker):
        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            encoded = super().run(arguments, timeout_seconds, mutation=mutation)
            if arguments[:2] != ["container", "inspect"]:
                return encoded
            if malformation == "bad-json":
                return "secret-marker definitely not valid JSON"
            records = json.loads(encoded)
            if malformation == "missing-state":
                records[0].pop("State")
            elif malformation == "missing-id":
                records[0].pop("Id")
            else:
                records[0]["Config"] = "secret-marker"
            return json.dumps(records)

    with pytest.raises(ContainerCommandError) as caught:
        MalformedDocker(ownership).inspect(ownership)
    assert "secret-marker" not in str(caught.value)


def test_cli_invalid_utf8_is_typed_and_preserves_mutation_uncertainty() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    with pytest.raises(ContainerCommandError, match="UTF-8") as caught:
        runtime.run(["-c", "import sys; sys.stdout.buffer.write(bytes([255]))"], mutation=True)
    assert caught.value.outcome_unknown


@pytest.mark.parametrize("encoded", [b"secret-marker invalid JSON", b"\xff"])
def test_corrupt_private_state_fails_closed_with_ownership_error(tmp_path: Path, encoded: bytes) -> None:
    with DeploymentRegistry("state", root=tmp_path) as registry:
        ownership = registry.prepare("generation", tmp_path)
        (registry.directory / "owner.json").write_bytes(encoded)
        with pytest.raises(RoutingOwnershipError) as caught:
            registry.read_owner()
        assert "secret-marker" not in str(caught.value)
    mutation = tmp_path / "docker-mutation.json"
    mutation.write_bytes(encoded)
    mutation.chmod(0o600)
    with pytest.raises(RoutingOwnershipError) as caught:
        DockerComposeRuntime().pending_operation_id(ownership)
    assert "secret-marker" not in str(caught.value)
