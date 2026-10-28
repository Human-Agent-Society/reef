"""The Modal cloud sandbox: its settings, its sandboxes, and the processes it launches, against a stand-in SDK."""

from __future__ import annotations

import io
import shlex
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest

import reef.sandbox.modal as modal_sandbox
import reef.sandbox.remote as remote
from reef.sandbox.config import build_executor, remote_executor
from reef.sandbox.modal import ModalExecutor, ModalSession
from reef.sandbox.process import RolloutLaunchError, RolloutTimeout, SandboxProgram, SandboxUnavailable
from reef.sandbox.remote import REMOTE_BASE, TUNNEL_PORT_BASE, deployment_owner


class Program(SandboxProgram):
    def __init__(self, *, npm_package: str = "pi@1.0.0", self_isolating: bool = False) -> None:
        self.name = "pi"
        self.binary = "pi"
        self.self_isolating = self_isolating
        self.npm_package = npm_package
        self.version = "1.0.0"

    def host_binary(self) -> str:
        raise AssertionError("a remote sandbox never installs the program on the Reef host")

    def host_install_dir(self) -> Path | None:
        return None


class Process:
    """A finished process with its exit status and output."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)

    def wait(self) -> None:
        return None


class Filesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def write_bytes(self, data: bytes, path: str) -> None:
        self.files[path] = bytes(data)

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]


class Sandbox:
    """Records each command; ``answers`` give the process of each ``exec`` in turn, then success."""

    def __init__(self, *answers: Process) -> None:
        self.filesystem = Filesystem()
        self.answers = list(answers)
        self.commands: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.terminated = False

    def exec(self, *args: str, **options: Any) -> Process:
        self.commands.append((args, options))
        return self.answers.pop(0) if self.answers else Process()

    def tunnels(self) -> dict[int, Any]:
        return {port: types.SimpleNamespace(url=f"https://tunnel-{port}.modal.test") for port in (21080, 21081)}

    def terminate(self) -> None:
        self.terminated = True


@pytest.mark.unit
def test_the_settings_take_the_modal_options_and_refuse_the_local_sandboxs() -> None:
    executor = ModalExecutor.from_config(
        {"modal_app": "reef-x", "modal_cpu": 2, "modal_memory_mb": 4096, "forward_ports": [8900, 8900, 8901]},
        {},
    )
    assert (executor.app_name, executor.cpu, executor.memory_mb) == ("reef-x", 2.0, 4096)
    assert executor.forward_ports == (8900, 8901)
    for section, message in [
        ({"egress_hosts": ["api.example"]}, "does not implement sandbox.egress_hosts"),
        ({"e2b_template": "x"}, "unknown Modal sandbox options: e2b_template"),
        ({"modal_cpu": True}, "modal_cpu must be a positive number"),
        ({"modal_memory_mb": 0}, "modal_memory_mb must be a positive int"),
        ({"modal_app": ""}, "modal_app must be a non-empty string"),
        ({"env_from": ["MISSING"]}, "requires environment variable 'MISSING'"),
    ]:
        with pytest.raises(SandboxUnavailable, match=message):
            ModalExecutor.from_config(section, {})


@pytest.mark.unit
def test_the_factory_builds_each_provider_from_its_own_options(monkeypatch) -> None:
    monkeypatch.setattr(ModalExecutor, "preflight", lambda self: None)
    built = build_executor({"executor": "modal", "sandbox": {"modal_app": "reef-y"}})
    assert isinstance(built, ModalExecutor) and built.app_name == "reef-y"
    # A caller's own settings beside the provider's options are not the provider's to read.
    chosen = remote_executor(
        "modal", {"timeout_s": 5, "modal_image": "img"}, {}, timeout_s=90, owner_anchor=Path("/x")
    )
    assert isinstance(chosen, ModalExecutor) and (chosen.image, chosen.timeout_s) == ("img", 90)
    assert chosen.owner == deployment_owner(Path("/x"))
    with pytest.raises(SandboxUnavailable, match="unknown cloud sandbox 'docker'"):
        remote_executor("docker", {}, {})
    with pytest.raises(Exception, match="use one of 'local', 'sandbox', 'e2b', 'modal'"):
        build_executor({"executor": "docker"})


@pytest.mark.unit
def test_a_program_runs_from_the_named_image_or_one_built_from_its_npm_package() -> None:
    bound, binary = ModalExecutor().for_program(Program())
    assert isinstance(bound, ModalExecutor) and (bound.npm_package, bound.binary, binary) == ("pi@1.0.0", "pi", "pi")
    named, _ = ModalExecutor(image="registry/pi:1").for_program(Program(npm_package=""))
    assert isinstance(named, ModalExecutor) and (named.image, named.npm_package) == ("registry/pi:1", "")
    with pytest.raises(SandboxUnavailable, match="requires an explicit modal_image"):
        ModalExecutor().for_program(Program(npm_package=""))
    with pytest.raises(SandboxUnavailable, match="manages its own containers"):
        ModalExecutor().for_program(Program(self_isolating=True))


def fake_modal(monkeypatch, sandboxes: list[Any]) -> list[dict[str, Any]]:
    """Install a stand-in ``modal`` module whose ``Sandbox.create`` records its arguments; return the records."""
    created: list[dict[str, Any]] = []

    class Image:
        def __init__(self, steps: tuple[str, ...]) -> None:
            self.steps = steps

        @classmethod
        def from_registry(cls, tag: str) -> Image:
            return cls((tag,))

        def apt_install(self, *packages: str) -> Image:
            return Image((*self.steps, "apt " + " ".join(packages)))

        def run_commands(self, command: str) -> Image:
            return Image((*self.steps, command))

    class App:
        app_id = "ap-1"

        @staticmethod
        def lookup(name: str, *, create_if_missing: bool) -> App:
            assert create_if_missing
            return App()

    class SandboxAPI:
        @staticmethod
        def create(**options: Any) -> Any:
            created.append(options)
            return sandboxes.pop(0)

        @staticmethod
        def list(*, app_id: str, tags: dict[str, str]) -> list[Any]:
            return [sandbox for sandbox in sandboxes if tags.items() <= sandbox.tags.items()]

    module = types.SimpleNamespace(Image=Image, App=App, Sandbox=SandboxAPI)
    monkeypatch.setitem(sys.modules, "modal", module)
    return created


@pytest.mark.unit
def test_a_session_starts_a_tagged_sandbox_with_a_tunnel_for_each_forwarded_port(monkeypatch) -> None:
    sandbox = Sandbox()
    created = fake_modal(monkeypatch, [sandbox])
    started: list[tuple[str, int]] = []
    monkeypatch.setattr(remote.TunnelPump, "start", lambda self, timeout=60.0: started.append((self.url, self.port)))
    executor = ModalExecutor(npm_package="pi@1.0.0", timeout_s=600, owner="team")
    session = executor.open_session(forward_ports=(8900, 8901), timeout_s=60)
    assert isinstance(session, ModalSession)
    (options,) = created
    assert options["timeout"] == 600 + remote.SANDBOX_MARGIN_S and options["tags"] == executor.labels()
    assert options["encrypted_ports"] == [TUNNEL_PORT_BASE, TUNNEL_PORT_BASE + 1]
    assert (
        options["image"].steps[0] == modal_sandbox.NODE_IMAGE and "npm install -g pi@1.0.0" in options["image"].steps
    )
    assert started == [("https://tunnel-21080.modal.test", 8900), ("https://tunnel-21081.modal.test", 8901)]
    assert "/root/reef-relay.py" in sandbox.filesystem.files
    relays = [args[-1] for args, _ in sandbox.commands if "reef-relay.py --port" in args[-1]]
    assert len(relays) == 2 and "--tunnel-port 21081" in relays[1]
    session.close()
    assert sandbox.terminated
    with pytest.raises(RolloutLaunchError, match="lives at most 86400 s"):
        ModalExecutor(npm_package="pi@1.0.0", timeout_s=86400).open()


class TaggedSandbox(Sandbox):
    def __init__(self, tags: dict[str, str]) -> None:
        super().__init__()
        self.tags = tags

    def get_tags(self) -> dict[str, str]:
        return self.tags


@pytest.mark.unit
def test_a_restart_stops_only_the_sandboxes_its_deployment_left(monkeypatch) -> None:
    mine, theirs = TaggedSandbox({"reef": "sandbox", "reef_owner": "team"}), TaggedSandbox(
        {"reef": "sandbox", "reef_owner": "other"}
    )
    fake_modal(monkeypatch, [mine, theirs])
    assert ModalExecutor(owner="team").reap() == 1
    assert mine.terminated and not theirs.terminated
    assert ModalExecutor().reap() == 0


@pytest.mark.unit
def test_a_process_runs_in_the_jail_under_its_limit_with_the_deployments_environment(tmp_path, monkeypatch) -> None:
    root = tmp_path / "rollout"
    (root / "workspace").mkdir(parents=True)
    sandbox = Sandbox(Process(0, "done", "note"))
    session = ModalSession(sandbox, env={"DEPLOYMENT": "yes"})
    monkeypatch.setattr(session, "push", lambda path: None)
    monkeypatch.setattr(session, "pull", lambda path, relative="": None)
    outcome = session.launch(
        ["pi", "--print", str(root / "workspace" / "out.txt")],
        root=root,
        workspace=root / "workspace",
        env={"CALLER": "1"},
        timeout=30,
    )
    assert (outcome.exit_code, outcome.stdout, outcome.stderr) == (0, "done", "note")
    ((args, options),) = sandbox.commands
    command = shlex.split(args[-1])
    assert command[:4] == ["timeout", f"--kill-after={modal_sandbox.KILL_AFTER_S}", "30", "/bin/sh"]
    jailed = command[-1]
    assert jailed.startswith("exec bwrap") and f"{REMOTE_BASE}/rollout/workspace/out.txt" in jailed
    assert options["workdir"] == f"{REMOTE_BASE}/rollout/workspace"
    assert options["env"] == {"CALLER": "1", "DEPLOYMENT": "yes"}


@pytest.mark.unit
def test_a_process_past_its_limit_is_a_timeout_and_a_failed_command_is_a_launch_error(tmp_path, monkeypatch) -> None:
    root = tmp_path / "rollout"
    (root / "workspace").mkdir(parents=True)
    clock = iter([0.0, 31.0])
    monkeypatch.setattr(modal_sandbox.time, "monotonic", lambda: next(clock))
    session = ModalSession(Sandbox(Process(124)))
    monkeypatch.setattr(session, "push", lambda path: None)
    pulled: list[Path] = []
    monkeypatch.setattr(session, "pull", lambda path, relative="": pulled.append(path))
    with pytest.raises(RolloutTimeout, match="past its 30 s limit"):
        session.launch(["pi"], root=root, workspace=root / "workspace", env={}, timeout=30)
    assert pulled == [root], "what the process wrote before its limit still comes back"
    with pytest.raises(RolloutLaunchError, match="exited with 2: no such file"):
        ModalSession(Sandbox(Process(2, stderr="no such file"))).run("false")


@pytest.mark.unit
def test_a_root_is_copied_in_and_back_through_the_sandboxs_files(tmp_path) -> None:
    class LocalSandbox(Sandbox):
        """Runs each shell command on this host with the sandbox's paths under ``base``."""

        def __init__(self, base: Path) -> None:
            super().__init__()
            self.base = base

        def exec(self, *args: str, **options: Any) -> Process:
            for path, data in self.filesystem.files.items():
                (self.base / path.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
                (self.base / path.lstrip("/")).write_bytes(data)
            command = args[-1].replace("/tmp/", f"{self.base}/tmp/").replace(REMOTE_BASE, f"{self.base}{REMOTE_BASE}")
            subprocess.run(["/bin/sh", "-c", command], check=True)
            for path in list(self.filesystem.files):
                if not (self.base / path.lstrip("/")).exists():
                    del self.filesystem.files[path]
            for archive in (self.base / "tmp").glob("reef-pull-*.tgz"):
                self.filesystem.files[f"/tmp/{archive.name}"] = archive.read_bytes()
            return Process()

    root = tmp_path / "host" / "rollout"
    (root / "workspace").mkdir(parents=True)
    (root / "workspace" / "task.md").write_text("solve it")
    sandbox = LocalSandbox(tmp_path / "sandbox")
    session = ModalSession(sandbox)
    session.push(root)
    remote_copy = tmp_path / "sandbox" / REMOTE_BASE.lstrip("/") / "rollout"
    assert (remote_copy / "workspace" / "task.md").read_text() == "solve it"
    (remote_copy / "workspace" / "answer.md").write_text("42")
    session.pull(root, "workspace")
    assert (root / "workspace" / "answer.md").read_text() == "42"
    with tarfile.open(fileobj=io.BytesIO(remote.pack(root)), mode="r:gz") as archive:
        assert sorted(archive.getnames()) == ["workspace", "workspace/answer.md", "workspace/task.md"]
