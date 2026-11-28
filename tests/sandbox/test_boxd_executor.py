"""The boxd cloud sandbox: its settings, its machines, and the processes it launches, against a stand-in SDK."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import reef.execution.sandbox.boxd as boxd_sandbox
import reef.execution.sandbox.remote as remote
from reef.execution.sandbox.boxd import BoxdExecutor, BoxdSession
from reef.execution.sandbox.config import build_executor, remote_executor
from reef.execution.sandbox.process import (
    ProgramLaunchError,
    ProgramTimeoutError,
    SandboxProgram,
    SandboxUnavailableError,
)
from reef.execution.sandbox.remote import REMOTE_BASE, TUNNEL_PORT_BASE, deployment_owner


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


class BoxdError(Exception):
    pass


class NotFoundError(BoxdError):
    pass


class ConflictError(BoxdError):
    pass


class Result:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class Files:
    def __init__(self) -> None:
        self.stored: dict[str, bytes] = {}

    def upload(self, machine_id: str, path: str, data: bytes) -> int:
        self.stored[path] = bytes(data)
        return len(data)

    def download(self, machine_id: str, path: str) -> bytes:
        return self.stored[path]


class Proxies:
    def list(self, machine_id: str) -> list[Any]:
        return [
            types.SimpleNamespace(name=None, domain="m.boxd.test"),
            types.SimpleNamespace(name="reef-relay-0", domain="reef-relay-0.m.boxd.test"),
            types.SimpleNamespace(name="reef-relay-1", domain="reef-relay-1.m.boxd.test"),
        ]


class Machines:
    """Records each call. ``answer`` gives the result of each command; by default it succeeds."""

    def __init__(self, listed: tuple[Any, ...] = ()) -> None:
        self.files = Files()
        self.proxies = Proxies()
        self.listed = listed
        self.created: list[dict[str, Any]] = []
        self.commands: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.deleted: list[str] = []
        self.hibernate_timeouts: list[tuple[str, int]] = []
        self.ready_error: Exception | None = None

    def create(self, name: str, **options: Any) -> Any:
        self.created.append({"name": name, **options})
        return types.SimpleNamespace(id="machine-1")

    def wait_until_ready(self, machine_id: str, *, timeout: float) -> None:
        if self.ready_error is not None:
            raise self.ready_error

    def set_auto_hibernate_timeout(self, machine_id: str, seconds: int) -> None:
        self.hibernate_timeouts.append((machine_id, seconds))

    def answer(self, command: list[str]) -> Result:
        if command[0] == "mktemp":
            return Result(stdout="/tmp/reef-upload-x\n")
        if command[-1] == f"cat {boxd_sandbox.BOOT_ID_FILE}":
            return Result(stdout="boot-1\n")
        return Result()

    def exec(self, machine_id: str, command: list[str], *, cwd: str = "", timeout: float | None = None) -> Result:
        self.commands.append(command)
        self.timeouts.append(timeout)
        return self.answer(command)

    def list(self, *, org: str | None, labels: dict[str, str]) -> list[Any]:
        return [machine for machine in self.listed if labels.items() <= machine.labels.items()]

    def delete(self, machine_id: str) -> None:
        self.deleted.append(machine_id)


def fake_boxd(monkeypatch, machines: Machines) -> list[dict[str, Any]]:
    """Install a stand-in ``boxd`` package; answer the arguments of each client."""
    clients: list[dict[str, Any]] = []

    class Boxd:
        def __init__(self, **options: Any) -> None:
            clients.append(options)
            self.machines = machines

    class ProxyEntry:
        def __init__(self, *, name: str, port: int) -> None:
            self.name = name
            self.port = port

    package = types.ModuleType("boxd")
    package.Boxd = Boxd  # type: ignore[attr-defined]
    errors = types.ModuleType("boxd.errors")
    errors.BoxdError = BoxdError  # type: ignore[attr-defined]
    errors.NotFoundError = NotFoundError  # type: ignore[attr-defined]
    errors.ConflictError = ConflictError  # type: ignore[attr-defined]
    models = types.ModuleType("boxd.models")
    models.ProxyEntry = ProxyEntry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boxd", package)
    monkeypatch.setitem(sys.modules, "boxd.errors", errors)
    monkeypatch.setitem(sys.modules, "boxd.models", models)
    return clients


@pytest.mark.unit
def test_the_settings_take_the_boxd_options_and_refuse_bad_values() -> None:
    executor = BoxdExecutor.from_config(
        {
            "boxd_api_key_env": "MY_KEY",
            "boxd_region": "nyc",
            "boxd_snapshot": "prepared",
            "boxd_vcpu": 4,
            "boxd_name_prefix": "team-",
            "boxd_labels": {"env": "ci"},
            "boxd_auto_destroy": False,
            "forward_ports": [8900, 8900],
        },
        {"MY_KEY": " key ", "BOXD_ORG": "acme"},
    )
    assert (executor.api_key, executor.org, executor.region, executor.snapshot) == ("key", "acme", "nyc", "prepared")
    assert (executor.vcpu, executor.name_prefix, executor.extra_labels) == (4, "team-", {"env": "ci"})
    assert executor.auto_destroy is False and executor.forward_ports == (8900,)
    assert "key" not in repr(executor).replace("api_key", "").replace("_key", "")
    for section, message in [
        ({"egress_hosts": ["api.example"]}, "does not implement sandbox.egress_hosts"),
        ({"modal_app": "x"}, "unknown boxd sandbox options: modal_app"),
        ({"isolated": False}, "unknown boxd sandbox options: isolated"),
        ({"boxd_vcpu": 3}, "boxd_vcpu must be one of 1, 2, 4, 8"),
        ({"boxd_vcpu": True}, "boxd_vcpu must be one of 1, 2, 4, 8"),
        ({"boxd_vcpu": 2.0}, "boxd_vcpu must be one of 1, 2, 4, 8"),
        ({"boxd_labels": {"reef_machine": "x"}}, "cannot set the reef, reef_owner or reef_machine label"),
        ({"boxd_name_prefix": "Bad_"}, "boxd_name_prefix must start with"),
        ({"boxd_labels": {"reef_owner": "x"}}, "cannot set the reef, reef_owner or reef_machine label"),
        ({"boxd_auto_destroy": "no"}, "boxd_auto_destroy must be true or false"),
        ({"boxd_api_key_env": "not a name"}, "must name an environment variable"),
        ({"env_from": ["MISSING"]}, "requires environment variable 'MISSING'"),
    ]:
        with pytest.raises(SandboxUnavailableError, match=message):
            BoxdExecutor.from_config(section, {})


@pytest.mark.unit
def test_the_factory_builds_boxd_from_its_own_options(monkeypatch) -> None:
    monkeypatch.setattr(BoxdExecutor, "preflight", lambda self: None)
    built = build_executor({"executor": "boxd", "sandbox": {"boxd_region": "fra"}}, environ={})
    assert isinstance(built, BoxdExecutor) and built.region == "fra"
    chosen = remote_executor("boxd", {"timeout_s": 5, "boxd_snapshot": "s"}, {}, timeout_s=90, owner_anchor=Path("/x"))
    assert isinstance(chosen, BoxdExecutor) and (chosen.snapshot, chosen.timeout_s) == ("s", 90)
    assert chosen.owner == deployment_owner(Path("/x"))


@pytest.mark.unit
def test_a_program_runs_from_the_named_snapshot_or_is_installed_from_its_npm_package() -> None:
    bound, binary = BoxdExecutor().for_program(Program())
    assert isinstance(bound, BoxdExecutor) and (bound.npm_package, bound.binary, binary) == ("pi@1.0.0", "pi", "pi")
    named, _ = BoxdExecutor(snapshot="prepared").for_program(Program(npm_package=""))
    assert isinstance(named, BoxdExecutor) and (named.snapshot, named.npm_package) == ("prepared", "")
    with pytest.raises(SandboxUnavailableError, match="requires an explicit boxd_snapshot"):
        BoxdExecutor().for_program(Program(npm_package=""))
    with pytest.raises(SandboxUnavailableError, match="manages its own containers"):
        BoxdExecutor().for_program(Program(self_isolating=True))


@pytest.mark.unit
def test_preflight_names_the_missing_key_and_the_missing_package(monkeypatch) -> None:
    with pytest.raises(SandboxUnavailableError, match="set BOXD_API_KEY"):
        BoxdExecutor().preflight()
    monkeypatch.setitem(sys.modules, "boxd.errors", None)
    with pytest.raises(SandboxUnavailableError, match=r"reef-infra\[boxd\]"):
        BoxdExecutor(api_key="key").preflight()


@pytest.mark.unit
def test_a_session_creates_an_isolated_labelled_machine_with_a_route_for_each_port(monkeypatch) -> None:
    machines = Machines()
    clients = fake_boxd(monkeypatch, machines)
    started: list[tuple[str, int, dict[str, str]]] = []

    class RecordingPump:
        def __init__(self, url: str, secret: str, port: int, *, tunnel_headers: dict[str, str]) -> None:
            started.append((url, port, tunnel_headers))

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(remote, "TunnelPump", RecordingPump)
    monkeypatch.setattr(BoxdSession, "public_url", lambda self, port: f"https://route-{port}.test")
    executor = BoxdExecutor(
        api_key="key",
        org="acme",
        region="nyc",
        vcpu=2,
        npm_package="pi@1.0.0",
        timeout_s=600,
        owner="team",
        extra_labels={"env": "ci"},
    )
    session = executor.open_session(forward_ports=(8900, 8901), timeout_s=60)
    assert isinstance(session, BoxdSession)
    assert clients[-1] == {"api_key": "key", "region": "nyc"}
    (options,) = machines.created
    assert options["isolated"] is True
    assert options["name"].startswith("reef-") and options["org"] == "acme"
    assert options["labels"] == {
        "env": "ci",
        "reef": "sandbox",
        "reef_owner": "team",
        "reef_machine": options["name"],
    }
    assert (options["vcpu"], options["memory"]) == (2, "8G")
    assert options["auto_suspend_timeout"] == 0 and machines.hibernate_timeouts == [("machine-1", 0)]
    assert options["delete_after"] == (
        600 + remote.SANDBOX_MARGIN_S + boxd_sandbox.MACHINE_READY_S + boxd_sandbox.COMMAND_TIMEOUT_S
    )
    assert "auto_destroy_timeout" not in options
    assert [(route.name, route.port) for route in options["proxies"]] == [
        ("reef-relay-0", TUNNEL_PORT_BASE),
        ("reef-relay-1", TUNNEL_PORT_BASE + 1),
    ]
    setup = machines.commands[0]
    assert setup[:4] == ["sudo", "-n", "-u", "root"] and REMOTE_BASE in setup[-1]
    assert f"useradd --system --no-create-home --shell /usr/sbin/nologin {boxd_sandbox.RUN_USER}" in setup[-1]
    assert f"install -d -o {boxd_sandbox.RUN_USER}" in setup[-1]
    assert "install -m 755" in setup[-1] and "/usr/local/bin/node" in setup[-1]
    assert f"--uid-owner {boxd_sandbox.RUN_USER} -j {boxd_sandbox.FIREWALL_CHAIN}" in setup[-1]
    assert "-j REJECT" in setup[-1]
    assert "apt-get install -y -qq bubblewrap" in setup[-1]
    assert "npm install -g --prefix /usr/local pi@1.0.0" in setup[-1]
    assert started == [
        (f"https://route-{TUNNEL_PORT_BASE}.test", 8900, {}),
        (f"https://route-{TUNNEL_PORT_BASE + 1}.test", 8901, {}),
    ]
    assert all("key" not in " ".join(command).replace("--secret-file", "") for command in machines.commands)
    session.close()
    assert machines.deleted == ["machine-1"]


@pytest.mark.unit
def test_a_snapshot_machine_installs_nothing_and_auto_destroy_can_be_off(monkeypatch) -> None:
    machines = Machines()
    fake_boxd(monkeypatch, machines)
    session = BoxdExecutor(api_key="key", snapshot="prepared", auto_destroy=False).open()
    (options,) = machines.created
    assert options["from_snapshot"] == "prepared" and options["isolated"] is True
    assert options["delete_after"] is None and options["vcpu"] is None and options["memory"] is None
    assert "npm install" not in machines.commands[0][-1] and "apt-get" not in machines.commands[0][-1]
    session.close()


@pytest.mark.unit
def test_a_machine_that_does_not_become_ready_is_deleted(monkeypatch) -> None:
    machines = Machines()
    machines.ready_error = TimeoutError("no answer")
    fake_boxd(monkeypatch, machines)
    with pytest.raises(ProgramLaunchError, match="did not become ready: no answer"):
        BoxdExecutor(api_key="key").open()
    assert machines.deleted == ["machine-1"]


@pytest.mark.unit
def test_a_restart_deletes_only_the_machines_its_deployment_left(monkeypatch) -> None:
    mine_labels = {"reef": "sandbox", "reef_owner": "team", "reef_machine": "reef-1"}
    mine = types.SimpleNamespace(id="mine", name="reef-1", labels=mine_labels)
    fork = types.SimpleNamespace(id="fork", name="my-debug-fork", labels=dict(mine_labels))
    theirs = types.SimpleNamespace(
        id="theirs", name="reef-2", labels={"reef": "sandbox", "reef_owner": "other", "reef_machine": "reef-2"}
    )
    unrelated = types.SimpleNamespace(id="unrelated", name="prod", labels={"env": "prod"})
    machines = Machines((mine, fork, theirs, unrelated))
    fake_boxd(monkeypatch, machines)
    assert BoxdExecutor(api_key="key", owner="team").reap() == 1
    assert machines.deleted == ["mine"]
    assert BoxdExecutor(api_key="key").reap() == 0


class ProcessMachines(Machines):
    """Answers a launched process: its exit status after ``polls`` checks, and its output files."""

    def __init__(self, status: str, *, polls: int = 1, stdout: bytes = b"", stderr: bytes = b"") -> None:
        super().__init__()
        self.status = status
        self.polls = polls
        self.stdout = stdout
        self.stderr = stderr
        self.errors: list[Exception] = []
        self.source = ""

    def answer(self, command: list[str]) -> Result:
        text = command[-1]
        if command[0] == "mktemp":
            return Result(stdout="/tmp/reef-download-x\n")
        if text.startswith("install -m 600 -o boxd "):
            self.source = shlex.split(text)[5]
        if text.startswith("if [ -f /tmp/reef-process-"):
            if self.errors:
                raise self.errors.pop(0)
            self.polls -= 1
            return Result(stdout=f"{self.status}\n" if self.polls <= 0 else "")
        return Result()

    def download(self, path: str) -> bytes:
        return self.stdout if self.source.endswith("stdout") else self.stderr


def process_session(monkeypatch, machines: ProcessMachines, env: dict[str, str] | None = None) -> BoxdSession:
    fake_boxd(monkeypatch, machines)
    monkeypatch.setattr(boxd_sandbox, "POLL_INTERVAL_S", 0)
    monkeypatch.setattr(machines.files, "download", lambda machine_id, path: machines.download(path))
    session = BoxdSession(types.SimpleNamespace(machines=machines), "machine-1", env=env)
    monkeypatch.setattr(session, "push", lambda path: None)
    monkeypatch.setattr(session, "pull", lambda path, relative="": None)
    return session


@pytest.mark.unit
def test_a_process_runs_detached_in_the_jail_with_its_exit_status_and_output(tmp_path, monkeypatch) -> None:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    machines = ProcessMachines("3", polls=2, stdout=b"done \xff", stderr=b"note")
    machines.errors.append(BoxdError("connection dropped"))
    session = process_session(monkeypatch, machines, env={"DEPLOYMENT": "yes"})
    outcome = session.launch(
        ["pi", "--print", str(root / "workspace" / "out.txt")],
        root=root,
        workspace=root / "workspace",
        env={"CALLER": "1"},
        timeout=30,
    )
    assert (outcome.exit_code, outcome.stdout, outcome.stderr) == (3, "done �", "note")
    launching = next(command for command in machines.commands if "nohup setsid" in command[-1])
    assert launching[:4] == ["sudo", "-n", "-u", boxd_sandbox.RUN_USER]
    launched = launching[-1]
    assert launched.startswith(f"cd {REMOTE_BASE}/run/workspace || exit 1;")
    script = shlex.split(launched.split("nohup setsid ", 1)[1])[2]
    assert f"timeout --kill-after={boxd_sandbox.KILL_AFTER_S} 30 /bin/sh -c" in script and "< /dev/null" in script
    assert "exec bwrap" in script and f"{REMOTE_BASE}/run/workspace/out.txt" in script
    assert "DEPLOYMENT yes" in script and "CALLER 1" in script
    assert any(command[-1].startswith("rm -rf /tmp/reef-process-") for command in machines.commands)


@pytest.mark.unit
def test_a_process_past_its_limit_is_a_timeout_and_what_it_wrote_comes_back(tmp_path, monkeypatch) -> None:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    machines = ProcessMachines("124")
    session = process_session(monkeypatch, machines)
    pulled: list[Path] = []
    monkeypatch.setattr(session, "pull", lambda path, relative="": pulled.append(path))
    clock = iter([0.0, 1.0, 31.0, 31.0])
    monkeypatch.setattr(boxd_sandbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(ProgramTimeoutError, match="past its 30 s limit"):
        session.launch(["pi"], root=root, workspace=root / "workspace", env={}, timeout=30)
    assert pulled == [root]
    killed = next(command for command in machines.commands if command[-1].startswith("pkill -KILL -s"))
    assert killed[:4] == ["sudo", "-n", "-u", "root"]


@pytest.mark.unit
def test_a_failed_command_is_a_launch_error_and_a_root_file_never_stays_readable(monkeypatch) -> None:
    machines = Machines()
    fake_boxd(monkeypatch, machines)
    session = BoxdSession(types.SimpleNamespace(machines=machines), "machine-1")
    session.write_file("/root/reef-relay-1.secret", b"secret", as_root=True)
    assert list(machines.files.stored) == ["/tmp/reef-upload-x/file"]
    install = machines.commands[-1]
    assert install[:4] == ["sudo", "-n", "-u", "root"] and "install -m 600 -o root -g root" in install[-1]
    assert "rm -rf /tmp/reef-upload-x" in install[-1]
    monkeypatch.setattr(machines, "answer", lambda command: Result(2, stderr="no such file"))
    with pytest.raises(ProgramLaunchError, match="exited with 2: no such file"):
        session.run("false")


@pytest.mark.unit
def test_a_session_refuses_to_launch_after_its_machine_restarted(tmp_path, monkeypatch) -> None:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    machines = ProcessMachines("0")
    session = process_session(monkeypatch, machines)
    session.boot_id = "first-boot"
    monkeypatch.setattr(
        machines,
        "answer",
        lambda command: Result(stdout="second-boot\n") if boxd_sandbox.BOOT_ID_FILE in command[-1] else Result(),
    )
    with pytest.raises(ProgramLaunchError, match="restarted"):
        session.launch(["pi"], root=root, workspace=root / "workspace", env={}, timeout=30)
    assert not any("nohup setsid" in command[-1] for command in machines.commands)


@pytest.mark.unit
def test_a_create_that_got_no_answer_deletes_only_the_machine_with_its_own_name(monkeypatch) -> None:
    class LostAnswer(Machines):
        def __init__(self, label_matches: bool) -> None:
            super().__init__()
            self.label_matches = label_matches
            self.name = ""

        def create(self, name: str, **options: Any) -> Any:
            self.name = name
            raise BoxdError("connection lost")

        def get(self, name: str) -> Any:
            label = name if self.label_matches else "another"
            return types.SimpleNamespace(id="made-1", labels={"reef_machine": label})

    for label_matches, deleted in ((True, ["made-1"]), (False, [])):
        machines = LostAnswer(label_matches)
        fake_boxd(monkeypatch, machines)
        with pytest.raises(ProgramLaunchError, match="did not start: connection lost"):
            BoxdExecutor(api_key="key").open()
        assert machines.deleted == deleted


@pytest.mark.unit
def test_a_failed_delete_does_not_hide_the_setup_error(monkeypatch) -> None:
    class FailedDelete(Machines):
        def delete(self, machine_id: str) -> None:
            raise BoxdError("delete refused")

    machines = FailedDelete()
    machines.ready_error = TimeoutError("no answer")
    fake_boxd(monkeypatch, machines)
    with pytest.raises(ProgramLaunchError, match="did not become ready: no answer"):
        BoxdExecutor(api_key="key").open()


@pytest.mark.unit
def test_closing_a_session_whose_machine_is_gone_succeeds(monkeypatch) -> None:
    class AlreadyGone(Machines):
        def delete(self, machine_id: str) -> None:
            raise NotFoundError("gone")

    machines = AlreadyGone()
    fake_boxd(monkeypatch, machines)
    session = BoxdSession(types.SimpleNamespace(machines=machines), "machine-1")
    session.close()
    assert session.closed


@pytest.mark.unit
def test_a_launch_that_never_started_leaves_the_local_root_alone(tmp_path, monkeypatch) -> None:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    machines = ProcessMachines("0")
    session = process_session(monkeypatch, machines)
    pulled: list[Path] = []
    monkeypatch.setattr(session, "pull", lambda path, relative="": pulled.append(path))

    def failed_push(path: Path) -> None:
        raise ProgramLaunchError("the copy failed")

    monkeypatch.setattr(session, "push", failed_push)
    with pytest.raises(ProgramLaunchError, match="the copy failed"):
        session.launch(["pi"], root=root, workspace=root / "workspace", env={}, timeout=30)
    assert pulled == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer, message", [("ended", "ended without an exit status"), ("restarted", "restarted while the process ran")]
)
def test_a_process_that_died_is_a_launch_error_and_not_a_timeout(tmp_path, monkeypatch, answer, message) -> None:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    session = process_session(monkeypatch, ProcessMachines(answer))
    with pytest.raises(ProgramLaunchError, match=message):
        session.launch(["pi"], root=root, workspace=root / "workspace", env={}, timeout=3600)


@pytest.mark.unit
def test_a_temporary_file_left_behind_does_not_fail_the_copy_back(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.txt").write_text("42")
    root = tmp_path / "run"
    root.mkdir()
    machines = Machines()
    monkeypatch.setattr(
        machines, "answer", lambda command: Result(1, stderr="busy") if "rm -f" in command[-1] else Result()
    )
    fake_boxd(monkeypatch, machines)
    session = BoxdSession(types.SimpleNamespace(machines=machines), "machine-1")
    monkeypatch.setattr(session, "read_file", lambda path: remote.pack(source))
    session.pull(root)
    assert (root / "answer.txt").read_text() == "42"


@pytest.mark.unit
def test_a_session_needs_a_machine_id() -> None:
    with pytest.raises(ValueError, match="non-empty machine ID"):
        BoxdSession(types.SimpleNamespace(), "")


def firewall_tools(directory: Path, *, ss_status: int = 0) -> dict[str, str]:
    """Stand-in tools on ``PATH`` that log each rule; ``-C`` always answers that the rule is missing."""
    directory.mkdir()
    log = directory / "rules.log"
    scripts = {
        "iptables": f'[ "$1" = -C ] && exit 1; echo "$0 $*" >> {log}',
        "ip6tables": f'[ "$1" = -C ] && exit 1; echo "$0 $*" >> {log}',
        "sysctl": f'echo "sysctl $*" >> {log}',
        "systemctl": "exit 1",
        "ss": (
            f"[ {ss_status} -eq 0 ] || exit {ss_status}; "
            'case "$1" in -ltnH) echo "LISTEN 0 128 0.0.0.0:57073 0.0.0.0:*";; '
            '*) echo "UNCONN 0 0 127.0.0.1:323 0.0.0.0:*";; esac'
        ),
    }
    for name, body in scripts.items():
        path = directory / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    return {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"}


@pytest.mark.unit
def test_the_firewall_opens_the_forwarded_port_and_refuses_the_machines_services(tmp_path) -> None:
    environ = firewall_tools(tmp_path / "tools")
    completed = subprocess.run(
        ["/bin/sh", "-c", boxd_sandbox.firewall_command((8900,))], env=environ, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    rules = (tmp_path / "tools" / "rules.log").read_text().splitlines()
    ipv4 = [rule for rule in rules if "/iptables " in rule]
    chain = boxd_sandbox.FIREWALL_CHAIN
    # A snapshot's old rules go first, so that they cannot refuse this session's forwarded port.
    assert ipv4.index(next(rule for rule in ipv4 if f"-F {chain}" in rule)) < ipv4.index(
        next(rule for rule in ipv4 if f"-A {chain}" in rule)
    )
    assert "--dport 8900 -j ACCEPT" in next(rule for rule in ipv4 if f"-A {chain}" in rule)
    assert any(f"-I OUTPUT 1 -m owner --uid-owner {boxd_sandbox.RUN_USER} -j {chain}" in rule for rule in ipv4)
    for expected in (
        "-p tcp --dport 57073 -j REJECT",
        "-p udp --dport 323 -j REJECT",
        "-p tcp --dport 0:1023 -j REJECT",
        "-p tcp --dport 32768:65535 -j REJECT",
        "-p udp --dport 32768:65535 -j REJECT",
    ):
        assert any(expected in rule for rule in ipv4) and any(
            expected in rule for rule in rules if "/ip6tables " in rule
        )
    assert any(f"net.ipv4.ip_local_port_range={boxd_sandbox.EPHEMERAL_PORTS}" in rule for rule in rules)


@pytest.mark.unit
def test_the_firewall_fails_when_it_cannot_list_the_ports_in_use(tmp_path) -> None:
    environ = firewall_tools(tmp_path / "tools", ss_status=3)
    completed = subprocess.run(
        ["/bin/sh", "-c", boxd_sandbox.firewall_command(())], env=environ, capture_output=True, text=True
    )
    assert completed.returncode != 0


@pytest.mark.unit
def test_every_machine_command_has_a_time_limit(monkeypatch) -> None:
    machines = Machines()
    fake_boxd(monkeypatch, machines)
    session = BoxdExecutor(api_key="key").open()
    machines.files.stored["/tmp/reef-upload-x/file"] = b"data"
    session.write_file("/tmp/a", b"data")
    assert session.read_file("/tmp/a") == b"data"
    assert machines.timeouts and all(timeout is not None and timeout > 0 for timeout in machines.timeouts)


@pytest.mark.unit
def test_preparing_a_machine_makes_its_own_users_home_and_environment_file_private(monkeypatch) -> None:
    machines = Machines()
    fake_boxd(monkeypatch, machines)
    BoxdExecutor(api_key="key").open()
    prepare = next(command[-1] for command in machines.commands if "useradd" in command[-1])
    assert 'chmod o-rwx "$owner_home"' in prepare
    assert f"chmod 640 {boxd_sandbox.MACHINE_ENV_FILE}" in prepare


@pytest.mark.unit
def test_a_failed_startup_probe_deletes_its_machine_and_names_the_check(monkeypatch) -> None:
    class FailedProbe(Machines):
        def answer(self, command: list[str]) -> Result:
            if "/bin/true" in command[-1]:
                return Result(exit_code=1, stderr="bwrap: no permission")
            return super().answer(command)

    machines = FailedProbe()
    fake_boxd(monkeypatch, machines)
    with pytest.raises(SandboxUnavailableError, match=r"boxd startup check failed: .*no permission"):
        BoxdExecutor(api_key="key", npm_package="pi@1.0.0", binary="pi").preflight()
    assert machines.deleted == ["machine-1"]
    assert "reef_owner" not in machines.created[0]["labels"]


@pytest.mark.unit
def test_a_startup_probe_whose_machine_cannot_be_deleted_is_unavailable(monkeypatch) -> None:
    class FailedDelete(Machines):
        def delete(self, machine_id: str) -> None:
            raise BoxdError("delete refused")

    machines = FailedDelete()
    fake_boxd(monkeypatch, machines)
    executor = BoxdExecutor(api_key="key", npm_package="pi@1.0.0", binary="pi")
    with pytest.raises(SandboxUnavailableError, match="could not be cleaned up"):
        executor.preflight()
    assert not executor.validated
