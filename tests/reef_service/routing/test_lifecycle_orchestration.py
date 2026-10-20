"""The public managed configuration reaches orchestration and its HTTP child."""

import json
import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
import yaml

from reef.service.deploy import orchestrator
from reef.service.deploy.routing.routing_containers import ContainerRuntime
from reef.service.deploy.service_config import ServiceConfig

pytestmark = pytest.mark.unit

#: A managed owner run in its own process: the real ``_run_orchestrator`` and ``_Stack`` signal handling,
#: run loop and shutdown, with a fake routing controller and no service start, so nothing touches Docker.
#: It raises one real SIGHUP when it reaches ``phase``; a default SIGHUP there would kill the owner.
MANAGED_OWNER_PROGRAM = textwrap.dedent(
    """
    import json
    import signal
    import sys
    from pathlib import Path

    from reef.service.deploy import orchestrator

    phase, config, previous_handler = sys.argv[1], sys.argv[2], sys.argv[3]
    root = Path(config).parent
    events, reloads, previous_calls, ticks = [], [], [], []

    def reach(name):
        events.append(name)
        if name == phase:
            signal.raise_signal(signal.SIGHUP)

    def previous(signum, frame):
        previous_calls.append(signum)

    class Controller:
        def __init__(self, settings, runtime):
            self.directory = root / "generation"
            self.stop_requested = False
            self.closed = False
            self.failed = False

        def prepare(self):
            reach("prepare")

        def request_reload(self):
            reloads.append(events[-1])

        def request_stop(self, *, force=False):
            self.stop_requested = True

        def tick(self):
            ticks.append(events[-1])
            if len(ticks) == 1:
                reach("steady")
            else:
                # The operator stops the deployment; a SIGHUP right after must not run another tick.
                signal.raise_signal(signal.SIGTERM)
                reach("stopping")

        def shutdown(self):
            self.closed = True
            reach("cleanup")

        def release_control_directory(self):
            reach("release")

    class Stack(orchestrator._Stack):
        def start(self):
            reach("start")

        def shutdown(self, grace=0):
            reach("shutdown")
            super().shutdown(grace=0)

    handler = previous if previous_handler == "custom" else signal.SIG_DFL
    signal.signal(signal.SIGHUP, handler)
    orchestrator.RoutingDeploymentController = Controller
    orchestrator._Stack = Stack
    code = orchestrator._run_orchestrator(config)
    restored = signal.getsignal(signal.SIGHUP) == handler
    if previous_handler == "custom":
        signal.raise_signal(signal.SIGHUP)
    print(json.dumps({
        "code": code,
        "events": events,
        "reloads": reloads,
        "ticks": len(ticks),
        "restored": restored,
        "previous_calls": previous_calls,
        "orchestrator": orchestrator.__file__,
    }))
    """
)


def write_managed_config(tmp_path: Path) -> Path:
    path = tmp_path / "reef.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "host": "127.0.0.1",
                    "tokens": ["test-gateway"],
                    "run_dir": str(tmp_path / "stack"),
                    "routing_manifest": str(tmp_path / "manifest.yaml"),
                    "routing_deployment": {"deployment_id": "test-managed"},
                },
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:8101", "upstream_model": "student"},
            }
        )
    )
    return path


def run_managed_owner(tmp_path: Path, phase: str, previous_handler: str) -> dict[str, object]:
    # The child imports the same Reef as this test: this checkout or the installed wheel.
    environment = {**os.environ, "PYTHONPATH": str(Path(orchestrator.__file__).resolve().parents[3])}
    result = subprocess.run(
        [sys.executable, "-c", MANAGED_OWNER_PROGRAM, phase, str(write_managed_config(tmp_path)), previous_handler],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"owner exited with {result.returncode}: {result.stderr[-2000:]}"
    owner = json.loads(result.stdout.splitlines()[-1])
    assert Path(owner["orchestrator"]).resolve() == Path(orchestrator.__file__).resolve()
    return owner


LIFECYCLE_EVENTS = ["prepare", "start", "steady", "stopping", "shutdown", "cleanup", "release"]


@pytest.mark.parametrize("phase", LIFECYCLE_EVENTS)
def test_sighup_anywhere_in_a_managed_lifecycle_only_records_a_reload(tmp_path: Path, phase: str) -> None:
    owner = run_managed_owner(tmp_path, phase, "default")
    # The owner survives and completes every cleanup step; the signal became one reload request.
    assert owner["code"] == 0
    assert owner["events"] == LIFECYCLE_EVENTS
    assert owner["reloads"] == [phase]
    # A stop takes precedence: the run loop never ticks the controller again, so no reload can start.
    assert owner["ticks"] == 2
    # After the last cleanup the handler that was installed before reef serve is back.
    assert owner["restored"] is True


def test_managed_owner_restores_a_custom_sighup_handler_it_never_calls(tmp_path: Path) -> None:
    owner = run_managed_owner(tmp_path, "prepare", "custom")
    assert owner["reloads"] == ["prepare"]
    assert owner["restored"] is True
    # Only the SIGHUP sent after reef serve returned reached the earlier handler.
    assert owner["previous_calls"] == [signal.SIGHUP]


def test_an_ordinary_deployment_never_installs_a_sighup_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = []
    monkeypatch.setattr(orchestrator.signal, "signal", lambda signum, handler: installed.append(signum))

    class Stack(orchestrator._Stack):
        """The real stack with no services: it starts and runs nothing, so only its signal setup is observed."""

        def start(self) -> None:
            pass

        def run_until_stopped(self) -> None:
            pass

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    path = tmp_path / "reef.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {"host": "127.0.0.1", "tokens": ["test-service"], "run_dir": str(tmp_path / "stack")},
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:8101", "upstream_model": "student"},
            }
        )
    )
    assert orchestrator._run_orchestrator(str(path)) == 0
    assert signal.SIGTERM in installed and signal.SIGINT in installed
    assert signal.SIGHUP not in installed


def test_managed_object_config_prepares_controller_and_private_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    class Controller:
        """Stands in for the routing controller: records what the orchestrator passes and calls, nothing else."""

        def __init__(self, settings: ServiceConfig, runtime: ContainerRuntime) -> None:
            observed["deployment_id"] = settings.routing_deployment.deployment_id
            self.directory = tmp_path / "generation"
            self.stop_requested = False
            self.closed = False

        def prepare(self) -> None:
            observed["prepared"] = True

        def shutdown(self) -> None:
            self.closed = True
            observed["closed"] = True

        def release_control_directory(self) -> None:
            # Only after the stack, and so the REEF child, has stopped.
            observed["control_released_after_close"] = self.closed

    class Stack:
        """Stands in for the stack: records the managed configuration and control file it receives."""

        def __init__(
            self,
            config: dict[str, dict[str, object]],
            services: Sequence[dict[str, object]],
            run_dir: Path,
            timeout: int,
            path: str | Path,
            *,
            source_root: Path | None,
            routing_controller: Controller,
        ) -> None:
            observed["manifest"] = config["reef"]["routing_manifest"]
            observed["control_file"] = services[0]["env"]["REEF_ROUTING_CONTROL_FILE"]
            observed["controller"] = routing_controller
            self.exit_code = 0

        def start(self) -> None:
            observed["started"] = True

        def block(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(orchestrator, "RoutingDeploymentController", Controller)
    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    path = tmp_path / "reef.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "host": "127.0.0.1",
                    "tokens": ["test-gateway"],
                    "run_dir": str(tmp_path / "stack"),
                    "routing_manifest": str(tmp_path / "manifest.yaml"),
                    # A routing manifest is enough: routing is always managed by reef serve.
                    "routing_deployment": {"deployment_id": "test-managed"},
                },
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:8101", "upstream_model": "student"},
            }
        )
    )
    assert orchestrator._run_orchestrator(str(path)) == 0
    assert observed["deployment_id"] == "test-managed"
    assert observed["prepared"] is True
    assert observed["started"] is True
    assert observed["closed"] is True
    assert Path(observed["manifest"]) == tmp_path / "generation" / "manifest.yaml"
    assert Path(observed["control_file"]) == tmp_path / "generation" / "control.json"
    assert observed["control_released_after_close"] is True


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({"gateway_port": 8899}, "unknown config fields: gateway_port"),
        ({"restart_policy": "always"}, "restart_policy must be fail_stop or bounded"),
    ],
)
def test_invalid_routing_settings_stop_before_any_process_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: dict[str, str | int], message: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("an invalid routing configuration reached the controller or stack")

    monkeypatch.setattr(orchestrator, "RoutingDeploymentController", forbidden)
    monkeypatch.setattr(orchestrator, "_Stack", forbidden)
    path = tmp_path / "reef.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "host": "127.0.0.1",
                    "tokens": ["test-gateway"],
                    "routing_manifest": str(tmp_path / "manifest.yaml"),
                    "routing_deployment": {"deployment_id": "test-managed", **invalid},
                },
                "recipe": {"implementation": "recipe"},
            }
        )
    )
    with pytest.raises(orchestrator.DeployConfigError, match=message):
        orchestrator._run_orchestrator(str(path))


def test_managed_child_existing_runtime_config_becomes_private(tmp_path: Path) -> None:
    import stat

    from reef.service.deploy.process import ProcessWorker

    directory = tmp_path / "worker"
    directory.mkdir(mode=0o755)
    path = directory / "runtime.yaml"
    path.write_text("old config")
    path.chmod(0o644)
    worker = ProcessWorker(
        {},
        [
            {
                "name": "reef",
                "command": ["unused"],
                "env": {"REEF_ROUTING_CONTROL_FILE": str(tmp_path / "control.json")},
            }
        ],
        directory,
        10,
        path,
    )
    worker.prepare({"reef": {"tokens": ["test-private-token"]}})
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_unexpected_routing_cleanup_exception_still_stops_native_executor(tmp_path: Path) -> None:
    actions = []

    class Controller:
        """A routing controller whose shutdown always fails with an unexpected error."""

        def request_stop(self, *, force: bool = False) -> None:
            actions.append("routing-stop")

        def shutdown(self) -> NoReturn:
            raise ValueError("malformed dependency reply")

    class Executor:
        """A native executor whose RPCs succeed with empty results; it logs each method it is asked to run."""

        def rpc(self, worker: int, method: str, **kwargs: object) -> tuple[str, int] | bool:
            actions.append(method)
            if method == "read_log":
                return "", 0
            return False

        def shutdown(self) -> None:
            actions.append("executor-shutdown")

    stack = orchestrator._Stack({}, [], tmp_path, 1, tmp_path / "config", routing_controller=Controller())
    stack._executors["reef"] = Executor()
    stack.shutdown(grace=0)
    assert "request_stop" in actions and "shutdown" in actions and "executor-shutdown" in actions
    assert stack.exit_code == 1


def test_watchdog_child_exit_interrupts_routing_recovery(tmp_path: Path) -> None:
    stopped = []

    class Controller:
        """A routing controller that only records stop requests."""

        def request_stop(self) -> None:
            stopped.append(True)

    class Executor:
        """A native executor whose ``reef`` child has exited with status 1."""

        def rpc(self, worker: int, method: str, **kwargs: object) -> tuple[str, int] | dict[str, int]:
            return ("", 0) if method == "read_log" else {"reef": 1}

    stack = orchestrator._Stack({}, [], tmp_path, 1, tmp_path / "config", routing_controller=Controller())
    stack._executors["reef"] = Executor()
    stack._watchdog()
    assert stopped == [True]
    assert stack.exit_code == 1
