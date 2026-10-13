"""CLI startup failures name the failed service and preserve its final output."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

from reef.cli import main as cli_main
from reef.service.deploy import orchestrator


def _write_stack(path: Path, run_dir: Path) -> None:
    """A provider deployment; each test replaces the processes Reef assembles for it."""
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {"run-dir": str(run_dir)},
                "inference": {"upstream-url": "http://127.0.0.1:9", "upstream-model": "demo"},
            }
        )
    )


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing-command"])
def test_startup_failure_reports_reason_and_logs_then_stops_processes(
    tmp_path: Path, capfd, monkeypatch, failure
) -> None:
    pid_file = tmp_path / "worker.pid"
    command = [
        sys.executable,
        "-u",
        "-c",
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('worker startup output', flush=True); "
        + ("raise SystemExit(7)" if failure == "exit" else "time.sleep(120)"),
    ]
    if failure == "missing-command":
        command = [str(tmp_path / "missing-program")]
    run_dir = tmp_path / "logs"
    config_path = tmp_path / "stack.yaml"
    _write_stack(config_path, run_dir)
    services = [
        {"name": "model", "command": command, "ready": "false", "ready_timeout": 1},
        {
            "name": "api",
            "command": [sys.executable, "-c", "raise RuntimeError('must not start')"],
            "depends_on": ["model"],
        },
    ]
    monkeypatch.setattr(orchestrator, "validate_services", lambda config, path: services)

    with pytest.raises(SystemExit) as caught:
        cli_main(["serve", "-c", str(config_path)])

    assert caught.value.code == 1
    output = capfd.readouterr()
    assert "model" in output.err
    assert str(run_dir) in output.err
    assert "Traceback" not in output.err
    if failure == "exit":
        assert "exit code 7" in output.err
    elif failure == "timeout":
        assert "did not become ready within 1s" in output.err
    else:
        assert "missing-program" in output.err
    if failure != "missing-command":
        assert "worker startup output" in output.out
        assert "worker startup output" in (run_dir / "model.log").read_text()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    assert not (run_dir / "api.worker.json").exists()


def test_sigterm_during_startup_stops_the_service_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "service.pid"
    _write_stack(tmp_path / "stack.yaml", tmp_path / "logs")
    services = [
        {
            "name": "loading-model",
            "command": [
                sys.executable,
                "-c",
                "import os,time; from pathlib import Path; "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)",
            ],
            "ready": "false",
            "ready_timeout": 60,
        }
    ]
    script = tmp_path / "serve.py"
    script.write_text(
        "from reef.cli import main\n"
        "from reef.service.deploy import orchestrator\n"
        f"orchestrator.validate_services = lambda config, path: {services!r}\n"
        "main(['serve', '-c', 'stack.yaml'])\n"
    )
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))}
    process = subprocess.Popen(
        [sys.executable, script.name],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert "Traceback" not in stderr
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
