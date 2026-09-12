"""Run example launchers with lightweight services, without GPUs or provider calls."""

import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("recipes/basic", "recipes/sao/examples/sao")


def _write_command(directory: Path, name: str, source: str) -> None:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize(
    "mode", ["exit", "exit-success", "timeout", "hanging-probe", "ready", "workload-error", "signal"]
)
def test_example_startup_and_cleanup(tmp_path: Path, example: str, mode: str) -> None:
    script = tmp_path / "run.sh"
    shutil.copyfile(ROOT / example / "run.sh", script)
    commands = tmp_path / "bin"
    commands.mkdir()
    _write_command(
        commands,
        "python3",
        """import os, signal, sys, time
from pathlib import Path

mode = os.environ['REEF_TEST_MODE']
if sys.argv[1:] == ['run.py']:
    Path('work/workload-started').touch()
    raise SystemExit(9 if mode == 'workload-error' else 0)
Path('work/service.pid').write_text(str(os.getpid()))
print('example startup output', flush=True)
if mode in ('exit', 'exit-success'):
    raise SystemExit(7 if mode == 'exit' else 0)
def stop(signum, frame):
    Path('work/service-stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path('work/service-started').touch()
while True:
    time.sleep(0.01)
""",
    )
    _write_command(
        commands,
        "curl",
        """import os, sys, time
from pathlib import Path

mode = os.environ['REEF_TEST_MODE']
if mode == 'hanging-probe':
    timeout = float(sys.argv[sys.argv.index('--max-time') + 1]) if '--max-time' in sys.argv else 120
    time.sleep(timeout)
    raise SystemExit(28)
time.sleep(0.05)
raise SystemExit(0 if mode in ('ready', 'workload-error') and Path('work/service-started').exists() else 7)
""",
    )
    env = {
        **os.environ,
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "REEF_TEST_MODE": mode,
        "REEF_STARTUP_TIMEOUT_S": "1" if mode in ("timeout", "hanging-probe") else "5",
    }
    process = subprocess.Popen(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        if mode == "signal":
            deadline = time.monotonic() + 5
            while not (tmp_path / "work/service-started").exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=8)
        expected = {"ready": 0, "workload-error": 9, "exit": 7, "signal": 143}.get(mode, 1)
        assert process.returncode == expected, stderr
        assert (tmp_path / "work/workload-started").exists() == (mode in ("ready", "workload-error"))
        if mode not in ("ready", "workload-error", "signal"):
            assert "example startup output" in stderr
            assert str(tmp_path / "work/reef.log") in stderr
            reason = {"exit": "exit code 7", "exit-success": "exit code 0"}.get(mode, "within 1s")
            assert reason in stderr
        if mode not in ("exit", "exit-success"):
            assert (tmp_path / "work/service-stopped").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "work/service.pid").read_text()), 0)
    finally:
        # The old infinite wait and failed assertions must not leave test workers alive.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("timeout", ["0", "-1", "abc", "1.5"])
def test_invalid_startup_timeout_fails_before_launch(tmp_path: Path, example: str, timeout: str) -> None:
    script = tmp_path / "run.sh"
    shutil.copyfile(ROOT / example / "run.sh", script)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "REEF_STARTUP_TIMEOUT_S": timeout},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert "REEF_STARTUP_TIMEOUT_S must be a positive integer" in result.stderr
    assert not (tmp_path / "work").exists()
