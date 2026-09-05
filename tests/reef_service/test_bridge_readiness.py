from __future__ import annotations

import subprocess
import sys

import pytest

from reef.service.deploy.config import validate_services
from reef.service.deploy.orchestrator import _Stack


@pytest.mark.parametrize("stale_marker", [False, True], ids=["fresh-start", "restart"])
def test_stack_waits_for_current_bridge_marker_before_starting_dependents(tmp_path, monkeypatch, stale_marker):
    marker = tmp_path / "bridge.ready"
    if stale_marker:
        marker.write_text("previous driver")
    release_driver = tmp_path / "release-driver"
    dependent_started = tmp_path / "dependent-started"
    config = {
        "services": [
            {
                "name": "slime-driver",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "import os, time\n"
                        "from pathlib import Path\n"
                        f"while not Path({str(release_driver)!r}).exists(): time.sleep(0.01)\n"
                        "Path(os.environ['REEF_BRIDGE_READY_FILE']).write_text('current driver')\n"
                        "time.sleep(120)\n"
                    ),
                ],
                "ready": 'test -f "$REEF_BRIDGE_READY_FILE"',
            },
            {
                "name": "reef",
                "depends_on": ["slime-driver"],
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "import time\n"
                        "from pathlib import Path\n"
                        f"assert Path({str(marker)!r}).read_text() == 'current driver'\n"
                        f"Path({str(dependent_started)!r}).touch()\n"
                        "time.sleep(120)\n"
                    ),
                ],
                "ready": 'test -f "$STARTED_FILE"',
                "env": {"STARTED_FILE": str(dependent_started)},
            },
        ],
    }
    spawn = subprocess.Popen
    probe = subprocess.run
    readiness = []

    def checked_spawn(command, **options):
        if command == config["services"][0]["command"]:
            assert not marker.exists(), "stale marker must be removed before the driver starts"
        return spawn(command, **options)

    def checked_probe(command, **options):
        result = probe(command, **options)
        if command == config["services"][0]["ready"]:
            readiness.append(result.returncode == 0)
            if result.returncode != 0:
                assert not dependent_started.exists()
                release_driver.touch()
        return result

    monkeypatch.setattr(subprocess, "Popen", checked_spawn)
    monkeypatch.setattr(subprocess, "run", checked_probe)
    stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 10, tmp_path / "input.yaml")
    try:
        stack.start()
        assert readiness[0] is False
        assert readiness[-1] is True
        assert marker.read_text() == "current driver"
        assert dependent_started.exists()
        assert stack._is_alive("slime-driver") and stack._is_alive("reef")
    finally:
        stack.shutdown(grace=1)
