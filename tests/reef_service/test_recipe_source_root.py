"""``reef serve`` puts a dotted recipe's package on every service's import path.

A cookbook recipe such as ``recipes.sao.recipe:SAORecipe`` is not
pip-installed; it lives beside the ``serve.yaml`` that selects it. The
orchestrator resolves that directory from the config file and exports it,
so example launchers no longer hand-roll ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from reef.service.deploy import orchestrator
from reef.service.deploy.config_utils import recipe_source_root
from reef.service.deploy.execution import service_executor_config
from reef.service.deploy.orchestrator import _run_orchestrator
from reef.service.deploy.process import ProcessWorker


def _checkout(tmp_path: Path, package: str = "recipes") -> tuple[Path, Path]:
    """A fake source checkout: ``<root>/<package>/`` and a config three levels down."""
    root = tmp_path / "checkout"
    (root / package).mkdir(parents=True)
    (root / package / "__init__.py").write_text("")
    config_path = root / package / "sao" / "examples" / "sao" / "serve.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("")
    return root, config_path


@pytest.mark.unit
@pytest.mark.parametrize(
    "reef_section, keep_init, holds_package",
    [
        ({"recipe": "recipes.sao.recipe:SAORecipe"}, True, True),
        # Bare recipe names and a missing recipe have no package.
        ({"recipe": "recipe"}, True, False),
        ({"recipe": "my-preset"}, True, False),
        (None, True, False),
        # A missing package leaves the import to the service.
        ({"recipe": "other_pkg.method:Recipe"}, True, False),
        # A directory without __init__.py is not the package.
        ({"recipe": "recipes.sao.recipe:SAORecipe"}, False, False),
    ],
)
def test_dotted_recipe_resolves_to_the_nearest_ancestor_holding_its_package(
    tmp_path: Path, reef_section: dict | None, keep_init: bool, holds_package: bool
) -> None:
    root, config_path = _checkout(tmp_path)
    if not keep_init:
        (root / "recipes" / "__init__.py").unlink()
    config = {} if reef_section is None else {"reef": reef_section}

    assert recipe_source_root(config, config_path) == (root if holds_package else None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "inherited, service_env, expected",
    [
        ("/opt/sglang/python:/root/Megatron-LM", {}, ["/opt/sglang/python", "/root/Megatron-LM", "<root>"]),
        (None, {}, ["<root>"]),
        # The service's env map still owns PYTHONPATH.
        (None, {"PYTHONPATH": "/explicit"}, ["/explicit"]),
    ],
)
def test_source_root_is_appended_to_pythonpath_once(
    tmp_path: Path, monkeypatch, inherited: str | None, service_env: dict[str, str], expected: list[str]
) -> None:
    root, config_path = _checkout(tmp_path)
    if inherited is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", inherited)
    stack = ProcessWorker({}, [{"name": "reef"}], tmp_path / "stack", 60, config_path, source_root=root)
    service = {"name": "reef", "env": service_env}

    env = stack._service_env(service)

    assert env["PYTHONPATH"].split(os.pathsep) == [str(root) if path == "<root>" else path for path in expected]
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    assert stack._service_env(service)["PYTHONPATH"] == env["PYTHONPATH"]


@pytest.mark.unit
def test_without_a_source_root_pythonpath_is_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    stack = ProcessWorker({}, [{"name": "reef"}], tmp_path / "stack", 60, tmp_path / "serve.yaml")

    assert "PYTHONPATH" not in stack._service_env({"name": "reef"})


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["uni", "mp"])
def test_a_service_imports_the_recipe_package_beside_the_config(tmp_path: Path, monkeypatch, backend: str) -> None:
    """End to end: a service started from an example directory imports the
    cookbook package without the launcher exporting ``PYTHONPATH``."""
    root, config_path = _checkout(tmp_path, package="probe_recipes")
    (root / "probe_recipes" / "method.py").write_text("from reef.recipe import Recipe\n")
    marker = tmp_path / "imported_from.txt"
    probe = "import probe_recipes.method as m, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(m.__file__)"
    config_path.write_text(
        "recipe:\n"
        "  implementation: probe_recipes.method:Recipe\n"
        "reef:\n"
        f"  run-dir: {tmp_path / 'run'}\n"
        "inference:\n"
        "  upstream-url: http://127.0.0.1:9\n"
        "  upstream-model: demo\n"
    )
    probe_service = {"name": "probe", "executor": backend, "command": [sys.executable, "-c", probe, str(marker)]}
    monkeypatch.setattr(orchestrator, "validate_services", lambda config, path: [probe_service])
    # The watchdog notices the probe's exit on its next check; a short interval ends the run at once.
    monkeypatch.setattr(orchestrator, "_WATCHDOG_INTERVAL", 0.05)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(config_path.parent)
    monkeypatch.setattr(sys, "path", list(sys.path))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # The probe exits as soon as it has written the marker, which the
        # watchdog reports as an unexpected exit: exit code 1 is the
        # expected outcome here, not a failure of the import.
        assert _run_orchestrator(str(config_path)) == 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

    assert Path(marker.read_text()).resolve() == (root / "probe_recipes" / "method.py").resolve()
    assert str(root) in sys.path


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["uni", "mp", "ray"])
def test_executor_passes_source_root_to_service_worker(tmp_path: Path, monkeypatch, backend: str) -> None:
    root, config_path = _checkout(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/root/Megatron-LM")
    service = {"name": "reef", "executor": backend}
    config = service_executor_config({}, service, tmp_path / "stack", 60, config_path, source_root=root)
    spec = config.workers[0]
    worker = spec.worker_cls(*spec.args, **spec.kwargs)
    try:
        assert worker._service_env(service)["PYTHONPATH"].split(os.pathsep) == ["/root/Megatron-LM", str(root)]
    finally:
        worker.shutdown(grace=0)
