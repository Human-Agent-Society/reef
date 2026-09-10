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

from reef.service.deploy.config import recipe_source_root
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
def test_dotted_recipe_resolves_to_the_nearest_ancestor_holding_its_package(tmp_path: Path) -> None:
    root, config_path = _checkout(tmp_path)

    assert recipe_source_root({"reef": {"recipe": "recipes.sao.recipe:SAORecipe"}}, config_path) == root


@pytest.mark.unit
def test_bare_recipe_names_have_no_source_root(tmp_path: Path) -> None:
    _, config_path = _checkout(tmp_path)

    assert recipe_source_root({"reef": {"recipe": "recipe"}}, config_path) is None
    assert recipe_source_root({"reef": {"recipe": "my-preset"}}, config_path) is None
    assert recipe_source_root({}, config_path) is None


@pytest.mark.unit
def test_missing_package_leaves_the_import_to_the_service(tmp_path: Path) -> None:
    _, config_path = _checkout(tmp_path)

    assert recipe_source_root({"reef": {"recipe": "other_pkg.method:Recipe"}}, config_path) is None


@pytest.mark.unit
def test_a_directory_without_init_is_not_the_package(tmp_path: Path) -> None:
    root, config_path = _checkout(tmp_path)
    (root / "recipes" / "__init__.py").unlink()

    assert recipe_source_root({"reef": {"recipe": "recipes.sao.recipe:SAORecipe"}}, config_path) is None


@pytest.mark.unit
def test_source_root_is_appended_to_pythonpath_once(tmp_path: Path, monkeypatch) -> None:
    root, config_path = _checkout(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/opt/sglang/python:/root/Megatron-LM")
    stack = ProcessWorker({}, [{"name": "reef"}], tmp_path / "stack", 60, config_path, source_root=root)

    env = stack._service_env({"name": "reef"})

    assert env["PYTHONPATH"].split(os.pathsep) == ["/opt/sglang/python", "/root/Megatron-LM", str(root)]

    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    assert stack._service_env({"name": "reef"})["PYTHONPATH"] == env["PYTHONPATH"]


@pytest.mark.unit
def test_source_root_starts_pythonpath_when_unset(tmp_path: Path, monkeypatch) -> None:
    root, config_path = _checkout(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    stack = ProcessWorker({}, [{"name": "reef"}], tmp_path / "stack", 60, config_path, source_root=root)

    assert stack._service_env({"name": "reef"})["PYTHONPATH"] == str(root)


@pytest.mark.unit
def test_service_env_map_still_owns_pythonpath(tmp_path: Path, monkeypatch) -> None:
    root, config_path = _checkout(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    stack = ProcessWorker({}, [{"name": "reef"}], tmp_path / "stack", 60, config_path, source_root=root)

    env = stack._service_env({"name": "reef", "env": {"PYTHONPATH": "/explicit"}})

    assert env["PYTHONPATH"] == "/explicit"


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
    (root / "probe_recipes" / "method.py").write_text("class Recipe: ...\n")
    marker = tmp_path / "imported_from.txt"
    probe = "import probe_recipes.method as m, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(m.__file__)"
    config_path.write_text(
        "reef:\n"
        "  recipe: probe_recipes.method:Recipe\n"
        f"run_dir: {tmp_path / 'run'}\n"
        "services:\n"
        "  - name: probe\n"
        f"    executor: {backend}\n"
        f'    command: ["${{REEF_PYTHON}}", "-c", {probe!r}, "{marker}"]\n'
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("REEF_PYTHON", raising=False)
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
