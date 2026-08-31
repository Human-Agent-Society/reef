from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_isolated_import(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )


def _imported_modules(tree: ast.AST, *, package: str = "", module_scope_only: bool = False) -> list[str]:
    """Absolute module targets of every import statement in *tree*.

    A full AST walk over Import/ImportFrom nodes: a module path in a comment
    or docstring cannot false-fail, and an aliased import cannot slip.
    ``from pkg import name`` contributes ``pkg.name`` so submodule imports
    (``from reef.train import slime``) are caught too. Relative imports are
    resolved against *package* (the dotted package containing the module).

    With ``module_scope_only`` imports inside function bodies are excluded —
    that is the lazy-import escape hatch the layering rules allow. Class
    bodies and conditional blocks still execute at import time and count.
    """
    targets: list[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if module_scope_only and isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            if isinstance(child, ast.Import):
                targets.extend(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - child.level + 1])
                    module = f"{base}.{child.module}" if child.module else base
                else:
                    module = child.module or ""
                targets.extend(f"{module}.{alias.name}" for alias in child.names)
            visit(child)

    visit(tree)
    return targets


def _imports_of(targets: list[str], forbidden_prefix: str) -> list[str]:
    return [target for target in targets if target == forbidden_prefix or target.startswith(forbidden_prefix + ".")]


METHOD_PACKAGES = ("harness_evolve", "openclawrl", "sao", "tttd")


def test_infra_never_imports_a_method_package() -> None:
    # The whole of reef is the machinery the method packages build on,
    # never the reverse. Dotted recipe references import cookbook code only
    # when the operator selects it, so no reef module has any reason to name a
    # method at module scope or inside a function.
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "reef").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        offenders.extend(f"{path.relative_to(REPO_ROOT)} -> {target}" for target in _imports_of(imported, "recipes"))
    assert offenders == []


def test_scenario_domain_does_not_depend_on_recipes() -> None:
    module = importlib.import_module("reef.scenario.scenario")
    tree = ast.parse(inspect.getsource(module))

    imported = _imported_modules(tree, package="reef.scenario")
    for method in METHOD_PACKAGES:
        assert _imports_of(imported, f"reef.{method}") == []


def test_core_package_does_not_depend_on_artifacts() -> None:
    # The pure identity types (ArtifactRef and friends) live in reef.core;
    # reef.artifact builds on them, never the other way around.

    for name in (
        "reef.core",
        "reef.core.records_types",
        "reef.core.artifact_ref",
        "reef.core.errors",
    ):
        module = importlib.import_module(name)
        tree = ast.parse(inspect.getsource(module))
        imported = _imported_modules(tree, package="reef.core")
        assert _imports_of(imported, "reef.artifact") == [], name


def test_artifact_version_chain_does_not_depend_on_scenario_or_trainer() -> None:
    module = importlib.import_module("reef.artifact.version_chain")
    tree = ast.parse(inspect.getsource(module))

    imported = _imported_modules(tree, package="reef.artifact")
    assert _imports_of(imported, "reef.scenario") == []
    assert _imports_of(imported, "reef.train") == []


def test_backend_agnostic_core_never_imports_slime_backend_at_module_scope() -> None:
    # The most-repeated layering rule: Reef's backend-agnostic core talks to
    # training through the abstract reef.train surface (types, trainer,
    # processors) and to the concrete runtime only through the runtime
    # registry / named Ray actors — it never imports reef.train.slime or
    # reef.train.slime_backend at module scope. A lazy import inside a
    # function (an adapter reaching for the backend it was configured with)
    # is the allowed escape hatch.
    core_packages = (
        "reef/core",
        "reef/service",
        "reef/scenario",
        "reef/artifact",
        "reef/recipe",
        "reef/runtime",
        "reef/surface",
    )
    core_modules = ["reef/dispatcher.py", "reef/records.py"]
    files = [path for package in core_packages for path in sorted((REPO_ROOT / package).rglob("*.py"))]
    files += [REPO_ROOT / module for module in core_modules]
    # A method package's public half too; its ``slime`` subpackage is the
    # backend half, imported by the training driver and workers only.
    files += [
        path
        for method in METHOD_PACKAGES
        for path in sorted((REPO_ROOT / "recipes" / method).rglob("*.py"))
        if "slime" not in path.relative_to(REPO_ROOT / "recipes" / method).parts[:-1]
    ]
    files += [REPO_ROOT / "reef" / "__init__.py"]
    assert len(files) > 30, "core package sweep looks wrong"

    offenders: list[str] = []
    for path in files:
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _imported_modules(tree, package=package, module_scope_only=True)
        for prefix in ("reef.train.slime", "reef.train.slime_backend"):
            offenders.extend(f"{path.relative_to(REPO_ROOT)}: {target}" for target in _imports_of(imported, prefix))
    assert offenders == []


def test_importing_reef_does_not_load_cookbook_or_slime_packages() -> None:
    _assert_isolated_import(
        "import sys; import reef; "
        "assert not [m for m in sys.modules if m == 'recipes' or m.startswith('recipes.')], "
        "[m for m in sys.modules if m == 'recipes' or m.startswith('recipes.')]; "
        "assert not [m for m in sys.modules if m.startswith('reef.train.slime_backend')], "
        "[m for m in sys.modules if m.startswith('reef.train.slime_backend')]"
    )


def test_scenario_factory_imports_without_a_cycle() -> None:
    # The factory bridges the scenario domain and the recipe registry; a fresh
    # interpreter import in either direction must not deadlock on a cycle.
    _assert_isolated_import("from reef.scenario.factory import ScenarioFactory; assert ScenarioFactory")
    _assert_isolated_import("import reef; from reef.scenario.factory import ScenarioFactory")


def test_checkpoint_strategy_is_a_leaf_module() -> None:
    module = importlib.import_module("reef.scenario.checkpoint_strategy")
    tree = ast.parse(inspect.getsource(module))
    reef_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reef")
    ]
    assert reef_imports == []


def test_checkpoint_storage_import_does_not_require_ray() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['ray'] = None; "
        "from reef.train.slime_backend.reef_adapters import RetentionConfig; "
        "from reef.train.slime_backend.reef_adapters.training_job "
        "import durable_io, storage"
    )


def test_slime_step_preparation_resolves_tttd_torch_lazily() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['torch'] = None; "
        "from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step; "
        "assert callable(prepare_slime_step)"
    )


def test_harness_training_backend_does_not_require_gpu_dependencies() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['ray'] = None; sys.modules['torch'] = None; "
        "from reef.train.cordis_backend import CordisBackend; "
        "assert CordisBackend"
    )


def test_slime_bridge_actor_import_does_not_load_megatron_stack() -> None:
    _assert_isolated_import(
        "import sys, types; "
        "ray = types.ModuleType('ray'); "
        "ray.remote = lambda **kwargs: lambda actor: actor; "
        "sys.modules['ray'] = ray; "
        "from reef.train.slime_backend.reef_adapters.bridge "
        "import TrainBridgeActorImpl; "
        "assert TrainBridgeActorImpl; "
        "assert 'slime.ray.placement_group' not in sys.modules"
    )
