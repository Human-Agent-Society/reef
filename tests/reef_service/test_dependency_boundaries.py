from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from graphlib import CycleError, TopologicalSorter
from itertools import pairwise
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


def test_release_id_chain_does_not_depend_on_scenario_or_trainer() -> None:
    module = importlib.import_module("reef.artifact.release_chain")
    tree = ast.parse(inspect.getsource(module))

    imported = _imported_modules(tree, package="reef.artifact")
    assert _imports_of(imported, "reef.scenario") == []
    assert _imports_of(imported, "reef.train") == []


def test_record_contracts_and_retention_policy_do_not_import_database_adapters() -> None:
    module = importlib.import_module("reef.storage.records")
    tree = ast.parse(inspect.getsource(module))

    imported = _imported_modules(tree, package="reef.storage")
    for dependency in (
        "reef.scenario",
        "reef.train",
        "reef.storage",
        "reef.artifact",
        "sqlite3",
        "sqlalchemy",
        "alembic",
    ):
        assert _imports_of(imported, dependency) == []


def test_importing_record_contracts_does_not_load_database_adapters() -> None:
    _assert_isolated_import(
        "import sys; from reef.storage.records import RecordStore; "
        "from reef.storage.scenario import ScenarioStorage, ScenarioStore; "
        "assert not [name for name in sys.modules if name.startswith("
        "('sqlalchemy', 'psycopg', 'sqlite3', 'reef.storage.sqlite', "
        "'reef.storage.postgres', 'reef.storage.sql_records'))]; "
        "from reef import SQLiteRecordStore, PostgresRecordStore; "
        "from reef.storage.sqlite import SQLiteRecordStore as SQLiteImplementation; "
        "from reef.storage.postgres import PostgresRecordStore as PostgresImplementation; "
        "from reef.storage.sqlite import SQLiteScenarioStorage; "
        "from reef.storage.postgres import PostgresScenarioStorage; "
        "assert issubclass(SQLiteScenarioStorage, ScenarioStorage); "
        "assert issubclass(PostgresScenarioStorage, ScenarioStorage); "
        "assert SQLiteRecordStore is SQLiteImplementation; "
        "assert PostgresRecordStore is PostgresImplementation"
    )


def test_sql_record_implementations_do_not_depend_on_scenario_or_training() -> None:
    for module_name in ("reef.storage.sql_records", "reef.storage.sqlite"):
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))

        imported = _imported_modules(tree, package="reef.storage")
        assert _imports_of(imported, "reef.scenario") == [], module_name
        assert _imports_of(imported, "reef.train") == [], module_name


def test_shared_sql_record_operations_do_not_select_sqlite() -> None:
    module = importlib.import_module("reef.storage.sql_records")
    imported = _imported_modules(ast.parse(inspect.getsource(module)), package="reef.storage")
    for dependency in ("reef.storage.sqlite", "sqlite3", "sqlalchemy.dialects.sqlite"):
        assert _imports_of(imported, dependency) == []


def test_dispatcher_does_not_select_a_record_backend() -> None:
    module = importlib.import_module("reef.dispatcher")
    imported = _imported_modules(ast.parse(inspect.getsource(module)), package="reef")
    assert all(
        any(
            target == contract or target.startswith(contract + ".")
            for contract in ("reef.storage.records", "reef.storage.scenario")
        )
        for target in _imports_of(imported, "reef.storage")
    )
    for dependency in ("sqlite3", "sqlalchemy", "psycopg"):
        assert _imports_of(imported, dependency) == []

    for constructor in (module.Dispatcher, module.build_default_dispatcher):
        parameter = inspect.signature(constructor).parameters["scenario_storage"]
        assert parameter.default is inspect.Parameter.empty


def test_http_app_requires_dispatcher_without_selecting_storage() -> None:
    module = importlib.import_module("reef.service.app")
    imported = _imported_modules(ast.parse(inspect.getsource(module)), package="reef.service")
    assert all(
        target == "reef.storage.records" or target.startswith("reef.storage.records.")
        for target in _imports_of(imported, "reef.storage")
    )
    assert _imports_of(imported, "reef.dispatcher.build_default_dispatcher") == []
    assert _imports_of(imported, "reef.service.assembly") == []
    assert inspect.signature(module.create_app).parameters["dispatcher"].default is inspect.Parameter.empty


def test_scenario_storage_contract_and_state_have_only_domain_dependencies() -> None:
    allowed_domains = {
        "reef.storage.commits": ("reef.core",),
        "reef.storage.scenario": ("reef.core", "reef.storage.records", "reef.storage.commits"),
    }
    for module_name, allowed in allowed_domains.items():
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
        imported = _imported_modules(tree, package=module_name.rpartition(".")[0])
        for dependency in imported:
            assert dependency.partition(".")[0] in sys.stdlib_module_names or any(
                dependency == domain or dependency.startswith(domain + ".") for domain in allowed
            ), f"{module_name} imports {dependency}"


def test_only_registry_imports_private_model_file_operations() -> None:
    for path in sorted((REPO_ROOT / "reef/scenario").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        # The registry composes fixed local model files directly. Record and
        # commit backends still enter through the injected ScenarioStorage.
        storage_imports = set(_imports_of(imported, "reef.storage")) - set(
            _imports_of(imported, "reef.storage.records")
            + _imports_of(imported, "reef.storage.scenario")
            + _imports_of(imported, "reef.storage.commits")
        )
        if path.name == "registry.py":
            assert all(
                target == "reef.storage.model_config" or target.startswith("reef.storage.model_config.")
                for target in storage_imports
            ), str(path.relative_to(REPO_ROOT))
        else:
            assert storage_imports == set(), str(path.relative_to(REPO_ROOT))
        for dependency in ("sqlite3", "sqlalchemy", "alembic"):
            assert _imports_of(imported, dependency) == [], str(path.relative_to(REPO_ROOT))


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
        "reef/storage",
        "reef/artifact",
        "reef/recipe",
        "reef/runtime",
        "reef/surface",
    )
    core_modules = ["reef/dispatcher.py"]
    # The explicit Slime process entrypoint assembles that integration; it is
    # never imported by the HTTP app or another backend-neutral package.
    files = [
        path
        for package in core_packages
        for path in sorted((REPO_ROOT / package).rglob("*.py"))
        if path != REPO_ROOT / "reef/service/slime_driver.py"
    ]
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
        "import sys; import reef; from reef.core.version import __version__; "
        "assert reef.__version__ == __version__; "
        "assert not [m for m in sys.modules if m == 'recipes' or m.startswith('recipes.')], "
        "[m for m in sys.modules if m == 'recipes' or m.startswith('recipes.')]; "
        "assert not [m for m in sys.modules if m.startswith('reef.train.slime_backend')], "
        "[m for m in sys.modules if m.startswith('reef.train.slime_backend')]"
    )


def test_scenario_factory_imports_without_a_cycle() -> None:
    # The factory bridges the scenario aggregate and the deployment recipe; a fresh
    # interpreter import in either direction must not deadlock on a cycle.
    _assert_isolated_import("from reef.scenario.factory import ScenarioFactory; assert ScenarioFactory")
    _assert_isolated_import("import reef; from reef.scenario.factory import ScenarioFactory")


def test_checkpoint_strategy_is_a_leaf_module() -> None:
    module = importlib.import_module("reef.recipe.checkpoint_strategy")
    tree = ast.parse(inspect.getsource(module))
    reef_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reef")
    ]
    assert reef_imports == []


def test_checkpoint_storage_import_does_not_require_ray() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['ray'] = None; "
        "from reef.train.slime_backend.reef_adapters import RetentionConfig; "
        "from reef.runtime.training_job import durable_io; "
        "from reef.train.slime_backend.reef_adapters.training_job import storage"
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


def test_model_config_and_file_operations_do_not_depend_on_scenario() -> None:
    for module_name in ("reef.runtime.model_config", "reef.storage.model_config"):
        module = importlib.import_module(module_name)
        imported = _imported_modules(ast.parse(inspect.getsource(module)), package=module_name.rpartition(".")[0])
        for dependency in ("reef.scenario", "reef.recipe", "reef.train", "reef.storage"):
            assert _imports_of(imported, dependency) == [], module_name


def test_reef_package_dependencies_are_acyclic() -> None:
    # Collapse submodules into their owning top-level Reef package. Walk all
    # imports, including local and relative ones: delaying an import cannot
    # turn a reversed dependency into a valid boundary.
    graph: dict[str, set[str]] = {}
    sources: dict[tuple[str, str], list[str]] = {}
    root = REPO_ROOT / "reef"
    owners = {
        path.stem if path.is_file() else path.name for path in root.iterdir() if path.is_dir() or path.suffix == ".py"
    }
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        owner = relative.parts[0].removesuffix(".py")
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        graph.setdefault(owner, set())
        for target in _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package):
            if target == "reef":
                dependency = "__init__"
            elif target.startswith("reef."):
                dependency = target.split(".")[1]
                if dependency not in owners:
                    # Imports from the public facade depend on that facade,
                    # even when the selected name is a class or constant.
                    dependency = "__init__"
            else:
                continue
            if dependency != owner:
                graph[owner].add(dependency)
                sources.setdefault((owner, dependency), []).append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        cycle = exc.args[1]
        details = [source for left, right in pairwise(cycle) for source in sources.get((right, left), [])]
        raise AssertionError(f"Reef package dependency cycle: {cycle}\n" + "\n".join(details)) from exc


def test_storage_and_core_only_depend_on_lower_layers() -> None:
    allowed = {"core": ("reef.core",), "storage": ("reef.core", "reef.storage")}
    for owner, prefixes in allowed.items():
        for path in sorted((REPO_ROOT / "reef" / owner).rglob("*.py")):
            package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
            for target in _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package):
                if target == "reef" or target.startswith("reef."):
                    assert any(
                        target == prefix or target.startswith(prefix + ".") for prefix in prefixes
                    ), f"{path.relative_to(REPO_ROOT)} imports {target}"


def test_package_scan_includes_relative_local_and_facade_imports() -> None:
    tree = ast.parse("from ..storage import commits\ndef load():\n    import reef\n    from reef import Scenario\n")
    assert _imported_modules(tree, package="reef.scenario") == ["reef.storage.commits", "reef", "reef.Scenario"]


def test_training_publication_import_requires_no_model_framework() -> None:
    _assert_isolated_import(
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron'))); "
        "from reef.runtime.training_job.publication import TrainingPublication, WeightPublisher; "
        "from reef.runtime.training_job.execution import TrainingExecution, TrainingJobBackend"
    )
