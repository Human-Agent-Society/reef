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
    # The pure identity types (SnapshotRef and friends) live in reef.core;
    # reef.artifact builds on them, never the other way around.

    for name in (
        "reef.core",
        "reef.core.records_types",
        "reef.core.snapshot",
        "reef.core.errors",
    ):
        module = importlib.import_module(name)
        tree = ast.parse(inspect.getsource(module))
        imported = _imported_modules(tree, package="reef.core")
        assert _imports_of(imported, "reef.artifact") == [], name


def test_harbor_task_values_and_validation_do_not_depend_on_harness() -> None:
    for module_name in ("reef.core.tasks.harbor", "reef.record2sample.generator.harbor"):
        module = importlib.import_module(module_name)
        imported = _imported_modules(ast.parse(inspect.getsource(module)), package=module_name.rpartition(".")[0])
        assert _imports_of(imported, "reef.harness") == [], module_name


def test_snapshot_chain_does_not_depend_on_scenario_or_trainer() -> None:
    module = importlib.import_module("reef.artifact.snapshot_chain")
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
    ):
        assert _imports_of(imported, dependency) == []


def test_importing_record_contracts_does_not_load_database_adapters() -> None:
    _assert_isolated_import(
        "import sys; from reef.storage.records import RecordStore; "
        "from reef.storage.scenario import ScenarioStorage, ScenarioStore; "
        "assert not [name for name in sys.modules if name.startswith("
        "('sqlalchemy', 'psycopg', 'sqlite3', 'reef.storage.sqlite', "
        "'reef.storage.postgres', 'reef.storage.sql_records'))]; "
        "from reef.storage.sqlite import SQLiteScenarioStorage; "
        "from reef.storage.postgres import PostgresScenarioStorage; "
        "assert issubclass(SQLiteScenarioStorage, ScenarioStorage); "
        "assert issubclass(PostgresScenarioStorage, ScenarioStorage)"
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
    imported = [
        target
        for path in sorted((REPO_ROOT / "reef/dispatcher").rglob("*.py"))
        for target in _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package="reef.dispatcher")
    ]
    assert all(
        any(
            target == contract or target.startswith(contract + ".")
            for contract in (
                "reef.storage.artifact_items",
                "reef.storage.control_lease",
                "reef.storage.coordination",
                "reef.storage.data",
                "reef.storage.data_queue",
                "reef.storage.data_sync",
                "reef.storage.job_data",
                "reef.storage.record_items",
                "reef.storage.records",
                "reef.storage.samples",
                "reef.storage.served_models",
                "reef.storage.scenario",
                "reef.storage.tasks",
                "reef.storage.work_queue",
            )
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
        "reef.storage.tasks": ("reef.core",),
        "reef.storage.segments": ("reef.core",),
        "reef.storage.samples": ("reef.core", "reef.storage.data", "reef.storage.tasks"),
        "reef.storage.item_kinds": ("reef.core", "reef.storage.tasks"),
        "reef.storage.job_data": ("reef.core", "reef.storage.segments"),
        "reef.storage.data": (
            "reef.core",
            "reef.storage.item_kinds",
            "reef.storage.job_data",
            "reef.storage.segments",
            "reef.storage.task_collections",
            "reef.storage.tasks",
        ),
        "reef.storage.task_collections": ("reef.core",),
        "reef.storage.evaluations": ("reef.core",),
        "reef.storage.scenario": (
            "reef.core",
            "reef.storage.data",
            "reef.storage.records",
            "reef.storage.commits",
            "reef.storage.tasks",
            "reef.storage.samples",
            "reef.storage.evaluations",
        ),
    }
    for module_name, allowed in allowed_domains.items():
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))
        imported = _imported_modules(tree, package=module_name.rpartition(".")[0])
        for dependency in imported:
            assert dependency.partition(".")[0] in sys.stdlib_module_names or any(
                dependency == domain or dependency.startswith(domain + ".") for domain in allowed
            ), f"{module_name} imports {dependency}"


def test_scenario_state_enters_only_through_injected_storage() -> None:
    for path in sorted((REPO_ROOT / "reef/scenario").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        # Scenario state and served routes enter through injected storage readers. The data store of the job data
        # directories is injected too: the process gives it in its ``LiveObjects``.
        storage_imports = set(_imports_of(imported, "reef.storage")) - set(
            _imports_of(imported, "reef.storage.records")
            + _imports_of(imported, "reef.storage.scenario")
            + _imports_of(imported, "reef.storage.commits")
            + ["reef.storage.served_models.ServedModels", "reef.storage.data.DataStore"]
        )
        assert storage_imports == set(), str(path.relative_to(REPO_ROOT))
        for dependency in ("sqlite3", "sqlalchemy"):
            assert _imports_of(imported, dependency) == [], str(path.relative_to(REPO_ROOT))


def test_backend_agnostic_core_never_imports_slime_backend_at_module_scope() -> None:
    # The most-repeated layering rule: Reef's backend-agnostic core talks to
    # training through the abstract reef.train surface (types, trainer,
    # processors) and to the concrete runtime only through the runtime
    # registry / named Ray actors — it never imports reef.train.slime or
    # reef.train.slime_backend at module scope. A lazy import inside a
    # function (an adapter reaching for the backend it was configured with)
    # is the allowed escape hatch. A method package's public half imports one
    # name of the backend: ``register_objective``, which records the reference
    # of the method's objective without importing it.
    core_packages = (
        "reef/core",
        "reef/dispatcher",
        "reef/service",
        "reef/scenario",
        "reef/storage",
        "reef/artifact",
        "reef/recipe",
        "reef/execution",
    )
    files = [path for package in core_packages for path in sorted((REPO_ROOT / package).rglob("*.py"))]
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
        allowed = (
            {"reef.train.slime_backend.objectives.register_objective"} if package.startswith("recipes") else set()
        )
        for prefix in ("reef.train.slime", "reef.train.slime_backend"):
            offenders.extend(
                f"{path.relative_to(REPO_ROOT)}: {target}"
                for target in _imports_of(imported, prefix)
                if target not in allowed
            )
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


def test_checkpoint_storage_import_does_not_require_ray() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['ray'] = None; "
        "from reef.train.slime_backend.reef_adapters import RetentionSettings; "
        "from reef.inference import recovery; "
        "from reef.train.slime_backend.reef_adapters.training_job import storage"
    )


def test_slime_job_preparation_resolves_tttd_torch_lazily() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['torch'] = None; "
        "from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job; "
        "assert callable(prepare_slime_job)"
    )


def test_harness_training_backend_does_not_require_gpu_dependencies() -> None:
    _assert_isolated_import(
        "import sys; sys.modules['ray'] = None; sys.modules['torch'] = None; "
        "from reef.train.harness_backend.backend import HarnessBackend; "
        "assert HarnessBackend"
    )


def test_executor_config_and_the_e2b_module_do_not_require_the_e2b_package() -> None:
    # The executor config imports reef.execution.sandbox.e2b for every deployment, and that module defines a
    # subclass of Harbor's E2B environment. A base installation has no e2b package. The modules still import, a
    # local executor is built and the E2B settings are read; only the E2B startup check names the missing extra.
    _assert_isolated_import(
        "import sys\n"
        "sys.modules['e2b'] = None\n"
        "from reef.execution.sandbox.config import REMOTE_EXECUTORS, build_executor, remote_executor\n"
        "from reef.execution.sandbox.e2b import E2BExecutor, KillOnTimeoutE2BEnvironment\n"
        "from reef.execution.sandbox.process import LocalExecutor, SandboxUnavailableError\n"
        "assert REMOTE_EXECUTORS['e2b'] is E2BExecutor and KillOnTimeoutE2BEnvironment.type() == 'e2b'\n"
        "assert isinstance(build_executor({'executor': 'local'}), LocalExecutor)\n"
        "executor = remote_executor('e2b', {'e2b_template': 'custom'}, {'E2B_API_KEY': 'key'})\n"
        "assert isinstance(executor, E2BExecutor) and executor.template == 'custom'\n"
        "try:\n"
        "    executor.preflight()\n"
        "except SandboxUnavailableError as error:\n"
        "    assert 'reef-infra[e2b]' in str(error), error\n"
        "else:\n"
        "    raise AssertionError('the E2B startup check passed without the e2b package')\n"
    )


def test_slime_training_component_import_does_not_load_megatron_stack() -> None:
    _assert_isolated_import(
        "import sys, types; "
        "ray = types.ModuleType('ray'); "
        "ray.remote = lambda **kwargs: lambda actor: actor; "
        "sys.modules['ray'] = ray; "
        "from reef.train.slime_backend.component import SlimeJobRunner, SlimeTrainingComponent; "
        "assert SlimeJobRunner and SlimeTrainingComponent; "
        "assert not [name for name in sys.modules if name.startswith(('megatron', 'torch', 'slime.ray'))]"
    )


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


def test_core_storage_and_sandbox_only_depend_on_lower_layers() -> None:
    allowed = {
        "core": ("reef.core",),
        "storage": ("reef.core", "reef.storage"),
        "execution/sandbox": (
            "reef.core",
            "reef.execution.lifecycle",
            "reef.execution.processes",
            "reef.execution.sandbox",
        ),
    }
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


def test_training_jobs_and_serving_import_require_no_model_framework() -> None:
    _assert_isolated_import(
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron'))); "
        "from reef.train.component import TrainingComponent, TrainingJob; "
        "from reef.inference.serving import AdapterServingRuntime, FullWeightServingRuntime; "
        "from reef.train.jobs import JobLoop; "
        "from reef.train.local_jobs import LocalJobComponent"
    )


def test_inference_recovery_requires_no_model_framework() -> None:
    _assert_isolated_import(
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron'))); "
        "from reef.inference.recovery import InferenceControl"
    )


def test_inference_contracts_import_only_lower_contracts_and_no_backend() -> None:
    # Include imports inside functions: delaying an import does not make a
    # contract depend on an integration or on a module above it.
    allowed = {
        "interfaces.py": (),
        "recovery.py": (),
        "residency.py": ("reef.inference.interfaces",),
        "deployment.py": ("reef.inference.interfaces",),
        "registry.py": ("reef.inference.interfaces",),
        "serving.py": ("reef.inference.interfaces", "reef.inference.residency"),
    }
    for name, prefixes in allowed.items():
        path = REPO_ROOT / "reef/inference" / name
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package="reef.inference")
        for dependency in ("reef.train", "slime", "sglang", "vllm", "megatron"):
            assert _imports_of(imported, dependency) == [], name
        for target in _imports_of(imported, "reef.inference"):
            assert any(target == prefix or target.startswith(prefix + ".") for prefix in prefixes), f"{name}: {target}"


def test_training_contracts_and_executors_never_import_model_backends() -> None:
    paths = [REPO_ROOT / "reef/train/component.py", *sorted((REPO_ROOT / "reef/execution/executor").rglob("*.py"))]
    for path in paths:
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        for dependency in ("reef.inference", "reef.train", "slime", "sglang", "megatron"):
            assert _imports_of(imported, dependency) == [], str(path.relative_to(REPO_ROOT))


def test_inference_backends_never_import_training_implementations() -> None:
    for path in sorted((REPO_ROOT / "reef/inference").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        for dependency in ("reef.train", "slime", "slime_plugins", "megatron"):
            assert _imports_of(imported, dependency) == [], str(path.relative_to(REPO_ROOT))


def test_inference_and_training_contracts_import_without_loading_backends() -> None:
    _assert_isolated_import(
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron', 'reef.inference.sglang'))); "
        "import reef.inference; from reef.inference import registry, serving; from reef.train import component; "
        "assert not [name for name in sys.modules "
        "if name.startswith(('reef.inference.sglang.', 'reef.train.slime_backend'))]"
    )


def test_only_the_training_service_builds_a_training_component() -> None:
    """One place builds a weight component and one a harness component, each only for a process that runs the
    training service. Every other process holds the backend's facts and the jobs queue."""
    builders = ("build_training_component", "from_settings")
    built: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "reef").rglob("*.py")):
        if path == REPO_ROOT / "reef/train/component.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if isinstance(called, ast.Name) and called.id == "build_training_component":
                built.setdefault(str(path.relative_to(REPO_ROOT)), []).append(called.id)
            if (
                isinstance(called, ast.Attribute)
                and called.attr == "from_settings"
                and isinstance(called.value, ast.Name)
                and called.value.id == "HarnessTrainingComponent"
            ):
                built.setdefault(str(path.relative_to(REPO_ROOT)), []).append(called.attr)
    assert built == {"reef/service/assembly.py": list(builders)}
    assembly = ast.parse((REPO_ROOT / "reef/service/assembly.py").read_text(encoding="utf-8"))
    guarded = [
        ast.unparse(node.test)
        for node in ast.walk(assembly)
        if isinstance(node, ast.IfExp)
        and any(name in ast.unparse(node.body) for name in ("build_training_component(", "from_settings("))
    ]
    assert guarded == ["runs_training", "runs_training"]
    # Control's view of a backend holds its facts: it has no component to call.
    from reef.train.job_queue import JobQueueComponent

    assert list(inspect.signature(JobQueueComponent).parameters)[:2] == ["facts", "queue"]


def test_training_backends_never_import_inference_implementations() -> None:
    for path in sorted((REPO_ROOT / "reef/train").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        assert _imports_of(imported, "reef.inference") == [], str(path.relative_to(REPO_ROOT))


def test_the_tinker_pieces_that_training_shares_never_import_inference_or_training() -> None:
    # Training imports reef.tinker, so what Tinker's sampler and the inference handlers share, such as the reasoning
    # split, is in reef.core.
    for path in sorted((REPO_ROOT / "reef/tinker").rglob("*.py")):
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package)
        for dependency in ("reef.inference", "reef.train"):
            assert _imports_of(imported, dependency) == [], str(path.relative_to(REPO_ROOT))


def test_task_generation_contracts_do_not_depend_on_their_consumers() -> None:
    # The generator service owns the request; the generated-task recipe owns the generator contract.
    for module_name, package, consumers in (
        (
            "reef.record2sample.generator.inputs",
            "reef.record2sample.generator",
            ("reef.train", "reef.recipe", "recipes"),
        ),
        ("reef.recipe.generated_tasks.generator", "reef.recipe.generated_tasks", ("reef.train", "recipes")),
    ):
        module = importlib.import_module(module_name)
        imported = _imported_modules(ast.parse(inspect.getsource(module)), package=package)
        for dependency in consumers:
            assert _imports_of(imported, dependency) == [], module_name


def test_execution_package_has_no_legacy_entry_points() -> None:
    for name in ("sandbox", "background_jobs.py", "processes.py"):
        assert not (REPO_ROOT / "reef" / name).exists()
    tree = ast.parse((REPO_ROOT / "reef/execution/__init__.py").read_text(encoding="utf-8"))
    assert _imported_modules(tree, package="reef.execution") == []


def test_execution_jobs_depend_only_on_processes_and_lifecycle() -> None:
    # Keep the shared worker independent of the sandbox subsystem.
    tree = ast.parse((REPO_ROOT / "reef/execution/jobs.py").read_text(encoding="utf-8"))
    targets = _imported_modules(tree, package="reef.execution")
    assert all(
        target.split(".")[0] in sys.stdlib_module_names
        or target.startswith(("reef.execution.processes.", "reef.execution.lifecycle."))
        for target in targets
    )


def test_processes_depend_only_on_the_standard_library_lifecycle_and_the_base_error() -> None:
    tree = ast.parse((REPO_ROOT / "reef/execution/processes.py").read_text(encoding="utf-8"))
    imports = _imported_modules(tree, package="reef.execution")
    assert all(
        target.split(".")[0] in sys.stdlib_module_names
        or target.startswith("reef.execution.lifecycle.")
        or target == "reef.core.errors.InternalError"
        for target in imports
    )
    tree = ast.parse((REPO_ROOT / "reef/execution/lifecycle.py").read_text(encoding="utf-8"))
    imports = _imported_modules(tree, package="reef.execution")
    assert all(target.split(".")[0] in sys.stdlib_module_names for target in imports)


def test_record_to_sample_stage_depends_only_on_contracts_below_training() -> None:
    # The trainer and the recipes import the stage, so it never imports them or the generator service.
    allowed = (
        "reef.core",
        "reef.execution.jobs",
        "reef.pipeline_context",
        "reef.record2sample",
    )
    for path in sorted((REPO_ROOT / "reef" / "record2sample").glob("*.py")):
        for target in _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package="reef.record2sample"):
            if target.startswith("reef.record2sample.generator"):
                raise AssertionError(f"{path.relative_to(REPO_ROOT)} imports the generator service: {target}")
            if target == "reef" or target.startswith("reef."):
                assert any(
                    target == prefix or target.startswith(prefix + ".") for prefix in allowed
                ), f"{path.relative_to(REPO_ROOT)} imports {target}"


def test_task_to_job_stage_depends_only_on_contracts_below_training() -> None:
    # The pipeline and the recipes import the stage, so it never imports them.
    allowed = (
        "reef.core",
        "reef.observability",
        "reef.pipeline_context",
        "reef.record2sample",
        "reef.storage.samples",
        "reef.storage.tasks",
        "reef.sample2job",
    )
    paths = [*sorted((REPO_ROOT / "reef" / "sample2job").rglob("*.py")), REPO_ROOT / "reef" / "pipeline_context.py"]
    for path in paths:
        package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
        for target in _imported_modules(ast.parse(path.read_text(encoding="utf-8")), package=package):
            if target == "reef" or target.startswith("reef."):
                assert any(
                    target == prefix or target.startswith(prefix + ".") for prefix in allowed
                ), f"{path.relative_to(REPO_ROOT)} imports {target}"


#: The methods that built a trainer or a pipeline on a recipe before RFC #1390 moved that work to the scenario
#: factory (``reef.scenario.factory.read_trainer_specs``) and ``reef.train.trainer_builder``.
TRAINER_CONSTRUCTION_METHODS = frozenset({"build", "build_trainers", "build_pipeline", "_build_trainer"})

#: ``(file, class, method)`` of each such method that a later step of RFC #1390 still has to remove; none is left.
KEPT_TRAINER_CONSTRUCTION_METHODS: frozenset[tuple[str, str, str]] = frozenset()


def test_a_recipe_builds_no_trainer_and_no_pipeline() -> None:
    # A recipe is configuration (RFC #1390): the scenario factory reads its values and builds the trainers.
    found: set[tuple[str, str, str]] = set()
    paths = [*(REPO_ROOT / "reef" / "recipe").rglob("*.py"), *(REPO_ROOT / "recipes").rglob("*.py")]
    for path in sorted(paths):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                found.update(
                    (str(path.relative_to(REPO_ROOT)), node.name, item.name)
                    for item in node.body
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name in TRAINER_CONSTRUCTION_METHODS
                )
    assert found == KEPT_TRAINER_CONSTRUCTION_METHODS


#: The methods that loaded a recipe, or decided for the launcher and the training service, on a recipe class
#: before RFC #1390 moved that work to ``reef.recipe.loading`` and ``reef.service.deploy.training``.
RECIPE_LOADING_METHODS = frozenset(
    {
        "from_environment",
        "from_resolved_config",
        "service_config",
        "select_weight_training",
        "resolve_training_component",
    }
)

#: ``(file, class, method)`` of each such method, or method whose name begins with an underscore, that a later step
#: of RFC #1390 still has to remove; none is left.
KEPT_RECIPE_LOADING_METHODS: frozenset[tuple[str, str, str]] = frozenset()


def test_a_recipe_class_does_not_load_itself_and_has_no_underscore_method() -> None:
    # One loader makes the recipe value (RFC #1390): a recipe class declares its configuration. No class of the
    # recipe packages has a loading method. A recipe class, which is a class with a base whose name ends in
    # "Recipe", has no method whose name begins with an underscore either, other than a special method.
    found: set[tuple[str, str, str]] = set()
    paths = [*(REPO_ROOT / "reef" / "recipe").rglob("*.py"), *(REPO_ROOT / "recipes").rglob("*.py")]
    for path in sorted(paths):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "") for base in node.bases]
            is_recipe = node.name == "Recipe" or any(base.endswith("Recipe") for base in bases)
            for item in node.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                special = item.name.startswith("__") and item.name.endswith("__")
                if item.name in RECIPE_LOADING_METHODS or (is_recipe and item.name.startswith("_") and not special):
                    found.add((str(path.relative_to(REPO_ROOT)), node.name, item.name))
    assert found == KEPT_RECIPE_LOADING_METHODS


#: The methods that built the objects of a harness job on a recipe before RFC #1390 moved that work into
#: ``reef.train.harness_backend.component.HarnessTrainingComponent``, and the two hooks through which a recipe
#: class changed its own loading before the loader read declarations alone (``section_defaults``, ``section_field``
#: and ``reader_field``).
HARNESS_CONSTRUCTION_METHODS = frozenset(
    {"build_backend", "build_evaluator", "build_policy", "built_fields", "config_with_defaults"}
)

#: ``(file, class, method)`` of each such method that a later step of RFC #1390 still has to remove; none is left.
KEPT_HARNESS_CONSTRUCTION_METHODS: frozenset[tuple[str, str, str]] = frozenset()

#: The classes of the objects that run a harness job: the component, the backend, the evaluator and the policies.
HARNESS_JOB_CLASSES = frozenset(
    {
        "HarnessTrainingComponent",
        "HarnessBackend",
        "HarnessTaskEvaluator",
        "ComparePolicy",
        "FloorPolicy",
        "NoEvaluationPolicy",
    }
)

#: The names of ``HARNESS_JOB_CLASSES`` that ``reef/recipe/harness.py`` still calls; none is left.
KEPT_HARNESS_JOB_CALLS: frozenset[str] = frozenset()


def test_a_recipe_builds_no_harness_backend_evaluator_or_policy() -> None:
    # A harness recipe is configuration (RFC #1390): the harness training component builds itself and the objects
    # of its jobs from the recipe's settings, so no recipe class defines a method that builds them, and no class
    # of the recipe packages defines a hook that builds fields or rewrites the config for the loader.
    found: set[tuple[str, str, str]] = set()
    paths = [*(REPO_ROOT / "reef" / "recipe").rglob("*.py"), *(REPO_ROOT / "recipes").rglob("*.py")]
    for path in sorted(paths):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                found.update(
                    (str(path.relative_to(REPO_ROOT)), node.name, item.name)
                    for item in node.body
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name in HARNESS_CONSTRUCTION_METHODS
                )
    assert found == KEPT_HARNESS_CONSTRUCTION_METHODS
    # The harness recipe's module calls none of the classes, by a name or as an attribute.
    called: set[str] = set()
    module = ast.parse((REPO_ROOT / "reef" / "recipe" / "harness.py").read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute) else None
            )
            if name in HARNESS_JOB_CLASSES:
                called.add(name)
    assert called == KEPT_HARNESS_JOB_CALLS


#: The types of the objects that run, and of the values that are built from a live runtime: a recipe class has no
#: field and no method that holds or returns one (RFC #1390). A name that ends in ``TrainingComponent`` counts too.
LIVE_OBJECT_TYPES = frozenset(
    {
        "InferenceRuntime",
        "TrainingComponent",
        "DataStore",
        "RecordStore",
        "ServedEndpoint",
        "ModelBinding",
        "ModelBindings",
        "HarnessModels",
    }
)

#: ``(file, class, member)`` of each field or method of a recipe package's class whose annotation names one of
#: ``LIVE_OBJECT_TYPES`` on purpose, with the reason.
KEPT_LIVE_OBJECT_ANNOTATIONS: dict[tuple[str, str, str], str] = {
    ("reef/recipe/harness.py", "HarnessEvolutionRecipe", "models"): (
        "configuration: the parsed evolution.models entries (URL, model name and key of each named model); "
        "none is built from a runtime"
    ),
    ("reef/recipe/harness.py", "NamedModels", "read"): (
        "the config reader of that field: it parses evolution.models entries from the config and the environment"
    ),
    ("recipes/gepa/examples/aime/run.py", "RunService", "binding"): (
        "not a recipe class: the example's own service, which owns a dispatcher; the endpoint that a run reaches it by"
    ),
    ("recipes/gepa/examples/aime/run.py", "RunService", "records"): (
        "not a recipe class: the example's own service, which owns a dispatcher; the records of its scenario"
    ),
}

#: The methods that held, swapped or asked a live object on a recipe before RFC #1390 gave the live objects to
#: ``reef.scenario.live_objects.LiveObjects``, and the three methods that a recipe may only define as a pure
#: function of its configuration.
LIVE_OBJECT_METHODS = frozenset(
    {
        "job_components",
        "with_job_components",
        "with_served_endpoint",
        "harness_served_model",
        "harness_models",
        "build_components",
        "default_start",
        "check_root_start",
    }
)

#: ``(file, class, method)`` of each such method that is kept on purpose, with the reason. Each one reads the
#: recipe's configuration fields alone; ``reef.scenario.live_objects`` adds what a live object knows.
KEPT_LIVE_OBJECT_METHODS: dict[tuple[str, str, str], str] = {
    ("reef/recipe/base.py", "Recipe", "build_components"): (
        "pure configuration: declares no component; a record or tree recipe overrides it with its own components"
    ),
    ("reef/recipe/base.py", "Recipe", "default_start"): (
        "pure configuration: the empty start; a tree recipe overrides it with the seed of its configuration"
    ),
    ("reef/recipe/base.py", "WeightTrainingRecipe", "build_components"): (
        "pure configuration: declares the weights component without kind and layout, which scenario_components "
        "takes from the runtime and the training component; step 6 of RFC #1390 gives both a source in "
        "configuration"
    ),
    ("reef/recipe/harness.py", "HarnessEvolutionRecipe", "build_components"): (
        "pure configuration: the harness component of the adapter that evolution.adapter names"
    ),
    ("reef/recipe/harness.py", "HarnessEvolutionRecipe", "default_start"): (
        "pure configuration: the seed tree that evolution.seed names"
    ),
}


def test_a_recipe_class_holds_and_returns_no_live_object() -> None:
    # A recipe is configuration (RFC #1390): the process holds the runtime, the training components, the stores and
    # the served endpoint in ``reef.scenario.live_objects.LiveObjects``. No class of the recipe packages has a field
    # annotation or a method return annotation that names a live object type, and none defines a method that held
    # or asked one, apart from the kept ones.
    def live_names(annotation: ast.expr | None) -> set[str]:
        names: set[str] = set()
        if annotation is None:
            return names
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        for node in ast.walk(annotation):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name is not None and (name in LIVE_OBJECT_TYPES or name.endswith("TrainingComponent")):
                names.add(name)
        return names

    annotated: set[tuple[str, str, str]] = set()
    methods: set[tuple[str, str, str]] = set()
    paths = [*(REPO_ROOT / "reef" / "recipe").rglob("*.py"), *(REPO_ROOT / "recipes").rglob("*.py")]
    for path in sorted(paths):
        relative = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if live_names(item.annotation):
                        annotated.add((relative, node.name, item.target.id))
                elif isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if live_names(item.returns):
                        annotated.add((relative, node.name, item.name))
                    if item.name in LIVE_OBJECT_METHODS:
                        methods.add((relative, node.name, item.name))
    assert annotated == set(KEPT_LIVE_OBJECT_ANNOTATIONS)
    assert methods == set(KEPT_LIVE_OBJECT_METHODS)


#: The methods that ``CompositeRecipe`` defines. ``__post_init__`` checks the invariants of the mapping.
COMPOSITE_RECIPE_METHODS = frozenset({"__post_init__"})

#: The members that answered for the recipes of a composite before RFC #1390 made the composite a mapping. No
#: recipe class of ``reef/recipe`` defines one: ``reef.recipe.composite.component_recipes`` gives the recipes of a
#: deployment, and the owner of each result asks them.
COMPOSITE_FAN_OUT_MEMBERS = frozenset(
    {
        "harness_adapter",
        "harness_state_dir",
        "scenario_state_dirs",
        "for_record2sample_configs",
        "report_contract",
    }
)


def test_a_composite_recipe_is_a_mapping_of_child_recipes() -> None:
    # A composite computes nothing from its children (RFC #1390): the class defines ``__post_init__`` alone, and
    # no recipe class keeps a member that existed to give a caller the composite's answer.
    fan_out: set[tuple[str, str]] = set()
    composite_methods: set[str] | None = None
    for path in sorted((REPO_ROOT / "reef" / "recipe").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)}
            fan_out.update((node.name, name) for name in methods & COMPOSITE_FAN_OUT_MEMBERS)
            if node.name == "CompositeRecipe":
                composite_methods = methods
    assert composite_methods == COMPOSITE_RECIPE_METHODS
    assert fan_out == set()


def test_only_configuration_writes_a_default_start() -> None:
    # A new scenario's root comes from configuration (RFC #1390): a recipe's own ``default_start`` or the
    # ``default_start`` of a training backend's facts. No training component, and no other class of ``reef``,
    # defines the method, so no running component makes a root.
    recipes = {("reef/recipe/base.py", "Recipe"), ("reef/recipe/harness.py", "HarnessEvolutionRecipe")}
    others: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "reef").rglob("*.py")):
        relative = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            defines = any(isinstance(item, ast.FunctionDef) and item.name == "default_start" for item in node.body)
            if defines and not node.name.endswith("Facts") and (relative, node.name) not in recipes:
                others.append((relative, node.name))
    assert others == []
