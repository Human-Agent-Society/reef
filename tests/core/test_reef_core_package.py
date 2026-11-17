from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
from pathlib import Path

import pytest

import reef
from reef.core import AgentRecord, RequestType
from reef.core.errors import ConfigError, DeployConfigError, ReefError, RuntimeConfigError
from reef.core.plugins import PluginConfigError
from reef.recipe.errors import RecipeConfigError
from reef.storage.record_items import ItemRecordStore
from reef.storage.records import RecordConflictError, RecordStore


@pytest.mark.unit
def test_core_package_exports_protocol_and_record_types() -> None:
    assert tuple(RequestType) == (RequestType.INFERENCE, RequestType.REPORT)
    assert AgentRecord.__module__ == "reef.core.records_types"
    assert RecordStore.__module__ == "reef.storage.records"
    assert inspect.isabstract(RecordStore)
    assert issubclass(ItemRecordStore, RecordStore)
    assert all(symbol is not None for symbol in (RecordConflictError,))


@pytest.mark.unit
@pytest.mark.parametrize("module", ["reef.headers", "reef.runtime", "reef.wire"])
def test_non_public_modules_are_not_shipped(module: str) -> None:
    assert importlib.util.find_spec(module) is None


@pytest.mark.unit
def test_every_exception_class_derives_from_reef_error_and_ends_in_error() -> None:
    builtin_errors = {
        name for name, value in vars(builtins).items() if isinstance(value, type) and issubclass(value, Exception)
    }
    bases_of: dict[str, set[str]] = {}
    for path in Path(reef.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases_of.setdefault(node.name, set()).update(ast.unparse(base).split(".")[-1] for base in node.bases)

    def ancestors(name: str) -> set[str]:
        found = set(bases_of.get(name, ()))
        for base in tuple(found):
            found |= ancestors(base)
        return found

    exception_classes = {name for name in bases_of if ancestors(name) & builtin_errors}
    assert "ReefError" in exception_classes
    assert {name for name in exception_classes if not name.endswith("Error")} == set()
    assert {name for name in exception_classes - {"ReefError"} if "ReefError" not in ancestors(name)} == set()
    # A ``RuntimeError`` is never the caller's fault, so the HTTP layer must not answer it as a refused request.
    runtime_errors = {name for name in exception_classes if "RuntimeError" in ancestors(name)}
    assert {name for name in runtime_errors if "InternalError" not in ancestors(name)} == set()


@pytest.mark.unit
def test_every_configuration_error_is_a_config_error_and_a_value_error() -> None:
    assert issubclass(ConfigError, ReefError) and issubclass(ConfigError, ValueError)
    for error_type in (DeployConfigError, RuntimeConfigError, RecipeConfigError, PluginConfigError):
        assert issubclass(error_type, ConfigError)
