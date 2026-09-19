import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_policy() -> ModuleType:
    script = Path(__file__).parents[2] / ".github" / "scripts" / "check_python_design.py"
    spec = importlib.util.spec_from_file_location("check_python_design", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_design_policy_rejects_dynamic_shortcuts() -> None:
    policy = _load_policy()
    source = """
from typing import TYPE_CHECKING
from collections.abc import Callable as Callback

if TYPE_CHECKING:
    from package import Hidden

Handler = Callback[[str], str]

class Service:
    callback: Handler

    def __init__(self, load: Handler):
        self.load = load

def compose(first: Handler, second: Handler):
    return first(second("value"))
"""

    findings = policy.inspect_source(source, "example.py")

    assert {finding.code for finding in findings} == {"PYD001", "PYD002", "PYD003", "PYD004"}


def test_design_policy_accepts_explicit_inheritance() -> None:
    policy = _load_policy()
    source = """
from abc import ABC, abstractmethod

class Loader(ABC):
    @abstractmethod
    def load(self, value: str) -> str: ...

class FileLoader(Loader):
    def load(self, value: str) -> str:
        return value

class Service:
    def __init__(self, loader: Loader):
        self.loader = loader
"""

    assert policy.inspect_source(source, "example.py") == []


@pytest.mark.parametrize(
    "source",
    [
        "from typing import Protocol",
        "from typing import Protocol as Interface",
        "from typing_extensions import Protocol as Interface",
        "import typing\nclass Loader(typing.Protocol): pass",
        "import typing as t\nclass Loader(t.Protocol): pass",
        "import typing_extensions as t\nclass Loader(t.Protocol): pass",
        "import typing\nalias = typing\nclass Loader(alias.Protocol): pass",
        "import typing_extensions as t\nfirst = t\nsecond = first\nclass Loader(second.Protocol): pass",
        "import typing\nalias: object = typing\nchecked = alias.runtime_checkable",
        "import typing\nfirst = second = typing\nclass Loader(second.Protocol): pass",
        "def load():\n    from typing import Protocol as Interface",
        "from typing import runtime_checkable as checked",
        "import typing_extensions as t\nchecked = t.runtime_checkable",
        "from typing import *\nclass Loader(Protocol): pass",
    ],
)
def test_design_policy_rejects_protocol_imports_and_aliases(source: str) -> None:
    policy = _load_policy()
    findings = policy.inspect_source(source, "example.py", check_callables=False)
    assert [finding.code for finding in findings] == ["PYD005"]
    assert findings[0].line >= 1


def test_design_policy_ignores_protocol_text_and_unrelated_classes() -> None:
    policy = _load_policy()
    source = """
# from typing import Protocol
fixture = "from typing_extensions import Protocol"
class TransportProtocol:
    name = "runtime_checkable"
"""
    assert policy.inspect_source(source, "example.py") == []


@pytest.mark.parametrize(
    "relative",
    [
        "reef/example.py",
        "recipes/demo/examples/main.py",
        "tests/test_example.py",
        "tutorials/demo/main.py",
        "docker/start.py",
        "docs/conf.py",
        ".github/scripts/check_example.py",
        "new_package/example.py",
        "setup.py",
    ],
)
def test_protocol_ban_cannot_be_baselined(tmp_path, monkeypatch, capsys, relative: str) -> None:
    policy = _load_policy()
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from typing_extensions import Protocol as Interface\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(f"PYD005|{relative}|<module>|Protocol\n")
    roots = tuple(tmp_path / name for name in ("reef", "recipes", "tests"))
    monkeypatch.setattr(policy, "ROOT", tmp_path)
    monkeypatch.setattr(policy, "BASELINE", baseline)
    monkeypatch.setattr(policy, "DESIGN_ROOTS", roots)
    monkeypatch.setattr(policy, "REQUIRED_ROOTS", roots)

    assert policy.main() == 1
    output = capsys.readouterr().out
    assert f"{relative}:1: PYD005" in output
    assert "Stale Python design baseline entries" in output


@pytest.mark.parametrize(
    "relative",
    [
        "third_party/library/example.py",
        ".venv/lib/example.py",
        ".venv-extra/lib/example.py",
        "docs/site/node_modules/package/example.py",
        "build/lib/example.py",
        "docs/_build/example.py",
        "dist/example.py",
        "recipes/skillclaw/harbor/environment/workspace/example.py",
        "recipes/skillclaw/harbor/tests/grade.py",
        "recipes/tttd/examples/tttd/harbor/circle_packing_10/environment/score.py",
        "recipes/tttd/examples/tttd/results/run/best_solution.py",
        "recipes/meta_harness/results/reef_harness.py",
        "tests/reef_service/data/harness_goldens/native/example.py",
    ],
)
def test_protocol_discovery_preserves_dependency_and_fixture_boundaries(tmp_path, monkeypatch, relative: str) -> None:
    policy = _load_policy()
    excluded = tmp_path / relative
    excluded.parent.mkdir(parents=True)
    excluded.write_text("from typing import Protocol\n")
    owned = tmp_path / "tutorials" / "owned.py"
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.write_text("from abc import ABC\n")
    monkeypatch.setattr(policy, "ROOT", tmp_path)

    assert list(policy._protocol_python_files()) == [owned]


def test_protocol_discovery_prunes_custom_virtual_environment(tmp_path, monkeypatch) -> None:
    policy = _load_policy()
    environment = tmp_path / "custom-python"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("home = /python\n")
    (environment / "package.py").write_text("from typing import Protocol\n")
    monkeypatch.setattr(policy, "ROOT", tmp_path)

    assert list(policy._protocol_python_files()) == []


@pytest.mark.parametrize(
    "cached_path",
    [
        ".uv-cache/archive-v0/package/python_discovery/_compat.py",
        ".ci-uv/lib/python3.12/site-packages/uv/_find_uv.py",
    ],
)
def test_protocol_ban_skips_ci_uv_dependencies_but_rejects_owned_source(
    tmp_path, monkeypatch, capsys, cached_path: str
) -> None:
    policy = _load_policy()
    dependency = tmp_path / cached_path
    dependency.parent.mkdir(parents=True)
    dependency.write_text("from typing import Protocol\n")
    owned = tmp_path / "reef" / "owned.py"
    owned.parent.mkdir()
    owned.write_text("from typing import Protocol\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    roots = tuple(tmp_path / name for name in ("reef", "recipes", "tests"))
    monkeypatch.setattr(policy, "ROOT", tmp_path)
    monkeypatch.setattr(policy, "BASELINE", baseline)
    monkeypatch.setattr(policy, "DESIGN_ROOTS", roots)
    monkeypatch.setattr(policy, "REQUIRED_ROOTS", roots)

    assert policy.main() == 1
    output = capsys.readouterr().out
    assert "reef/owned.py:1: PYD005" in output
    assert cached_path not in output

    owned.write_text("from abc import ABC\n")
    assert policy.main() == 0


@pytest.mark.parametrize("relative", ["tutorials/example.py", ".github/scripts/example.py", "setup.py"])
def test_protocol_scope_does_not_expand_legacy_design_rules(tmp_path, monkeypatch, relative: str) -> None:
    policy = _load_policy()
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from typing import TYPE_CHECKING, Callable\n"
        "class Service:\n"
        "    def __init__(self, load: Callable[[], str]):\n"
        "        self.load = load\n"
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    roots = tuple(tmp_path / name for name in ("reef", "recipes", "tests"))
    monkeypatch.setattr(policy, "ROOT", tmp_path)
    monkeypatch.setattr(policy, "BASELINE", baseline)
    monkeypatch.setattr(policy, "DESIGN_ROOTS", roots)
    monkeypatch.setattr(policy, "REQUIRED_ROOTS", roots)

    assert policy.main() == 0
