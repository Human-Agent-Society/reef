import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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


def test_design_policy_accepts_named_protocol() -> None:
    policy = _load_policy()
    source = """
from typing import Protocol

class Loader(Protocol):
    def load(self, value: str) -> str: ...

class Service:
    def __init__(self, loader: Loader):
        self.loader = loader
"""

    assert policy.inspect_source(source, "example.py") == []
