import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_policy() -> ModuleType:
    script = Path(__file__).parents[2] / ".github" / "scripts" / "check_python_statements.py"
    spec = importlib.util.spec_from_file_location("check_python_statements", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_statement_policy_rejects_assert_and_delete_statements() -> None:
    policy = _load_policy()

    findings = policy.inspect_source(
        """
assert ready
del items[key]
""",
        "reef/example.py",
    )

    assert [finding.statement for finding in findings] == ["assert", "del"]


def test_statement_policy_ignores_words_in_comments_and_strings() -> None:
    policy = _load_policy()

    findings = policy.inspect_source(
        """
# assert ready
message = "del items[key]"
""",
        "reef/example.py",
    )

    assert findings == []
