"""LaTeX powers must agree in the harness and each isolated verifier."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "recipes/sao/examples/imo_answerbench"
GRADERS = [ROOT / "harness/grader.py", *sorted(ROOT.glob("harbor/*/tests/grade.py"))]


@pytest.mark.parametrize("path", GRADERS, ids=lambda path: str(path.relative_to(ROOT)))
def test_latex_powers_are_not_xor(path):
    spec = importlib.util.spec_from_file_location("imo_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.answers_equal("8", "2^3")
    assert module.answers_equal("8", "2^{3}")
    assert module.answers_equal("0.25", "2^{-2}")
    assert module.answers_equal("0.125", r"\frac{1}{2^3}")
    assert not module.answers_equal("1", "2^3")
