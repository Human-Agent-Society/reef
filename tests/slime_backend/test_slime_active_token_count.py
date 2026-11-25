"""Slime's per-token loss divisor counts only the tokens with loss.

``docker/Dockerfile.reef`` applies ``docker/patch/slime_active_token_count.patch`` to Slime's
``slime/backends/megatron_utils/loss.py`` and ``slime/observability/train_metric_utils.py``. These tests apply the
patch to copies of the installed files, or read the installed files when the image build already applied it. They
skip when Slime is missing.

Both modules import packages that Reef's CI does not install (Megatron, wandb), so the tests run only the patched
statement and function.

Remove these tests with the patch when the Slime pin has a fix.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

PATCH = Path(__file__).resolve().parents[2] / "docker" / "patch" / "slime_active_token_count.patch"
LOSS = Path("slime/backends/megatron_utils/loss.py")
METRICS = Path("slime/observability/train_metric_utils.py")


@pytest.fixture
def patched_slime(tmp_path) -> Path:
    """A directory with the two patched Slime files at their paths in the Slime package."""
    if importlib.util.find_spec("slime") is None:
        pytest.skip("Slime is not installed")
    site = Path(importlib.util.find_spec("slime").origin).parents[1]
    for path in (LOSS, METRICS):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(site / path, tmp_path / path)
    if "torch.clamp_min(loss_mask.sum(), 1)" in (tmp_path / LOSS).read_text():
        subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    return tmp_path


def function_node(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_rows_without_a_token_with_loss_add_nothing_to_the_divisor(patched_slime):
    loss_function = function_node(patched_slime / LOSS, "loss_function")
    (statement,) = [
        node
        for node in loss_function.body
        if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["num_tokens"]
    ]
    divisor = compile(ast.Expression(statement.value), str(LOSS), "eval")
    removed, live = torch.zeros(4, dtype=torch.int), torch.tensor([0, 1, 1, 0], dtype=torch.int)

    assert eval(divisor, {"torch": torch}, {"batch": {"loss_masks": [removed, live, removed]}}) == 2
    assert eval(divisor, {"torch": torch}, {"batch": {"loss_masks": [removed, removed]}}) == 0


def test_a_step_without_a_token_with_loss_reports_zero_metrics(patched_slime):
    module = ast.Module([function_node(patched_slime / METRICS, "reduce_train_step_metrics")], type_ignores=[])
    namespace = {"dist": types.SimpleNamespace(all_reduce=lambda values, group: None)}
    exec(compile(module, str(METRICS), "exec"), namespace)
    # values[0] is the step's tokens with loss; the metrics follow it.
    losses = [{"keys": ["pg_loss", "entropy"], "values": torch.zeros(3)}]

    reduced = namespace["reduce_train_step_metrics"](
        losses, calculate_per_token_loss=True, step_global_batch_size=2, cp_size=1, dp_with_cp_group=None
    )

    assert reduced == {"pg_loss": 0.0, "entropy": 0.0}
