"""Check the pinned Slime policy loss with mismatch diagnostics and IcePop."""

import ast
import subprocess
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit
PATCH = Path(__file__).resolve().parents[2] / "docker/patch/slime_mismatch_metrics.patch"


def test_mismatch_metrics_keep_the_policy_gradient_without_tis(tmp_path):
    slime = pytest.importorskip("slime")
    from slime.utils.ppo_utils import compute_policy_loss, get_pg_loss_type, importance_weights

    source = Path(next(iter(slime.__path__))) / "backends/megatron_utils"
    relative_path = Path("slime/backends/megatron_utils/loss.py")
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes((source / "loss.py").read_bytes())
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=tmp_path, capture_output=True)
    if check.returncode != 0:
        subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)

    def score(logits, **kwargs):
        return None, {"log_probs": [logits.log_softmax(-1)[:, 0]], "entropy": [logits.new_zeros(2)]}

    namespace = {
        "torch": torch,
        "Namespace": Namespace,
        "RolloutBatch": dict,
        "Callable": Callable,
        "Any": Any,
        "mpu": SimpleNamespace(get_context_parallel_world_size=lambda: 1),
        "get_log_probs_and_entropy": score,
        "_maybe_capture_log_probs": lambda batch, log_probs: None,
        "get_rollout_top_p_logprob_kwargs": lambda args, batch: {},
        "compute_policy_loss": compute_policy_loss.__wrapped__,
        "get_pg_loss_type": get_pg_loss_type,
        "importance_weights": importance_weights,
    }
    namespace["load_function"] = lambda path: namespace[path]
    # Execute the native policy loss, IcePop, and mask reducer without GPU imports.
    for path, names in (
        (source / "cp_utils.py", {"get_sum_of_sample_mean"}),
        (target, {"icepop_function", "policy_loss_function"}),
    ):
        definitions = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)

    for diagnostics, use_tis in ((False, False), (True, False), (False, True)):
        args = Namespace(
            advantage_estimator="grpo",
            use_rollout_logprobs=False,
            use_opsm=False,
            eps_clip=0.2,
            eps_clip_high=0.2,
            eps_clip_c=None,
            get_mismatch_metrics=diagnostics,
            use_tis=use_tis,
            use_kl_loss=False,
            custom_tis_function_path="icepop_function",
            calculate_per_token_loss=False,
            entropy_coef=0.0,
            tis_clip_low=0.5,
            tis_clip=2.0,
        )
        batch = {
            "advantages": [torch.ones(2)],
            "log_probs": [torch.full((2,), -torch.log(torch.tensor(2.0)))],
            "rollout_log_probs": [torch.full((2,), -torch.log(torch.tensor(8.0)))],
            "response_lengths": [2],
            "total_lengths": [3],
            "unconcat_tokens": [torch.tensor([0, 0, 0])],
            "loss_masks": [torch.tensor([1, 0])],
            "rollout_mask_sums": [torch.tensor(1)],
        }
        logits = torch.zeros((2, 2), requires_grad=True)
        reducer = namespace["get_sum_of_sample_mean"]([3], [2], batch["loss_masks"])
        loss, metrics = namespace["policy_loss_function"](args, batch, logits, reducer)
        loss.backward()

        expected_gradient = torch.zeros((2, 2)) if use_tis else torch.tensor([[-0.5, 0.5], [0.0, 0.0]])
        torch.testing.assert_close(logits.grad, expected_gradient)
        assert loss.item() == pytest.approx(0 if use_tis else -1)
        if diagnostics or use_tis:
            assert metrics["tis"].item() == pytest.approx(4)
            assert metrics["tis_clipfrac"].item() == pytest.approx(1)
