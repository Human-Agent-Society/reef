"""Exercise the pinned Slime OPSM policy loss and step metric on CPU."""

import ast
import math
import subprocess
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit
PATCH = Path(__file__).resolve().parents[2] / "docker/patch/slime_opsm_clip_fraction.patch"


@pytest.fixture
def slime_opsm_loss(tmp_path):
    slime = pytest.importorskip("slime")
    from slime.utils.ppo_utils import compute_opsm_mask, compute_policy_loss, get_pg_loss_type

    source = Path(next(iter(slime.__path__)))
    target = tmp_path / "slime/backends/megatron_utils/loss.py"
    target.parent.mkdir(parents=True)
    target.write_bytes((source / "backends/megatron_utils/loss.py").read_bytes())
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=tmp_path, capture_output=True)
    if check.returncode != 0:
        subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)

    parallel = SimpleNamespace(size=1, rank=0)
    parallel.get_context_parallel_world_size = lambda: parallel.size
    parallel.get_context_parallel_rank = lambda: parallel.rank
    parallel.get_data_parallel_world_size = lambda **kwargs: parallel.size

    def score(logits, *, response_lengths, total_lengths, **kwargs):
        lengths = [
            namespace["slice_log_prob_with_cp"](torch.arange(response), total, response).numel()
            for total, response in zip(total_lengths, response_lengths, strict=True)
        ]
        return None, {
            "log_probs": list(logits.log_softmax(-1)[:, 0].split(lengths)),
            "entropy": [logits.new_zeros(length) for length in lengths],
        }

    def gather_constant_response(values, total_length, response_length):
        # The fixture uses constant full rows. Rebuild them without a real collective.
        return values.new_full((response_length,), -math.log(2) if values.requires_grad else -0.56)

    namespace = {
        "torch": torch,
        "Namespace": Namespace,
        "RolloutBatch": dict,
        "Callable": Callable,
        "mpu": parallel,
        "dist": SimpleNamespace(all_reduce=lambda *args, **kwargs: None),
        "all_gather_with_cp": gather_constant_response,
        "_maybe_capture_log_probs": lambda *args: None,
        "get_rollout_top_p_logprob_kwargs": lambda *args: {},
        "compute_opsm_mask": compute_opsm_mask,
        "compute_policy_loss": compute_policy_loss.__wrapped__,
        "get_pg_loss_type": get_pg_loss_type,
        "get_log_probs_and_entropy": score,
    }
    # Keep the real loss and both reducers without importing the GPU runtime.
    for path, names in (
        (
            source / "backends/megatron_utils/cp_utils.py",
            {"get_sum_of_sample_mean", "get_logits_and_tokens_offset_with_cp", "slice_log_prob_with_cp"},
        ),
        (source / "observability/train_metric_utils.py", {"reduce_train_step_metrics"}),
        (target, {"loss_function", "policy_loss_function"}),
    ):
        definitions = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(("per_token", "expected"), [(False, 0.5), (True, 0.25)])
@pytest.mark.parametrize("masked_context", [0, 128])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_opsm_clip_fraction_counts_only_active_rejected_tokens(
    slime_opsm_loss, per_token, expected, masked_context, cp_size
):
    args = Namespace(
        loss_type="policy_loss",
        advantage_estimator="grpo",
        use_rollout_logprobs=False,
        use_opsm=True,
        opsm_delta=0.1,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=False,
        use_kl_loss=False,
        calculate_per_token_loss=per_token,
        entropy_coef=0.0,
        allgather_cp=False,
        recompute_loss_function=False,
    )
    response_lengths = [1 + masked_context, 3 + masked_context]
    masks = [torch.tensor([1] * active + [0] * masked_context) for active in (1, 3)]
    batch = {
        "advantages": [
            torch.full((length,), advantage) for length, advantage in zip(response_lengths, (-1.0, 1.0), strict=True)
        ],
        "log_probs": [torch.full((length,), -0.56) for length in response_lengths],
        "response_lengths": response_lengths,
        "total_lengths": [length + 1 for length in response_lengths],
        "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in response_lengths],
        "loss_masks": masks,
        "rollout_mask_sums": [mask.sum() for mask in masks],
    }
    gradients = [torch.zeros((length, 2)) for length in response_lengths]
    metrics_by_rank = []
    slime_opsm_loss["mpu"].size = cp_size
    for rank in range(cp_size):
        slime_opsm_loss["mpu"].rank = rank
        indices = [
            slime_opsm_loss["slice_log_prob_with_cp"](torch.arange(response), total, response)
            for total, response in zip(batch["total_lengths"], response_lengths, strict=True)
        ]
        local_batch = dict(batch)
        for key in ("advantages", "log_probs"):
            local_batch[key] = [row[index] for row, index in zip(batch[key], indices, strict=True)]
        local_lengths = [index.numel() for index in indices]
        logits = torch.zeros((sum(local_lengths), 2), requires_grad=True)
        loss, _, metrics = slime_opsm_loss["loss_function"](
            args, local_batch, num_microbatches=1, step_global_batch_size=2, logits=logits
        )
        loss.backward()
        for full, local, index in zip(gradients, logits.grad.split(local_lengths), indices, strict=True):
            full[index] = local
        metrics_by_rank.append(metrics)

    report = slime_opsm_loss["reduce_train_step_metrics"](
        metrics_by_rank,
        calculate_per_token_loss=per_token,
        step_global_batch_size=2,
        cp_size=cp_size,
        dp_with_cp_group=None,
    )
    rejected, retained = gradients
    assert torch.count_nonzero(rejected) == 0
    assert torch.count_nonzero(retained[:3]) == 6
    assert torch.count_nonzero(retained[3:]) == 0
    assert report["opsm_clipfrac"] == pytest.approx(expected)
