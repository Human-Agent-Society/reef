"""Run Slime's pinned REINFORCE++ dispatch on CPU with the Reef image patch."""

import ast
import importlib.util
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

PATCH = Path(__file__).resolve().parents[2] / "docker/patch/slime_reinforce_plus_plus_masks.patch"


def test_reinforce_plus_plus_ignores_a_fully_masked_response(tmp_path, monkeypatch):
    pytest.importorskip("slime")
    import slime
    from slime.utils.distributed_utils import distributed_masked_whiten

    source = Path(next(iter(slime.__path__)))
    for relative_path in ("slime/utils/ppo_utils.py", "slime/backends/megatron_utils/loss.py"):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / relative_path, target)
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=tmp_path, capture_output=True)
    if check.returncode != 0:
        subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)

    # Supply single-rank Megatron state and keep Slime's return and whitening math.
    mpu = SimpleNamespace(
        is_pipeline_last_stage=lambda: True,
        get_context_parallel_world_size=lambda: 1,
        get_data_parallel_group=lambda *, with_context_parallel: None,
    )
    megatron_core = ModuleType("megatron.core")
    megatron_core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *, group: None)

    utility_path = tmp_path / "slime/utils/ppo_utils.py"
    spec = importlib.util.spec_from_file_location("slime_ppo_utils", utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    # The rest of the loss module imports the GPU training stack.
    loss_path = tmp_path / "slime/backends/megatron_utils/loss.py"
    tree = ast.parse(loss_path.read_text())
    dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_advantages_and_returns"
    )
    namespace = {
        "Namespace": Namespace,
        "RolloutBatch": dict,
        "torch": torch,
        "mpu": mpu,
        "get_reinforce_plus_plus_returns": utility.get_reinforce_plus_plus_returns,
        "distributed_masked_whiten": distributed_masked_whiten,
    }
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(loss_path), "exec"), namespace)

    args = Namespace(
        advantage_estimator="reinforce_plus_plus",
        custom_advantage_function_path=None,
        gamma=1.0,
        kl_coef=0.0,
        normalize_advantages=True,
        use_opd=False,
        use_rollout_logprobs=False,
    )
    batch = {
        "rewards": [1.0, 0.0, 9.0],
        "response_lengths": [1, 1, 1],
        "total_lengths": [2, 2, 2],
        "log_probs": [torch.zeros(1) for _ in range(3)],
        "loss_masks": [torch.tensor([1]), torch.tensor([1]), torch.tensor([0])],
    }
    namespace["compute_advantages_and_returns"](args, batch)
    torch.testing.assert_close(torch.stack(batch["returns"]), torch.tensor([[1.0], [0.0], [0.0]]))
    torch.testing.assert_close(torch.stack(batch["advantages"]), torch.tensor([[0.70710678], [-0.70710678], [0.0]]))
    assert [mask.tolist() for mask in batch["loss_masks"]] == [[1], [1], [0]]
