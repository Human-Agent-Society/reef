"""Run the pinned Slime CP gather on two CPU/Gloo ranks with integer masks."""

import sys
import types
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("slime")


def context_parallel_worker(rank, rendezvous):
    import torch.distributed as dist
    import torch.distributed.nn

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_context_parallel_rank=lambda: rank,
        get_context_parallel_group=lambda: dist.group.WORLD,
    )
    sys.modules["megatron.core"] = core
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
    from tests.support.reference_algorithms import sao

    from recipes.sao.slime.objective import get_skip_observation_advantages_and_returns

    values = torch.tensor([0.2, 0.3, 8, 9, 0.4, 0.6, 7, 0.1], dtype=torch.float64)
    mask = [1, 1, 0, 0, 1, 1, 0, 1]
    rewards = [0.0] * 7 + [1.0]
    expected = sao.skip_observation_gae(values.tolist(), rewards, mask, gamma=1, lambd=0.9)
    local_values = slice_log_prob_with_cp(values, 10, 8)
    try:
        for dtype in (torch.int64, torch.int32, torch.bool):
            local_mask = slice_log_prob_with_cp(torch.tensor(mask, dtype=dtype), 10, 8)
            # Reproduce the upstream precondition: its destination requires gradients.
            with pytest.raises(RuntimeError, match=r"floating point|complex dtype"):
                all_gather_with_cp(local_mask, 10, 8)
            advantages, returns = get_skip_observation_advantages_and_returns(
                10, 8, local_values, 1, local_mask, 1, 0.9
            )
            for result, reference in zip((advantages, returns), expected, strict=True):
                expected_slice = slice_log_prob_with_cp(torch.tensor(reference, dtype=torch.float64), 10, 8)
                torch.testing.assert_close(result, expected_slice)
            assert not advantages.requires_grad
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_integer_action_masks_match_single_rank_gae(tmp_path):
    torch.multiprocessing.spawn(context_parallel_worker, args=((tmp_path / "cp-gloo").as_uri(),), nprocs=2)
