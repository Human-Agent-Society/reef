from __future__ import annotations

from types import SimpleNamespace

import pytest

from recipes.ppo_rlhf.objective import PpoRlhfObjective
from recipes.ppo_rlhf.slime import PpoRlhfAlgorithm


@pytest.mark.unit
def test_standard_family_configures_stock_ppo_advantage_path() -> None:
    algorithm = PpoRlhfAlgorithm()
    args = SimpleNamespace(
        loss_type="policy_loss",
        use_rollout_logprobs=True,
        use_critic=True,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=False,
    )

    algorithm.configure_backend_args(args)
    algorithm.validate_backend_args(args)

    assert args.compute_advantages_and_returns is True
    assert args.advantage_estimator == "ppo"
    assert algorithm.advantages == "forbidden"
    assert algorithm.allows_slime_advantage_computation is True
    assert PpoRlhfObjective.loss_family == "ppo_rlhf_reference_reward"


@pytest.mark.unit
def test_standard_family_rejects_missing_critic() -> None:
    algorithm = PpoRlhfAlgorithm()
    args = SimpleNamespace(
        loss_type="policy_loss",
        use_rollout_logprobs=True,
        use_critic=False,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=False,
    )
    with pytest.raises(RuntimeError, match="value model"):
        algorithm.validate_backend_args(args)
