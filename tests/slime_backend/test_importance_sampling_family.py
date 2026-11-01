"""The ``importance_sampling`` loss family on Slime: its spec, and its loss on CPU torch."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.slime_backend.test_score_centering import sao_args

from reef.train.slime_backend.algorithm import PolicyGradientWeight
from reef.train.slime_backend.importance_sampling import ImportanceSamplingAlgorithm
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.score_centering import ROLLOUT_KEYS


@pytest.fixture
def slime_loss(monkeypatch):
    """Slime's log-prob helpers, where a row's log-probs are its logits."""
    torch = pytest.importorskip("torch")
    module_name = "slime.backends.megatron_utils.loss"
    fake = ModuleType(module_name)
    fake.get_log_probs_and_entropy = lambda logits, **_: (torch.empty(0), {"log_probs": [logits.reshape(-1)]})
    fake.get_rollout_top_p_logprob_kwargs = lambda *_: {}
    monkeypatch.setitem(sys.modules, module_name, fake)
    return torch


def batch(torch, **overrides):
    values = {
        "rollout_log_probs": [torch.zeros(2)],
        "advantages": [torch.tensor([2.0, -3.0])],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
    }
    values.update(overrides)
    return values


@pytest.mark.unit
def test_the_loss_truncates_the_ratio_at_tis_clip_on_reefs_advantages(slime_loss) -> None:
    torch = slime_loss
    from reef.train.slime_backend.importance_sampling.objective import importance_sampling_loss

    # p / q is 1 at the first token and 4 at the second, which --tis-clip 2 truncates to 2.
    logits = torch.tensor([0.0, math.log(4.0)], requires_grad=True)
    loss, metrics = importance_sampling_loss(SimpleNamespace(tis_clip=2.0), batch(torch), logits, torch.mean)

    # The mean of -A * min(r, 2) * log p: (0 + 3 * 2 * log 4) / 2.
    assert loss.item() == pytest.approx(3 * math.log(4.0))
    assert metrics["importance_ratio"].item() == pytest.approx(2.5)
    assert metrics["truncated_fraction"].item() == pytest.approx(0.5)
    loss.backward()
    # The gradient on log p is -A * sg[min(r, 2)], reduced by the mean.
    assert logits.grad.tolist() == pytest.approx([-1.0, 3.0])


@pytest.mark.unit
def test_the_loss_trains_only_rows_with_the_samplers_log_probs(slime_loss) -> None:
    torch = slime_loss
    from reef.train.slime_backend.importance_sampling.objective import importance_sampling_loss

    logits = torch.zeros(2, requires_grad=True)
    for missing in ("rollout_log_probs", "advantages"):
        with pytest.raises(ValueError, match="recorded rollout log-probs and Reef's advantages"):
            importance_sampling_loss(
                SimpleNamespace(tis_clip=2.0), batch(torch, **{missing: None}), logits, torch.mean
            )


@pytest.mark.unit
def test_the_family_declares_its_truncation_so_score_centering_can_correct_it() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    spec = resolve_loss_family("importance_sampling")
    assert isinstance(spec, ImportanceSamplingAlgorithm)
    assert (spec.loss_type, spec.advantages, spec.allows_slime_advantage_computation) == (
        "custom_loss",
        "required",
        False,
    )
    assert spec.policy_gradient_weight(SimpleNamespace(tis_clip=1.5)) == PolicyGradientWeight("truncated", upper=1.5)
    for cap in (math.inf, 0.0, math.nan):
        with pytest.raises(RuntimeError, match="--tis-clip, which must be finite and > 0"):
            spec.validate_specific_args(SimpleNamespace(tis_clip=cap), "the recipe")

    args = sao_args(loss_family="importance_sampling", tis_clip=2.0)
    configure_reef_loss_args(args)
    assert set(ROLLOUT_KEYS) <= set(args.custom_rollout_data_keys)
