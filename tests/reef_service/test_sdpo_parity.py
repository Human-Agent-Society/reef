"""CPU numerical checks of SDPO weighting through the shared worker loss."""

from __future__ import annotations

import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sdpo.slime import SdpoSettings
from recipes.sdpo.slime.objective import sdpo_loss
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lengths", "active", "expected_loss", "expected_gradient"),
    [
        ([1, 3], [True, True], 2.0, [0.5, 1 / 6, 1 / 6, 1 / 6]),
        ([1] * 8, [True, *[False] * 7], 0.125, [0.125, *[0.0] * 7]),
        ([1, 3], [False, False], 0.0, [0.0] * 4),
    ],
)
def test_sdpo_matches_reference_response_weighting(
    monkeypatch: pytest.MonkeyPatch,
    lengths: list[int],
    active: list[bool],
    expected_loss: float,
    expected_gradient: list[float],
) -> None:
    # The pinned author uses one response per microbatch, then averages those
    # token means over the minibatch, including inactive rows as zero.
    from reef.train.slime_backend.distill import objective as distill_objective

    family = resolve_loss_family("sdpo")
    args = Namespace(
        loss_type="custom_loss",
        use_rollout_logprobs=True,
        num_steps_per_rollout=1,
        calculate_per_token_loss=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    family.apply_driver_options(args, SdpoSettings(importance_sampling_cap=0))
    family.validate_backend_args(args)
    samples = [
        [f"i{index}", [9, *[2] * length], [1] * length, [-0.5] * length, 0.0, [8, *[2] * length], float(enabled)]
        for index, (length, enabled) in enumerate(zip(lengths, active, strict=True))
    ]
    data = to_slime_rollout_data({"samples": samples, "rollout_ids": list(range(len(samples))), "loss": "sdpo"})
    masks = [torch.tensor(mask) for mask in data["loss_masks"]]
    values = torch.tensor([1.0, *[3.0] * (sum(lengths) - 1)], requires_grad=True)

    def reduce_samples(tokens: torch.Tensor) -> torch.Tensor:
        return sum(
            (
                (part * mask).sum() / mask.sum().clamp_min(1)
                for part, mask in zip(tokens.split(lengths), masks, strict=True)
            ),
            tokens.new_zeros(()),
        )

    modules = {
        "megatron.core": SimpleNamespace(mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1)),
        "slime.backends.megatron_utils.cp_utils": SimpleNamespace(get_sum_of_sample_mean=None),
        "slime.backends.megatron_utils.loss": SimpleNamespace(
            get_log_probs_and_entropy=None, get_responses=lambda *args, **kwargs: []
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(distill_objective, "student_topk_tail_divergences", lambda *args: list(values.split(lengths)))
    batch = {
        "total_lengths": [len(tokens) for tokens in data["tokens"]],
        "response_lengths": lengths,
        "unconcat_tokens": data["tokens"],
        "loss_masks": masks,
        "distill_sample_weights": data["distill_sample_weights"],
    }
    loss_sum, metrics = sdpo_loss(args, batch, values, reduce_samples)
    loss = loss_sum / len(samples)  # Slime's sequence-mode global-batch normalization.
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert values.grad.tolist() == pytest.approx(expected_gradient)

    assert metrics["distill_divergence"].item() == pytest.approx(
        sum(part.mean().item() for part in values.detach().split(lengths))
    )
    assert metrics["distill_sample_weight"].item() == pytest.approx(sum(active))
