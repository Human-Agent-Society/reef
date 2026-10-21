"""The MLX loss families: resolution, settings and the rows they build from reserved samples."""

from __future__ import annotations

import pytest
from tests.support._trajectories import policy_trajectory

mx = pytest.importorskip("mlx.core", reason="the MLX losses need the optional mlx extra")

import recipes.openclawrl
import recipes.tttd  # noqa: F401  (registers the tttd loss family)
from recipes.openclawrl.mlx import (
    OPDNormalization,
    OpenClawRLMLXLoss,
    OpenClawRLMLXSettings,
    opd_gap_statistics,
    opd_one_sample,
    select_hint,
)
from recipes.tttd.mlx import TttdMLXLoss, frozen_base_kl_advantages
from reef.train.mlx_backend.contracts import LossOptions
from reef.train.mlx_backend.losses import resolve_mlx_loss
from reef.train.mlx_backend.training import TrainingRow

pytestmark = pytest.mark.unit


def distillation_sample(*, topk: bool = True, teacher: bool = True):
    extras = {"teacher_cands": ({"hint": "Be terse.", "teacher_tokens": [7, 8, 1, 2]},)} if teacher else {}
    return policy_trajectory(
        "turn-1",
        (5, 6, 1, 2),
        (1, 1),
        (-0.5, -0.25),
        1.0,
        topk_indices=((1, 3), (2, 4)) if topk else (),
        topk_log_probs=((-0.5, -2.0), (-0.25, -3.0)) if topk else (),
        extras=extras,
    )


@pytest.mark.parametrize(("family", "loss_type"), [("tttd", TttdMLXLoss), ("openclawrl", OpenClawRLMLXLoss)])
def test_a_recipe_registers_its_loss_for_mlx(family: str, loss_type: type) -> None:
    assert isinstance(resolve_mlx_loss(family), loss_type)


def test_an_unregistered_family_is_refused_with_the_registered_ones() -> None:
    with pytest.raises(ValueError, match=r"unsupported MLX loss family 'fake-family'; registered: .*openclawrl"):
        resolve_mlx_loss("fake-family")


def test_the_tttd_base_kl_term_moves_each_advantage_by_its_centered_difference() -> None:
    rows = [TrainingRow((1, 2, 3, 4), 2, (1, 1), (-0.5, -0.25), (advantage, advantage)) for advantage in (1.0, -1.0)]

    shaped = frozen_base_kl_advantages(rows, [[-1.0, -1.0], [-1.0, -1.0]], 0.5)

    # The differences are (0.5, 0.75) with mean 0.625.
    assert shaped[0].advantages == pytest.approx((1.0 + 0.5 * 0.125, 1.0 - 0.5 * 0.125))
    assert shaped[1].advantages == pytest.approx((-1.0 + 0.5 * 0.125, -1.0 - 0.5 * 0.125))


def test_a_tttd_step_refuses_a_sample_without_response_tensors() -> None:
    with pytest.raises(ValueError, match="invalid MLX training row"):
        TttdMLXLoss().step([policy_trajectory("empty", (1, 2), (), (), 1.0)], [1.0], LossOptions())


@pytest.mark.parametrize(
    ("sample", "message"),
    [(distillation_sample(topk=False), "capture_topk"), (distillation_sample(teacher=False), "teacher_cands")],
    ids=["no-top-k", "no-teacher"],
)
def test_an_openclawrl_step_refuses_a_sample_without_its_distillation_inputs(sample, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenClawRLMLXLoss().step([sample], [1.0], LossOptions())


def test_an_openclawrl_row_carries_the_reward_and_the_hints() -> None:
    step = OpenClawRLMLXLoss().step([distillation_sample()], [1.0], LossOptions())

    [row] = step.rows
    assert row.policy.advantages == (1.0, 1.0)
    assert row.candidates[0].hint == "Be terse."
    assert row.topk_indices == ((1, 3), (2, 4))


@pytest.mark.parametrize(
    ("options", "kl_coef"),
    [
        (LossOptions(0.05), 0.05),
        (LossOptions(0.05, {"kl_coef": 0.5}), 0.5),
        (LossOptions(0.05, {"kl-coef": 0.5}), 0.5),
    ],
    ids=["deployment", "loss-options", "hyphenated"],
)
def test_the_deployment_kl_coef_reaches_the_openclawrl_loss_unless_its_options_set_one(
    options: LossOptions, kl_coef: float
) -> None:
    assert OpenClawRLMLXLoss().settings(options).kl_coef == kl_coef


def test_the_openclawrl_defaults_match_the_slime_recipe() -> None:
    settings = OpenClawRLMLXLoss().settings(LossOptions())

    assert (settings.w_rl, settings.w_opd, settings.diff_clip, settings.hint_selection) == (
        1.0,
        1.0,
        1.0,
        "sequence_optimal",
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"hint_selection": "token_optimal"}, "hint_selection"),
        ({"w_rl": 0.0, "w_opd": 0.0}, "non-zero w_rl or w_opd"),
        ({"w_rl": 1.0, "temperature": 0.6}, "unknown config fields: temperature"),
        ({"opd_advantage_clip": 5.0}, "set diff_clip to null"),
        ({"opd_advantage_clip": 0.0, "diff_clip": None}, "opd_advantage_clip must be finite and positive"),
    ],
    ids=["hint-selection", "no-term", "unknown-option", "two-clips", "zero-advantage-clip"],
)
def test_invalid_openclawrl_options_are_refused(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenClawRLMLXSettings.from_options(LossOptions(0.0, options))


def test_sequence_optimal_takes_the_hint_whose_tokens_the_policy_considered() -> None:
    student = mx.array([[1, 2], [3, 4]])
    unrelated = mx.array([[9, 8], [7, 6]])
    overlapping = mx.array([[1, 9], [4, 3]])

    assert select_hint([unrelated, overlapping], student, "sequence_optimal") == 1
    assert select_hint([unrelated, overlapping], student, "shortest") == 0


def test_the_gap_statistics_cover_the_masked_tokens_of_every_row() -> None:
    # One candidate per token, so each token's gap is teacher minus rollout.
    captured = [mx.array([[-1.0], [-1.0]]), mx.array([[-2.0], [-2.0]])]
    teacher = [mx.array([[0.0], [-3.0]]), mx.array([[0.0], [9.0]])]
    masks = [mx.array([1.0, 1.0]), mx.array([1.0, 0.0])]

    statistics = opd_gap_statistics(captured, teacher, masks, diff_clip=1.5, advantage_clip=None)

    # The gaps 1, -2 and 2; the masked 11 is left out.
    assert statistics.tokens == 3
    assert statistics.mean == pytest.approx(1 / 3)
    assert statistics.std == pytest.approx((((2 / 3) ** 2 + (7 / 3) ** 2 + (5 / 3) ** 2) / 3) ** 0.5)
    assert statistics.clip_fraction == pytest.approx(2 / 3)
    normalized = opd_gap_statistics(captured, teacher, masks, diff_clip=None, advantage_clip=1.0)
    assert normalized.clip_fraction == pytest.approx(1 / 3)


def test_the_normalized_advantage_is_the_centered_scaled_and_clipped_rollout_gap() -> None:
    indices = mx.array([[1, 2], [3, 4]])
    captured = mx.array([[-0.5, -1.0], [-0.2, -2.0]])
    teacher = mx.array([[0.5, -4.0], [-0.2, 10.0]])
    current = mx.array([[-0.6, -0.9], [-0.3, -1.9]])
    normalization = OPDNormalization(center=0.5, scale=2.0, clip=3.0)

    result = opd_one_sample(
        current,
        student_indices=indices,
        student_captured_log_probs=captured,
        teacher_indices=indices,
        teacher_log_probs=teacher,
        eps_lo=0.2,
        eps_hi=0.28,
        diff_clip=None,
        normalization=normalization,
    )

    gap = mx.clip((teacher - captured - 0.5) / 2.0, -3.0, 3.0)
    weight = mx.softmax(current, axis=-1)
    # At ratio 1 the surrogate is minus the advantage, summed over the candidates.
    assert mx.allclose(result.per_token_pg, -mx.sum(gap * weight, axis=-1), atol=1e-6)
    assert float(gap[1, 1].item()) == 3.0
