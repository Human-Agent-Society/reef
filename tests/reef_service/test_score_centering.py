"""Score centering's contract around a family's loss: settings, driver checks, payload columns, bridge check, wrapper."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pytest
from reef_service._trajectories import policy_trajectory

from reef.runtime.interfaces import TrainingMethod
from reef.train.algos import StepScheduling
from reef.train.slime_backend.algorithm import PolicyGradientWeight
from reef.train.slime_backend.distill import DistillSettings
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.slime_backend.score_centering import (
    ROLLOUT_KEYS,
    TOPK_INDICES_KEY,
    TOPK_LOG_PROBS_KEY,
    ScoreCenteringSettings,
    sampler_topk_columns,
    settings_from_args,
)
from reef.train.types import TrainingBatch


def log_probs_fixture(*probabilities: float) -> list[float]:
    return [math.log(value) for value in probabilities]


def sao_args(**overrides: object) -> SimpleNamespace:
    """The namespace fields ``configure_reef_loss_args`` reads for SAO with score centering on."""
    values: dict[str, object] = {
        "loss_family": "sao",
        "custom_rollout_data_keys": None,
        "score_centering": True,
        "score_centering_top_k": 128,
        "score_centering_min_tail_mass": 1e-6,
        "context_parallel_size": 1,
        "use_tis": False,
        "get_mismatch_metrics": False,
        "use_opsm": False,
        "custom_pg_loss_reducer_function_path": None,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "advantage_estimator": "grpo",
        "reef_configured_advantage_estimator": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"kind": "ppo"}, "kind must be one of"),
        ({"kind": "truncated"}, "finite cap"),
        ({"kind": "truncated", "upper": 0.0}, "finite cap"),
        ({"kind": "masked", "lower": 2.0, "upper": 1.0}, "0 <= lower < upper"),
        ({"kind": "masked", "lower": 0.5, "upper": math.nan}, "0 <= lower < upper"),
    ],
)
def test_policy_gradient_weight_rejects_invalid_values(values: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PolicyGradientWeight(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"), [("top_k", 0), ("top_k", True), ("min_tail_mass", 0.0), ("min_tail_mass", 1.0)]
)
def test_settings_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="score centering"):
        ScoreCenteringSettings(**{field: value})


@pytest.mark.unit
def test_flags_are_off_by_default_and_travel_onto_args() -> None:
    # slime_arguments imports the Megatron LoRA adapter, which needs torch.
    pytest.importorskip("torch")
    from reef.train.slime_backend.reef_adapters.slime_arguments import add_reef_slime_arguments

    parser = add_reef_slime_arguments(argparse.ArgumentParser())
    assert settings_from_args(parser.parse_args([])) is None
    args = parser.parse_args(["--score-centering", "--score-centering-top-k", "32"])
    assert settings_from_args(args) == ScoreCenteringSettings(top_k=32, min_tail_mass=1e-6)


@pytest.mark.unit
def test_driver_adds_the_rollout_keys_for_a_family_that_declares_its_weight() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    args = sao_args()
    configure_reef_loss_args(args)
    # SAO's own wire keys stay; the term's join them.
    assert args.custom_rollout_data_keys == ("action_masks", *ROLLOUT_KEYS)
    assert args.reef_rollout_tensor_dtypes == {
        "action_masks": "int",
        TOPK_INDICES_KEY: "long",
        TOPK_LOG_PROBS_KEY: "float32",
    }
    assert args.reef_external_batch_keys == ("action_masks", *ROLLOUT_KEYS)
    assert set(ROLLOUT_KEYS) <= set(args.reef_rollout_log_skip_keys)

    off = sao_args(score_centering=False)
    configure_reef_loss_args(off)
    assert off.custom_rollout_data_keys == ("action_masks",)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_family": "tttd", "kl_coef": 0.1}, "does not declare the weight of its policy-gradient loss"),
        ({"loss_family": "pg"}, "does not declare the weight of its policy-gradient loss"),
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"use_tis": True}, "--use-tis"),
        ({"get_mismatch_metrics": True, "use_opsm": True}, "--get-mismatch-metrics, --use-opsm"),
        ({"custom_pg_loss_reducer_function_path": "my.reducer"}, "--custom-pg-loss-reducer-function-path"),
    ],
)
def test_driver_refuses_what_the_term_cannot_correct(overrides: dict, message: str) -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    with pytest.raises(RuntimeError, match=message):
        configure_reef_loss_args(sao_args(**overrides))


@pytest.mark.unit
@pytest.mark.parametrize("family", ["sdft", "sdpo"])
@pytest.mark.parametrize("cap", [0.0, 1.2])
def test_driver_accepts_sampled_opd_and_ships_sampler_heads(family: str, cap: float) -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    spec = resolve_loss_family(family)
    args = sao_args(loss_family=family)
    settings = spec.settings_type(
        divergence="reverse",
        top_k=32,
        top_k_distribution="renormalized",
        importance_sampling_cap=cap,
        importance_sampling_level="token",
    )
    spec.apply_driver_options(args, settings)
    configure_reef_loss_args(args)

    expected = PolicyGradientWeight("none") if cap == 0 else PolicyGradientWeight("truncated", upper=cap)
    assert spec.policy_gradient_weight(args) == expected
    assert set(ROLLOUT_KEYS) <= set(args.reef_external_batch_keys)
    assert set(ROLLOUT_KEYS) <= set(args.custom_rollout_data_keys)
    assert "distill_teacher_sampled_log_probs" in args.reef_external_batch_keys
    assert spec.advantages == "forbidden"  # The teacher signal is computed inside the loss.


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes",
    [
        {"divergence": "forward"},
        {"divergence": "jsd"},
        {"top_k": 0},
        {"top_k_distribution": "tail"},
        {"importance_sampling_level": "sequence"},
    ],
)
def test_driver_refuses_other_distillation_losses(changes: dict[str, object]) -> None:
    pytest.importorskip("torch")
    from recipes.sdft.slime import SdftAlgorithm, SdftSettings
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    args = sao_args(loss_family="sdft")
    settings = SdftSettings(
        **{
            "divergence": "reverse",
            "top_k": 32,
            "importance_sampling_level": "token",
            **changes,
        }
    )
    SdftAlgorithm().apply_driver_options(args, settings)
    with pytest.raises(RuntimeError, match="score centering for distillation requires"):
        configure_reef_loss_args(args)

    args.score_centering = False
    configure_reef_loss_args(args)  # Existing configurations remain valid when centering is off.


@pytest.mark.unit
def test_disabled_importance_sampling_does_not_require_token_level() -> None:
    settings = DistillSettings(divergence="reverse", top_k=32, importance_sampling_cap=0)
    assert settings.score_centering_weight == PolicyGradientWeight("none")


@pytest.mark.unit
def test_payload_carries_each_wire_rows_top_k_in_schedule_order() -> None:
    samples = [
        policy_trajectory(
            f"i{index}", [1, 2, 3], [1], [-0.5], 0.0, "v0", topk_indices=[[3, index]], topk_log_probs=[[-0.5, -1.0]]
        )
        for index in range(2)
    ]
    assert sampler_topk_columns(samples, [1, 0, 1]) == {
        TOPK_INDICES_KEY: [[[3, 1]], [[3, 0]], [[3, 1]]],
        TOPK_LOG_PROBS_KEY: [[[-0.5, -1.0]]] * 3,
    }
    batch = TrainingBatch("batch", tuple(samples))
    # Two epochs repeat every row: the columns follow the wire rows.
    step = prepare_slime_step(
        batch, TrainingMethod("sao"), {}, StepScheduling(unit="sample", epochs=2), sampler_topk=True
    )
    assert step.payload is not None
    rows = [row[0] for row in step.payload["samples"]]
    assert [ids[0][1] for ids in step.payload[TOPK_INDICES_KEY]] == [int(row[1:]) for row in rows]
    plain = prepare_slime_step(batch, TrainingMethod("sao"), {}, StepScheduling(unit="sample"))
    assert plain.payload is not None and TOPK_INDICES_KEY not in plain.payload


def wire_data(
    *,
    loss_mask: list[int] | None = None,
    indices: object = None,
    log_probs: object = None,
    rollout_log_probs: bool = True,
) -> tuple[dict, dict]:
    """Rollout data for one sample (prompt [1, 2], response [5, 6]) and a payload with its top-3 columns."""
    data: dict = {"tokens": [[1, 2, 5, 6]], "loss_masks": [loss_mask if loss_mask is not None else [1, 1]]}
    if rollout_log_probs:
        data["rollout_log_probs"] = [log_probs_fixture(0.5, 0.3)]
    payload = {
        TOPK_INDICES_KEY: [indices if indices is not None else [[5, 7, 8], [9, 6, 3]]],
        TOPK_LOG_PROBS_KEY: [
            (
                log_probs
                if log_probs is not None
                else [log_probs_fixture(0.5, 0.2, 0.1), log_probs_fixture(0.4, 0.3, 0.2)]
            )
        ],
    }
    return data, payload


def attached(top_k: int, data: dict, payload: dict) -> dict:
    pytest.importorskip("torch")
    from reef.train.slime_backend.score_centering.heads import attach_sampler_heads

    attach_sampler_heads(data, payload, ScoreCenteringSettings(top_k=top_k))
    return data


@pytest.mark.unit
def test_bridge_keeps_the_first_top_k_entries_and_pads_untrained_positions() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.score_centering.heads import PADDING_LOG_PROB

    data = attached(2, *wire_data(loss_mask=[0, 1]))
    assert [ids.tolist() for ids in data[TOPK_INDICES_KEY]] == [[[0, 0], [9, 6]]]
    assert [values.tolist() for values in data[TOPK_LOG_PROBS_KEY]] == [
        [[PADDING_LOG_PROB] * 2, pytest.approx(log_probs_fixture(0.4, 0.3))]
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "top_k", "message"),
    [
        (wire_data(indices=[], log_probs=[]), 3, "capture_topk >= 3"),
        (
            wire_data(indices=[[5, 7, 8]], log_probs=[log_probs_fixture(0.5, 0.2, 0.1)]),
            3,
            "1 rows for a 2-token response",
        ),
        (wire_data(), 4, "fewer than top_k=4"),
        (wire_data(indices=[[5, 7, 7], [9, 6, 3]]), 3, "distinct"),
        (wire_data(indices=[[5, -1, 8], [9, 6, 3]]), 3, "position 0 topk_indices must be non-negative"),
        (wire_data(indices=[[5, 7.5, 8], [9, 6, 3]]), 3, "non-negative integers"),
        (wire_data(log_probs=[["x", -1.0, -2.0], log_probs_fixture(0.4, 0.3, 0.2)]), 3, "must hold numbers"),
        (
            wire_data(log_probs=[[-0.1, -0.2, math.nan], log_probs_fixture(0.4, 0.3, 0.2)]),
            3,
            "finite log-probabilities",
        ),
        (
            wire_data(log_probs=[log_probs_fixture(0.5, 0.4, 0.3), log_probs_fixture(0.4, 0.3, 0.2)]),
            3,
            "more than one",
        ),
        # Position 1 (after an untrained position 0) repeats an id.
        (wire_data(loss_mask=[0, 1], indices=[[], [9, 9, 3]], log_probs=[[], [-1, -2, -3]]), 3, "position 1"),
        # Position 1's head records the sampled token 6 at 0.2, not the 0.3 it was sampled at.
        (wire_data(log_probs=[log_probs_fixture(0.5, 0.2, 0.1), log_probs_fixture(0.4, 0.2, 0.3)]), 3, "not aligned"),
    ],
)
def test_bridge_refuses_incomplete_or_misaligned_top_k(row: tuple[dict, dict], top_k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        attached(top_k, *row)


@pytest.mark.unit
def test_bridge_checks_alignment_only_against_recorded_rollout_log_probs() -> None:
    misaligned = [log_probs_fixture(0.5, 0.2, 0.1), log_probs_fixture(0.4, 0.2, 0.3)]
    data = attached(3, *wire_data(log_probs=misaligned, rollout_log_probs=False))
    assert [ids.shape for ids in data[TOPK_INDICES_KEY]] == [(2, 3)]
    data, payload = wire_data()
    del payload[TOPK_LOG_PROBS_KEY]
    with pytest.raises(ValueError, match=f"needs {TOPK_LOG_PROBS_KEY} with one entry per sample"):
        attached(3, data, payload)


@pytest.mark.unit
def test_the_term_is_added_to_the_family_loss_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from reef.train.slime_backend.score_centering import term

    seen: list[PolicyGradientWeight] = []

    def fake_term(_args, _batch, _logits, _reduce, weight, settings):
        seen.append(weight)
        assert settings == ScoreCenteringSettings()
        return torch.tensor(0.25), {"score_centering_term": torch.tensor(0.25)}

    monkeypatch.setattr(term, "score_centering_term", fake_term)
    args = sao_args()
    loss, log = term.add_score_centering(
        args, {}, None, None, torch.tensor(1.0), {"loss": torch.tensor(1.0), "pg_loss": torch.tensor(1.0)}
    )
    assert float(loss) == 1.25
    assert {name: float(value) for name, value in log.items()} == {
        "loss": 1.25,
        "pg_loss": 1.0,
        "score_centering_term": 0.25,
    }
    assert seen == [PolicyGradientWeight("masked", lower=0.8, upper=1.28)]
    with pytest.raises(RuntimeError, match="declares no policy-gradient weight"):
        term.add_score_centering(sao_args(loss_family="tttd"), {}, None, None, torch.tensor(1.0), {})
