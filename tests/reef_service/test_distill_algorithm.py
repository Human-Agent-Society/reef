"""The distillation base of the Slime backend: what every distilling family gets from ``DistillAlgorithm``.

Torch/ray free: a toy family subclasses the base under its own prefix, the
way ``recipes/<name>/slime/`` does, and the tests exercise the flags, the
settings stamped on ``args``, the seven-column wire row and the payload
checks. The kernels are pinned in ``test_distill_parity.py``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reef_service._trajectories import policy_trajectory

from reef.core.trajectories import source_record_id
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings, settings_from_args
from reef.train.slime_backend.loss_families import register_loss_family, resolve_loss_family, unregister_loss_family
from reef.train.types import TrajectoryItem

STUDENT_TOKENS = [5, 6, 7, 1, 2, 3]  # three prompt ids, three response ids
STUDENT_LOSS_MASK = [1, 1, 1]
STUDENT_LOG_PROBS = [-0.1, -0.2, -0.3]
TEACHER_TOKENS = [9, 9, 9, 9, 1, 2, 3]


@dataclass(frozen=True)
class ToySettings(DistillSettings):
    """A recipe's defaults: a separate teacher with the reverse KL at the sampled token."""

    teacher: str = "separate"
    divergence: str = "reverse"
    top_k: int = 32
    teacher_checkpoint: str = "/models/teacher"
    importance_sampling_cap: float = 0.0


class ToyAlgorithm(DistillAlgorithm):
    loss_family = "toydistill"
    settings_type = ToySettings


@pytest.fixture
def toy_family():
    register_loss_family(ToyAlgorithm())
    try:
        yield resolve_loss_family("toydistill")
    finally:
        unregister_loss_family("toydistill")


def _sample() -> TrajectoryItem:
    sample = policy_trajectory("i1", STUDENT_TOKENS, STUDENT_LOSS_MASK, STUDENT_LOG_PROBS, 0.0, "v0")
    return sample.with_training(teacher_tokens=TEACHER_TOKENS)


def _payload(teacher_tokens: list[Any], sample_weight: Any = 1.0, **overrides: Any) -> dict[str, Any]:
    row = ["i1", STUDENT_TOKENS, STUDENT_LOSS_MASK, STUDENT_LOG_PROBS, 0.0, teacher_tokens, sample_weight]
    return {"samples": [row], "rollout_ids": [0], "loss": "toydistill", **overrides}


@pytest.mark.unit
def test_flags_carry_the_family_prefix_and_land_on_args_under_distill_names(toy_family) -> None:
    settings, remaining = toy_family.parse_driver_options(
        [
            "--toydistill-teacher=self",
            "--toydistill-divergence=jsd",
            "--toydistill-top-k=0",
            "--toydistill-teacher-update-rate=0.02",
            "--toydistill-importance-sampling-cap=2",
            "--toydistill-skip-response-tokens=3",
            "--toydistill-jsd-beta=0.3",
            "--lr=1",
        ]
    )

    assert settings == ToySettings(
        teacher="self",
        divergence="jsd",
        top_k=0,
        teacher_update_rate=0.02,
        importance_sampling_cap=2.0,
        skip_response_tokens=3,
        jsd_beta=0.3,
    )
    assert remaining == ["--lr=1"]
    args = SimpleNamespace()
    toy_family.apply_driver_options(args, settings)
    assert args.loss_family == "toydistill"
    assert settings_from_args(args) == DistillSettings(
        teacher="self",
        divergence="jsd",
        top_k=0,
        teacher_update_rate=0.02,
        teacher_checkpoint="/models/teacher",
        importance_sampling_cap=2.0,
        skip_response_tokens=3,
        jsd_beta=0.3,
    )
    assert settings_from_args(args).exact

    # The recipe's own defaults apply when the driver passes no flags.
    defaults = SimpleNamespace()
    toy_family.apply_driver_options(defaults, None)
    assert settings_from_args(defaults) == DistillSettings(**vars(ToySettings()))
    assert (defaults.distill_teacher, defaults.distill_divergence, defaults.distill_top_k) == (
        "separate",
        "reverse",
        32,
    )
    assert toy_family.bind(settings) is toy_family
    with pytest.raises(TypeError, match="ToySettings"):
        toy_family.bind(SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("teacher", "oracle"),
        ("divergence", "hellinger"),
        ("top_k", -1),
        ("top_k", 2.5),
        ("teacher_update_rate", 1.5),
        ("teacher_update_rate", -0.1),
        ("teacher_checkpoint", 3),
        ("importance_sampling_cap", -1.0),
        ("importance_sampling_cap", float("nan")),
        ("skip_response_tokens", -1),
        ("jsd_beta", 1.0),
        ("top_k_source", "oracle"),
        ("top_k_distribution", "flat"),
        ("importance_sampling_level", "batch"),
    ],
)
def test_settings_reject_invalid_values(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        DistillSettings(**{name: value})


@pytest.mark.unit
def test_a_separate_teacher_needs_its_checkpoint() -> None:
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        DistillSettings(teacher="separate")
    assert DistillSettings(teacher="separate", teacher_checkpoint="/models/teacher").teacher == "separate"


@pytest.mark.unit
def test_backend_validation_pins_the_loss_type_rollout_logprobs_and_one_step_per_rollout(toy_family) -> None:
    accepted = {"loss_type": "custom_loss", "use_rollout_logprobs": True, "num_steps_per_rollout": 1}

    def _args(**changes: Any) -> SimpleNamespace:
        # The driver stamps the family's settings on args before it validates them.
        args = SimpleNamespace(**{**accepted, **changes})
        toy_family.apply_driver_options(args, None)
        return args

    toy_family.validate_backend_args(_args())
    toy_family.validate_backend_args(_args(num_steps_per_rollout=None))  # Slime's default

    with pytest.raises(RuntimeError, match="loss-type custom_loss"):
        toy_family.validate_backend_args(_args(loss_type="policy_loss"))
    with pytest.raises(RuntimeError, match="use-rollout-logprobs"):
        toy_family.validate_backend_args(_args(use_rollout_logprobs=False))
    with pytest.raises(RuntimeError, match="num-steps-per-rollout"):
        toy_family.validate_backend_args(_args(num_steps_per_rollout=2))


@pytest.mark.unit
def test_selecting_the_support_with_the_student_needs_zero_dropout(toy_family) -> None:
    args = SimpleNamespace(
        loss_type="custom_loss",
        use_rollout_logprobs=True,
        num_steps_per_rollout=1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    toy_family.apply_driver_options(args, ToySettings(top_k_source="student"))
    toy_family.validate_backend_args(args)
    with pytest.raises(RuntimeError, match="dropout"):
        toy_family.validate_backend_args(SimpleNamespace(**{**vars(args), "attention_dropout": 0.1}))
    # The exact representation keeps the teacher's whole distribution: nothing is selected.
    exact = SimpleNamespace(**{**vars(args), "attention_dropout": 0.1})
    toy_family.apply_driver_options(exact, ToySettings(top_k=0, top_k_source="student"))
    toy_family.validate_backend_args(exact)


@pytest.mark.unit
def test_the_wire_row_is_the_policy_row_plus_the_teacher_sequence(toy_family) -> None:
    sample = _sample()

    row = toy_family.shape_sample_row(sample)

    expected = [source_record_id(sample), STUDENT_TOKENS, STUDENT_LOSS_MASK, STUDENT_LOG_PROBS, 0.0, TEACHER_TOKENS]
    # A sample without a weight of its own carries 1; a recipe's weight rides along.
    assert row == [*expected, 1.0]
    assert toy_family.shape_sample_row(sample.with_training(distill_sample_weight=0))[6] == 0.0
    data = to_slime_rollout_data(_payload(TEACHER_TOKENS))
    assert data["loss"] == "toydistill"
    assert data["tokens"] == [STUDENT_TOKENS]
    assert data["response_lengths"] == [3]
    assert data["rollout_log_probs"] == [STUDENT_LOG_PROBS]
    assert data["teacher_tokens"] == [TEACHER_TOKENS]
    assert data["distill_sample_weights"] == [1.0]
    assert toy_family.rollout_data_keys == ("teacher_tokens", "distill_sample_weights")
    assert set(toy_family.external_batch_keys) >= {
        "rollout_log_probs",
        "distill_sample_weights",
        "distill_teacher_log_probs",
    }
    assert set(toy_family.rollout_log_skip_keys) >= {"teacher_tokens", "distill_teacher_topk_ids"}


@pytest.mark.unit
@pytest.mark.parametrize("sample_weight", [-1.0, "1", None, float("nan"), True])
def test_the_payload_rejects_an_invalid_sample_weight(toy_family, sample_weight: Any) -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        to_slime_rollout_data(_payload(TEACHER_TOKENS, sample_weight=sample_weight))


@pytest.mark.unit
def test_the_payload_rejects_a_teacher_sequence_that_does_not_end_with_the_response(toy_family) -> None:
    with pytest.raises(ValueError, match="end with the student's response ids"):
        to_slime_rollout_data(_payload([9, 9, 1, 2, 4]))
    with pytest.raises(ValueError, match="carry a prompt"):
        to_slime_rollout_data(_payload([1, 2, 3]))
    with pytest.raises(ValueError, match="sequence of integers"):
        to_slime_rollout_data(_payload([9, "1", 2, 3]))
    with pytest.raises(ValueError, match="must be \\[source_id"):
        to_slime_rollout_data(
            {"samples": [["i1", [5, 1], [1], [-0.1], 0.0]], "rollout_ids": [0], "loss": "toydistill"}
        )
    with pytest.raises(ValueError, match="must omit advantages"):
        to_slime_rollout_data(_payload(TEACHER_TOKENS, advantages=[1.0]))


@pytest.mark.unit
def test_the_base_registers_no_family_and_imports_no_adapters() -> None:
    # The base is backend machinery: recipes register their families on it,
    # and it stays family-blind toward reef_adapters like the rest of the backend contract.
    package = Path(__file__).resolve().parents[2] / "reef" / "train" / "slime_backend" / "distill"
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "reef_adapters" not in node.module, path
                assert node.module != "reef.train.slime_backend.loss_families", path
            if isinstance(node, ast.Import):
                assert all("reef_adapters" not in alias.name for alias in node.names), path
    with pytest.raises(Exception, match="available families"):
        resolve_loss_family("distill")
