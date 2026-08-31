from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor
from recipes.tttd.preparer import TttdPreparer
from reef.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.train import ProcessorContext
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step


class _ExperimentLogger:
    def __init__(self) -> None:
        self.events = []

    def log(self, metrics, *, namespace):
        self.events.append((namespace, dict(metrics)))


def _inference(
    agent_record_id: str,
    token: int,
    *,
    artifact_ref: ArtifactRef | None = None,
) -> AgentRecord:
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        artifact_ref=artifact_ref,
        payload={
            "input_ids": [10, 11],
            "response": {
                "training": {
                    "tokens": [10, 11, token],
                    "loss_mask": [1],
                    "rollout_log_probs": [-0.2],
                }
            },
        },
    )


def _report(step: int, group: int, rollout: int, score: float) -> AgentRecord:
    reference = f"i-{step}-{group}-{rollout}"
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.REPORT,
        agent_record_id=f"r-{step}-{group}-{rollout}",
        references=(reference,),
        payload={
            "score": score,
            "references": [reference],
            "metadata": {
                "algorithm": "ttt-discover",
                "comparison_set": f"tttd-step-{step}-group-{group}",
                "step": step,
                "group": group,
                "rollout": rollout,
                "groups_per_step": 2,
                "rollouts_per_group": 3,
            },
        },
    )


def _processor(experiment_logger=None) -> TTTDProcessor:
    return TTTDProcessor(
        ProcessorContext(
            "discovery",
            {
                "groups_per_step": 2,
                "rollouts_per_group": 3,
            },
            report_type=TTTDGroupedRolloutReport,
            experiment_logger=experiment_logger or _ExperimentLogger(),
        )
    )


@pytest.mark.unit
def test_tttd_backend_contract_owns_its_dynamic_function_paths() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.algorithm import resolve_objective_paths

    args = SimpleNamespace(
        loss_family="tttd",
        compute_advantages_and_returns=True,
        kl_coef=0.1,
    )

    resolve_objective_paths(args)

    assert args.custom_loss_function_path.endswith("recipes.tttd.slime.objective.tttd_loss")
    assert args.custom_advantage_function_path.endswith("recipes.tttd.slime.objective.tttd_advantages")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"compute_advantages_and_returns": False}, "advantage computation"),
        ({"kl_coef": 0.0}, "kl-coef"),
        ({"kl_coef": -0.1}, "kl-coef"),
        ({"kl_coef": float("nan")}, "kl-coef"),
        ({"kl_coef": float("inf")}, "kl-coef"),
        ({"kl_coef": float("-inf")}, "kl-coef"),
        ({"kl_coef": True}, "kl-coef"),
        ({"kl_coef": None}, "kl-coef"),
    ],
)
def test_tttd_backend_contract_rejects_objective_drift(overrides, message) -> None:
    tttd = resolve_loss_family("tttd")
    values = {
        "compute_advantages_and_returns": True,
        "kl_coef": 0.1,
    }
    values.update(overrides)

    with pytest.raises(RuntimeError, match=message):
        tttd.validate_specific_args(SimpleNamespace(**values), "REEF_TRAINING_LOSS=tttd")


@pytest.mark.unit
def test_tttd_accepts_any_positive_kl_coef() -> None:
    # The paper's 0.1 is a default, not an invariant: any positive coefficient
    # keeps the reference-model group alive and scales the KL correction.
    tttd = resolve_loss_family("tttd")
    tttd.validate_specific_args(
        SimpleNamespace(compute_advantages_and_returns=True, kl_coef=0.2),
        "REEF_TRAINING_LOSS=tttd",
    )


@pytest.mark.unit
def test_tttd_waits_for_every_rollout_in_one_policy_step() -> None:
    processor = _processor()
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))
            if (group, rollout) != (1, 2):
                processor.ingest(_report(0, group, rollout, float(group + rollout)))

    assert not processor.ready()

    processor.ingest(_report(0, 1, 2, 3.0))

    assert processor.ready()
    batch = processor.build_batch()
    assert batch.batch_id == "discovery:tttd:0"
    assert len(batch.comparison_sets) == 2
    assert [[sample.reward for sample in group] for group in batch.comparison_sets] == [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ]


@pytest.mark.unit
def test_tttd_does_not_mix_complete_groups_from_different_steps() -> None:
    processor = _processor()
    for step, group in ((0, 0), (1, 1)):
        for rollout in range(3):
            processor.ingest(_inference(f"i-{step}-{group}-{rollout}", 100 + rollout))
            processor.ingest(_report(step, group, rollout, float(rollout)))

    assert not processor.ready()


@pytest.mark.unit
def test_tttd_accepts_one_durable_artifact_version() -> None:
    processor = _processor()
    artifact_ref = ArtifactRef("artifact-1", "checkpoint-1", None)
    for group in range(2):
        for rollout in range(3):
            processor.ingest(
                _inference(
                    f"i-0-{group}-{rollout}",
                    100 + group * 3 + rollout,
                    artifact_ref=artifact_ref,
                )
            )
            processor.ingest(_report(0, group, rollout, float(group + rollout)))

    assert processor.ready()


@pytest.mark.unit
def test_tttd_rejects_and_releases_a_step_spanning_artifact_versions(caplog) -> None:
    processor = _processor()
    with caplog.at_level("WARNING", logger="recipes.tttd.processor"):
        for group in range(2):
            for rollout in range(3):
                version = "checkpoint-1" if (group, rollout) != (1, 2) else "checkpoint-2"
                processor.ingest(
                    _inference(
                        f"i-0-{group}-{rollout}",
                        100 + group * 3 + rollout,
                        artifact_ref=ArtifactRef(f"artifact-{version}", version, None),
                    )
                )
                processor.ingest(_report(0, group, rollout, float(group + rollout)))

    assert not processor.ready()
    assert "span artifact versions" in caplog.text
    assert processor.status() == {
        "failed_steps": [
            {
                "step": 0,
                "reason": "mixed_artifact_versions",
                "artifact_versions": ["checkpoint-1", "checkpoint-2"],
            }
        ]
    }
    retention = processor.retention_decision()
    expected_ids = {
        f"{prefix}-0-{group}-{rollout}" for prefix in ("i", "r") for group in range(2) for rollout in range(3)
    }
    assert expected_ids <= retention.releasable_agent_record_ids


@pytest.mark.unit
def test_tttd_caches_policy_samples_at_ingest(monkeypatch) -> None:
    from reef.train.processors import reported as reported_module

    calls = 0
    make_sample = reported_module.make_policy_sample

    def counted_make_sample(*args, **kwargs):
        nonlocal calls
        calls += 1
        return make_sample(*args, **kwargs)

    monkeypatch.setattr(reported_module, "make_policy_sample", counted_make_sample)
    processor = _processor()
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))
            processor.ingest(_report(0, group, rollout, float(group + rollout)))
            processor.ready()

    for _ in range(5):
        assert processor.ready()
    assert calls == 6


@pytest.mark.unit
def test_tttd_resolves_reports_replayed_before_their_inferences() -> None:
    processor = _processor()
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_report(0, group, rollout, float(group + rollout)))

    assert not processor.ready()

    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))

    batch = processor.build_batch()
    assert [[sample.reward for sample in group] for group in batch.comparison_sets] == [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ]


@pytest.mark.unit
def test_tttd_invalid_single_report_does_not_wait_for_missing_inference() -> None:
    processor = _processor()
    invalid_report = _report(0, processor.groups_per_step, 0, 1.0)

    processor.ingest(invalid_report)

    retention = processor.retention_decision()
    assert retention.protected_agent_record_ids == frozenset()
    assert retention.releasable_agent_record_ids == frozenset(
        {invalid_report.agent_record_id, *invalid_report.references}
    )


@pytest.mark.unit
def test_tttd_filters_constant_groups_after_step_barrier() -> None:
    processor = _processor()
    rewards = ((1.0, 1.0, 1.0), (1.0, 2.0, 3.0))
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))
            processor.ingest(_report(0, group, rollout, rewards[group][rollout]))

    batch = processor.build_batch()

    assert len(batch.comparison_sets) == 1
    assert [sample.reward for sample in batch.comparison_sets[0]] == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_tttd_logs_full_grid_reward_and_filter_metrics() -> None:
    experiment_logger = _ExperimentLogger()
    processor = _processor(experiment_logger)
    rewards = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))
            processor.ingest(_report(0, group, rollout, rewards[group][rollout]))

    processor.build_batch()

    assert len(experiment_logger.events) == 1
    namespace, metrics = experiment_logger.events[0]
    assert namespace == "tttd"
    assert metrics == {
        "step": 0,
        "grid_groups": 2,
        "grid_rollouts": 6,
        "reward_min": 0.0,
        "reward_max": 3.0,
        "reward_mean": 1.0,
        "reward_std": pytest.approx(math.sqrt(4 / 3)),
        "reward_zero_fraction": 0.5,
        "constant_groups": 1,
        "constant_groups_removed": 1,
        "training_groups": 1,
        "training_rollouts": 3,
    }


@pytest.mark.unit
def test_tttd_keeps_one_group_when_all_rewards_are_constant() -> None:
    processor = _processor()
    for group in range(2):
        for rollout in range(3):
            processor.ingest(_inference(f"i-0-{group}-{rollout}", 100 + group * 3 + rollout))
            processor.ingest(_report(0, group, rollout, 1.0))

    batch = processor.build_batch()
    result = prepare_slime_step(batch, "tttd", {})

    assert len(batch.comparison_sets) == 1
    assert result.payload is not None
    assert result.payload["advantages"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
    assert result.payload["loss"] == "tttd"
    assert result.payload["rollout_ids"] == [0, 1, 2]
    assert result.payload["external_step_sizes"] == [3]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        (
            [1.0, 2.0, 3.0, 4.0],
            [-0.9696331024169922, -0.8666980266571045, -0.338700532913208, 8.865418434143066],
        ),
        (
            [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
            [
                -0.8313751220703125,
                -0.7453605532646179,
                -0.8313751220703125,
                -0.7453605532646179,
                -0.6131024360656738,
                -0.4065307378768921,
                0.47480976581573486,
                8.595550537109375,
            ],
        ),
    ],
)
def test_adaptive_entropic_advantages_match_pinned_reference(rewards, expected) -> None:
    advantages, beta = TttdPreparer.adaptive_entropic_advantages(rewards)

    # Pinned from the original float32 torch implementation; the pure-math
    # port uses float64 so values match well within this tolerance.
    assert advantages == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert math.isfinite(beta)


@pytest.mark.unit
def test_importance_sampling_surrogate_is_unclipped() -> None:
    torch = pytest.importorskip("torch")
    from recipes.tttd.slime.objective import importance_sampling_surrogate

    current = torch.tensor([0.0, math.log(4.0)], requires_grad=True)
    rollout = torch.zeros(2)
    advantages = torch.tensor([2.0, -3.0])

    loss = importance_sampling_surrogate(current, rollout, advantages)

    assert loss.tolist() == pytest.approx([-2.0, 12.0])
    loss.sum().backward()
    assert current.grad is not None
    assert current.grad.tolist() == pytest.approx([-2.0, 12.0])


@pytest.mark.unit
def test_tttd_custom_loss_uses_rollout_log_probs_without_clipping(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from recipes.tttd.slime.objective import tttd_loss

    module_name = "slime.backends.megatron_utils.loss"
    fake_loss_module = ModuleType(module_name)
    fake_loss_module.get_log_probs_and_entropy = lambda logits, **_: (
        torch.empty(0),
        {"log_probs": [logits.reshape(-1)]},
    )
    fake_loss_module.get_rollout_top_p_logprob_kwargs = lambda *_: {}
    monkeypatch.setitem(sys.modules, module_name, fake_loss_module)

    logits = torch.tensor([0.0, math.log(4.0)], requires_grad=True)
    batch = {
        "rollout_log_probs": [torch.zeros(2)],
        "advantages": [torch.tensor([2.0, -3.0])],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        # The reference removes this mask before applying policy loss. A
        # forced two-phase prefill token therefore still contributes.
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "step_global_batch_size": 1,
    }

    loss, metrics = tttd_loss(
        SimpleNamespace(use_rollout_logprobs=True),
        batch,
        logits,
        torch.mean,
    )

    # Tinker's built-in importance_sampling primitive sum-reduces tokens.
    assert loss.item() == pytest.approx(10.0)
    assert metrics["importance_ratio"].item() == pytest.approx(2.5)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == pytest.approx([-2.0, 12.0])


@pytest.mark.unit
def test_tttd_custom_loss_cancels_slime_dynamic_batch_normalization(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from recipes.tttd.slime.objective import tttd_loss

    module_name = "slime.backends.megatron_utils.loss"
    fake_loss_module = ModuleType(module_name)
    fake_loss_module.get_log_probs_and_entropy = lambda logits, **_: (
        torch.empty(0),
        {"log_probs": [logits.reshape(-1)]},
    )
    fake_loss_module.get_rollout_top_p_logprob_kwargs = lambda *_: {}
    monkeypatch.setitem(sys.modules, module_name, fake_loss_module)

    logits = torch.zeros(2, requires_grad=True)
    batch = {
        "rollout_log_probs": [torch.zeros(2)],
        "advantages": [torch.tensor([1.0, 1.0])],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "step_global_batch_size": 7,
    }

    scaled_loss, _ = tttd_loss(
        SimpleNamespace(use_rollout_logprobs=True),
        batch,
        logits,
        torch.mean,
    )

    # Slime's outer adapter divides by this dynamic value. The quotient is the
    # exact Tinker token sum (-2), independent of configured/sampled batch size.
    assert (scaled_loss / batch["step_global_batch_size"]).item() == pytest.approx(-2.0)


@pytest.mark.unit
def test_tttd_frozen_base_kl_executes_and_centers_masked_tokens(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from recipes.tttd.slime.objective import tttd_advantages

    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        is_pipeline_last_stage=lambda: True,
        get_data_parallel_group=lambda: None,
    )
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)

    rollout_data = {
        "advantages": [torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
        "rollout_log_probs": [torch.tensor([-0.2, -0.5]), torch.tensor([-0.3])],
        "ref_log_probs": [torch.tensor([-0.4, -0.1]), torch.tensor([-0.2])],
        "loss_masks": [torch.tensor([1.0, 0.0]), torch.tensor([1.0])],
    }

    tttd_advantages(SimpleNamespace(kl_coef=0.1), rollout_data)

    # Masked differences are [0.2, -0.1], whose distributed mean is 0.05.
    assert rollout_data["advantages"][0].tolist() == pytest.approx([0.985, 2.0])
    assert rollout_data["advantages"][1].tolist() == pytest.approx([3.015])
    assert rollout_data["returns"][0].tolist() == pytest.approx([0.985, 2.0])
    assert rollout_data["tttd_base_logp_diff"][0].tolist() == pytest.approx([0.05, 0.05])


def test_tttd_preparer_flags_a_batch_of_constant_groups() -> None:
    # The processor keeps one constant-reward group when every group is
    # constant; the preparer reports that batch, whose advantages carry no
    # signal, through constant_groups_retained.
    from reef.train.algos.registry import resolve_preparer
    from reef.train.types import GroupedPolicyBatch, PolicySample

    def sample(record_id: str, reward: float) -> PolicySample:
        return PolicySample(record_id, (5, 1), (1,), (-0.1,), reward)

    preparer = resolve_preparer("tttd")
    constant = GroupedPolicyBatch("b", ((sample("a", 1.0), sample("b", 1.0)),))
    mixed = GroupedPolicyBatch("b", ((sample("a", 1.0), sample("b", 0.0)),))

    assert preparer(constant, {}).metrics["constant_groups_retained"] == 1
    assert preparer(mixed, {}).metrics["constant_groups_retained"] == 0
