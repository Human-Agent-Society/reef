"""The GRPO(+DIS) control arm of the SAO comparison: SAO's wire surface and DIS primitive, no critic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sao import SAOGrpoControlRecipe, SAORecipe
from recipes.sao.processor import SAOProcessor
from recipes.sao.slime.grpo_dis import SaoGrpoControlAlgorithm
from reef.train.slime_backend.algorithm import TrainResult
from reef.train.slime_backend.loss_families import resolve_loss_family


def _control_backend_args(**overrides):
    values = {
        "loss_type": "policy_loss",
        "use_rollout_logprobs": True,
        "use_critic": False,
        "advantage_estimator": "grpo",
        "n_samples_per_prompt": 8,
        "eps_clip": 0.3,
        "eps_clip_high": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_control_recipe_shares_the_sao_processor_and_batch_size() -> None:
    recipe = SAOGrpoControlRecipe(**runtime_bindings(StubTrainingRuntime()))
    spec = recipe.training_spec()
    assert recipe.name == "sao-grpo-dis"
    assert spec.objective == "sao-grpo-dis"
    assert spec.processor is SAOProcessor
    # A step is batch_size accepted rollouts for both arms; the control's groups fit inside it.
    assert recipe.batch_size == SAORecipe(**runtime_bindings(StubTrainingRuntime())).batch_size == 128


@pytest.mark.unit
def test_control_loss_family_takes_the_dis_primitive_without_a_critic() -> None:
    control = resolve_loss_family("sao-grpo-dis")
    sao = resolve_loss_family("sao")
    assert isinstance(control, SaoGrpoControlAlgorithm)
    assert control.uses_pg_loss_primitive and control.requires_rollout_logprobs
    assert control.required_objective_hooks == ("custom_pg_loss_function_path",)
    assert control.rollout_data_keys == sao.rollout_data_keys
    assert control.external_batch_keys == sao.external_batch_keys
    control.validate_backend_args(_control_backend_args())
    control.validate_backend_args(_control_backend_args(advantage_estimator="cispo"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_control_backend_args(use_critic=True), "do not pass --use-critic"),
        (_control_backend_args(advantage_estimator="gae"), "advantage-estimator=grpo"),
        (_control_backend_args(n_samples_per_prompt=1), "n-samples-per-prompt"),
        (_control_backend_args(eps_clip=1.0), "eps-clip in"),
        (_control_backend_args(eps_clip_high=-0.1), "eps-clip-high"),
    ],
)
def test_control_backend_contract_rejects_sao_and_vanilla_grpo_settings(args, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("sao-grpo-dis").validate_backend_args(args)


class _ActorGroup:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.calls.append((rollout_id, rollout_data_ref))
        return [{"loss": 0.5}]


@pytest.mark.unit
def test_control_train_step_is_one_actor_update_without_a_critic() -> None:
    algorithm = resolve_loss_family("sao-grpo-dis").bind(None)
    actor = _ActorGroup()
    result = algorithm.train(3, "refs", actor_group=actor, critic_group=None, resolve=lambda value: value)
    assert isinstance(result, TrainResult)
    assert actor.calls == [(3, "refs")]
    assert result.worker_results == [{"loss": 0.5}]
    assert result.durable_metrics == {"control/actor_trained": 1}
    with pytest.raises(RuntimeError, match="critic"):
        algorithm.train(4, "refs", actor_group=actor, critic_group=object(), resolve=lambda value: value)


@pytest.mark.unit
def test_control_reports_the_same_asynchrony_telemetry_as_sao() -> None:
    rollout_data = {"rollout_created_ats": [0.0], "response_lengths": [4], "loss_masks": [[1, 1, 0, 0]]}
    control = resolve_loss_family("sao-grpo-dis").rollout_metrics(rollout_data, "run:3")
    sao = resolve_loss_family("sao").rollout_metrics(rollout_data, "run:3")
    assert control.keys() == sao.keys()
    assert control["sao/effective_token_rate"] == 0.5
