"""The built-in ``importance_sampling`` family of the Slime backend: Tinker's loss of the same name.

The generated-task recipe names this family, so these tests also check that
the recipe resolves on Slime and that Slime and Tinker give its samples the
same token advantages and the same loss. The loss test needs torch; the fake
``slime`` loss module returns the trainer's log probabilities as given.
"""

from __future__ import annotations

import math
import os
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from tests.support._trajectories import policy_trajectory

from reef.recipe.generated_tasks import GeneratedTaskRecipe
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.importance_sampling import ImportanceSamplingAlgorithm
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_step
from reef.train.types import TrainingBatch

REFERENCE = "reef.train.slime_backend.importance_sampling:ImportanceSamplingAlgorithm"
RECIPE = "reef.recipe.generated_tasks:GeneratedTaskRecipe"
#: Two prompt ids, then three response ids; the middle one is a tool's output, outside the loss mask.
TOKENS = (5, 6, 7, 8, 9)
LOSS_MASK = (1, 0, 1)
ROLLOUT_LOG_PROBS = (-0.5, 0.0, -0.25)
#: Two task groups of two episodes: group relative advantages +1, -1, -1, +1.
REWARDS = (("a", 1.0), ("a", 0.0), ("b", 0.25), ("b", 0.75))


def slime_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "loss_type": "custom_loss",
        "use_rollout_logprobs": True,
        "kl_coef": 0.0,
        "use_kl_loss": False,
        "calculate_per_token_loss": False,
    }
    values.update(overrides)
    return Namespace(**values)


def generated_task_batch() -> TrainingBatch:
    samples = tuple(
        replace(policy_trajectory(f"episode-{index}", TOKENS, LOSS_MASK, ROLLOUT_LOG_PROBS, reward), group_id=group)
        for index, (group, reward) in enumerate(REWARDS)
    )
    return TrainingBatch("rollout", samples)


@pytest.mark.unit
def test_slime_has_the_importance_sampling_family_without_a_method_package() -> None:
    assert loss_family_refs()["importance_sampling"] == REFERENCE
    family = resolve_loss_family("importance_sampling")
    assert isinstance(family, ImportanceSamplingAlgorithm)
    assert (family.loss_type, family.requires_rollout_logprobs, family.advantages) == (
        "custom_loss",
        True,
        "required",
    )
    family.validate_backend_args(slime_args(), recipe=RECIPE)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kl_coef": 0.1}, "no KL penalty"),
        ({"use_kl_loss": True}, "no KL penalty"),
        ({"calculate_per_token_loss": True}, "omit --calculate-per-token-loss"),
        ({"use_rollout_logprobs": False}, "--use-rollout-logprobs"),
        ({"loss_type": "policy_loss"}, "--loss-type custom_loss"),
    ],
)
def test_importance_sampling_refuses_flags_of_another_loss(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("importance_sampling").validate_backend_args(slime_args(**overrides), recipe=RECIPE)


@pytest.mark.unit
def test_the_generated_task_recipe_resolves_on_slime_at_config_time(tmp_path: Path, monkeypatch) -> None:
    from reef.service.deploy.orchestrator import resolve_deployment_config
    from reef.service.training_driver import _resolve_training_recipe
    from reef.train.slime_backend.driver import _stamp_loss_family_reference
    from reef.train.slime_backend.launch import driver_arguments

    for key in list(os.environ):
        if key.startswith("REEF_") or key == "RAY_ADDRESS":
            monkeypatch.delenv(key)
    options = {
        "global-batch-size": 8,
        "loss-type": "custom_loss",
        "use-rollout-logprobs": True,
        "disable-compute-advantages-and-returns": True,
    }
    raw = {
        "recipe": {"implementation": RECIPE},
        "inference": {"model-path": "/models/demo"},
        "training": {"backend": "slime", "options": options},
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend"] == "slime"

    # The driver's steps before Slime parses its flags.
    loss_family, recipe = _resolve_training_recipe(config)
    assert (loss_family, recipe) == ("importance_sampling", RECIPE)
    family = resolve_loss_family(loss_family)
    family_options, arguments = family.parse_driver_options(driver_arguments(config, loss_family=loss_family))
    assert family_options is None
    assert {"--loss-type=custom_loss", "--use-rollout-logprobs"} <= set(arguments)
    args = slime_args()
    family.apply_driver_options(args, family_options)
    _stamp_loss_family_reference(args, loss_family)
    assert (args.loss_family, args.loss_family_ref) == ("importance_sampling", REFERENCE)
    family.validate_backend_args(args, recipe=recipe)


@pytest.mark.unit
def test_slime_and_tinker_give_generated_task_samples_the_same_token_advantages_and_loss(monkeypatch) -> None:
    batch = generated_task_batch()
    method = TrainingMethod(GeneratedTaskRecipe.training_spec().objective)
    scheduling = GeneratedTaskRecipe.training_spec().scheduling

    slime_step = prepare_slime_step(batch, method, {}, scheduling, step_size=len(REWARDS))
    assert slime_step.action == "train" and slime_step.payload is not None
    data = to_slime_rollout_data(slime_step.payload)
    assert data["rollout_log_probs"] == [list(ROLLOUT_LOG_PROBS)] * len(REWARDS)

    tinker_step = prepare_tinker_step(batch, method, {}, scheduling, batch_size=len(REWARDS), kl_coef=0.0)
    assert tinker_step.payload is not None
    (tinker_batch,) = tinker_step.payload["batches"]
    rows = [row_from_payload(row) for row in tinker_batch]
    tinker_inputs = resolve_tinker_loss("importance_sampling").inputs(rows, [], kl_coef=0.0)
    tinker_advantages = [values["advantages"][-len(LOSS_MASK) :] for values in tinker_inputs]
    # The advantage trains each token the loss mask selects; the tool's output token gets 0.
    assert (
        data["advantages"]
        == tinker_advantages
        == [
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
        ]
    )

    torch = pytest.importorskip("torch")
    from reef.train.slime_backend.importance_sampling.objective import importance_sampling_loss

    fake_loss_module = ModuleType("slime.backends.megatron_utils.loss")
    fake_loss_module.get_log_probs_and_entropy = lambda logits, **_: (  # type: ignore[attr-defined]
        torch.empty(0),
        {"log_probs": [logits.reshape(-1)]},
    )
    fake_loss_module.get_rollout_top_p_logprob_kwargs = lambda *_: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", fake_loss_module)

    trainer_log_probs = [-0.25, -1.0, -0.5] * len(REWARDS)
    logits = torch.tensor(trainer_log_probs, requires_grad=True)
    step_global_batch_size = 3
    slime_batch = {
        "rollout_log_probs": [torch.tensor(row) for row in data["rollout_log_probs"]],
        "advantages": [torch.tensor(row) for row in data["advantages"]],
        "unconcat_tokens": [torch.tensor(TOKENS)] * len(REWARDS),
        "total_lengths": [len(TOKENS)] * len(REWARDS),
        "response_lengths": [len(LOSS_MASK)] * len(REWARDS),
        "step_global_batch_size": step_global_batch_size,
    }
    loss, metrics = importance_sampling_loss(Namespace(use_rollout_logprobs=True), slime_batch, logits, torch.mean)

    # Tinker's importance_sampling: -(exp(logp - logq) * advantage), summed over every token of the step.
    # Each token's loss is also its gradient with respect to logp.
    tinker_logq = [logq for row in rows for logq in row.logprobs]
    tinker_token_advantages = [advantage for row in tinker_advantages for advantage in row]
    tinker_token_losses = [
        -math.exp(logp - logq) * advantage
        for logp, logq, advantage in zip(trainer_log_probs, tinker_logq, tinker_token_advantages, strict=True)
    ]
    # Slime's outer adapter divides the loss by the step's global batch size.
    assert (loss / step_global_batch_size).item() == pytest.approx(sum(tinker_token_losses))
    assert metrics["pg_loss"].item() == pytest.approx(loss.item())
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad / step_global_batch_size).tolist() == pytest.approx(tinker_token_losses)


@pytest.mark.unit
def test_workers_resolve_the_family_loss_by_its_objective_path() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.algorithm import resolve_objective_paths

    args = Namespace(loss_family="importance_sampling")
    resolve_objective_paths(args)
    assert args.custom_loss_function_path == (
        "reef.train.slime_backend.importance_sampling.objective.importance_sampling_loss"
    )
