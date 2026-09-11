from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from reef.train.slime_backend.reef_adapters.slime_arguments import (
    REEF_MEGATRON_INIT_PATH,
    REEF_MODEL_PROVIDER_PATH,
    add_reef_slime_arguments,
    configure_reef_loss_args,
    finalize_reef_slime_args,
)


def _args(**overrides):
    values = {
        "critic_steps_per_actor": 2,
        "use_critic": False,
        "offload_train": False,
        "disable_grad_buffers_cpu_backup": False,
        "disable_param_buffers_cpu_backup": False,
        "megatron_lora_rank": 0,
        "megatron_lora_alpha": None,
        "megatron_lora_dropout": 0.0,
        "megatron_lora_target_modules": None,
        "megatron_to_hf_mode": "raw",
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "custom_megatron_init_path": "custom.initialize",
        "custom_model_provider_path": "custom.model_provider",
        "loss_family": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_reef_slime_argument_hook_owns_only_reef_options() -> None:
    parser = add_reef_slime_arguments(argparse.ArgumentParser())

    args = parser.parse_args(["--megatron-to-hf-mode=bridge", "--megatron-lora-rank=4", "--use-critic"])

    assert args.megatron_to_hf_mode == "bridge"
    assert args.megatron_lora_rank == 4
    assert args.use_critic is True
    assert args.check_lora_weight_equal is True
    assert args.verify_lora_base_weights is True
    assert args.reef_executor_backend == "auto"
    assert args.reef_rollout_executor_backend == "auto"


@pytest.mark.unit
def test_finalize_arguments_chains_user_hooks_and_enables_explicit_critic() -> None:
    args = _args(
        disable_grad_buffers_cpu_backup=True,
        disable_param_buffers_cpu_backup=True,
        megatron_to_hf_mode="bridge",
        megatron_lora_rank=8,
        custom_megatron_init_path="custom.initialize",
        custom_model_provider_path="custom.model_provider",
    )

    finalize_reef_slime_args(args, ["--use-critic", "--megatron-lora-rank=8"])

    assert args.use_critic is True
    assert args.offload_train is True
    assert args.disable_grad_buffers_cpu_backup is True
    assert args.disable_param_buffers_cpu_backup is True
    assert args.megatron_lora_alpha == 8
    assert args.custom_megatron_init_path == REEF_MEGATRON_INIT_PATH
    assert args.reef_chained_megatron_init_path == "custom.initialize"
    assert args.custom_model_provider_path == REEF_MODEL_PROVIDER_PATH
    assert args.reef_chained_model_provider_path == "custom.model_provider"


@pytest.mark.unit
def test_finalize_arguments_rejects_nonpositive_critic_steps() -> None:
    with pytest.raises(ValueError, match="critic-steps-per-actor"):
        finalize_reef_slime_args(_args(critic_steps_per_actor=0), [])


@pytest.mark.unit
def test_loss_family_projection_uses_public_slime_primitives() -> None:
    sao = _args(loss_family="sao")
    configure_reef_loss_args(sao)
    assert sao.advantage_estimator == "cispo"
    assert sao.compute_advantages_and_returns is True

    topk = _args(loss_family="openclawrl")
    configure_reef_loss_args(topk)
    assert topk.compute_advantages_and_returns is True

    neutral = _args(loss_family="sft")
    configure_reef_loss_args(neutral)
    assert not hasattr(neutral, "compute_advantages_and_returns")


@pytest.mark.unit
def test_keeping_the_lora_base_resident_needs_lora_and_colocation() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_bridge_args

    def args(**overrides):
        values = {
            "num_rollout": 1,
            "save_hf": "/tmp/hf/checkpoint-{rollout_id}",
            "save": "/tmp/megatron",
            "debug_train_only": False,
            "debug_rollout_only": False,
            "rollout_num_gpus": 1,
            "rollout_external": False,
            "colocate": True,
            "offload_train": True,
            "offload_rollout": True,
            "megatron_lora_rank": 32,
            "keep_lora_base_resident": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    with pytest.raises(ValueError, match="requires LoRA training"):
        validate_bridge_args(args(megatron_lora_rank=0), None)
    with pytest.raises(ValueError, match="colocated training"):
        validate_bridge_args(args(colocate=False, offload_rollout=False), None)


@pytest.mark.unit
def test_the_preflight_and_the_bridge_agree_on_what_a_lora_run_is() -> None:
    """The preflight spells the predicate out; this pins it to the shared helper."""
    from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled
    from reef.train.slime_backend.reef_adapters.preflight import validate_bridge_args

    def args(rank):
        return SimpleNamespace(
            num_rollout=1,
            save_hf="/tmp/hf/checkpoint-{rollout_id}",
            save="/tmp/megatron",
            debug_train_only=False,
            debug_rollout_only=False,
            rollout_num_gpus=1,
            rollout_external=False,
            colocate=True,
            offload_train=True,
            offload_rollout=True,
            megatron_lora_rank=rank,
            keep_lora_base_resident=True,
        )

    for rank in (0, 32):
        refused = False
        try:
            validate_bridge_args(args(rank), None)
        except ValueError as exc:
            refused = "requires LoRA training" in str(exc)
        assert refused is not megatron_lora_enabled(args(rank))
