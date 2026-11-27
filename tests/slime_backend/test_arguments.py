from __future__ import annotations

import argparse
import re
import sys
from types import SimpleNamespace

import pytest

from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.slime_arguments import (
    REEF_BEFORE_TRAIN_STEP_HOOK_PATH,
    REEF_MEGATRON_INIT_PATH,
    REEF_MODEL_PROVIDER_PATH,
    add_reef_slime_arguments,
    configure_reef_loss_args,
    finalize_reef_slime_args,
)


def slime_args(**overrides):
    values = {
        "critic_steps_per_actor": 2,
        "critic_save_interval": 1,
        "save": "/tmp/megatron",
        "use_critic": False,
        "offload_train": False,
        "disable_grad_buffers_cpu_backup": False,
        "disable_param_buffers_cpu_backup": False,
        "train_env_vars": {},
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
        "score_centering": False,
        "advantage_estimator": "grpo",
    }
    values.update(overrides)
    return SlimeArguments(**values)


@pytest.mark.unit
def test_reef_slime_argument_hook_owns_only_reef_options() -> None:
    parser = add_reef_slime_arguments(argparse.ArgumentParser())

    args = parser.parse_args(["--megatron-to-hf-mode=bridge", "--megatron-lora-rank=4", "--use-critic"])

    assert args.megatron_to_hf_mode == "bridge"
    assert args.megatron_lora_rank == 4
    assert args.use_critic is True
    assert args.reef_executor_backend == "auto"
    assert args.is_optimizer_restart_accepted is False
    assert parser.parse_args(["--reef-accept-optimizer-restart"]).is_optimizer_restart_accepted is True


@pytest.mark.unit
def test_finalize_arguments_chains_user_hooks_and_enables_explicit_critic() -> None:
    args = slime_args(
        disable_grad_buffers_cpu_backup=True,
        disable_param_buffers_cpu_backup=True,
        megatron_to_hf_mode="bridge",
        megatron_lora_rank=8,
        custom_megatron_init_path="custom.initialize",
        custom_model_provider_path="custom.model_provider",
        custom_megatron_before_train_step_hook_path="custom.before_step",
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
    # Every optimizer step passes Reef's learning-rate hook, which runs the user's hook first.
    assert args.custom_megatron_before_train_step_hook_path == REEF_BEFORE_TRAIN_STEP_HOOK_PATH
    assert args.reef_chained_before_train_step_hook_path == "custom.before_step"


@pytest.mark.unit
def test_finalize_arguments_lets_an_explicit_no_offload_train_stand() -> None:
    # --use-critic offloads the idle model by default; a launch that says
    # --no-offload-train keeps the actor and the critic resident.
    args = slime_args(offload_train=False)

    finalize_reef_slime_args(args, ["--use-critic", "--no-offload-train"])

    assert args.use_critic is True
    assert args.offload_train is False


@pytest.mark.unit
def test_finalize_arguments_rejects_nonpositive_critic_steps() -> None:
    with pytest.raises(ValueError, match="critic-steps-per-actor"):
        finalize_reef_slime_args(slime_args(critic_steps_per_actor=0), [])


@pytest.mark.unit
def test_loss_family_projection_uses_public_slime_primitives() -> None:
    sao = slime_args(loss_family="sao")
    configure_reef_loss_args(sao)
    assert sao.advantage_estimator == "cispo"
    # The configured estimator stays on record for a job of another family.
    assert sao.reef_configured_advantage_estimator == "grpo"
    assert sao.compute_advantages_and_returns is True

    topk = slime_args(loss_family="openclawrl")
    configure_reef_loss_args(topk)
    assert topk.compute_advantages_and_returns is True

    neutral = slime_args(loss_family="sft")
    configure_reef_loss_args(neutral)
    assert not hasattr(neutral, "compute_advantages_and_returns")


def component_args(**overrides):
    values = {
        "compute_advantages_and_returns": False,
        "num_rollout": 1,
        "save_hf": "/tmp/hf/checkpoint-{rollout_id}",
        "save": "/tmp/megatron",
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "offload_train": False,
        "offload_rollout": False,
        "disable_grad_buffers_cpu_backup": False,
        "disable_param_buffers_cpu_backup": False,
        "train_env_vars": {},
        "release_train": False,
        "megatron_lora_rank": 0,
        "rollout_num_gpus": 0,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 4,
        **overrides,
    }
    return SlimeArguments(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"megatron_lora_rank": 32},
        {"offload_train": True},
        # The component's own rollout engines: on GPUs of their own, or on some or all of the training GPUs.
        {"rollout_num_gpus": 2},
        {"colocate": True, "offload_train": True, "offload_rollout": True, "rollout_num_gpus": 4},
        {"colocate": True, "offload_train": True, "offload_rollout": True, "rollout_num_gpus": 2},
        # A deployment that accepts the optimizer restart may save weights only.
        {"no_save_optim": True, "is_optimizer_restart_accepted": True},
    ],
)
def test_slime_args_accept_a_run_the_training_component_can_run(overrides) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_slime_args

    validate_slime_args(component_args(**overrides), None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        # Reef's LoRA adapters are not wired into Slime's weight transfer to its engines.
        ({"rollout_num_gpus": 2, "megatron_lora_rank": 32}, "remove --megatron-lora-rank"),
        ({"debug_train_only": True}, "--debug-train-only"),
        ({"debug_rollout_only": True}, "--debug-rollout-only"),
        ({"num_rollout": 0}, "positive --num-rollout"),
        ({"save_hf": "/tmp/hf/checkpoint"}, "{rollout_id}"),
        # The workers keep running through a job, so nothing releases and recreates them.
        ({"release_train": True}, "remove --release-train"),
        # Each job's driver loads the latest checkpoint: without its optimizer state, Adam starts again every job.
        ({"no_save_optim": True}, "--no-save-optim restarts the optimizer at every job"),
        ({"is_optimizer_restart_accepted": True}, "--reef-accept-optimizer-restart applies only with --no-save-optim"),
    ],
)
def test_slime_args_refuse_what_the_training_component_cannot_run(overrides, refusal) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_slime_args

    with pytest.raises(ValueError, match=re.escape(refusal)):
        validate_slime_args(component_args(**overrides), None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("allocator_settings", "enabled"),
    [
        ("expandable_segments:True", True),
        ("expandable_segments:true", True),
        (" max_split_size_mb:512 , expandable_segments : True ", True),
        ("roundup_power2_divisions:[64:8,256:4],expandable_segments:True", True),
        # Readers can take either pair of a repeated key.
        ("expandable_segments:True,expandable_segments:False", True),
        ("expandable_segments:False,expandable_segments:True", True),
        ("expandable_segments:False", False),
        ("max_split_size_mb:512", False),
        ("expandable_segments", False),
        ("", False),
    ],
)
def test_allocator_settings_turn_on_expandable_segments_with_a_true_pair(allocator_settings, enabled) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import enables_expandable_segments

    assert enables_expandable_segments(allocator_settings) is enabled


EXPANDABLE_SEGMENTS_ENVIRONMENTS = [
    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    {"PYTORCH_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:true"},
    {"SLIME_ENABLE_EXPANDABLE_SEGMENTS": "1"},
]
# Slime turns off the CPU copy of the buffers of a trainer it offloads.
NO_CPU_COPY = {"disable_grad_buffers_cpu_backup": True, "disable_param_buffers_cpu_backup": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    "trainer",
    [
        # Slime offloads the trainer with --colocate, with --offload or --offload-train, and with a ppo critic.
        {"colocate": True, "offload_train": True, "offload_rollout": True, "rollout_num_gpus": 4, **NO_CPU_COPY},
        {"offload_train": True, **NO_CPU_COPY},
        # The critic's role gets the CPU copy of its parameter buffers back, but not of its gradient buffers.
        {"use_critic": True, "offload_train": True, "disable_grad_buffers_cpu_backup": True},
        # A role that turns off the CPU copy of its parameter buffers only.
        {"offload_train": True, "disable_param_buffers_cpu_backup": True},
    ],
)
@pytest.mark.parametrize("environ", EXPANDABLE_SEGMENTS_ENVIRONMENTS)
def test_an_offloaded_trainer_without_a_cpu_copy_refuses_expandable_segments(trainer, environ) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_training_worker_environment

    [(name, value)] = environ.items()
    with pytest.raises(ValueError) as refusal:
        validate_training_worker_environment(component_args(**trainer), environ)
    message = str(refusal.value)
    assert f"({name}={value} in the environment)" in message
    assert "writes the zeros into the weights" in message
    fix = (
        f"set {name} to 0" if name == "SLIME_ENABLE_EXPANDABLE_SEGMENTS" else f"remove expandable_segments from {name}"
    )
    assert f"To fix this, {fix}, or keep the workers on the GPU" in message


@pytest.mark.unit
def test_the_refusal_names_the_train_env_vars_that_reach_the_workers() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_training_worker_environment

    trainer = component_args(
        offload_train=True, train_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, **NO_CPU_COPY
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in --train-env-vars, "
            "SLIME_ENABLE_EXPANDABLE_SEGMENTS=1 in the environment)"
        ),
    ):
        validate_training_worker_environment(
            trainer, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512", "SLIME_ENABLE_EXPANDABLE_SEGMENTS": "1"}
        )
    # Slime's --train-env-vars go over the environment.
    trainer.train_env_vars = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}
    validate_training_worker_environment(trainer, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


@pytest.mark.unit
@pytest.mark.parametrize("environ", EXPANDABLE_SEGMENTS_ENVIRONMENTS)
def test_a_trainer_that_keeps_its_weights_keeps_expandable_segments(environ) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_training_worker_environment

    # The trainer stays on the GPU.
    validate_training_worker_environment(component_args(rollout_num_gpus=2, **NO_CPU_COPY), environ)
    # The trainer sleeps, but every page keeps its CPU copy.
    validate_training_worker_environment(component_args(use_critic=True, offload_train=True), environ)
    # An offloaded trainer keeps the allocator settings that do not turn them on.
    validate_training_worker_environment(
        component_args(offload_train=True, **NO_CPU_COPY),
        {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512", "SLIME_ENABLE_EXPANDABLE_SEGMENTS": "0"},
    )


@pytest.mark.unit
def test_a_critic_refuses_expandable_segments_only_when_slime_turns_off_the_cpu_copy() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_training_worker_environment

    environ = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    # Slime's ppo critic offloads the trainer without the CPU copy.
    ppo = slime_args(advantage_estimator="ppo", use_critic=True, offload_train=True, **NO_CPU_COPY)
    finalize_reef_slime_args(ppo, [])
    with pytest.raises(ValueError, match="cannot use expandable segments"):
        validate_training_worker_environment(ppo, environ)
    # --use-critic with --no-offload-train keeps the actor and the critic on the GPU.
    resident = slime_args(advantage_estimator="ppo", use_critic=True, offload_train=True, **NO_CPU_COPY)
    finalize_reef_slime_args(resident, ["--use-critic", "--no-offload-train"])
    validate_training_worker_environment(resident, environ)
    # Reef's --use-critic offloads a grpo trainer, as SAO's imo_answerbench example does, with the CPU copy.
    offloaded = slime_args()
    finalize_reef_slime_args(offloaded, ["--use-critic"])
    assert offloaded.offload_train is True
    validate_training_worker_environment(offloaded, environ)


@pytest.mark.unit
@pytest.mark.parametrize("interval", [0, -1])
def test_finalize_arguments_rejects_nonpositive_critic_save_interval(interval: int) -> None:
    with pytest.raises(ValueError, match="critic-save-interval"):
        finalize_reef_slime_args(slime_args(critic_save_interval=interval), [])


@pytest.mark.unit
def test_finalize_arguments_refuses_no_load_optim() -> None:
    # Each job's driver loads the latest checkpoint, so the flag would restart Adam at every job.
    with pytest.raises(ValueError, match="remove --no-load-optim"):
        finalize_reef_slime_args(slime_args(), ["--no-load-optim"])


@pytest.mark.unit
@pytest.mark.parametrize("save", [None, " "])
def test_finalize_arguments_requires_save(save: str | None) -> None:
    # Megatron checks --save-interval only with --save, and the bridge reads --save-interval.
    with pytest.raises(ValueError, match="requires --save"):
        finalize_reef_slime_args(slime_args(save=save), [])


@pytest.mark.unit
@pytest.mark.parametrize("interval", [0, -1])
def test_checkpoint_preflight_preserves_invalid_interval_for_validation(tmp_path, interval: int) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import prepare_checkpoint_storage
    from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionSettings

    args = SlimeArguments(
        save_hf=str(tmp_path / "hf" / "{rollout_id}"),
        save=str(tmp_path / "megatron"),
        use_critic=False,
        hf_checkpoint=None,
        load=None,
        megatron_lora_rank=0,
        critic_save_interval=interval,
    )
    with pytest.raises(ValueError, match="critic_save_interval"):
        prepare_checkpoint_storage(args, RetentionSettings())
    assert not (tmp_path / "megatron").exists()


@pytest.mark.unit
def test_slime_parses_tensor_parallel_arguments_without_the_workers_cuda_variable(monkeypatch) -> None:
    """Megatron's check wants CUDA_DEVICE_MAX_CONNECTIONS=1 in the parsing process: the HTTP service and each driver."""
    from reef.train.slime_backend.reef_adapters import slime_arguments, startup
    from reef.train.slime_backend.reef_adapters.megatron import hf_arguments

    seen: list[str | None] = []

    def parse_args(add_custom_arguments):
        import os

        seen.append(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"))
        return argparse.Namespace()

    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", SimpleNamespace(parse_args=parse_args))
    monkeypatch.setattr(hf_arguments, "add_hf_architecture_arguments", lambda arguments: list(arguments))
    monkeypatch.setattr(slime_arguments, "finalize_reef_slime_args", lambda args, arguments: None)
    startup.parse_slime_args(["--tensor-model-parallel-size=4"])
    assert seen == ["1"]
