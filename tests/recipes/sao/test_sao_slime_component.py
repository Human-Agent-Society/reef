"""Container-only tests for SAO on Slime: its wire rows and its jobs on the Slime training component.

These exercise the SAO payload builder (``build_sao_rollout_data`` in
``recipes.sao.slime``), SAO jobs on ``SlimeTrainingComponent`` over fake
actor and critic groups, the critic's arguments and its checkpoint storage.
They transitively import ``ray`` and ``torch`` and touch no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory
from tests.support.slime_component import FakeTrainGroup, RecordingBatchProcessor, checkpoint_storage, finished

from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import JobStatus, TrainingJob
from reef.train.slime_backend.component import SlimeTrainingComponent
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.types import JobData


def _sao_row(
    source_id: str = "i1",
    *,
    tokens: list[int] | None = None,
    loss_mask: list[int] | None = None,
    action_mask: list[int] | None = None,
    log_probs: list[float] | None = None,
    reward: float = 0.5,
    producing_runtime_load_id: str | None = "slime-v3",
    rollout_created_at: float | None = 1234.5,
) -> list[object]:
    """One 8-element SAO row as emitted by Reef's remote handle."""
    tokens = [9, 1, 2, 3] if tokens is None else tokens
    loss_mask = [1, 1, 1] if loss_mask is None else loss_mask
    action_mask = list(loss_mask) if action_mask is None else action_mask
    log_probs = [-0.1, -0.2, -0.3] if log_probs is None else log_probs
    return [
        source_id,
        tokens,
        loss_mask,
        log_probs,
        reward,
        action_mask,
        producing_runtime_load_id,
        rollout_created_at,
    ]


def _payload(rows: list[list[object]], rollout_ids: list[int] | None = None) -> dict:
    return {
        "samples": rows,
        "rollout_ids": list(range(len(rows))) if rollout_ids is None else rollout_ids,
        "loss": "sao",
        "method": TrainingMethod("sao").to_dict(),
        # The rows train as one optimizer step.
        "external_step_sizes": [len(rows)],
    }


@pytest.mark.unit
def test_sao_parser_converts_rows_and_preserves_source_fields() -> None:
    converted = to_slime_rollout_data(
        _payload([_sao_row("a"), _sao_row("b", reward=1.0, producing_runtime_load_id="slime-v4")])
    )

    assert converted["loss"] == "sao"
    assert converted["tokens"] == [[9, 1, 2, 3], [9, 1, 2, 3]]
    assert converted["loss_masks"] == [[1, 1, 1], [1, 1, 1]]
    assert converted["action_masks"] == [[1, 1, 1], [1, 1, 1]]
    assert converted["rewards"] == [0.5, 1.0]
    assert converted["response_lengths"] == [3, 3]
    assert converted["rollout_log_probs"] == [[-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]]
    # The producing version and timestamp support policy lag / queue age.
    assert converted["producing_runtime_load_ids"] == ["slime-v3", "slime-v4"]
    assert converted["rollout_created_ats"] == [1234.5, 1234.5]
    # No comparison-group barrier: one rollout per id, and no advantages.
    assert converted["rollout_ids"] == [0, 1]
    assert "advantages" not in converted


@pytest.mark.unit
def test_sao_parser_keeps_observation_boundaries_distinct_from_loss_mask() -> None:
    converted = to_slime_rollout_data(_payload([_sao_row(loss_mask=[1, 0, 1], action_mask=[1, 0, 1])]))

    assert converted["loss_masks"] == [[1, 0, 1]]
    assert converted["action_masks"] == [[1, 0, 1]]


@pytest.mark.unit
def test_sao_parser_rejects_wrong_row_length() -> None:
    short_row = _sao_row()[:-1]  # 7 elements
    with pytest.raises(ValueError, match="SAO sample 0 must be"):
        to_slime_rollout_data(_payload([short_row]))


@pytest.mark.unit
def test_sao_parser_rejects_action_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="action_mask length"):
        to_slime_rollout_data(_payload([_sao_row(loss_mask=[1, 1, 1], action_mask=[1, 1])]))


@pytest.mark.unit
def test_sao_parser_rejects_training_a_non_action_token() -> None:
    # loss_mask trains index 1, action_mask marks it observation. Skip-obs GAE
    # would give it zero advantage: refuse rather than waste gradient.
    with pytest.raises(ValueError, match="not an action token"):
        to_slime_rollout_data(_payload([_sao_row(loss_mask=[1, 1, 1], action_mask=[1, 0, 1])]))


@pytest.mark.unit
def test_sao_parser_rejects_empty_and_all_zero_masks() -> None:
    with pytest.raises(ValueError, match="empty loss_mask"):
        to_slime_rollout_data(_payload([_sao_row(loss_mask=[], action_mask=[], log_probs=[])]))
    with pytest.raises(ValueError, match="at least one response token"):
        to_slime_rollout_data(_payload([_sao_row(loss_mask=[0, 0, 0], action_mask=[0, 0, 0])]))


@pytest.mark.unit
def test_sao_parser_rejects_logprob_length_mismatch() -> None:
    with pytest.raises(ValueError, match="rollout_log_probs length"):
        to_slime_rollout_data(_payload([_sao_row(loss_mask=[1, 1, 1], log_probs=[-0.1, -0.2])]))


@pytest.mark.unit
def test_sao_parser_rejects_prompt_shorter_than_response() -> None:
    # tokens must carry at least one prompt token beyond the response tokens.
    with pytest.raises(ValueError, match="at least one prompt token"):
        to_slime_rollout_data(_payload([_sao_row(tokens=[1, 2, 3], loss_mask=[1, 1, 1], action_mask=[1, 1, 1])]))


@pytest.mark.unit
def test_sao_parser_requires_rollout_ids_for_every_sample() -> None:
    with pytest.raises(ValueError, match="rollout_ids length"):
        to_slime_rollout_data(_payload([_sao_row("a"), _sao_row("b")], rollout_ids=[0]))


@pytest.mark.unit
def test_sao_parser_tolerates_missing_source_fields() -> None:
    converted = to_slime_rollout_data(_payload([_sao_row(producing_runtime_load_id=None, rollout_created_at=None)]))

    assert converted["producing_runtime_load_ids"] == [None]
    assert converted["rollout_created_ats"] == [None]


@pytest.mark.unit
def test_sao_row_versions_must_match_the_shared_producing_versions() -> None:
    payload = {
        **_payload([_sao_row(producing_runtime_load_id="engine:1")]),
        "producing_runtime_load_ids": ["engine:2"],
    }
    with pytest.raises(ValueError, match="row runtime load IDs do not match"):
        to_slime_rollout_data(payload)


@pytest.mark.unit
def test_sao_rollout_metrics_reports_version_spread_queue_age_and_effective_tokens() -> None:
    sao = resolve_loss_family("sao")
    rollout_data = to_slime_rollout_data(
        _payload(
            [
                _sao_row("a", loss_mask=[1, 1, 1], action_mask=[1, 1, 1], producing_runtime_load_id="inc:2"),
                _sao_row("b", loss_mask=[1, 0, 1], action_mask=[1, 1, 1], producing_runtime_load_id="inc:4"),
            ]
        )
    )

    metrics = sao.rollout_metrics(rollout_data)

    # The spread counts versions behind the newest producing version in the rollout's data.
    assert metrics["sao/producing_version_spread_max"] == 2
    assert metrics["sao/producing_version_spread_mean"] == 1.0
    assert metrics["sao/effective_token_rate"] == 5 / 6
    assert metrics["sao/queue_age_s_max"] >= 0.0


@pytest.mark.unit
def test_sao_rollout_metrics_skip_the_spread_of_versions_without_a_sequence() -> None:
    sao = resolve_loss_family("sao")
    rollout_data = to_slime_rollout_data(_payload([_sao_row("a", producing_runtime_load_id="no-sequence")]))

    metrics = sao.rollout_metrics(rollout_data)

    assert "sao/producing_version_spread_max" not in metrics
    assert metrics["sao/effective_token_rate"] == 1.0


class SaoStack:
    """A Slime component whose workers started with SAO, over fake actor and critic groups."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        critic: bool = True,
        critic_steps_per_actor: int | None = 2,
        critic_only_steps: int = 0,
        critic_save_interval: int = 1,
        rank_metrics=None,
    ) -> None:
        template = str(tmp_path / "hf" / "{rollout_id}")
        critic_root = tmp_path / "megatron-critic" if critic else None
        storage = checkpoint_storage(tmp_path, critic_root=critic_root, critic_save_interval=critic_save_interval)
        args = SlimeArguments(
            loss_family="sao",
            global_batch_size=2,
            save_hf=template,
            start_rollout_id=0,
            critic_save=None if critic_root is None else str(critic_root),
            critic_save_interval=critic_save_interval,
            critic_steps_per_actor=critic_steps_per_actor,
            num_critic_only_steps=critic_only_steps,
            decoupled_lr=None,
            score_centering=False,
            reef_rollout_tensor_dtypes={},
        )
        self.actor = FakeTrainGroup(template, megatron_root=storage.megatron_root, rank_metrics=rank_metrics)
        self.actor.train_result = [{"values": [0.25]}]
        self.critic = None if critic_root is None else FakeTrainGroup(None, megatron_root=critic_root)
        if self.critic is not None:
            self.critic.train_result = [{"values": [0.25]}]
        self.critic_external_data: list[object] = []
        self.component = SlimeTrainingComponent(
            args,
            self.actor,
            self.critic,
            batch_processor=RecordingBatchProcessor(),
            storage=storage,
            lora=False,
            loss_family="sao",
        )
        self.start: str | None = None

    def run(self, step: int = 0) -> JobStatus:
        batch = JobData(
            f"batch-{step}",
            tuple(
                policy_trajectory(
                    f"s{step}-{index}",
                    (9, 1, 2, 3),
                    (1, 1, 1),
                    (-0.1, -0.2, -0.3),
                    reward,
                    producing,
                    action_mask=(1, 1, 1),
                    rollout_created_at=1234.5,
                )
                for index, (reward, producing) in enumerate(((0.5, "inc:4"), (1.0, "inc:3")))
            ),
        )
        method = {"method": TrainingMethod("sao").to_dict()}
        job_id = f"job-{step}"
        self.component.submit(TrainingJob(job_id, "slime", self.start, batch, method))
        status = finished(self.component, job_id)
        if status.checkpoints:
            self.start = status.checkpoints[-1].path
        return status


@pytest.fixture
def sao_stacks():
    built: list[SaoStack] = []

    def build(tmp_path: Path, **options) -> SaoStack:
        stack = SaoStack(tmp_path, **options)
        built.append(stack)
        return stack

    yield build
    for stack in built:
        stack.component.close()


def checkpoint_metrics(status: JobStatus):
    assert status.state == "finished", status.error
    (checkpoint,) = status.checkpoints
    return checkpoint.metrics


@pytest.mark.unit
def test_a_sao_job_runs_k_critic_steps_then_the_actor(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path)

    metrics = checkpoint_metrics(stack.run())

    # Two-time-scale update: K critic steps, then one actor step fed the last critic's values.
    assert len(stack.critic.train_calls) == 2
    assert len(stack.actor.train_calls) == 1
    assert metrics["sao/critic_updates"] == 2
    assert metrics["sao/actor_trained"] == 1
    assert metrics["sao/effective_token_rate"] == 1.0


@pytest.mark.unit
def test_a_sao_critic_only_warmup_saves_without_moving_the_policy(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path, critic_only_steps=1)

    metrics = checkpoint_metrics(stack.run())

    assert len(stack.critic.train_calls) == 2
    assert stack.actor.train_calls == []
    assert metrics["sao/actor_trained"] == 0


@pytest.mark.unit
def test_consecutive_sao_jobs_train_and_save_consecutive_rollouts(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path)
    stack.run(0)
    stack.run(1)
    assert [rollout_id for rollout_id, _ in stack.actor.train_calls] == [0, 1]
    assert stack.actor.saved == [(0, True), (1, True)]


@pytest.mark.unit
def test_ten_valid_batches_warm_up_critic_before_first_actor_update(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path, critic_only_steps=10)
    for step in range(11):
        assert checkpoint_metrics(stack.run(step))["sao/actor_trained"] == int(step == 10)
    assert len(stack.critic.train_calls) == 22
    assert [rollout_id for rollout_id, _ in stack.actor.train_calls] == [10]


@pytest.mark.unit
def test_a_sao_checkpoint_carries_the_backend_and_worker_metrics(tmp_path, sao_stacks) -> None:
    # Slime is only a producer here: Reef records the checkpoint's metrics and decides who sees them.
    stack = sao_stacks(tmp_path, rank_metrics=[{"slime/lr": 1e-6}])

    metrics = checkpoint_metrics(stack.run())

    assert metrics["slime/lr"] == pytest.approx(1e-6)
    assert set(metrics) == {
        "slime/lr",
        "values",
        "sao/critic_updates",
        "sao/actor_trained",
        "sao/producing_version_spread_max",
        "sao/producing_version_spread_mean",
        "sao/queue_age_s_max",
        "sao/queue_age_s_mean",
        "sao/effective_token_rate",
    }


@pytest.mark.unit
@pytest.mark.parametrize("critic_only_steps", [0, 1])
def test_a_sao_rollout_saves_the_critic_alongside_the_actor(tmp_path, sao_stacks, critic_only_steps) -> None:
    # The warmup steps are exactly the ones whose lost progress the paper's
    # cold-start concern is about: a warmup rollout persists the critic too.
    stack = sao_stacks(tmp_path, critic_only_steps=critic_only_steps)

    stack.run()

    assert stack.actor.saved == [(0, True)]
    # The critic's weights and optimizer persist on the same rollout, so a restart keeps the value head.
    assert stack.critic.saved == [(0, True)]


@pytest.mark.unit
def test_the_sao_critic_save_interval_skips_the_rollouts_in_between(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path, critic_save_interval=2)

    stack.run(0)
    stack.run(1)

    assert stack.actor.saved == [(0, True), (1, True)]
    # The second rollout is the first that the cadence saves the critic on.
    assert stack.critic.saved == [(1, True)]
    with pytest.raises(ValueError, match="critic_save_interval"):
        SaoStack(tmp_path / "again", critic_save_interval=0)


@pytest.mark.unit
def test_the_default_critic_cadence_matches_the_paper(tmp_path, sao_stacks) -> None:
    # Without an explicit --critic-steps-per-actor the component trains the paper's K=2.
    stack = sao_stacks(tmp_path, critic_steps_per_actor=None)

    metrics = checkpoint_metrics(stack.run())

    assert metrics["sao/critic_updates"] == 2
    assert len(stack.critic.train_calls) == 2


@pytest.mark.unit
def test_sao_requires_a_value_model(tmp_path, sao_stacks) -> None:
    stack = sao_stacks(tmp_path, critic=False)

    status = stack.run()

    assert status.state == "failed" and "SAO requires a value model" in status.error
    assert status.checkpoints == ()


def _install_slime_parser_stubs() -> None:
    """Stub the GPU-only modules ``utils.arguments`` imports at module scope.

    Only genuinely missing modules are stubbed (mirrors
    ``test_sao_configs._install_slime_parser_stubs``); inside the full Slime
    image the real modules win.
    """
    import importlib.machinery
    import sys
    import types

    stubs: dict[str, dict[str, object]] = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.server_args": {"ServerArgs": type("ServerArgs", (), {})},
        "sglang_router": {},
        "sglang_router.launch_router": {"RouterArgs": type("RouterArgs", (), {})},
        "wandb": {},
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            # A bare ModuleType has __spec__ set to None, and these stubs
            # stay in sys.modules after the test that installed them. Later
            # code that asks the import system about the name then fails:
            # importing peft reaches accelerate, which looks up wandb this way.
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module


def critic_prep_args(**overrides):
    from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments

    values = {
        "megatron_config_path": None,
        "loss_family": "sao",
        "custom_advantage_function_path": "recipes.sao.slime.objective.sao_advantages",
        "num_experts": 0,
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "sao_length_adaptive_lambda": True,
        "sao_lambda_alpha": 1.5,
        "sao_critic_lambda": 1.0,
        "lambd": 1.0,
        "critic_save": None,
        "critic_init": None,
        "critic_lr": None,
        "megatron_lora_rank": 0,
        "custom_model_provider_path": None,
        "custom_megatron_init_path": None,
        "load": "/ckpt/megatron",
        "save": "/ckpt/megatron",
        "no_load_optim": True,
        "no_load_rng": True,
        "finetune": True,
        "ckpt_step": None,
        "disable_param_buffers_cpu_backup": True,
    }
    values.update(overrides)
    return SlimeArguments(**values)


@pytest.mark.unit
def test_prepare_critic_args_keeps_the_custom_advantage_path_and_pins_lambda() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    args = critic_prep_args()
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    # Role-separated lambda policy: adaptive lambda off, lambd = critic lambda.
    assert critic_args.sao_length_adaptive_lambda is False
    assert critic_args.lambd == 1.0
    # The actor args are untouched.
    assert args.sao_length_adaptive_lambda is True


@pytest.mark.unit
def test_prepare_critic_args_starts_the_critic_from_an_init_checkpoint_until_it_has_its_own(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    critic_save, critic_init = tmp_path / "megatron-critic", tmp_path / "critic-init"
    critic_init.mkdir()
    (critic_init / "latest_checkpointed_iteration.txt").write_text("23", encoding="utf-8")

    # No checkpoint of its own yet: the value model trained on an earlier
    # rollout is loaded, weights and optimizer, and saves go to the save root.
    args = critic_prep_args(critic_save=str(critic_save), critic_init=str(critic_init))
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)
    assert (critic_args.load, critic_args.save) == (str(critic_init), str(critic_save))
    assert (critic_args.no_load_optim, critic_args.finetune) == (False, False)
    # The schedule is this run's: the earlier run's batch size sets a different iteration total.
    assert critic_args.override_opt_param_scheduler is True

    # Once the critic has saved, its own checkpoint wins over the init.
    critic_save.mkdir()
    (critic_save / "latest_checkpointed_iteration.txt").write_text("7", encoding="utf-8")
    critic_args = prepare_critic_args(critic_prep_args(critic_save=str(critic_save), critic_init=str(critic_init)))
    assert critic_args.load == str(critic_save)

    # An explicitly requested initialization must not silently start cold.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=r"critic-init.*has no latest"):
        prepare_critic_args(critic_prep_args(critic_save=str(tmp_path / "fresh"), critic_init=str(empty)))


@pytest.mark.unit
def test_prepare_critic_args_applies_the_critic_learning_rate() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    args = critic_prep_args(critic_lr=5e-6, lr=1e-6)
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    assert critic_args.lr == 5e-6
    assert args.lr == 1e-6  # the actor keeps the policy lr


@pytest.mark.unit
def test_prepare_critic_args_inherits_the_policy_lr_when_unset() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    args = critic_prep_args(lr=1e-6)
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    assert critic_args.lr == 1e-6


@pytest.mark.unit
def test_prepare_critic_args_keeps_lora_for_the_critic() -> None:
    """A LoRA actor's critic trains adapters on the same frozen base.

    The critic is the policy's size; without adapters it would train every
    parameter and could not sit beside a LoRA actor of the same model.
    """
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.slime_arguments import REEF_MODEL_PROVIDER_PATH
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    # The state finalize_reef_slime_args leaves for a LoRA actor: alpha
    # defaulted, Reef's provider installed, the user's provider chained.
    args = critic_prep_args(
        megatron_lora_rank=32,
        megatron_lora_alpha=32,
        megatron_lora_dropout=0.0,
        megatron_lora_target_modules=["linear_qkv", "linear_fc1"],
        custom_model_provider_path=REEF_MODEL_PROVIDER_PATH,
        reef_chained_model_provider_path=None,
    )
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    assert critic_args.megatron_lora_rank == 32
    assert critic_args.megatron_lora_alpha == 32
    assert critic_args.megatron_lora_target_modules == ["linear_qkv", "linear_fc1"]
    assert critic_args.custom_model_provider_path == REEF_MODEL_PROVIDER_PATH


@pytest.mark.unit
@pytest.mark.parametrize("provider", [None, "recipes.sao.examples.servelearnbench.long_context.provide_model"])
def test_prepare_critic_args_without_lora_preserves_the_user_provider(provider) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    args = critic_prep_args(megatron_lora_rank=0, custom_model_provider_path=provider)
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    assert critic_args.megatron_lora_rank == 0
    assert critic_args.megatron_lora_alpha is None
    assert critic_args.custom_model_provider_path == provider
    assert args.custom_model_provider_path == provider


@pytest.mark.unit
@pytest.mark.parametrize("provider", [None, "my.provider:build"])
def test_prepare_critic_args_without_lora_unwraps_only_the_reef_provider(provider) -> None:
    from reef.train.slime_backend.reef_adapters.slime_arguments import REEF_MODEL_PROVIDER_PATH
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    args = critic_prep_args(
        custom_model_provider_path=REEF_MODEL_PROVIDER_PATH,
        reef_chained_model_provider_path=provider,
    )
    critic_args = prepare_critic_args(args)

    assert critic_args.custom_model_provider_path == provider
    assert args.custom_model_provider_path == REEF_MODEL_PROVIDER_PATH


@pytest.mark.unit
def test_prepare_critic_args_restores_from_the_critic_root_when_present(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    critic_root = tmp_path / "megatron-critic"
    critic_root.mkdir()
    (critic_root / "latest_checkpointed_iteration.txt").write_text("7\n", encoding="utf-8")

    critic_args = prepare_critic_args(critic_prep_args(critic_save=str(critic_root)))

    assert critic_args.save == str(critic_root)
    assert critic_args.load == str(critic_root)
    assert critic_args.no_load_optim is False
    assert critic_args.no_load_rng is False
    assert critic_args.finetune is False


@pytest.mark.unit
def test_prepare_critic_args_first_boot_keeps_the_inherited_load_fallback(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    critic_root = tmp_path / "megatron-critic"  # no checkpoint tracker yet

    critic_args = prepare_critic_args(critic_prep_args(critic_save=str(critic_root)))

    # Saves go to the critic root from the first commit on, but the first boot
    # still loads from the inherited fallback (actor checkpoint / base model).
    assert critic_args.save == str(critic_root)
    assert critic_args.load == "/ckpt/megatron"
    assert critic_args.no_load_optim is True


@pytest.mark.unit
def test_megatron_config_critic_role_pins_lambda(tmp_path) -> None:
    # The --megatron-config-path role surgery must still apply SAO's
    # critic-role lambda policy even with a megatron config.
    from reef.train.slime_backend.reef_adapters.preflight import configure_critic_args_hook
    from reef.train.slime_backend.reef_adapters.train_groups import prepare_critic_args

    _install_slime_parser_stubs()
    config_path = tmp_path / "megatron.yaml"
    config_path.write_text(
        "megatron:\n  - name: critic\n    role: critic\n    overrides: {}\n",
        encoding="utf-8",
    )

    args = critic_prep_args(megatron_config_path=str(config_path))
    configure_critic_args_hook(args)
    critic_args = prepare_critic_args(args)

    assert critic_args.sao_length_adaptive_lambda is False
    assert critic_args.lambd == 1.0


def _logical_bytes(path: Path) -> int:
    if not path.exists() or path.is_symlink() or path.is_file():
        return path.lstat().st_size if path.exists() or path.is_symlink() else 0
    return sum(_logical_bytes(child) for child in path.iterdir())


def _write_bytes(path: Path, size: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").write_bytes(b"x" * size)


def _critic_storage(tmp_path: Path, *, critic: bool = True, cap: int = 1000, free: int = 1000):
    from collections import namedtuple

    from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

    usage = namedtuple("Usage", "total used free")
    root = tmp_path / "checkpoints"
    source_hf, source_megatron = tmp_path / "source-hf", tmp_path / "source-megatron"
    _write_bytes(source_hf, 10)
    _write_bytes(source_megatron / "iter_0000000", 90)
    (source_megatron / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    return CheckpointStorage(
        RetentionConfig(max_storage_bytes=cap),
        hf_template=str(root / "hf" / "{rollout_id}"),
        megatron_root=root / "megatron",
        critic_root=(root / "megatron-critic") if critic else None,
        source_hf=source_hf,
        source_megatron=source_megatron,
        measure=_logical_bytes,
        disk_usage=lambda path: usage(1000, 1000 - free, free),
    )


def _complete_with_critic(storage, rollout_id: int, *, critic_bytes: int | None = 10) -> None:
    with storage.admit(rollout_id=rollout_id) as plan:
        assert not plan["blocked"], plan
        pair = storage.pair_paths(rollout_id)
        for path, size in zip(pair, (40, 60), strict=True):
            _write_bytes(path, size)
        (storage.megatron_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id), encoding="utf-8")
        if critic_bytes is not None and storage.critic_root is not None:
            _write_bytes(storage.critic_root / f"iter_{rollout_id:07d}", critic_bytes)
            (storage.critic_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id), encoding="utf-8")
        storage.complete(f"job-{rollout_id}", rollout_id, reward=None)


@pytest.mark.unit
def test_storage_retention_deletes_the_critic_asset_with_its_pair(tmp_path) -> None:
    # cap 1000 with a 110-byte reservation and 110-byte groups: admitting
    # rollout 2 must reclaim rollout 0 — HF, Megatron, and critic assets alike
    # — while the newest group (recovery pair + critic) stays protected.
    storage = _critic_storage(tmp_path, cap=330)
    _complete_with_critic(storage, 0)
    _complete_with_critic(storage, 1)

    _complete_with_critic(storage, 2)

    for path in storage.asset_paths(0):
        assert not path.exists(), path
    for path in storage.asset_paths(2):
        assert path.is_dir(), path


@pytest.mark.unit
def test_storage_complete_requires_the_critic_asset_when_configured(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorageError

    storage = _critic_storage(tmp_path)

    with pytest.raises(CheckpointStorageError, match="missing or unsafe"):
        _complete_with_critic(storage, 0, critic_bytes=None)


@pytest.mark.unit
def test_storage_records_from_before_the_critic_root_stay_valid(tmp_path) -> None:
    # Upgrade path: pairs recorded by an actor-only deployment have no critic
    # asset. They must stay valid (the recovery-pair invariant is about the HF
    # + Megatron pair) and reclaimable after a critic root is configured.
    plain = _critic_storage(tmp_path, critic=False)
    _complete_with_critic(plain, 0)

    upgraded = _critic_storage(tmp_path, cap=330)
    _complete_with_critic(upgraded, 1)
    _complete_with_critic(upgraded, 2)
    _complete_with_critic(upgraded, 3)

    # Rollout 0 (pair-only) was reclaimed without tripping on the missing
    # critic asset; the newest group survives whole.
    for path in upgraded.pair_paths(0):
        assert not path.exists(), path
    for path in upgraded.asset_paths(3):
        assert path.is_dir(), path


@pytest.mark.unit
def test_storage_flags_unknown_assets_in_the_critic_root(tmp_path) -> None:
    storage = _critic_storage(tmp_path)
    _complete_with_critic(storage, 0)
    _write_bytes(storage.critic_root / "not-a-checkpoint", 5)

    plan = storage.validate_capacity()

    assert plan["blocked"]
    assert any("unowned" in reason for reason in plan["reasons"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
