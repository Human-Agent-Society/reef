"""Container-only tests for the SAO branch of the Reef -> Slime bridge.

These exercise the SAO payload builder (``recipes.sao.slime.utils.data_builder``) and
the SAO path through
``execute_training_job``, which live in the Slime plugin package and
transitively import ``ray`` and ``torch``. Like ``test_slime_bridge.py`` they
need the full Slime image and are excluded from the minimal CI gate (see
``.github/workflows/ci.yml``); run them in the ``slime`` container. They are
deterministic Python and touch no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters import bridge


def _sao_row(
    source_id: str = "i1",
    *,
    tokens: list[int] | None = None,
    loss_mask: list[int] | None = None,
    action_mask: list[int] | None = None,
    log_probs: list[float] | None = None,
    reward: float = 0.5,
    producing_weight_version: str | None = "slime-v3",
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
        producing_weight_version,
        rollout_created_at,
    ]


def _payload(rows: list[list[object]], rollout_ids: list[int] | None = None) -> dict:
    return {
        "samples": rows,
        "rollout_ids": list(range(len(rows))) if rollout_ids is None else rollout_ids,
        "loss": "sao",
    }


def _execute_and_update_weights(actor, payload):
    result = actor.execute_training_job(payload)
    if result.outcome != "checkpoint":
        return result
    assert result.training_job_id is not None
    return actor.update_serving_weights(result.training_job_id)


@pytest.mark.unit
def test_sao_parser_converts_rows_and_preserves_provenance() -> None:
    converted = to_slime_rollout_data(
        _payload([_sao_row("a"), _sao_row("b", reward=1.0, producing_weight_version="slime-v4")])
    )

    assert converted["loss"] == "sao"
    assert converted["tokens"] == [[9, 1, 2, 3], [9, 1, 2, 3]]
    assert converted["loss_masks"] == [[1, 1, 1], [1, 1, 1]]
    assert converted["action_masks"] == [[1, 1, 1], [1, 1, 1]]
    assert converted["rewards"] == [0.5, 1.0]
    assert converted["response_lengths"] == [3, 3]
    assert converted["rollout_log_probs"] == [[-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]]
    # Provenance rides through for policy lag / queue age.
    assert converted["producing_weight_versions"] == ["slime-v3", "slime-v4"]
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
def test_sao_parser_tolerates_missing_provenance() -> None:
    converted = to_slime_rollout_data(_payload([_sao_row(producing_weight_version=None, rollout_created_at=None)]))

    assert converted["producing_weight_versions"] == [None]
    assert converted["rollout_created_ats"] == [None]


@pytest.mark.unit
def test_weight_version_sequence_parses_matching_incarnation() -> None:
    from recipes.sao.slime import _weight_version_sequence

    assert _weight_version_sequence("abc:7", "abc") == 7


@pytest.mark.unit
def test_weight_version_sequence_is_none_across_incarnations() -> None:
    from recipes.sao.slime import _weight_version_sequence

    assert _weight_version_sequence("old:7", "new") is None
    assert _weight_version_sequence("no-colon", "no-colon") is None


@pytest.mark.unit
def test_sao_provenance_metrics_reports_lag_queue_age_and_effective_tokens() -> None:
    sao = resolve_loss_family("sao")
    rollout_data = to_slime_rollout_data(
        _payload(
            [
                _sao_row("a", loss_mask=[1, 1, 1], action_mask=[1, 1, 1], producing_weight_version="inc:2"),
                _sao_row("b", loss_mask=[1, 0, 1], action_mask=[1, 1, 1], producing_weight_version="inc:4"),
            ]
        )
    )

    metrics = sao.provenance_metrics(rollout_data, serving_version="inc:5")

    assert metrics["sao/policy_lag_max"] == 3
    assert metrics["sao/policy_lag_mean"] == 2.0
    assert metrics["sao/effective_token_rate"] == 5 / 6
    assert metrics["sao/queue_age_s_max"] >= 0.0


@pytest.mark.unit
def test_sao_provenance_metrics_warns_when_dropping_future_producing_steps(caplog) -> None:
    sao = resolve_loss_family("sao")
    rollout_data = to_slime_rollout_data(
        _payload(
            [
                _sao_row("a", producing_weight_version="inc:2"),
                _sao_row("b", producing_weight_version="inc:9"),
            ]
        )
    )

    with caplog.at_level("WARNING", logger="recipes.sao.slime"):
        metrics = sao.provenance_metrics(rollout_data, serving_version="inc:5")

    assert metrics["sao/policy_lag_max"] == 3
    assert any("ahead of serving step 5" in record.message for record in caplog.records)


@pytest.mark.unit
def test_sao_provenance_metrics_skips_lag_across_incarnation() -> None:
    sao = resolve_loss_family("sao")
    rollout_data = to_slime_rollout_data(_payload([_sao_row("a", producing_weight_version="oldinc:2")]))

    metrics = sao.provenance_metrics(rollout_data, serving_version="newinc:5")

    assert "sao/policy_lag_max" not in metrics
    assert metrics["sao/effective_token_rate"] == 1.0


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRank:
    def __init__(self, version="v1", metrics=None):
        self.version = version
        self.metrics = {} if metrics is None else metrics
        self.get_weight_version = _RemoteMethod(lambda: self.version)
        self.pop_metrics = _RemoteMethod(lambda: self.metrics)


# The serving head and the rows' producing head share an incarnation so policy
# lag is defined; ``_weight_version_sequence`` drops lag across a restart.
SERVING_VERSION = "inc:5"


class _FakeRolloutManager:
    def __init__(self, packed):
        self.packed = packed
        self.prepare_external_train_data = _RemoteMethod(lambda data: "packed-ref")
        self.inference_url = _RemoteMethod(lambda: "http://10.0.0.7:30000")
        self.get_weight_versions = _RemoteMethod(lambda: [SERVING_VERSION])
        self.terminate_updatable_engines = _RemoteMethod(lambda: 1)
        self.pause_generation_for_update = _RemoteMethod(lambda: None)
        self.continue_generation_after_update = _RemoteMethod(lambda: None)


class _RecordingGroup:
    """Actor or critic group that records its ``async_train`` calls."""

    def __init__(self, template: str, worker_metrics=None, *, critic: bool = False):
        self.template = template
        self.critic = critic
        self.train_calls: list[tuple[int, object]] = []
        self.external_data: list[object] = []
        self.saved_model_rollouts: list[int] = []
        self.saved_training_checkpoint_rollouts: list[int] = []
        self._actor_handlers = [_FakeRank(version=SERVING_VERSION, metrics=worker_metrics)]

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append((rollout_id, rollout_data_ref))
        self.external_data.append(external_data)
        return [{"values": [0.25]}]

    def async_pop_rank0_metrics(self):
        return self._actor_handlers[0].pop_metrics.remote()

    def async_get_rank0_weight_version(self):
        return self._actor_handlers[0].get_weight_version.remote()

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
        del force_full, manage_generation

    def restore_weight_version_for_republication(self, weight_version):
        pass

    def save_model(self, rollout_id, force_sync=False):
        if self.critic:
            self.saved_training_checkpoint_rollouts.append(rollout_id)
            return
        self.saved_model_rollouts.append(rollout_id)
        checkpoint = Path(self.template.format(rollout_id=rollout_id))
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("hf", encoding="utf-8")


def _sao_actor(
    tmp_path,
    *,
    worker_metrics=None,
    critic_steps_per_actor=2,
    critic_only_steps=0,
    critic_save_root=None,
):
    template = str(tmp_path / "checkpoint-{rollout_id}")
    actor_group = _RecordingGroup(template, worker_metrics=worker_metrics)
    critic_group = _RecordingGroup(template, critic=True)
    actor = bridge.TrainBridgeActorImpl(
        actor_group,
        _FakeRolloutManager(["packed"]),
        save_hf_template=template,
        critic_group=critic_group,
        critic_save_root=critic_save_root,
        critic_steps_per_actor=critic_steps_per_actor,
        critic_only_steps=critic_only_steps,
        loss_family="sao",
    )
    payload = _payload(
        [
            _sao_row("a", producing_weight_version="inc:4"),
            _sao_row("b", reward=1.0, producing_weight_version="inc:3"),
        ]
    )
    payload.update(rollout_id=0, expected_weight_version=SERVING_VERSION)
    return actor, actor_group, critic_group, payload


@pytest.fixture
def _local_ray_get(monkeypatch):
    monkeypatch.setattr(bridge.ray, "get", lambda value, **kwargs: value)


@pytest.mark.unit
def test_sao_durable_job_runs_k_critic_steps_then_the_actor(tmp_path, _local_ray_get) -> None:
    actor, actor_group, critic_group, payload = _sao_actor(tmp_path)

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    # Two-time-scale update: K critic steps, then one actor step fed the last
    # critic's values through external_data.
    assert len(critic_group.train_calls) == 2
    assert len(actor_group.train_calls) == 1
    assert actor_group.external_data == [[{"values": [0.25]}]]
    assert result.metrics["sao/critic_updates"] == 2
    assert result.metrics["sao/actor_trained"] == 1
    assert result.metrics["sao/effective_token_rate"] == 1.0


@pytest.mark.unit
def test_sao_critic_only_warmup_commits_without_moving_the_policy(tmp_path, _local_ray_get) -> None:
    actor, actor_group, critic_group, payload = _sao_actor(tmp_path, critic_only_steps=1)

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert len(critic_group.train_calls) == 2
    assert actor_group.train_calls == []
    assert result.metrics["sao/actor_trained"] == 0


@pytest.mark.unit
def test_sao_durable_record_returns_backend_and_worker_metrics(tmp_path, _local_ray_get) -> None:
    # Slime is only a producer here: worker metrics are returned in the generic
    # TrainingJobResult and Dispatcher decides which experiment provider sees them.
    actor, _, _, payload = _sao_actor(tmp_path, worker_metrics={"slime/lr": 1e-6})

    result = _execute_and_update_weights(actor, payload)

    assert result.metrics["slime/lr"] == pytest.approx(1e-6)
    assert set(result.metrics) == {
        "slime/lr",
        "values",
        "sao/critic_updates",
        "sao/actor_trained",
        "sao/policy_lag_max",
        "sao/policy_lag_mean",
        "sao/queue_age_s_max",
        "sao/queue_age_s_mean",
        "sao/effective_token_rate",
    }
    assert actor.health()["last_train_metrics"]["slime/lr"] == 1e-6


@pytest.mark.unit
def test_sao_commit_saves_the_critic_alongside_the_actor(tmp_path, _local_ray_get) -> None:
    actor, actor_group, critic_group, payload = _sao_actor(
        tmp_path, critic_save_root=str(tmp_path / "megatron-critic")
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert actor_group.saved_model_rollouts == [0]
    # The critic's weights + optimizer are persisted on the same commit — a
    # restart must not cold-start the value head.
    assert critic_group.saved_training_checkpoint_rollouts == [0]
    # The critic never serves; it must not export an HF checkpoint.
    assert critic_group.saved_model_rollouts == []


@pytest.mark.unit
def test_sao_critic_only_warmup_commit_also_saves_the_critic(tmp_path, _local_ray_get) -> None:
    # The warmup steps are exactly the ones whose lost progress the paper's
    # cold-start concern is about: a warmup commit must persist the critic too.
    actor, _actor_group, critic_group, payload = _sao_actor(
        tmp_path, critic_only_steps=1, critic_save_root=str(tmp_path / "megatron-critic")
    )

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert result.metrics["sao/actor_trained"] == 0
    assert critic_group.saved_training_checkpoint_rollouts == [0]


@pytest.mark.unit
def test_sao_without_a_critic_root_keeps_the_actor_only_save(tmp_path, _local_ray_get) -> None:
    # Without a distinct critic root the critic's Megatron save would land in
    # the actor's tree; the bridge must then fall back to actor-only saves.
    actor, actor_group, critic_group, payload = _sao_actor(tmp_path, critic_save_root=None)

    result = _execute_and_update_weights(actor, payload)

    assert result.outcome == "complete"
    assert actor_group.saved_model_rollouts == [0]
    assert critic_group.saved_training_checkpoint_rollouts == []


@pytest.mark.unit
def test_bridge_defaults_match_the_paper_critic_cadence(tmp_path, _local_ray_get) -> None:
    # Constructing the bridge without an explicit critic_steps_per_actor must
    # yield the paper's K=2 (matching the --critic-steps-per-actor default and
    # the SaoSchedule dataclass default).
    template = str(tmp_path / "checkpoint-{rollout_id}")
    actor_group = _RecordingGroup(template)
    critic_group = _RecordingGroup(template)
    actor = bridge.TrainBridgeActorImpl(
        actor_group,
        _FakeRolloutManager(["packed"]),
        save_hf_template=template,
        critic_group=critic_group,
        loss_family="sao",
    )
    payload = _payload([_sao_row("a", producing_weight_version="inc:4")])
    payload.update(rollout_id=0, expected_weight_version=SERVING_VERSION)

    result = _execute_and_update_weights(actor, payload)

    assert result.metrics["sao/critic_updates"] == 2
    assert len(critic_group.train_calls) == 2


def _install_slime_parser_stubs() -> None:
    """Stub the GPU-only modules ``utils.arguments`` imports at module scope.

    Only genuinely missing modules are stubbed (mirrors
    ``test_sao_configs._install_slime_parser_stubs``); inside the full Slime
    image the real modules win.
    """
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
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module


def _critic_prep_args(**overrides):
    from types import SimpleNamespace

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
        "load": "/ckpt/megatron",
        "save": "/ckpt/megatron",
        "no_load_optim": True,
        "no_load_rng": True,
        "finetune": True,
        "ckpt_step": None,
        "disable_param_buffers_cpu_backup": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_prepare_critic_args_keeps_the_custom_advantage_path_and_pins_lambda() -> None:
    from reef.train.slime_backend.reef_adapters.preflight import configure_megatron_runtime
    from reef.train.slime_backend.reef_adapters.ray_train_groups import prepare_critic_args

    args = _critic_prep_args()
    configure_megatron_runtime(args)
    critic_args = prepare_critic_args(args)

    # Role-separated lambda policy: adaptive lambda off, lambd = critic lambda.
    assert critic_args.sao_length_adaptive_lambda is False
    assert critic_args.lambd == 1.0
    # The actor args are untouched.
    assert args.sao_length_adaptive_lambda is True


@pytest.mark.unit
def test_prepare_critic_args_restores_from_the_critic_root_when_present(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.ray_train_groups import prepare_critic_args

    critic_root = tmp_path / "megatron-critic"
    critic_root.mkdir()
    (critic_root / "latest_checkpointed_iteration.txt").write_text("7\n", encoding="utf-8")

    critic_args = prepare_critic_args(_critic_prep_args(critic_save=str(critic_root)))

    assert critic_args.save == str(critic_root)
    assert critic_args.load == str(critic_root)
    assert critic_args.no_load_optim is False
    assert critic_args.no_load_rng is False
    assert critic_args.finetune is False


@pytest.mark.unit
def test_prepare_critic_args_first_boot_keeps_the_inherited_load_fallback(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.ray_train_groups import prepare_critic_args

    critic_root = tmp_path / "megatron-critic"  # no checkpoint tracker yet

    critic_args = prepare_critic_args(_critic_prep_args(critic_save=str(critic_root)))

    # Saves go to the critic root from the first commit on, but the first boot
    # still loads from the inherited fallback (actor checkpoint / base model).
    assert critic_args.save == str(critic_root)
    assert critic_args.load == "/ckpt/megatron"
    assert critic_args.no_load_optim is True


@pytest.mark.unit
def test_megatron_config_critic_role_pins_lambda(tmp_path) -> None:
    # The --megatron-config-path role surgery must still apply SAO's
    # critic-role lambda policy even with a megatron config.
    from reef.train.slime_backend.reef_adapters.preflight import configure_megatron_runtime
    from reef.train.slime_backend.reef_adapters.ray_train_groups import prepare_critic_args

    _install_slime_parser_stubs()
    config_path = tmp_path / "megatron.yaml"
    config_path.write_text(
        "megatron:\n  - name: critic\n    role: critic\n    overrides: {}\n",
        encoding="utf-8",
    )

    args = _critic_prep_args(megatron_config_path=str(config_path))
    configure_megatron_runtime(args)
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


@pytest.mark.unit
def test_sao_requires_a_value_model(tmp_path, _local_ray_get) -> None:
    template = str(tmp_path / "checkpoint-{rollout_id}")
    group = _RecordingGroup(template)
    actor = bridge.TrainBridgeActorImpl(
        group,
        _FakeRolloutManager(["packed"]),
        save_hf_template=template,
        loss_family="sao",
    )
    payload = _payload([_sao_row("a")])
    payload.update(rollout_id=0, expected_weight_version=SERVING_VERSION)

    with pytest.raises(RuntimeError, match="SAO requires a value model"):
        _execute_and_update_weights(actor, payload)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
