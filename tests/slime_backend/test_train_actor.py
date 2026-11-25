"""CPU tests of Reef's Slime training worker (``ReefMegatronTrainRayActor``).

The worker module loads with stubs for Slime's Megatron actor, its timer and
the Megatron LoRA helpers, so the tests run without a GPU or Megatron.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


class _StubTimer:
    """Slime's timer as the adapter meets it: a singleton that refuses a
    second start under one name (``assert name not in self.start_time``)."""

    _instance: _StubTimer | None = None
    started: list[str] = []
    _open: set[str] = set()

    def __new__(cls) -> _StubTimer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls.started = []
        cls._open = set()

    @contextmanager
    def context(self, name: str):
        assert name not in self._open, f"Timer {name} already started."
        self._open.add(name)
        self.started.append(name)
        try:
            yield
        finally:
            self._open.discard(name)


def _stub_timer_decorator(function):
    """Slime's ``@timer``: time the call under the function's own name."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with _StubTimer().context(function.__name__):
            return function(*args, **kwargs)

    return wrapper


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_train_actor(monkeypatch: pytest.MonkeyPatch):
    _stub_module(monkeypatch, "slime.backends.megatron_utils.actor", MegatronTrainRayActor=type("Actor", (), {}))
    _stub_module(
        monkeypatch,
        "slime.utils.reloadable_process_group",
        destroy_process_groups=lambda: None,
        reload_process_groups=lambda: None,
        monkey_patch_torch_dist=lambda: None,
    )
    _StubTimer.reset()
    _stub_module(monkeypatch, "megatron.core", mpu=types.SimpleNamespace(get_data_parallel_group=lambda: "dp"))
    _stub_module(monkeypatch, "slime.observability.timer", timer=_stub_timer_decorator, Timer=_StubTimer)
    _stub_module(
        monkeypatch, "torch_memory_saver", torch_memory_saver=types.SimpleNamespace(disable=lambda: nullcontext())
    )
    _stub_module(
        monkeypatch,
        "reef.train.slime_backend.reef_adapters.megatron.lora",
        collect_lora_train_metrics=lambda _model: {},
        megatron_lora_enabled=lambda _args: False,
        zero_megatron_lora_adapters=lambda _model: None,
    )
    _stub_module(
        monkeypatch,
        "reef.train.slime_backend.reef_adapters.megatron.hf_export",
        MegatronToHfWeightIterator=type("MegatronToHfWeightIterator", (), {}),
    )
    _stub_module(
        monkeypatch,
        "reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint",
        export_lora_adapter_tensors=lambda _iterator, _weights: [],
        save_lora_adapter_to_path=lambda *_args, **_kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "reef.train.slime_backend.reef_adapters.worker_hooks",
        drain_worker_metrics=lambda: {"train/loss": 0.5},
        record_worker_metrics=lambda _metrics: None,
        reef_node_ip_and_free_port=lambda: ("127.0.0.1", 1234),
        _loss_family_spec=lambda _args: None,
        switch_loss_family=lambda _args: None,
        resolve_tensor_dtype=lambda name: name,
    )
    name = "reef.train.slime_backend.reef_adapters._train_actor_test"
    path = (
        Path(__file__).parents[2]
        / "reef"
        / "train"
        / "slime_backend"
        / "reef_adapters"
        / "megatron"
        / "train_actor.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_full_weight_save_delegates_without_reopening_slimes_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Slime times its own save_model. A full-weight save delegates to it, so
    # the adapter must not hold a timer of the same name open across the call:
    # Slime's Timer is a singleton and asserts "Timer save_model already
    # started", which failed every non-LoRA save after a training step.
    module = _load_train_actor(monkeypatch)
    saved: list[tuple[int, bool]] = []
    base = module.ReefMegatronTrainRayActor.__mro__[1]

    def base_save_model(self, rollout_id, force_sync=False):
        with _StubTimer().context("save_model"):
            saved.append((rollout_id, force_sync))

    monkeypatch.setattr(base, "save_model", base_save_model, raising=False)
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = object()

    actor.save_model(7, force_sync=True)

    assert saved == [(7, True)]
    assert _StubTimer.started == ["save_model"]


@pytest.mark.unit
def test_lora_save_times_the_branch_that_does_not_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The LoRA branch saves the adapter itself, so it keeps the timing the
    # decorator used to provide.
    module = _load_train_actor(monkeypatch)
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda _args: True)
    saved: list[tuple[int, bool]] = []
    base = module.ReefMegatronTrainRayActor.__mro__[1]
    monkeypatch.setattr(
        base,
        "save_model",
        lambda *_args, **_kwargs: pytest.fail("the LoRA branch must not delegate"),
        raising=False,
    )
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = object()
    monkeypatch.setattr(
        actor,
        "_save_lora_model",
        lambda rollout_id, force_sync=False: saved.append((rollout_id, force_sync)),
        raising=False,
    )

    actor.save_model(3)

    assert saved == [(3, False)]
    assert _StubTimer.started == ["save_model"]


@pytest.mark.unit
@pytest.mark.parametrize("offload_train", [False, True])
def test_lora_save_writes_the_rollouts_megatron_checkpoint_and_adapter(monkeypatch, tmp_path, offload_train):
    module = _load_train_actor(monkeypatch)
    events = []
    _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.model",
        save=lambda rollout_id, *_state: events.append(("megatron", rollout_id)),
    )
    actor_weights = {"vp_stages.0.layer.adapter.linear_in.weight": "weight"}
    weight_iterator = object()

    def export(iterator, weights):
        assert (iterator, weights) == (weight_iterator, actor_weights)
        return ["lora_A"]

    def save_adapter(args, path, tensors, *, rollout_id):
        events.append(("adapter", path, tensors, rollout_id))

    monkeypatch.setattr(module, "export_lora_adapter_tensors", export)
    monkeypatch.setattr(module, "save_lora_adapter_to_path", save_adapter)
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = types.SimpleNamespace(
        debug_rollout_only=False,
        offload_train=offload_train,
        async_save=False,
        save_hf=str(tmp_path / "hf" / "checkpoint-{rollout_id}"),
    )
    actor.role = "actor"
    actor.model = actor.optimizer = actor.opt_param_scheduler = None
    actor.lora_weight_iterator = weight_iterator
    actor.weights_backuper = types.SimpleNamespace(get=lambda tag: actor_weights if tag == "actor" else None)
    actor.wake_up = lambda: events.append(("wake_up",))
    actor.sleep = lambda: events.append(("sleep",))

    actor._save_lora_model(5, force_sync=True)

    saved = [("megatron", 5), ("adapter", tmp_path / "hf" / "checkpoint-5", ["lora_A"], 5)]
    # An offloaded model is on the GPU only while it saves.
    assert events == ([("wake_up",), *saved, ("sleep",)] if offload_train else saved)


@pytest.mark.unit
def test_the_worker_reports_its_parallel_layout_and_only_its_own_metrics(monkeypatch) -> None:
    module = _load_train_actor(monkeypatch)
    worker = object.__new__(module.ReefMegatronTrainRayActor)
    worker.train_parallel_config = {"dp_size": 2, "cp_size": 1}

    layout = worker.parallel_layout()

    assert layout == worker.train_parallel_config
    assert layout is not worker.train_parallel_config
    assert worker.pop_metrics() == {"train/loss": 0.5}


@pytest.mark.unit
def test_an_actor_switches_to_a_jobs_family_and_initializes_it_once(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_actor(monkeypatch)
    initialized: list[str] = []
    misc = importlib.import_module("slime.utils.misc")
    monkeypatch.setattr(misc, "load_function", lambda path: lambda actor: initialized.append(path))
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = types.SimpleNamespace(loss_family="sft", loss_type="sft_loss", reef_actor_init_hook_path=None)
    actor.initialized_loss_families = {"sft"}
    schedule_state = {"schedule": {"name": "rl"}, "completed_steps": 4}
    worker_switch = {
        "loss_family_args": {"loss_family": "pg", "loss_type": "policy_loss", "reef_actor_init_hook_path": "pg.init"},
        "learning_rate_schedule_state": schedule_state,
    }

    actor.switch_training_method(worker_switch)
    actor.switch_training_method(worker_switch)

    assert (actor.args.loss_family, actor.args.loss_type) == ("pg", "policy_loss")
    assert actor.args.reef_learning_rate_schedule_state == schedule_state
    # A family's actor init hook runs when the actor first trains it, not on every switch.
    assert initialized == ["pg.init"]
    actor.switch_training_method({**worker_switch, "learning_rate_schedule_state": None})
    assert actor.args.reef_learning_rate_schedule_state is None


@pytest.mark.unit
@pytest.mark.parametrize("tag", ["ref", "distill_teacher", "actor"])
@pytest.mark.parametrize("enabled", [True, False])
def test_external_checkpoint_clears_student_adapter_only_for_frozen_roles(monkeypatch, tag, enabled):
    module = _load_train_actor(monkeypatch)
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = types.SimpleNamespace()
    actor.model = object()
    calls = []
    base = module.ReefMegatronTrainRayActor.__mro__[1]
    monkeypatch.setattr(
        base, "load_other_checkpoint", lambda self, role, path: calls.append(("load", role)), raising=False
    )
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda args: enabled)
    monkeypatch.setattr(module, "zero_megatron_lora_adapters", lambda model: calls.append(("zero", model)))
    actor.weights_backuper = types.SimpleNamespace(backup=lambda role: calls.append(("backup", role)))
    actor.load_other_checkpoint(tag, "/teacher")
    expected = [("load", tag)]
    if enabled and tag in ("ref", "distill_teacher"):
        expected += [("zero", actor.model), ("backup", tag)]
    assert calls == expected


def load_on_rank_zero(monkeypatch, module, reduced_groups, *, other_rank_mask: list[int], raw_reward_count: int = 2):
    """Load one rollout on DP rank 0 of 2: one optimizer step of 2 rollouts, one sample on each rank.

    Rank 0's sample has no token with loss, as Slime makes a removed sample. The all-reduce adds the counts of
    rank 1, whose sample has ``other_rank_mask``, and records its group in ``reduced_groups``. ``raw_reward_count``
    is the samples Slime converted.
    """
    rank_data = {
        "loss_masks": [torch.zeros(3, dtype=torch.int)],
        "micro_batch_indices": [[0]],
        "num_microbatches": [1],
        "global_batch_sizes": [2],
        "raw_reward": [0.0] * raw_reward_count,
    }
    base = module.ReefMegatronTrainRayActor.__mro__[1]
    monkeypatch.setattr(base, "_get_rollout_data", lambda self, ref: dict(rank_data), raising=False)
    _stub_module(monkeypatch, "slime.backends.megatron_utils.cp_utils", slice_log_prob_with_cp=None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # Rank 1's counts of the step, as ``rank_step_counts`` gives them: samples, samples without loss, tokens.
    other_rank = [1, int(not any(other_rank_mask)), sum(other_rank_mask)]

    def all_reduce(tensor, group):
        reduced_groups.append(group)
        tensor += torch.tensor(other_rank)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = types.SimpleNamespace()
    actor.role = "actor"
    return actor._get_rollout_data("ref")


@pytest.mark.unit
def test_a_step_whose_samples_are_all_removed_fails_before_it_trains(monkeypatch) -> None:
    module = _load_train_actor(monkeypatch)
    reduced_groups: list[str] = []

    with pytest.raises(RuntimeError) as failure:
        load_on_rank_zero(monkeypatch, module, reduced_groups, other_rank_mask=[0, 0])

    assert str(failure.value) == (
        "optimizer step 0 of the rollout trains nothing: it has 2 rollouts and 2 samples, "
        "2 of them removed or fully masked, and 0 tokens with loss"
    )
    # The data-parallel ranks sum their counts: each holds a part of the step.
    assert reduced_groups == ["dp"]


@pytest.mark.unit
def test_a_step_with_one_live_sample_on_another_rank_trains(monkeypatch) -> None:
    module = _load_train_actor(monkeypatch)

    loaded = load_on_rank_zero(monkeypatch, module, [], other_rank_mask=[0, 1, 1])

    assert loaded["global_batch_sizes"] == [2]


@pytest.mark.unit
def test_a_step_plan_that_drops_converted_samples_trains_with_a_warning(monkeypatch, caplog) -> None:
    # Slime's build_dp_schedule drops the rollouts after the last full step by design, for example when a rollout
    # function returns a rollout count that is not a multiple of the step. The job must not fail for it.
    module = _load_train_actor(monkeypatch)

    with caplog.at_level("WARNING", logger=module.__name__):
        loaded = load_on_rank_zero(monkeypatch, module, [], other_rank_mask=[1], raw_reward_count=3)

    assert loaded["global_batch_sizes"] == [2]
    assert "trains 2 of its 3 samples, in steps of [2] rollouts" in caplog.text


@pytest.mark.unit
def test_a_step_plan_that_trains_every_converted_sample_logs_nothing(monkeypatch, caplog) -> None:
    module = _load_train_actor(monkeypatch)

    with caplog.at_level("WARNING", logger=module.__name__):
        load_on_rank_zero(monkeypatch, module, [], other_rank_mask=[1])

    assert "step plan" not in caplog.text
