from __future__ import annotations

from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from executor_helpers import AttachedTestGroup

from reef.runtime.executor import Executor, ExecutorFuture, resolve
from reef.train.slime_backend.reef_adapters import train_groups
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup


class _Worker:
    def __init__(self, rank):
        self.rank = rank
        self.init_calls = []
        self.manager_calls = []
        self.train_calls = []
        self.save_calls = []
        self.update_calls = []
        self.shutdown_calls = 0
        self.version = "deployment:6"

    def init(self, args, role, *, with_ref, with_opd_teacher):
        self.args = args
        self.init_calls.append((args.update_weight_start_version, args.load, role, with_ref, with_opd_teacher))
        if args.fail_init_rank == self.rank:
            raise RuntimeError("worker init failed")
        return args.start_ids[self.rank]

    def set_rollout_manager(self, manager):
        self.manager_calls.append(manager)

    def train(self, rollout_id, data, *, external_data):
        self.train_calls.append((rollout_id, data, external_data))
        if self.args.train_barrier is not None:
            self.args.train_barrier.wait(timeout=5)
        return {"rank": self.rank, "external_data": external_data}

    def save_model(self, rollout_id, *, force_sync):
        self.save_calls.append((rollout_id, force_sync))
        if self.args.fail_save_rank == self.rank:
            raise RuntimeError("worker save failed")
        return self.rank

    def update_weights(self, **kwargs):
        self.update_calls.append(kwargs)
        return self.rank

    def get_runtime_load_id(self):
        return self.version

    def shutdown(self):
        self.shutdown_calls += 1


class CpuSlimeExecutor(Executor):
    """Custom launcher consumes the real Slime config but runs CPU test workers."""

    def _init_executor(self):
        self.launch_options = dict(self.config.options)
        count = self.launch_options["num_nodes"] * self.launch_options["num_gpus_per_node"]
        self.workers = [self.launch_options["actor_cls"](rank) for rank in range(count)]
        self._local = AttachedTestGroup.from_workers(self.workers, owned=True)
        self.launch_options["args"].launches.append(self)

    def collective_rpc(self, method, *, args=(), kwargs=None, timeout=None, non_block=False):
        return self._local.collective_rpc(method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

    def rpc(self, rank, method, *, args=(), kwargs=None, timeout=None, non_block=False):
        return self._local.rpc(rank, method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

    def check_health(self, timeout=None):
        self._local.check_health(timeout=timeout)

    def shutdown(self):
        self._local.shutdown()


@pytest.fixture
def make_group():
    groups = []

    def make(*, role="actor", **overrides):
        values = {
            "reef_executor_backend": f"{__name__}:CpuSlimeExecutor",
            "update_weight_start_version": 5,
            "release_train": False,
            "update_weight_mode": "full",
            "update_weight_transport": "nccl",
            "update_weight_disk_dir": "/unused/weight-export",
            "update_weight_disk_keep_files": True,
            "update_weight_local_checkpoint_dir": None,
            "offload_rollout": False,
            "ci_test": False,
            "save": "checkpoint-root",
            "load": "initial-checkpoint",
            "ckpt_step": 3,
            "finetune": True,
            "no_load_optim": True,
            "no_save_optim": False,
            "no_load_rng": True,
            "launches": [],
            "start_ids": [4, 4],
            "fail_init_rank": None,
            "fail_save_rank": None,
            "train_barrier": None,
        }
        values.update(overrides)
        args = SimpleNamespace(**values)
        group = SlimeTrainGroup(
            args,
            1,
            2,
            pg="shared-placement",
            role=role,
            actor_cls=_Worker,
            num_gpus_per_actor=0.4,
            with_ref=True,
            with_opd_teacher=True,
        )
        groups.append(group)
        return group

    yield make
    for group in groups:
        group.release()


@pytest.mark.unit
def test_custom_executor_factory_recreates_workers_with_checkpoint_and_manager(make_group) -> None:
    group = make_group(release_train=True)
    manager = object()

    assert group.create(rollout_manager=manager) == [4, 4]
    first = group.executor
    assert isinstance(first, CpuSlimeExecutor)
    assert first.launch_options["pg"] == "shared-placement"
    assert first.launch_options["num_gpus_per_actor"] == 0.4
    for worker in first.workers:
        assert worker.init_calls == [(5, "initial-checkpoint", "actor", True, True)]
        assert worker.manager_calls == [manager]
    assert group.create() is None
    assert len(group.args.launches) == 1

    assert group.save_model(4, force_sync=True) == [0, 1]
    group.release()
    group.release()
    assert [worker.shutdown_calls for worker in first.workers] == [1, 1]
    with pytest.raises(RuntimeError, match="not running"):
        _ = group.executor

    assert resolve(group.async_train(5, "batch")) == [
        {"rank": 0, "external_data": None},
        {"rank": 1, "external_data": None},
    ]
    assert group.create() is None
    second = group.executor
    assert second is not first
    assert len(group.args.launches) == 2
    for worker in second.workers:
        assert worker.init_calls == [(5, "checkpoint-root", "actor", True, True)]
        assert worker.manager_calls == [manager]
        assert worker.train_calls == [(5, "batch", None)]
    assert group.args.ckpt_step is None
    assert group.args.finetune is False
    assert group.args.no_load_optim is False
    assert group.args.no_load_rng is False


@pytest.mark.unit
def test_async_train_preserves_rank_critic_values_and_dispatches_all_before_waiting(make_group) -> None:
    group = make_group(train_barrier=Barrier(2))
    group.create()
    batch = object()
    critic_values = [{}, {"values": [0.25, 0.5]}]

    pending = group.async_train(6, batch, external_data=critic_values)
    assert len(pending) == 2
    assert all(isinstance(value, ExecutorFuture) for value in pending)
    assert resolve(pending, timeout=10) == [
        {"rank": 0, "external_data": {}},
        {"rank": 1, "external_data": {"values": [0.25, 0.5]}},
    ]
    for rank, worker in enumerate(group.executor.workers):
        assert worker.train_calls == [(6, batch, critic_values[rank])]
        assert worker.train_calls[0][2] is critic_values[rank]


@pytest.mark.unit
def test_invalid_critic_rank_count_is_rejected_before_any_worker_is_called(make_group) -> None:
    group = make_group()
    group.create()

    with pytest.raises(ValueError, match="one entry per training worker"):
        group.async_train(6, "batch", external_data=[{}])

    assert all(not worker.train_calls for worker in group.executor.workers)
    broadcast = {"values": [0.75]}
    resolve(group.async_train(6, "batch", external_data=broadcast))
    assert all(worker.train_calls[0][2] is broadcast for worker in group.executor.workers)


@pytest.mark.unit
def test_failed_save_does_not_repoint_checkpoint_resume_parameters(make_group) -> None:
    group = make_group(release_train=True, fail_save_rank=1)
    group.create()
    before = (
        group.args.load,
        group.args.ckpt_step,
        group.args.finetune,
        group.args.no_load_optim,
        group.args.no_load_rng,
    )

    with pytest.raises(RuntimeError, match="worker save failed"):
        group.save_model(6, force_sync=True)

    assert (
        group.args.load,
        group.args.ckpt_step,
        group.args.finetune,
        group.args.no_load_optim,
        group.args.no_load_rng,
    ) == before
    assert all(worker.save_calls == [(6, True)] for worker in group.executor.workers)


@pytest.mark.unit
@pytest.mark.parametrize("overrides", [{"fail_init_rank": 1}, {"start_ids": [4, 5]}])
def test_failed_or_inconsistent_initialization_releases_the_entire_group(make_group, overrides) -> None:
    group = make_group(**overrides)

    with pytest.raises(RuntimeError):
        group.create()

    assert [worker.shutdown_calls for worker in group.args.launches[0].workers] == [1, 1]
    with pytest.raises(RuntimeError, match="not running"):
        _ = group.executor


@pytest.mark.unit
@pytest.mark.parametrize("manage_generation,force_full", [(True, False), (False, True)])
def test_disk_update_preserves_flags_version_and_recreate_sequence(
    make_group,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manage_generation,
    force_full,
) -> None:
    group = make_group(release_train=True, update_weight_transport="disk", update_weight_disk_dir=str(tmp_path))
    manager = object()
    group.create(rollout_manager=manager)
    first = group.executor
    publications = []

    def publish(path, sequence, runtime_load_id, *, manage_generation):
        assert [worker.shutdown_calls for worker in first.workers] == [1, 1]
        publications.append((path, sequence, runtime_load_id, manage_generation))

    monkeypatch.setattr(group, "_reload_rollout_weights_from_disk", publish)
    group.update_weights(manage_generation=manage_generation, force_full=force_full)

    expected_kwargs = (
        {}
        if manage_generation and not force_full
        else {
            "manage_generation": manage_generation,
            "force_full": force_full,
        }
    )
    assert all(worker.update_calls == [expected_kwargs] for worker in first.workers)
    assert publications == [(tmp_path / "weight_v000006", 6, "deployment:6", manage_generation)]
    assert resolve(group.async_get_rank0_runtime_load_id()) == "deployment:6"
    assert group._disk_weight_version == 6
    assert group.create() == [4, 4]
    assert group.args.update_weight_start_version == 6
    assert all(worker.init_calls[0][0] == 6 for worker in group.executor.workers)
    assert all(worker.manager_calls == [manager] for worker in group.executor.workers)
    group.executor.workers[0].version = "new-worker-incarnation:6"
    assert resolve(group.async_get_rank0_runtime_load_id()) == "new-worker-incarnation:6"


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["critic_config", "critic_init"])
def test_create_train_groups_releases_actor_when_critic_setup_fails(
    make_group,
    monkeypatch: pytest.MonkeyPatch,
    failure,
) -> None:
    args = make_group(
        megatron_config_path=None,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        critic_num_nodes=1,
        critic_num_gpus_per_node=2,
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        use_critic=True,
        start_rollout_id=None,
    ).args
    manager = object()

    def prepare_critic(config):
        assert all(worker.manager_calls == [manager] for worker in config.launches[0].workers)
        if failure == "critic_config":
            raise ValueError("invalid critic configuration")
        critic = SimpleNamespace(**vars(config))
        critic.fail_init_rank = 1
        return critic

    monkeypatch.setattr(train_groups, "prepare_critic_args", prepare_critic)
    error = ValueError if failure == "critic_config" else RuntimeError
    message = "invalid critic configuration" if failure == "critic_config" else "worker init failed"

    try:
        with pytest.raises(error, match=message):
            train_groups.create_train_groups(
                args,
                {"actor": "shared-placement", "critic": "shared-placement"},
                manager,
                actor_cls=_Worker,
            )

        assert len(args.launches) == (1 if failure == "critic_config" else 2)
        assert args.launches[0].launch_options["role"] == "actor"
        if failure == "critic_init":
            assert args.launches[1].launch_options["role"] == "critic"
        for launched in args.launches:
            assert [worker.shutdown_calls for worker in launched.workers] == [1, 1]
    finally:
        for launched in args.launches:
            launched.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize(
    "manage_generation,mismatch",
    [(True, False), (False, False), (True, True)],
)
def test_disk_reload_orders_serving_operations_and_checks_published_version(
    make_group,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manage_generation,
    mismatch,
) -> None:
    events = []
    local_checkpoint = str(tmp_path / "engine-local-checkpoint")

    class Engine:
        def __init__(self, rank):
            self.rank = rank
            self.version = None

        def pull_weights(self, sequence):
            events.append(("pull", self.rank, sequence))

        def pause_generation(self, mode):
            events.append(("pause", self.rank, mode))

        def flush_cache(self):
            events.append(("flush", self.rank, None))

        def update_weights_from_disk(self, *, model_path, runtime_load_id):
            events.append(("update", self.rank, (model_path, runtime_load_id)))
            self.version = runtime_load_id

        def get_runtime_load_id(self):
            events.append(("verify", self.rank, None))
            return "other-incarnation:6" if mismatch and self.rank == 1 else self.version

        def continue_generation(self):
            events.append(("continue", self.rank, None))

    engines = [Engine(0), Engine(1)]

    class Manager:
        def onload_weights(self):
            events.append(("onload", None, None))

        def get_updatable_engines_and_lock(self):
            events.append(("engines", None, None))
            return engines, None

    attached = []

    def attach(workers):
        executor = AttachedTestGroup.from_workers(workers)
        attached.append(executor)
        return executor

    monkeypatch.setattr(train_groups.RayExecutor, "from_workers", staticmethod(attach))
    group = make_group(
        release_train=True,
        update_weight_transport="disk",
        update_weight_disk_dir=str(tmp_path),
        update_weight_local_checkpoint_dir=local_checkpoint,
        offload_rollout=True,
        weight_update_pause_mode="in_place",
        ci_test=True,
    )
    group.create(rollout_manager=Manager())
    try:
        if mismatch:
            with pytest.raises(RuntimeError, match=r"runtime load ID mismatch.*deployment:6"):
                group.update_weights(manage_generation=manage_generation, force_full=not manage_generation)
        else:
            group.update_weights(manage_generation=manage_generation, force_full=not manage_generation)

        phases = ["onload", "engines", "pull", "pull"]
        if manage_generation:
            phases += ["pause", "pause", "flush", "flush"]
        phases += ["update", "update", "verify", "verify"]
        if manage_generation and not mismatch:
            phases += ["continue", "continue"]
        assert [event[0] for event in events] == phases
        assert sorted(event for event in events if event[0] == "pull") == [("pull", 0, 6), ("pull", 1, 6)]
        assert sorted(event for event in events if event[0] == "update") == [
            ("update", 0, (local_checkpoint, "deployment:6")),
            ("update", 1, (local_checkpoint, "deployment:6")),
        ]
        if manage_generation:
            assert sorted(event for event in events if event[0] == "pause") == [
                ("pause", 0, "in_place"),
                ("pause", 1, "in_place"),
            ]
    finally:
        for executor in attached:
            executor.shutdown()
