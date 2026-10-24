from __future__ import annotations

from threading import Barrier
from types import SimpleNamespace

import pytest
from tests.support.executor_helpers import AttachedTestGroup

from reef.runtime.executor import Executor, ExecutorFuture, resolve
from reef.runtime.executor.failure import ExecutorFailureListener
from reef.train.slime_backend.reef_adapters import train_groups
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup


class _Worker:
    def __init__(self, rank):
        self.rank = rank
        self.init_calls = []
        self.manager_calls = []
        self.train_calls = []
        self.save_calls = []
        self.shutdown_calls = 0

    def init(self, args, role, *, with_ref, with_opd_teacher):
        self.args = args
        self.init_calls.append((args.load, role, with_ref, with_opd_teacher))
        if args.fail_init_rank == self.rank:
            raise RuntimeError("worker init failed")
        return args.start_ids[self.rank]

    def set_rollout_manager(self, manager):
        self.manager_calls.append(manager)
        return {"dp_size": 2, "cp_size": 1}

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

    def shutdown(self):
        self.shutdown_calls += 1


class CpuSlimeExecutor(Executor):
    """Custom launcher consumes the real Slime config but runs CPU test workers."""

    def _init_executor(self):
        self.rpc_waits = []
        self.launch_options = dict(self.config.options)
        count = self.launch_options["num_nodes"] * self.launch_options["num_gpus_per_node"]
        self.workers = [self.launch_options["actor_cls"](rank) for rank in range(count)]
        self._local = AttachedTestGroup.from_workers(self.workers, owned=True)
        self.launch_options["args"].launches.append(self)

    def collective_rpc(self, method, *, args=(), kwargs=None, timeout=None, non_block=False):
        self.rpc_waits.append((method, timeout))
        return self._local.collective_rpc(method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

    def rpc(self, rank, method, *, args=(), kwargs=None, timeout=None, non_block=False):
        self.rpc_waits.append((method, timeout))
        return self._local.rpc(rank, method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

    def check_health(self, timeout=None):
        self._local.check_health(timeout=timeout)

    def shutdown(self):
        self._local.shutdown()


@pytest.fixture
def make_group():
    groups = []

    def make(*, role="actor", train_rpc_timeout_s=14400, **overrides):
        values = {
            "reef_executor_backend": f"{__name__}:CpuSlimeExecutor",
            "release_train": False,
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
            train_rpc_timeout_s=train_rpc_timeout_s,
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
    assert group.train_parallel_config == {"dp_size": 2, "cp_size": 1}
    first = group.executor
    assert isinstance(first, CpuSlimeExecutor)
    assert first.launch_options["pg"] == "shared-placement"
    assert first.launch_options["num_gpus_per_actor"] == 0.4
    for worker in first.workers:
        assert worker.init_calls == [("initial-checkpoint", "actor", True, True)]
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
        assert worker.init_calls == [("checkpoint-root", "actor", True, True)]
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


@pytest.mark.parametrize("invalid", [None, {"dp_size": 99}])
def test_attachment_rejects_missing_or_inconsistent_training_layout(make_group, monkeypatch, invalid):
    group = make_group()

    def attach(worker, inference):
        return {"dp_size": 2, "cp_size": 1} if worker.rank == 0 else invalid

    monkeypatch.setattr(_Worker, "set_rollout_manager", attach)
    with pytest.raises(RuntimeError, match="training parallel config"):
        group.create(rollout_manager=object())
    assert group._executor is None


def test_failure_observer_follows_recreated_training_workers(make_group):
    failures = []

    class Observer(ExecutorFailureListener):
        def on_executor_failure(self, failure):
            failures.append(failure)

    group = make_group(release_train=True)
    group.register_failure_listener(Observer())
    group.create()
    first = group.executor
    group.release()
    group.create()
    assert group.executor is not first
    assert failures == []
    group.executor._fail("replacement worker died", rank=1)
    assert len(failures) == 1
    assert failures[0].reason == "replacement worker died"
    assert failures[0].rank == 1


@pytest.mark.parametrize("timeout", [14400, 43200, 0.5])
def test_actor_and_critic_use_job_timeout_without_changing_initialization(make_group, monkeypatch, timeout):
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
    monkeypatch.setattr(train_groups, "prepare_critic_args", lambda config: SimpleNamespace(**vars(config)))
    # The Slime component starts its groups with no rollout manager and attaches none.
    groups = train_groups.create_train_groups(
        args,
        {"actor": "actor-pg", "critic": "critic-pg"},
        None,
        actor_cls=_Worker,
        train_rpc_timeout_s=timeout,
    )
    try:
        for group in groups:
            group.set_rollout_manager(None)
            group.save_model(4, force_sync=True)
            assert all(worker.manager_calls == [None] for worker in group.executor.workers)
            assert all(worker.save_calls == [(4, True)] for worker in group.executor.workers)
            assert group.executor.rpc_waits == [
                ("init", 14400),
                ("set_rollout_manager", 14400),
                ("save_model", timeout),
            ]
    finally:
        for group in groups:
            group.release()
