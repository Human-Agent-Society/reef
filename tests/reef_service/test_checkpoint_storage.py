from __future__ import annotations

import multiprocessing
import os
from collections import namedtuple
from pathlib import Path

import pytest

from reef.train.slime_backend.reef_adapters.training_job import durable_io
from reef.train.slime_backend.reef_adapters.training_job import storage as checkpoint_storage
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

Usage = namedtuple("Usage", "total used free")


def _logical_bytes(path: Path) -> int:
    if not path.exists() or path.is_symlink() or path.is_file():
        return path.lstat().st_size if path.exists() or path.is_symlink() else 0
    return sum(_logical_bytes(child) for child in path.iterdir())


def _write_bytes(path: Path, size: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").write_bytes(b"x" * size)


def _storage(
    tmp_path: Path,
    *,
    policy: str = "latest",
    cap: int = 1000,
    free: int = 1000,
    min_free: int = 0,
) -> CheckpointStorage:
    root = tmp_path / "checkpoints"
    source_hf, source_megatron = tmp_path / "source-hf", tmp_path / "source-megatron"
    _write_bytes(source_hf, 10)
    _write_bytes(source_megatron / "iter_0000000", 90)
    (source_megatron / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    return CheckpointStorage(
        RetentionConfig(policy=policy, max_storage_bytes=cap, min_free_space_bytes=min_free),
        hf_template=str(root / "hf" / "{rollout_id}"),
        megatron_root=root / "megatron",
        source_hf=source_hf,
        source_megatron=source_megatron,
        measure=_logical_bytes,
        disk_usage=lambda path: Usage(1000, 1000 - free, free),
    )


def _complete(storage: CheckpointStorage, rollout_id: int, *, reward=None) -> None:
    with storage.admit(rollout_id=rollout_id) as plan:
        assert not plan["blocked"], plan
        for path, size in zip(storage.pair_paths(rollout_id), (40, 60), strict=True):
            _write_bytes(path, size)
        (storage.megatron_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id), encoding="utf-8")
        storage.complete(f"job-{rollout_id}", rollout_id, reward=reward)


def _completed_storage(tmp_path: Path, rewards, **options) -> CheckpointStorage:
    storage = _storage(tmp_path, **options)
    for rollout_id, reward in enumerate(rewards):
        _complete(storage, rollout_id, reward=reward)
    return storage


@pytest.mark.unit
class TestCheckpointStorage:
    def test_retention_config_validates_fraction_budget(self) -> None:
        assert RetentionConfig().capacity_bytes(1000) == 800
        assert RetentionConfig().free_floor_bytes(1000) == 100
        assert RetentionConfig(max_storage_bytes=950).capacity_bytes(1000) == 900
        with pytest.raises(ValueError, match="must not exceed"):
            RetentionConfig(max_storage_fraction=0.9, min_free_space_fraction=0.2)

    def test_admission_blocks_when_one_estimated_pair_cannot_fit(self, tmp_path: Path) -> None:
        assert _storage(tmp_path, cap=99).validate_capacity()["blocked"]
        assert _storage(tmp_path, free=199, min_free=100).validate_capacity()["blocked"]
        assert not _storage(tmp_path, cap=150).validate_capacity()["blocked"]

    def test_initial_estimate_includes_adam_checkpoint_state(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        _write_bytes(storage._source_megatron_checkpoint, 10)
        assert storage.validate_capacity()["reservation_bytes"] == 80

    def test_cold_start_estimates_from_the_hf_source_alone(self, tmp_path: Path) -> None:
        # The first boot of a fresh deployment has saved nothing yet, and
        # Slime clears --load when the load root holds no checkpoint, so the
        # store is built with source_megatron unset. The HF source is enough:
        # refusing here blocked every cold start of a training stack.
        root, source_hf = tmp_path / "checkpoints", tmp_path / "source-hf"
        _write_bytes(source_hf, 10)
        storage = CheckpointStorage(
            RetentionConfig(max_storage_bytes=1000, min_free_space_bytes=0),
            hf_template=str(root / "hf" / "{rollout_id}"),
            megatron_root=root / "megatron",
            source_hf=source_hf,
            source_megatron=None,
            measure=_logical_bytes,
            disk_usage=lambda path: Usage(1000, 0, 1000),
        )

        assert storage.validate_capacity()["reservation_bytes"] == 80

    def test_megatron_source_alone_stands_in_for_the_pair(self, tmp_path: Path) -> None:
        root, source_megatron = tmp_path / "checkpoints", tmp_path / "source-megatron"
        _write_bytes(source_megatron / "iter_0000000", 90)
        (source_megatron / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
        storage = CheckpointStorage(
            RetentionConfig(max_storage_bytes=1000, min_free_space_bytes=0),
            hf_template=str(root / "hf" / "{rollout_id}"),
            megatron_root=root / "megatron",
            source_hf=None,
            source_megatron=source_megatron,
            measure=_logical_bytes,
            disk_usage=lambda path: Usage(1000, 0, 1000),
        )

        assert storage.validate_capacity()["reservation_bytes"] == 90

    def test_capacity_refuses_only_when_no_source_can_size_a_pair(self, tmp_path: Path) -> None:
        root = tmp_path / "checkpoints"
        storage = CheckpointStorage(
            RetentionConfig(max_storage_bytes=1000, min_free_space_bytes=0),
            hf_template=str(root / "hf" / "{rollout_id}"),
            megatron_root=root / "megatron",
            source_hf=None,
            source_megatron=None,
            measure=_logical_bytes,
            disk_usage=lambda path: Usage(1000, 0, 1000),
        )

        with pytest.raises(ValueError, match="cannot estimate one paired checkpoint"):
            storage.validate_capacity()

    def test_source_measurement_follows_nested_symlinks(self, tmp_path: Path) -> None:
        target = tmp_path / "weights"
        target.write_bytes(b"x" * 4096)
        link = tmp_path / "link"
        link.symlink_to(target)
        assert checkpoint_storage._allocated_bytes(link, follow_symlinks=True) == checkpoint_storage._allocated_bytes(
            target
        )

    @pytest.mark.parametrize(
        ("policy", "metrics", "deleted"),
        [
            ("latest", [9.0, 1.0, 5.0], 0),
            ("best_reward", [9.0, 1.0, 5.0], 1),
            ("best_reward", [None, 0.2, 0.1], 0),
        ],
    )
    def test_policy_ranks_pairs_but_always_protects_latest(self, tmp_path: Path, policy, metrics, deleted) -> None:
        _completed_storage(tmp_path, metrics)

        constrained = _storage(tmp_path, policy=policy, cap=300)
        plan = constrained.validate_capacity(active_rollouts={2})

        assert not plan["blocked"]
        assert plan["delete"] == [deleted]

    @pytest.mark.parametrize("protection", ["tracker", "source"])
    def test_recovery_pair_is_never_eligible_for_deletion(self, tmp_path: Path, protection: str) -> None:
        storage = _completed_storage(tmp_path, (None,) * 3)
        constrained = _storage(tmp_path, cap=300)
        if protection == "tracker":
            (storage.megatron_root / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
        else:
            constrained.source_hf, constrained.source_megatron = storage.pair_paths(0)

        plan = constrained.validate_capacity(active_rollouts={2})

        assert plan["delete"] == [1]

    def test_tracker_symlink_target_is_protected(self, tmp_path: Path) -> None:
        storage = _completed_storage(tmp_path, (None,) * 3)
        source = tmp_path / "tracker-source"
        source.mkdir()
        (source / "iter_0000000").symlink_to(storage.pair_paths(0)[1], target_is_directory=True)
        (source / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
        constrained = _storage(tmp_path, cap=300)
        constrained.source_megatron = source.resolve()

        assert constrained.validate_capacity(active_rollouts={2})["delete"] == [1]

    def test_free_space_floor_reclaims_an_eligible_pair(self, tmp_path: Path) -> None:
        _completed_storage(tmp_path, (None,) * 2)

        constrained = _storage(tmp_path, free=150, min_free=100)
        plan = constrained.validate_capacity(active_rollouts={1})

        assert not plan["blocked"]
        assert plan["delete"] == [0]

    def test_admission_finishes_followup_reclamation(self, tmp_path: Path) -> None:
        storage = _completed_storage(tmp_path, (None,) * 3, min_free=100)
        frees = iter((150, 150, 250))

        def disk_usage(_path: Path) -> Usage:
            free = next(frees)
            return Usage(1000, 1000 - free, free)

        storage._disk_usage = disk_usage
        with storage.admit(rollout_id=3, active_rollouts={2}) as plan:
            assert not plan["blocked"] and plan["delete"] == []
        assert [int(record["rollout_id"]) for record in storage._records()[0]] == [2]

    def test_cataloged_pairs_are_not_recursively_remeasured(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        _complete(storage, 0)
        storage._measure = lambda path: pytest.fail(f"unexpected measurement: {path}")
        assert not storage.validate_capacity(active_rollouts={0})["blocked"]

    def test_atomic_json_temp_cannot_clobber_symlink(self, tmp_path: Path) -> None:
        path, victim = tmp_path / "state.json", tmp_path / "victim"
        victim.write_text("safe", encoding="utf-8")
        predictable = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        predictable.symlink_to(victim)
        durable_io.write_json(path, {"ok": True})
        assert victim.read_text(encoding="utf-8") == "safe"
        assert durable_io.read_json(path) == {"ok": True}

    def test_marker_temp_from_crash_is_ignored(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        storage.hf_root.mkdir(parents=True)
        (storage.hf_root / ".reef-latest-job.json.crash.tmp").write_text("partial", encoding="utf-8")
        assert not storage.validate_capacity()["blocked"]

    def test_symlinked_checkpoint_root_is_rejected(self, tmp_path: Path) -> None:
        root, outside = tmp_path / "checkpoints", tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "hf").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="under one checkpoint root"):
            CheckpointStorage(
                RetentionConfig(),
                hf_template=str(root / "hf" / "{rollout_id}"),
                megatron_root=root / "megatron",
            )

    def test_symlinked_storage_lock_is_rejected(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        durable_io.mkdir_durable(storage.meta_root)
        storage.lock_path.symlink_to(tmp_path / "victim")
        with pytest.raises(OSError):
            storage.validate_capacity()

    def test_storage_lock_serializes_processes(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        context = multiprocessing.get_context("fork")
        go, attempting, entered = (context.Event() for _ in range(3))

        def enter() -> None:
            go.wait()
            attempting.set()
            with storage._locked():
                entered.set()

        process = context.Process(target=enter, daemon=True)
        process.start()
        with storage._locked():
            go.set()
            assert attempting.wait(5) and not entered.wait(0.1)
        assert entered.wait(5)
        process.join(5)
        assert process.exitcode == 0

    def test_unknown_and_symlinked_entries_are_protected_and_never_deleted(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        _complete(storage, 0)
        unknown = storage.hf_root / "operator-copy"
        _write_bytes(unknown, 20)
        target = tmp_path / "outside"
        _write_bytes(target, 30)
        symlink = storage.megatron_root / "iter_0000009"
        symlink.symlink_to(target, target_is_directory=True)
        orphan = storage.hf_root / ".reef-delete-4-unowned"
        _write_bytes(orphan, 10)

        plan = _storage(tmp_path, cap=500).validate_capacity(active_rollouts={0})

        assert plan["blocked"]
        assert all(name in " ".join(plan["reasons"]) for name in (unknown.name, symlink.name, orphan.name))
        assert unknown.is_dir() and symlink.is_symlink() and target.is_dir()
        hf_target, _ = storage.pair_paths(1)
        hf_target.symlink_to(target, target_is_directory=True)
        with (
            pytest.raises(RuntimeError, match="target already exists"),
            storage.admit(rollout_id=1),
        ):
            pass

    def test_lora_control_files_are_owned_not_unknown(self, tmp_path: Path) -> None:
        """The scenario ledger and adapter-slot snapshots live in the managed roots by design."""
        from reef.runtime.names import ADAPTER_SLOTS_DIRNAME, SCENARIO_LEDGER_FILENAME

        storage = _storage(tmp_path)
        _complete(storage, 0)
        (storage.hf_root / SCENARIO_LEDGER_FILENAME).write_text("{}", encoding="utf-8")
        _write_bytes(storage.megatron_root / ADAPTER_SLOTS_DIRNAME / "bWF0aA" / "rank_00000.pt", 5)

        plan = _storage(tmp_path).validate_capacity(active_rollouts={0})

        assert not plan["blocked"], plan

    def test_protected_pair_blocks_before_any_new_checkpoint_can_fit(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        _complete(storage, 0)

        with _storage(tmp_path, cap=150).admit(rollout_id=1, active_rollouts={0}) as plan:
            assert plan["blocked"]
            assert "protected checkpoints" in " ".join(plan["reasons"])

    def test_oversized_completed_pair_stays_protected_and_blocks_later_jobs(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path, cap=250)
        with storage.admit(rollout_id=0) as plan:
            assert not plan["blocked"]
            hf, megatron = storage.pair_paths(0)
            _write_bytes(hf, 150)
            _write_bytes(megatron, 150)
            storage.complete("large", 0, reward=1.0)

        plan = storage.validate_capacity(active_rollouts={0})

        assert plan["blocked"]
        assert plan["delete"] == []

    @pytest.mark.parametrize("fault", ["rename", "remove"])
    def test_deletion_fault_is_reconciled_on_restart(self, tmp_path: Path, monkeypatch, fault: str) -> None:
        _completed_storage(tmp_path, (0.0, 1.0))
        crashing = _storage(tmp_path, cap=250)
        hf, megatron = crashing.pair_paths(0)

        if fault == "rename":
            replace = checkpoint_storage.os.replace

            def fail(source, target) -> None:
                if Path(source) == megatron:
                    raise OSError("crash during deletion")
                replace(source, target)

            monkeypatch.setattr(checkpoint_storage.os, "replace", fail)
        else:
            monkeypatch.setattr(
                checkpoint_storage.shutil,
                "rmtree",
                lambda path: (_ for _ in ()).throw(OSError("crash during deletion")),
            )
        with (
            pytest.raises(OSError, match="crash during deletion"),
            crashing.admit(rollout_id=2, active_rollouts={1}),
        ):
            pass

        record = checkpoint_storage._read_json(crashing._record_path(0))
        tombstone = crashing._tombstone_path(hf, 0, record["job_id"])
        assert record["status"] == "DELETING"
        assert tombstone.exists()
        assert fault != "rename" or megatron.exists()

        monkeypatch.undo()
        recovered = _storage(tmp_path, cap=250)
        plan = recovered.validate_capacity(active_rollouts={1})
        assert not recovered._record_path(0).exists()
        assert not tombstone.exists()
        assert not megatron.exists()
        assert not plan["blocked"]

    def test_reclamation_never_touches_unowned_sibling_storage(self, tmp_path: Path) -> None:
        storage = _completed_storage(tmp_path, (None,) * 2)
        artifacts = storage.root / "artifacts.git"
        _write_bytes(artifacts, 50)

        constrained = _storage(tmp_path, cap=250)
        with constrained.admit(rollout_id=2, active_rollouts={1}) as plan:
            assert not plan["blocked"]
            assert not constrained._record_path(0).exists()
            assert artifacts.is_dir()

    def test_corrupt_catalog_fails_closed(self, tmp_path: Path) -> None:
        storage = _storage(tmp_path)
        storage.records_root.mkdir(parents=True)
        (storage.records_root / "broken.json").write_text("{", encoding="utf-8")

        plan = storage.validate_capacity()

        assert plan["blocked"]
        assert "invalid checkpoint catalog" in " ".join(plan["reasons"])
