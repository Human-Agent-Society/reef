"""Bound Slime checkpoint pairs with locked admission, protected recovery assets,
and restart-safe tombstone reclamation."""

from __future__ import annotations

import fcntl
import math
import os
import re
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from reef.runtime.names import ADAPTER_SLOTS_DIRNAME, LATEST_JOB_MARKER_FILENAME, SCENARIO_LEDGER_FILENAME
from reef.train.slime_backend.reef_adapters.training_job.durable_io import fsync_dir as _fsync_dir
from reef.train.slime_backend.reef_adapters.training_job.durable_io import mkdir_durable as _mkdir_durable
from reef.train.slime_backend.reef_adapters.training_job.durable_io import read_json as _read_json
from reef.train.slime_backend.reef_adapters.training_job.durable_io import write_json as _write_json
from reef.train.slime_backend.reef_adapters.training_job.marker import marker_path as _marker_path

POLICIES = {"latest", "best_reward"}
Inventory = tuple[list[dict[str, Any]], list[str]]


class CheckpointStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetentionConfig:
    policy: str = "latest"
    max_storage_fraction: float | None = 0.8
    min_free_space_fraction: float = 0.1
    max_storage_bytes: int | None = None
    min_free_space_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"checkpoint retention policy must be one of {sorted(POLICIES)}, got {self.policy!r}")
        if self.max_storage_bytes is None and self.max_storage_fraction is None:
            raise ValueError("checkpoint retention requires a storage cap")
        if self.max_storage_bytes is not None and self.max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be positive when set")
        if self.min_free_space_bytes is not None and self.min_free_space_bytes < 0:
            raise ValueError("min_free_space_bytes must be non-negative")
        if (self.max_storage_fraction is not None and not 0 < self.max_storage_fraction <= 1) or not (
            0 <= self.min_free_space_fraction < 1
        ):
            raise ValueError("checkpoint storage fractions are out of range")
        if self.max_storage_bytes is self.min_free_space_bytes is None and (
            (self.max_storage_fraction or 0) + self.min_free_space_fraction > 1
        ):
            raise ValueError("max_storage_fraction + min_free_space_fraction must not exceed 1")

    def capacity_bytes(self, total: int) -> int:
        configured = self.max_storage_bytes or int(total * (self.max_storage_fraction or 0))
        return min(configured, total - self.free_floor_bytes(total))

    def free_floor_bytes(self, total: int) -> int:
        return (
            int(total * self.min_free_space_fraction)
            if self.min_free_space_bytes is None
            else self.min_free_space_bytes
        )


class CheckpointStorage:
    """Own paired Slime HF and Megatron checkpoints under one local byte cap."""

    def __init__(
        self,
        config: RetentionConfig,
        *,
        hf_template: str,
        megatron_root: str | Path,
        critic_root: str | Path | None = None,
        source_hf: str | Path | None = None,
        source_megatron: str | Path | None = None,
        measure: Callable[[Path], int] | None = None,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    ) -> None:
        self.config = config
        template = Path(hf_template).expanduser()
        if "{rollout_id}" not in template.name:
            raise ValueError("HF checkpoint template must contain {rollout_id} in its basename")
        self.hf_root = template.parent.resolve()
        self.hf_template = str(self.hf_root / template.name)
        self.megatron_root = Path(megatron_root).expanduser().resolve()
        if self.hf_root == self.megatron_root or self.hf_root.parent != self.megatron_root.parent:
            raise ValueError("HF and Megatron checkpoints must use distinct directories under one checkpoint root")
        self.critic_root = None if critic_root is None else Path(critic_root).expanduser().resolve()
        if self.critic_root is not None and (
            self.critic_root in {self.hf_root, self.megatron_root}
            or self.critic_root.parent != self.megatron_root.parent
        ):
            raise ValueError("critic checkpoints must use a distinct directory under the same checkpoint root")
        self.root = self.hf_root.parent
        self.source_hf = _optional_path(source_hf)
        self.source_megatron = _optional_path(source_megatron)
        self.meta_root = self.root / ".reef-retention"
        self.records_root = self.meta_root / "records"
        self.estimate_path = self.meta_root / "estimate.json"
        self.lock_path = self.meta_root / "storage.lock"
        self._measure = measure or _allocated_bytes
        self._source_measure = (
            self._measure if measure is not None else lambda path: _allocated_bytes(path, follow_symlinks=True)
        )
        self._disk_usage = disk_usage

    @property
    def marker_path(self) -> Path:
        """Location of the durable training-job marker for these checkpoints."""
        return _marker_path(self.hf_template)

    def validate_capacity(self, *, active_rollouts: set[int] | None = None) -> dict[str, Any]:
        with self._locked():
            inventory = self._reconcile()
            plan = self._plan(active_rollouts or set(), inventory)
            if int(plan["reservation_bytes"]) <= 0:
                raise ValueError("cannot estimate one paired checkpoint from source or completed checkpoints")
            return plan

    def _estimate(self, records: list[dict[str, Any]]) -> int:
        # Retain the high-water mark after deletion so future reservations
        # cannot underestimate a previously observed checkpoint pair.
        persisted = (_read_json(self.estimate_path) or {}).get("bytes", 0)
        persisted = persisted if isinstance(persisted, int) else 0
        return max(
            self._source_estimate,
            persisted,
            *(int(record["bytes"]) for record in records),
        )

    @contextmanager
    def admit(
        self,
        *,
        rollout_id: int,
        active_rollouts: set[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        # Hold the lock through the caller's train/save work so another job
        # cannot consume this reservation.
        with self._locked():
            active = active_rollouts or set()
            inventory = self._reconcile()
            if any(path.exists() or path.is_symlink() for path in self.asset_paths(rollout_id)):
                raise CheckpointStorageError(f"checkpoint pair target already exists for rollout {rollout_id}")
            plan = self._plan(active, inventory)
            while not plan["blocked"] and plan["delete"]:
                records, errors = inventory
                catalog = {int(record["rollout_id"]): record for record in records}
                for candidate in plan["delete"]:
                    self._delete(catalog.pop(int(candidate)))
                inventory = list(catalog.values()), errors
                plan = self._plan(active, inventory)
            yield plan

    def complete(self, job_id: str, rollout_id: int, *, reward: float | None) -> None:
        # The critic asset (when a critic root is configured) is required at
        # completion time — the bridge saves it before completing — so a
        # failed critic save cannot be recorded as a durable checkpoint.
        for path in self.asset_paths(rollout_id):
            if path.is_symlink() or not path.is_dir():
                raise CheckpointStorageError(f"checkpoint asset is missing or unsafe: {path}")
        pair_bytes = sum(self._measure(path) for path in self.asset_paths(rollout_id))
        previous = (_read_json(self.estimate_path) or {}).get("bytes", 0)
        previous = previous if isinstance(previous, int) else 0
        _write_json(self.estimate_path, {"bytes": max(previous, pair_bytes)})
        _write_json(
            self._record_path(rollout_id),
            {
                "version": 1,
                "status": "COMPLETE",
                "job_id": job_id,
                "rollout_id": rollout_id,
                "bytes": pair_bytes,
                "reward": _finite_or_none(reward),
            },
        )

    def _plan(self, active_rollouts: set[int], inventory: Inventory) -> dict[str, Any]:
        usage = self._disk_usage(self.root)
        free_floor = self.config.free_floor_bytes(usage.total)
        capacity = self.config.capacity_bytes(usage.total)
        records, errors = inventory
        reservation = self._estimate(records)
        errors = list(errors) + (["cannot estimate one paired checkpoint"] if reservation <= 0 else [])
        pairs = {int(record["rollout_id"]): record for record in records if record["status"] == "COMPLETE"}
        protected = set(active_rollouts)
        protected.update(self._tracker_rollouts())
        if pairs:
            protected.add(max(pairs))
        source_paths = {path for path in (self.source_hf, self._source_megatron_checkpoint) if path is not None}
        protected.update(
            rollout_id
            for rollout_id in pairs
            if source_paths.intersection(path.resolve() for path in self.asset_paths(rollout_id))
        )
        known_paths = {path for rollout_id in pairs for path in self.asset_paths(rollout_id)}
        unknown = self._unknown_assets(known_paths)
        live_sizes = {rollout_id: int(record["bytes"]) for rollout_id, record in pairs.items()}
        keep_ids = protected & live_sizes.keys()
        kept_bytes = sum(live_sizes[rollout_id] for rollout_id in keep_ids)
        live_bytes = sum(live_sizes.values())
        budget = min(
            capacity - reservation,
            usage.free + live_bytes - reservation - free_floor,
        )
        for rollout_id in sorted(
            (rollout_id for rollout_id in pairs if rollout_id not in protected),
            key=lambda rollout_id: self._rank_key(pairs[rollout_id]),
        ):
            if kept_bytes + live_sizes[rollout_id] <= budget:
                keep_ids.add(rollout_id)
                kept_bytes += live_sizes[rollout_id]
        blocked_reasons = errors + ([f"unowned checkpoint assets: {', '.join(unknown)}"] if unknown else [])
        if kept_bytes + reservation > capacity:
            blocked_reasons.append("protected checkpoints plus reservation exceed the managed storage cap")
        if usage.free + (live_bytes - kept_bytes) - reservation < free_floor:
            blocked_reasons.append("checkpoint reservation would violate the filesystem free-space floor")
        return {
            "blocked": bool(blocked_reasons),
            "reasons": blocked_reasons,
            "capacity_bytes": capacity,
            "min_free_bytes": free_floor,
            "filesystem_free_bytes": usage.free,
            "reservation_bytes": reservation,
            "delete": sorted(live_sizes.keys() - keep_ids),
        }

    def _rank_key(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        rollout_id = int(record["rollout_id"])
        if self.config.policy == "latest":
            return (-rollout_id,)
        if (metric := record.get("reward")) is None:
            return (1, 0.0, -rollout_id)
        return (0, -float(metric), -rollout_id)

    def _delete(self, record: Mapping[str, Any]) -> None:
        self._validate_record(record)
        rollout_id = int(record["rollout_id"])
        # Persist intent before renames so reconciliation can finish after a crash.
        record = {**record, "status": "DELETING"}
        _write_json(self._record_path(rollout_id), record)
        self._finish_delete(record)

    def _reconcile(self) -> Inventory:
        records, errors = self._records(allow_deleting=True)
        if errors:
            return records, errors
        for record in records:
            if record["status"] == "DELETING":
                self._finish_delete(record)
        return [record for record in records if record["status"] == "COMPLETE"], []

    def _finish_delete(self, record: Mapping[str, Any]) -> None:
        self._validate_record(record, deleting=True)
        rollout_id, job_id = int(record["rollout_id"]), str(record["job_id"])
        pairs = [(path, self._tombstone_path(path, rollout_id, job_id)) for path in self.asset_paths(rollout_id)]
        for original, tombstone in pairs:
            if original.is_symlink() or tombstone.is_symlink() or (original.exists() and tombstone.exists()):
                raise CheckpointStorageError(f"checkpoint and tombstone state is unsafe: {original}")
        for original, tombstone in pairs:
            if original.exists():
                os.replace(original, tombstone)
                _fsync_dir(original.parent)
        for _, tombstone in pairs:
            if tombstone.exists():
                shutil.rmtree(tombstone)
                _fsync_dir(tombstone.parent)
        self._record_path(rollout_id).unlink(missing_ok=True)
        _fsync_dir(self.records_root)

    def _records(self, *, allow_deleting: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in sorted(self.records_root.glob("*.json")):
            try:
                if (record := _read_json(path)) is None:
                    continue
                self._validate_record(record, deleting=allow_deleting and record.get("status") == "DELETING")
                if path != self._record_path(int(record["rollout_id"])):
                    raise CheckpointStorageError("checkpoint record has non-canonical identity")
                records.append(record)
            except (CheckpointStorageError, OSError, ValueError) as exc:
                errors.append(f"invalid checkpoint catalog record {path.name}: {exc}")
        return records, errors

    def _validate_record(self, record: Mapping[str, Any], *, deleting: bool = False) -> None:
        if record.get("version") != 1 or record.get("status") not in {"COMPLETE", "DELETING"}:
            raise CheckpointStorageError("unsupported checkpoint record")
        rollout_id, job_id = record.get("rollout_id"), record.get("job_id")
        valid_rollout = isinstance(rollout_id, int) and not isinstance(rollout_id, bool) and rollout_id >= 0
        if not valid_rollout or not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", job_id) is None:
            raise CheckpointStorageError("checkpoint record has invalid identity")
        size = record.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CheckpointStorageError("checkpoint record has invalid byte count")
        if record.get("reward") is not None and _finite_or_none(record.get("reward")) is None:
            raise CheckpointStorageError("checkpoint record has invalid reward")
        if deleting != (record["status"] == "DELETING"):
            raise CheckpointStorageError("checkpoint record status mismatch")
        if not deleting and any(
            path.is_symlink() or not path.is_dir() for path in self.pair_paths(int(record["rollout_id"]))
        ):
            raise CheckpointStorageError("checkpoint record has missing or symlinked asset")

    def _unknown_assets(self, known: set[Path]) -> list[str]:
        unknown: list[str] = []
        # Control files Reef itself keeps in the managed roots: the job marker
        # and the LoRA scenario ledger beside the HF exports, and the
        # adapter-slot snapshots beside the Megatron checkpoints.
        roots = [
            (self.hf_root, LATEST_JOB_MARKER_FILENAME, {SCENARIO_LEDGER_FILENAME}),
            (self.megatron_root, "latest_checkpointed_iteration.txt", {ADAPTER_SLOTS_DIRNAME}),
        ]
        if self.critic_root is not None:
            roots.append((self.critic_root, "latest_checkpointed_iteration.txt", set()))
        for root, control, owned in roots:
            if root.exists():
                unknown.extend(
                    str(path)
                    for path in root.iterdir()
                    if path not in known
                    and path.name != control
                    and path.name not in owned
                    and not (
                        not path.is_symlink()
                        and path.is_file()
                        and path.name.startswith(f".{control.lstrip('.')}.")
                        and path.name.endswith(".tmp")
                    )
                )
        return unknown

    @cached_property
    def _source_estimate(self) -> int:
        """Size one paired checkpoint from whichever sources this job has.

        A cold start has only the HF source. Slime clears ``--load`` when the
        load root holds no checkpoint yet, so the first boot of a fresh
        deployment reaches here with ``source_megatron`` unset — and that is
        the case the estimate is *least* unsure about, since the multiplier
        below is derived from the HF model size alone. Requiring both sources
        refused every such boot outright ("cannot estimate one paired
        checkpoint from source or completed checkpoints").

        With no HF source the Megatron checkpoint stands in for the pair; it
        already holds the training state, and the first completed pair
        replaces this estimate with a measured one either way.
        """
        hf = self.source_hf if self.source_hf is not None and self.source_hf.exists() else None
        megatron = self._source_megatron_checkpoint
        if hf is None and megatron is None:
            return 0
        hf_bytes = self._source_measure(hf) if hf is not None else 0
        megatron_bytes = self._source_measure(megatron) if megatron is not None else 0
        # HF export + model, FP32 master weights, and two Adam moments; the
        # critic checkpoint (when configured) is a second full model plus
        # optimizer state of roughly the same footprint.
        training_state = 8 * hf_bytes * (2 if self.critic_root is not None else 1)
        return max(hf_bytes + megatron_bytes, training_state)

    @cached_property
    def _source_megatron_checkpoint(self) -> Path | None:
        path = self.source_megatron
        if path is None or not path.exists():
            return None
        if re.fullmatch(r"iter_\d{7}", path.name):
            return path
        try:
            iteration = int((path / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            return path if path.is_dir() else None
        except (IsADirectoryError, ValueError):
            return None
        candidate = path / f"iter_{iteration:07d}"
        return candidate.resolve() if candidate.is_dir() else None

    def _tracker_rollouts(self) -> set[int]:
        rollouts: set[int] = set()
        roots = [self.megatron_root] + ([self.critic_root] if self.critic_root is not None else [])
        for root in roots:
            tracker = root / "latest_checkpointed_iteration.txt"
            try:
                rollouts.add(int(tracker.read_text(encoding="utf-8").strip()))
            except (FileNotFoundError, ValueError):
                continue
        return rollouts

    def pair_paths(self, rollout_id: int) -> tuple[Path, Path]:
        """The recovery pair: the HF export and the actor Megatron checkpoint.

        These two always exist for a COMPLETE record (the recovery-pair
        invariant); the critic asset is additional and tracked by
        :meth:`asset_paths`.
        """
        hf = Path(self.hf_template.format(rollout_id=rollout_id))
        return hf, self.megatron_root / f"iter_{rollout_id:07d}"

    def asset_paths(self, rollout_id: int) -> tuple[Path, ...]:
        """All storage-owned assets for one rollout: the pair plus, when a
        critic root is configured, the critic Megatron checkpoint.

        The critic asset participates in admission, retention accounting, and
        deletion, but record validation intentionally requires only the pair —
        records written before a critic root existed stay valid, and a missing
        critic checkpoint degrades to a value-model cold start, not a blocked
        store."""
        paths: tuple[Path, ...] = self.pair_paths(rollout_id)
        if self.critic_root is not None:
            paths = (*paths, self.critic_root / f"iter_{rollout_id:07d}")
        return paths

    def _record_path(self, rollout_id: int) -> Path:
        return self.records_root / f"{rollout_id:020d}.json"

    @staticmethod
    def _tombstone_path(original: Path, rollout_id: int, job_id: str) -> Path:
        return original.with_name(f".reef-delete-{rollout_id}-{job_id[:12]}")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _mkdir_durable(self.meta_root)
        roots: tuple[Path, ...] = (self.root, self.hf_root, self.megatron_root, self.meta_root)
        if self.critic_root is not None:
            roots = (*roots, self.critic_root)
        if any(path.is_symlink() for path in roots):
            raise CheckpointStorageError("checkpoint storage paths must not be symlinks")
        if len({path.stat().st_dev for path in roots if path.exists()}) != 1:
            raise CheckpointStorageError("checkpoint pair paths must share one filesystem")
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CheckpointStorageError("checkpoint storage lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()


def _finite_or_none(value: float | None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _allocated_bytes(
    path: Path,
    *,
    follow_symlinks: bool = False,
    _seen: set[tuple[int, int]] | None = None,
) -> int:
    try:
        metadata = path.stat() if follow_symlinks else path.lstat()
    except FileNotFoundError:
        return 0
    if follow_symlinks:
        _seen = set() if _seen is None else _seen
        identity = metadata.st_dev, metadata.st_ino
        if identity in _seen:
            return 0
        _seen.add(identity)
    total = metadata.st_blocks * 512
    if not stat.S_ISDIR(metadata.st_mode):
        return total
    with os.scandir(path) as entries:
        for entry in entries:
            total += _allocated_bytes(
                Path(entry.path),
                follow_symlinks=follow_symlinks,
                _seen=_seen,
            )
    return total
