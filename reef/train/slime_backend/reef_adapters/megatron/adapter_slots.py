"""Time-slice one Megatron LoRA parameter slot between several scenarios.

Megatron Bridge attaches exactly one adapter to each wrapped linear, so a
training group can hold one adapter's parameters at a time. Several scenarios
share the frozen base by taking turns in that slot: before a scenario trains,
publishes, or checkpoints, its adapter parameters, the optimizer's main
parameters and moments for them, and its schedule position are restored;
afterwards they are snapshotted again.

Every snapshot is rank-local — the adapter tensors this rank's model chunks
hold and the shards this rank's optimizer owns — so switching costs no
collective and stays exact (fp32 main parameters are preserved, not rederived
from the bf16 model copy). Persisted snapshots therefore require the same
parallel topology on restart, exactly like the Megatron checkpoint they sit
next to.
"""

from __future__ import annotations

import base64
import contextlib
import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT = 1


def scenario_key(scenario: str) -> str:
    """A filesystem-safe, lossless key for one scenario name."""
    if not scenario:
        raise ValueError("scenario key requires a non-empty scenario name")
    return base64.urlsafe_b64encode(scenario.encode("utf-8")).decode("ascii").rstrip("=")


def is_megatron_lora_parameter(name: str) -> bool:
    """Whether a Megatron Bridge parameter belongs to the adapter (``adapter.`` prefix)."""
    return name.startswith("adapter.") or ".adapter." in name


def _inner_optimizers(optimizer: Any) -> Iterator[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained:
        for member in chained:
            yield from _inner_optimizers(member)
        return
    inner = getattr(optimizer, "optimizer", None)
    yield inner if inner is not None else optimizer


def _main_params(optimizer: Any) -> Iterator[tuple[Any, torch.Tensor]]:
    """Every parameter the torch-level optimizer steps, with its owner."""
    for inner in _inner_optimizers(optimizer):
        for group in inner.param_groups:
            for param in group["params"]:
                yield inner, param


def _adapter_params(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            if is_megatron_lora_parameter(name):
                yield f"chunk{index}.{name}", parameter


def _write_snapshot(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` with ``payload``; an interrupted write keeps the old file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


class AdapterSlotSwitcher:
    """Own the rank-local snapshots of every scenario's adapter state.

    ``capture(scenario)`` records the slot as scenario ``scenario``'s state;
    ``restore(scenario)`` puts it back, falling back to the pristine
    post-init snapshot for a scenario seen for the first time. ``active`` is
    the scenario whose state the slot currently holds.
    """

    def __init__(
        self,
        model: Sequence[torch.nn.Module],
        optimizer: Any,
        opt_param_scheduler: Any = None,
        *,
        store_dir: str | Path | None = None,
        rank: int = 0,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = opt_param_scheduler
        self._store_dir = None if store_dir is None else Path(store_dir)
        self._rank = rank
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._active: str | None = None
        # One writer, so snapshot writes stay ordered among themselves and a
        # slot movement has exactly one write to wait for.
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adapter-slot-persist")
        self._pending_write: Future[None] | None = None
        adapter_count = sum(1 for _ in _adapter_params(model))
        if adapter_count == 0:
            raise RuntimeError("adapter slot switching requires a model with Megatron LoRA parameters")
        self._pristine = self._snapshot()

    @property
    def active(self) -> str | None:
        return self._active

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    # -- Slot movement ---------------------------------------------------

    def capture(self, scenario: str) -> None:
        """Record the slot's current state as ``scenario``'s."""
        _require(scenario)
        self._snapshots[scenario] = self._snapshot()
        self._active = scenario

    @torch.no_grad()
    def restore(self, scenario: str) -> bool:
        """Put ``scenario``'s state into the slot; True when it already existed."""
        _require(scenario)
        if self._active == scenario:
            return True
        # The occupant is about to leave the slot, so its file must be on
        # disk before anything else can overwrite the state it holds.
        self.wait_for_persist()
        snapshot = self._snapshots.get(scenario)
        existed = snapshot is not None
        if snapshot is None:
            snapshot = self._load_persisted(scenario)
            existed = snapshot is not None
        if snapshot is None:
            snapshot = self._pristine
        self._apply(snapshot)
        self._active = scenario
        if not existed:
            # A first-time scenario starts from the pristine adapter; record
            # it so a later capture/persist has a baseline to compare against.
            self._snapshots[scenario] = self._snapshot()
        return existed

    # -- Persistence -----------------------------------------------------

    def persist(self, scenario: str) -> Path | None:
        """Start writing ``scenario``'s snapshot for this rank; None without a store.

        The write runs on the switcher's writer thread and the returned path
        is where it will land. A colocated save runs with serving paused,
        while the bytes being written are a host-side copy no later step
        mutates, so the write overlaps the publication that follows instead
        of extending the pause. :meth:`wait_for_persist` joins it, and both
        a slot movement and the next ``persist`` join it first: a scenario's
        file is durable before its state can leave the slot, and a write that
        failed is reported rather than dropped.
        """
        _require(scenario)
        if self._store_dir is None:
            return None
        snapshot = self._snapshots.get(scenario)
        if snapshot is None:
            raise RuntimeError(f"scenario {scenario!r} has no captured adapter state to persist")
        self.wait_for_persist()
        path = self.snapshot_path(scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": SNAPSHOT_FORMAT, "scenario": scenario, **snapshot}
        self._pending_write = self._writer.submit(_write_snapshot, path, payload)
        return path

    def wait_for_persist(self) -> None:
        """Finish the pending snapshot write, raising the failure it hit."""
        pending, self._pending_write = self._pending_write, None
        if pending is not None:
            pending.result()

    def snapshot_path(self, scenario: str) -> Path:
        if self._store_dir is None:
            raise RuntimeError("adapter slot store is not configured")
        return self._store_dir / scenario_key(scenario) / f"rank_{self._rank:05d}.pt"

    def _load_persisted(self, scenario: str) -> dict[str, Any] | None:
        if self._store_dir is None:
            return None
        path = self.snapshot_path(scenario)
        if not path.is_file():
            return None
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(loaded, dict)
            or loaded.get("format") != SNAPSHOT_FORMAT
            or loaded.get("scenario") != scenario
        ):
            raise RuntimeError(f"adapter snapshot {path} is not a rank snapshot for scenario {scenario!r}")
        snapshot = {key: loaded[key] for key in ("adapter", "main", "state", "schedule")}
        self._validate(snapshot, path)
        self._snapshots[scenario] = snapshot
        logger.info("restored persisted adapter state for scenario %r from %s", scenario, path)
        return snapshot

    # -- Snapshot mechanics ----------------------------------------------

    @torch.no_grad()
    def _snapshot(self) -> dict[str, Any]:
        adapter = {name: _to_cpu(parameter) for name, parameter in _adapter_params(self._model)}
        main: list[torch.Tensor] = []
        state: list[dict[str, Any] | None] = []
        for inner, param in _main_params(self._optimizer):
            main.append(_to_cpu(param))
            param_state = inner.state.get(param)
            state.append(None if not param_state else _to_cpu(dict(param_state)))
        return {"adapter": adapter, "main": main, "state": state, "schedule": self._schedule_state()}

    @torch.no_grad()
    def _apply(self, snapshot: dict[str, Any]) -> None:
        self._validate(snapshot)
        adapter = snapshot["adapter"]
        for name, parameter in _adapter_params(self._model):
            parameter.copy_(adapter[name].to(parameter.device, parameter.dtype), non_blocking=True)
        for (inner, param), saved, saved_state in zip(
            _main_params(self._optimizer), snapshot["main"], snapshot["state"], strict=True
        ):
            param.copy_(saved.to(param.device, param.dtype), non_blocking=True)
            if not saved_state:
                inner.state.pop(param, None)
                continue
            current = inner.state.setdefault(param, {})
            for key, value in saved_state.items():
                target = current.get(key)
                if (
                    isinstance(value, torch.Tensor)
                    and isinstance(target, torch.Tensor)
                    and target.shape == value.shape
                ):
                    target.copy_(value.to(target.device, target.dtype), non_blocking=True)
                elif isinstance(value, torch.Tensor):
                    current[key] = value.to(param.device).clone()
                else:
                    current[key] = value
        self._set_schedule_state(snapshot["schedule"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _validate(self, snapshot: dict[str, Any], path: Path | None = None) -> None:
        origin = f" ({path})" if path is not None else ""
        adapter = snapshot["adapter"]
        expected = {name: tuple(parameter.shape) for name, parameter in _adapter_params(self._model)}
        found = {name: tuple(tensor.shape) for name, tensor in adapter.items()}
        if expected != found:
            raise RuntimeError(f"adapter snapshot does not match this rank's LoRA layout{origin}")
        main_shapes = [tuple(param.shape) for _, param in _main_params(self._optimizer)]
        if [tuple(tensor.shape) for tensor in snapshot["main"]] != main_shapes:
            raise RuntimeError(f"adapter snapshot does not match this rank's optimizer shards{origin}")
        if len(snapshot["state"]) != len(main_shapes):
            raise RuntimeError(f"adapter snapshot optimizer state is malformed{origin}")

    def _schedule_state(self) -> dict[str, Any]:
        scheduler = self._scheduler
        if scheduler is None:
            return {}
        return {key: getattr(scheduler, key) for key in ("num_steps",) if hasattr(scheduler, key)}

    def _set_schedule_state(self, state: dict[str, Any]) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        for key, value in state.items():
            setattr(scheduler, key, value)
        if hasattr(scheduler, "step") and "num_steps" in state and hasattr(scheduler, "num_steps"):
            # Re-derive the learning rate for the restored position without
            # advancing it.
            with_step = getattr(scheduler, "step", None)
            if callable(with_step):
                with contextlib.suppress(TypeError):
                    with_step(0)


def _require(scenario: str) -> None:
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("adapter slot operations require a non-empty scenario name")


__all__ = ["SNAPSHOT_FORMAT", "AdapterSlotSwitcher", "scenario_key"]
