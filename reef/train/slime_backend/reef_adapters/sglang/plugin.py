"""Scheduler extensions loaded through SGLang's native plugin hook."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from reef.train.slime_backend.reef_adapters.weight_version import WeightVersion

TOKEN_WEIGHT_VERSIONS_KEY = "_reef_token_weight_versions"
SGLANG_PLUGIN_NAME = "reef"
REEF_SGLANG_PLUGIN_ENV = "SGLANG_REEF_PLUGIN"
_UNSET_WEIGHT_VERSION = object()
_PROCESSING_WEIGHT_VERSION: ContextVar[object] = ContextVar(
    "_reef_processing_weight_version",
    default=_UNSET_WEIGHT_VERSION,
)


def install_scheduler_weight_version_tracking() -> None:
    """Stamp each decoded token with the weight head that produced it."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.output_streamer import _GenerationStreamAccumulator
    from sglang.srt.server_args import get_global_server_args

    _install_weight_version_updates()
    if getattr(_GenerationStreamAccumulator, "_reef_token_weight_versions_installed", False):
        return

    original_accept = _GenerationStreamAccumulator.accept

    def current_weight_version() -> str | None:
        version = getattr(get_global_server_args(), "weight_version", None)
        return str(version) if version is not None and str(version) else None

    def accept_with_weight_version(self: Any, *, req: Any):
        output_len = len(req.output_ids_through_stop)
        if req.customized_info is None:
            req.customized_info = {}
        versions = req.customized_info.setdefault(TOKEN_WEIGHT_VERSIONS_KEY, [])
        if len(versions) < output_len:
            stamped = _PROCESSING_WEIGHT_VERSION.get()
            if stamped is _UNSET_WEIGHT_VERSION:
                stamped = current_weight_version()
            # Never let a later tokenizer observation backfill an older token.
            versions.extend([stamped] * (output_len - len(versions)))
        return original_accept(self, req=req)

    original_run_batch = Scheduler.run_batch

    def run_batch_with_weight_version(self: Any, batch: Any):
        result = original_run_batch(self, batch)
        pending = getattr(self, "_reef_result_weight_versions", None)
        if pending is None:
            pending = {}
            self._reef_result_weight_versions = pending
        pending.setdefault(id(result), []).append(current_weight_version())
        return result

    original_process_batch_result = Scheduler.process_batch_result

    def process_batch_result_with_weight_version(self: Any, batch: Any, result: Any):
        pending = getattr(self, "_reef_result_weight_versions", {})
        queued = pending.get(id(result), [])
        stamped = queued.pop(0) if queued else _UNSET_WEIGHT_VERSION
        if not queued:
            pending.pop(id(result), None)
        token = _PROCESSING_WEIGHT_VERSION.set(stamped)
        try:
            return original_process_batch_result(self, batch, result)
        finally:
            _PROCESSING_WEIGHT_VERSION.reset(token)

    original_set_internal_state = Scheduler.set_internal_state

    def set_internal_state_with_weight_version(self: Any, recv_req: Any):
        version = recv_req.server_args.pop("weight_version", None)
        result = original_set_internal_state(self, recv_req)
        if version is not None and result.updated:
            if not isinstance(version, str) or not version:
                result.updated = False
            else:
                get_global_server_args().weight_version = version
                result.server_args["weight_version"] = version
        return result

    _GenerationStreamAccumulator.accept = accept_with_weight_version
    _GenerationStreamAccumulator._reef_token_weight_versions_installed = True
    Scheduler.run_batch = run_batch_with_weight_version
    Scheduler.process_batch_result = process_batch_result_with_weight_version
    Scheduler.set_internal_state = set_internal_state_with_weight_version


@contextmanager
def _weight_update_session(updater: Any) -> Iterator[None]:
    """Hold SGLang's weight-update session open around one weight mutation.

    SGLang's RL weight path became two-phase in sglang ``6f950d95``
    (2026-07-13, before the revision this image pins):
    ``update_weights_from_distributed`` and ``update_weights_from_tensor``
    now assert a session that ``begin_weight_update`` opens — restoring
    in-place-packed weights to a loadable state — and ``end_weight_update``
    finalizes (quantization finalize, plus ``post_load_weights`` when the load
    bypassed it). Slime still issues the single-call form, so without this the
    first weight sync of any training deployment dies on that assertion and
    takes the scheduler with it.

    Both ends barrier across the TP group, so every scheduler process must
    reach the same decision. They do: each rank receives the same update
    request, and a session is opened only when none is open. A caller that
    opens its own session keeps it, and closes it itself.
    """
    begin = getattr(updater, "begin_weight_update", None)
    end = getattr(updater, "end_weight_update", None)
    # An SGLang without the two-phase protocol, or a session someone else owns.
    if begin is None or end is None or getattr(updater, "_weight_update_in_progress", False):
        yield
        return
    from sglang.srt.managers.io_struct import BeginWeightUpdateReqInput, EndWeightUpdateReqInput

    begin(BeginWeightUpdateReqInput())
    try:
        yield
    finally:
        end(EndWeightUpdateReqInput())


def _install_weight_version_updates() -> None:
    """Commit scheduler stamps with successful weight mutations.

    SGLang updates the tokenizer manager's ``weight_version`` after a load,
    while generation happens in separate scheduler processes. Keep each
    scheduler's stamp in the same success branch as its local weight update,
    and reject a delayed delta before it can overwrite a recovered full load.

    The same wrappers open SGLang's weight-update session, because they are
    already the one place every distributed and tensor update passes through;
    see :func:`_weight_update_session`.
    """
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqOutput
    from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
    from sglang.srt.server_args import get_global_server_args

    if getattr(SchedulerWeightUpdaterManager, "_reef_weight_version_updates_installed", False):
        return

    original_disk = SchedulerWeightUpdaterManager.update_weights_from_disk
    original_distributed = SchedulerWeightUpdaterManager.update_weights_from_distributed
    original_tensor = SchedulerWeightUpdaterManager.update_weights_from_tensor

    def commit_version(recv_req: Any, result: Any) -> Any:
        version = getattr(recv_req, "weight_version", None)
        if getattr(result, "success", False) and version is not None:
            if not isinstance(version, str) or not version:
                result.success = False
                result.message = "weight_version must be a non-empty string"
            else:
                get_global_server_args().weight_version = version
        return result

    @wraps(original_disk)
    def update_weights_from_disk(self: Any, recv_req: Any):
        if getattr(recv_req, "load_format", None) == "delta" and getattr(recv_req, "weight_version", None) is not None:
            try:
                incoming = WeightVersion.parse(recv_req.weight_version)
                current = WeightVersion.parse(get_global_server_args().weight_version)
            except (TypeError, ValueError):
                return UpdateWeightFromDiskReqOutput(
                    success=False,
                    message="delta updates require canonical current and incoming weight versions",
                    num_paused_requests=0,
                )
            if incoming.incarnation != current.incarnation or incoming.sequence < current.sequence:
                return UpdateWeightFromDiskReqOutput(
                    success=False,
                    message=f"refusing stale delta weight version {incoming}; current serving version is {current}",
                    num_paused_requests=0,
                )
        return commit_version(recv_req, original_disk(self, recv_req))

    @wraps(original_distributed)
    def update_weights_from_distributed(self: Any, recv_req: Any):
        with _weight_update_session(self):
            return commit_version(recv_req, original_distributed(self, recv_req))

    @wraps(original_tensor)
    def update_weights_from_tensor(self: Any, recv_req: Any):
        with _weight_update_session(self):
            return commit_version(recv_req, original_tensor(self, recv_req))

    SchedulerWeightUpdaterManager.update_weights_from_disk = update_weights_from_disk
    SchedulerWeightUpdaterManager.update_weights_from_distributed = update_weights_from_distributed
    SchedulerWeightUpdaterManager.update_weights_from_tensor = update_weights_from_tensor
    SchedulerWeightUpdaterManager._reef_weight_version_updates_installed = True


def install_colocated_retract_offload() -> None:
    """Allow paused requests to survive SGLang's colocated memory offload.

    SGLang's ``retract`` pause releases every running request's KV allocation
    and moves its CPU-side request state into ``waiting_queue``.  Its generic
    memory-saver guard nevertheless defines *fully idle* as an empty waiting
    queue, which prevents ``release_memory_occupation`` and the cache flush
    used by weight updates from consuming that already-released state.

    Reef closes admission before this pause, so no new durable request can
    bind the old head.  While the engine is paused, treat the waiting queue as
    CPU-resident suspended work for destructive GPU-memory operations.  All
    other scheduler queues and outstanding batches still have to satisfy
    SGLang's original idle predicate.  On resume, SGLang re-prefills these
    requests under the newly committed weights.
    """
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_reef_colocated_retract_offload_installed", False):
        return

    original_is_fully_idle = Scheduler.is_fully_idle
    idle_signature = inspect.signature(original_is_fully_idle)
    supports_ignore_waiting = "ignore_waiting" in idle_signature.parameters
    original_pause_generation = Scheduler.pause_generation
    original_continue_generation = Scheduler.continue_generation

    @wraps(original_pause_generation)
    def pause_generation_with_mode(self: Any, recv_req: Any):
        result = original_pause_generation(self, recv_req)
        self._reef_pause_mode = getattr(recv_req, "mode", None)
        return result

    @wraps(original_continue_generation)
    def continue_generation_with_mode(self: Any, recv_req: Any):
        result = original_continue_generation(self, recv_req)
        self._reef_pause_mode = None
        return result

    @wraps(original_is_fully_idle)
    def is_offload_idle_with_suspended_requests(self: Any, *args: Any, **kwargs: Any) -> bool:
        bound = idle_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        for_health_check = bool(bound.arguments.get("for_health_check", False))
        if for_health_check or getattr(self, "_reef_pause_mode", None) != "retract":
            return original_is_fully_idle(*bound.args, **bound.kwargs)
        if supports_ignore_waiting:
            bound.arguments["ignore_waiting"] = True
            return original_is_fully_idle(*bound.args, **bound.kwargs)
        waiting = self.waiting_queue
        if not waiting:
            return original_is_fully_idle(*bound.args, **bound.kwargs)
        # Scheduler control messages run serially on this process. Temporarily
        # hide only the CPU-side suspended queue while the upstream predicate
        # proves that no GPU batch, overlap result, grammar task, or
        # disaggregation transfer remains active.
        self.waiting_queue = []
        try:
            return original_is_fully_idle(*bound.args, **bound.kwargs)
        finally:
            self.waiting_queue = waiting

    Scheduler.pause_generation = pause_generation_with_mode
    Scheduler.continue_generation = continue_generation_with_mode
    Scheduler.is_fully_idle = is_offload_idle_with_suspended_requests
    Scheduler._reef_colocated_retract_offload_installed = True


def install_sglang_plugin() -> None:
    """Install hooks when a Reef-configured SGLang process loads plugins."""
    if os.environ.get(REEF_SGLANG_PLUGIN_ENV) != "1":
        return
    install_scheduler_weight_version_tracking()
    # This patch is dormant unless generation is paused in retract mode, so
    # installing it for both Reef layouts keeps one native plugin lifecycle.
    install_colocated_retract_offload()
