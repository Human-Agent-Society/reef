"""Scheduler extensions loaded through SGLang's native plugin hook."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from reef.inference.interfaces import RuntimeLoadId

REEF_SGLANG_PLUGIN_ENV = "SGLANG_REEF_PLUGIN"


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


def install_runtime_load_id_updates() -> None:
    """Check the runtime load ID of every weight update that a scheduler receives.

    A successful update must carry a non-empty runtime load ID when it carries
    one. A delayed delta is rejected before it can overwrite a recovered full load.

    The same wrappers open SGLang's weight-update session, because they are
    already the one place every distributed and tensor update passes through;
    see :func:`_weight_update_session`.
    """
    from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqOutput
    from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
    from sglang.srt.runtime_context import get_serving

    if getattr(SchedulerWeightUpdaterManager, "_reef_runtime_load_id_updates_installed", False):
        return

    original_disk = SchedulerWeightUpdaterManager.update_weights_from_disk
    original_distributed = SchedulerWeightUpdaterManager.update_weights_from_distributed
    original_tensor = SchedulerWeightUpdaterManager.update_weights_from_tensor

    def commit_version(recv_req: Any, result: Any) -> Any:
        # Weight-update requests use SGLang's field name "weight_version" for
        # what Reef calls the runtime-load-ID. SGLang's updater records it on
        # the scheduler after a successful load; Reef refuses an empty one.
        version = getattr(recv_req, "weight_version", None)
        if getattr(result, "success", False) and version is not None and (not isinstance(version, str) or not version):
            result.success = False
            result.message = "weight_version must be a non-empty string"
        return result

    @wraps(original_disk)
    def update_weights_from_disk(self: Any, recv_req: Any):
        if getattr(recv_req, "load_format", None) == "delta" and getattr(recv_req, "weight_version", None) is not None:
            try:
                incoming = RuntimeLoadId.parse(recv_req.weight_version)
                current = RuntimeLoadId.parse(get_serving().weight_version)
            except (TypeError, ValueError):
                return UpdateWeightFromDiskReqOutput(
                    success=False,
                    message="delta updates require canonical current and incoming runtime load IDs",
                    num_paused_requests=0,
                )
            if incoming.incarnation != current.incarnation or incoming.sequence < current.sequence:
                return UpdateWeightFromDiskReqOutput(
                    success=False,
                    message=f"refusing stale delta runtime load ID {incoming}; current serving version is {current}",
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
    SchedulerWeightUpdaterManager._reef_runtime_load_id_updates_installed = True


def install_colocated_retract_offload() -> None:
    """Allow paused requests to survive SGLang's colocated memory offload.

    SGLang's ``retract`` pause releases every running request's KV allocation
    and moves its CPU-side request state into ``waiting_queue``.  Its generic
    memory-saver guard nevertheless defines *fully idle* as an empty waiting
    queue, which prevents ``release_memory_occupation`` and the cache flush
    used by weight updates from consuming that already-released state.

    Reef closes admission before this pause, so no new durable request can
    bind the old head. While the engine is paused, waiting requests and requests
    awaiting grammar compilation are CPU-resident work without GPU KV. Grammar
    polling stops during a pause, so requiring that queue to drain would prevent
    every subsequent flush. Preserve both queues while checking all outstanding
    GPU work with SGLang's original predicate. Its grammar-cache reset clears
    compiled cache entries, not queued requests or their compilation futures.
    On resume, those requests continue through normal grammar polling and prefill.
    """
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_reef_colocated_retract_offload_installed", False):
        return

    original_is_fully_idle = Scheduler.is_fully_idle
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
    def is_offload_idle_with_suspended_requests(
        self: Any, for_health_check: bool = False, ignore_waiting: bool = False
    ) -> bool:
        if for_health_check or getattr(self, "_reef_pause_mode", None) != "retract":
            return original_is_fully_idle(self, for_health_check=for_health_check, ignore_waiting=ignore_waiting)
        # Scheduler control messages run serially on this process. Temporarily
        # hide the CPU-only grammar queue and use the pinned SGLang API to
        # ignore waiting requests. All GPU idle checks remain active.
        grammar_queue = self.grammar_manager.grammar_queue
        self.grammar_manager.grammar_queue = []
        try:
            return original_is_fully_idle(self, for_health_check=False, ignore_waiting=True)
        finally:
            self.grammar_manager.grammar_queue = grammar_queue

    Scheduler.pause_generation = pause_generation_with_mode
    Scheduler.continue_generation = continue_generation_with_mode
    Scheduler.is_fully_idle = is_offload_idle_with_suspended_requests
    Scheduler._reef_colocated_retract_offload_installed = True


def install_sglang_plugin() -> None:
    """Install hooks when a Reef-configured SGLang process loads plugins."""
    if os.environ.get(REEF_SGLANG_PLUGIN_ENV) != "1":
        return
    install_runtime_load_id_updates()
    # This patch is dormant unless generation is paused in retract mode, so
    # installing it for both Reef layouts keeps one native plugin lifecycle.
    install_colocated_retract_offload()
