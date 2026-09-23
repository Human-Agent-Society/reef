"""Reef's vLLM KV connector: stamp every sampled token with the weight version that produced it.

Select it on the engine with::

    --kv-transfer-config '{"kv_connector": "ReefConnector",
                           "kv_connector_module_path": "reef.inference.vllm.connector",
                           "kv_role": "kv_both"}'

The connector moves no KV. Its scheduler-side role sees every scheduled batch
in ``build_connector_meta`` (called inside ``schedule()``, before the forward
pass) and records, per request, the output index from which the current weight
version applies. When a request finishes, ``request_finished`` returns the
per-token versions, which vLLM places into the response's
``kv_transfer_params``. Reading the version at schedule time is what keeps the
stamps exact under async scheduling and across a pause for a weight update.

vLLM exposes the engine's weight version only on ``EngineCore``, which the
connector cannot reach, so the scheduler-side role wraps
``EngineCore.set_weight_version`` once to observe ``/update_weight_version`` and
``/finish_weight_update``. That wrapper is the one non-public vLLM symbol this
module touches. This module is imported only inside vLLM processes.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.engine.core import EngineCore

from reef.inference.vllm.versions import TOKEN_RUNTIME_LOAD_IDS_KEY, TokenVersionTracker

logger = logging.getLogger(__name__)

_observing_trackers: list[TokenVersionTracker] = []
_observer_installed = False


def install_weight_version_observer(tracker: TokenVersionTracker) -> None:
    """Forward every engine weight-version change to ``tracker``, wrapping ``EngineCore`` once per process."""
    global _observer_installed
    _observing_trackers.append(tracker)
    if _observer_installed:
        return
    original_set_weight_version = EngineCore.set_weight_version

    def set_weight_version(self: Any, weight_version: str) -> None:
        original_set_weight_version(self, weight_version)
        for observing in _observing_trackers:
            observing.set_version(weight_version)
        logger.info("engine weight version is now %r", weight_version)

    EngineCore.set_weight_version = set_weight_version
    _observer_installed = True


class ReefConnector(KVConnectorBase_V1, SupportsHMA):
    """A KV connector that carries per-token runtime load IDs and no KV."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._tracker = TokenVersionTracker()
        # Live requests by id: their output-token count survives preemption,
        # while a NewRequestData for a resumed request would suggest index zero.
        self._requests: dict[str, Any] = {}
        if role == KVConnectorRole.SCHEDULER:
            install_weight_version_observer(self._tracker)
            logger.info("Reef connector installed on the scheduler")

    # -- Scheduler side -----------------------------------------------------------

    def on_new_request(self, request: Any) -> None:
        self._requests[request.request_id] = request

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        self._requests[request.request_id] = request

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        for new_request in scheduler_output.scheduled_new_reqs:
            self._tracker.note(new_request.req_id, self._output_index(new_request.req_id, 0))
        cached = scheduler_output.scheduled_cached_reqs
        for request_id, output_count in zip(cached.req_ids, cached.num_output_tokens, strict=True):
            self._tracker.note(request_id, self._output_index(request_id, output_count))
        return KVConnectorMetadata()

    def _output_index(self, request_id: str, fallback: int) -> int:
        """The index of the token this step produces: sampled tokens plus async-scheduling placeholders."""
        request = self._requests.get(request_id)
        if request is None:
            return fallback
        return int(request.num_output_tokens) + int(request.num_output_placeholders)

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self._finish(request)

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._finish(request)

    def _finish(self, request: Any) -> tuple[bool, dict[str, Any]]:
        self._requests.pop(request.request_id, None)
        stamps = self._tracker.finish(request.request_id, len(request.output_token_ids))
        return False, {TOKEN_RUNTIME_LOAD_IDS_KEY: stamps}

    # -- Worker side: nothing to load or save -------------------------------------

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        return None


__all__ = ["ReefConnector", "install_weight_version_observer"]
