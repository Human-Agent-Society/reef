"""The vLLM version connector stamps every sampled token with the weight version that produced it."""

from __future__ import annotations

import enum
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from reef.inference.vllm.versions import TOKEN_RUNTIME_LOAD_IDS_KEY, UNKNOWN_RUNTIME_LOAD_ID, TokenVersionTracker

# -- tracker ------------------------------------------------------------------------


@pytest.mark.unit
def test_tracker_splits_tokens_at_the_step_where_the_version_changed() -> None:
    tracker = TokenVersionTracker("engine:1")
    tracker.note("r", 0)
    tracker.note("r", 3)  # same version: no new event
    tracker.set_version("engine:2")
    tracker.note("r", 5)  # first step under the new version produces token 5
    tracker.note("r", 6)
    assert tracker.finish("r", 8) == ["engine:1"] * 5 + ["engine:2"] * 3
    assert tracker.finish("r", 8) == [UNKNOWN_RUNTIME_LOAD_ID] * 8  # events are gone after finish


@pytest.mark.unit
def test_tracker_marks_tokens_it_never_saw_scheduled_and_handles_empty_requests() -> None:
    tracker = TokenVersionTracker("engine:1")
    tracker.note("late", 2)
    assert tracker.finish("late", 4) == [UNKNOWN_RUNTIME_LOAD_ID, UNKNOWN_RUNTIME_LOAD_ID, "engine:1", "engine:1"]
    assert tracker.finish("aborted-before-first-token", 0) == []
    tracker.note("dropped", 0)
    tracker.forget("dropped")
    assert tracker.finish("dropped", 1) == [UNKNOWN_RUNTIME_LOAD_ID]


@pytest.mark.unit
def test_tracker_lets_a_version_bump_during_prefill_win_for_that_index() -> None:
    tracker = TokenVersionTracker("engine:1")
    tracker.note("r", 0)  # prefill chunk one
    tracker.set_version("engine:2")
    tracker.note("r", 0)  # prefill chunk two, still no token produced
    assert tracker.finish("r", 2) == ["engine:2", "engine:2"]


# -- connector, against stubbed vLLM modules ---------------------------------------


def _import_connector_with_fake_vllm(monkeypatch):
    """Install the vLLM symbols the connector imports, then import it fresh."""
    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.v1",
        "vllm.v1.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:  # noqa: N801 - vLLM's spelling
        def __init__(self, vllm_config, role, kv_cache_config):
            self.init_args = (vllm_config, role, kv_cache_config)

    class SupportsHMA:
        pass

    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorRole = KVConnectorRole
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.SupportsHMA = SupportsHMA

    class EngineCore:
        def __init__(self):
            self._weight_version = "default"

        def set_weight_version(self, weight_version):
            self._weight_version = weight_version

    core = types.ModuleType("vllm.v1.engine.core")
    core.EngineCore = EngineCore
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, core.__name__, core)
    monkeypatch.delitem(sys.modules, "reef.inference.vllm.connector", raising=False)
    module = importlib.import_module("reef.inference.vllm.connector")
    return module, KVConnectorRole, EngineCore, KVConnectorMetadata


def _request(request_id: str, *, output_tokens: int = 0, placeholders: int = 0):
    return SimpleNamespace(
        request_id=request_id,
        num_output_tokens=output_tokens,
        num_output_placeholders=placeholders,
        output_token_ids=list(range(output_tokens)),
    )


def _step(*, new=(), cached=()):
    """A SchedulerOutput with the fields the connector reads."""
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id) for request_id in new],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[request_id for request_id, _ in cached],
            num_output_tokens=[count for _, count in cached],
        ),
    )


@pytest.mark.unit
def test_connector_stamps_a_request_that_spans_a_weight_update(monkeypatch) -> None:
    module, role, engine_core_type, metadata_type = _import_connector_with_fake_vllm(monkeypatch)
    connector = module.ReefVersionConnector("config", role.SCHEDULER, "kv-cache-config")
    assert connector.init_args == ("config", role.SCHEDULER, "kv-cache-config")

    request = _request("r1")
    connector.on_new_request(request)
    assert isinstance(connector.build_connector_meta(_step(new=["r1"])), metadata_type)
    request.num_output_tokens = 1
    connector.build_connector_meta(_step(cached=[("r1", 1)]))

    engine_core = engine_core_type()
    engine_core.set_weight_version("engine:7")  # the observer sees /update_weight_version
    assert engine_core._weight_version == "engine:7"
    request.num_output_tokens = 5
    connector.build_connector_meta(_step(cached=[("r1", 5)]))

    request.output_token_ids = list(range(8))
    finished, params = connector.request_finished_all_groups(request, ([1, 2],))
    assert finished is False
    assert params == {TOKEN_RUNTIME_LOAD_IDS_KEY: ["default"] * 5 + ["engine:7"] * 3}
    assert connector.request_finished(_request("never-scheduled"), []) == (False, {TOKEN_RUNTIME_LOAD_IDS_KEY: []})


@pytest.mark.unit
def test_connector_reads_the_index_from_the_live_request_for_resumed_and_async_batches(monkeypatch) -> None:
    module, role, engine_core_type, _ = _import_connector_with_fake_vllm(monkeypatch)
    connector = module.ReefVersionConnector(None, role.SCHEDULER, None)
    request = _request("r1")
    connector.on_new_request(request)
    connector.build_connector_meta(_step(new=["r1"]))
    engine_core_type().set_weight_version("engine:2")
    # Retracted and resumed: vLLM lists the request as new again, but it kept its 4 tokens.
    request.num_output_tokens = 4
    connector.build_connector_meta(_step(new=["r1"]))
    # Async scheduling: one token scheduled but not yet sampled counts as produced.
    request.num_output_tokens, request.num_output_placeholders = 5, 1
    engine_core_type().set_weight_version("engine:3")
    connector.build_connector_meta(_step(cached=[("r1", 6)]))
    request.output_token_ids = list(range(7))
    _, params = connector.request_finished(request, [])
    assert params[TOKEN_RUNTIME_LOAD_IDS_KEY] == ["default"] * 4 + ["engine:2"] * 2 + ["engine:3"]


@pytest.mark.unit
def test_worker_role_does_not_observe_versions_and_the_observer_installs_once(monkeypatch) -> None:
    module, role, engine_core_type, _ = _import_connector_with_fake_vllm(monkeypatch)
    original = engine_core_type.set_weight_version
    module.ReefVersionConnector(None, role.WORKER, None)
    assert engine_core_type.set_weight_version is original
    first = module.ReefVersionConnector(None, role.SCHEDULER, None)
    wrapped = engine_core_type.set_weight_version
    second = module.ReefVersionConnector(None, role.SCHEDULER, None)
    assert engine_core_type.set_weight_version is wrapped
    engine_core_type().set_weight_version("engine:9")
    assert first._tracker.version == second._tracker.version == "engine:9"
    assert first.start_load_kv(None) is None and first.wait_for_save() is None
    assert first.get_num_new_matched_tokens(_request("r"), 0) == (0, False)


@pytest.mark.unit
def test_chat_client_refuses_a_response_whose_stamps_include_unknown() -> None:
    from reef.artifact import Artifact, LiveWeightArtifactRef
    from reef.inference.vllm.chat import VLLMGenerateClient

    artifact = Artifact(
        LiveWeightArtifactRef(
            content_id="live", release_id="live:engine:1", parent_release_id="checkpoint", runtime_load_id="engine:1"
        ),
        None,
    )
    response = {
        "choices": [
            {
                "index": 0,
                "token_ids": [20, 21],
                "logprobs": {
                    "content": [{"token": "token_id:20", "logprob": -0.1}, {"token": "token_id:21", "logprob": -0.2}]
                },
                "finish_reason": "stop",
            }
        ],
        "kv_transfer_params": {TOKEN_RUNTIME_LOAD_IDS_KEY: [UNKNOWN_RUNTIME_LOAD_ID, "engine:1"]},
    }
    with pytest.raises(ValueError, match="incomplete token runtime load IDs"):
        VLLMGenerateClient().parse(artifact, response, capture_topk=0)
