"""Slime health probes use captured identities and bounded Ray operations."""

from types import SimpleNamespace

import pytest

from reef.runtime.sglang import health


class Engine:
    def __init__(self, name):
        self.name = name
        self.health_generate = SimpleNamespace(remote=lambda **kwargs: (name, "probe", kwargs))
        self.shutdown = SimpleNamespace(remote=lambda: (name, "shutdown"))


@pytest.fixture
def transport(monkeypatch):
    calls = []
    kills = []

    def get(value, **kwargs):
        calls.append((value, kwargs))
        return True

    monkeypatch.setattr(health, "ray", SimpleNamespace(get=get, kill=lambda target, **kwargs: kills.append(target)))
    return calls, kills


def test_probe_and_group_shutdown_have_outer_ray_deadlines(transport):
    calls, kills = transport
    engines = [Engine("leader"), Engine("follower")]
    group = SimpleNamespace(all_engines=list(engines), nodes_per_engine=2)
    target = health.SGLangEngineHealthChecks(group).targets()[0]
    target.check(3)
    target.retire(4)
    assert calls == [
        (("leader", "probe", {"timeout": 3}), {"timeout": 3}),
        ([("leader", "shutdown"), ("follower", "shutdown")], {"timeout": 4}),
    ]
    assert kills == engines
    assert group.all_engines == [None, None]


def test_failed_old_probe_cannot_retire_replacement_in_same_slot(transport):
    _, kills = transport
    old = Engine("old")
    replacement = Engine("new")
    group = SimpleNamespace(all_engines=[old], nodes_per_engine=1)
    target = health.SGLangEngineHealthChecks(group).targets()[0]
    group.all_engines[0] = replacement
    target.retire(1)
    assert kills == []
    assert group.all_engines == [replacement]


def test_slot_replaced_during_shutdown_is_not_cleared_or_killed(transport, monkeypatch):
    _, kills = transport
    old, replacement = Engine("old"), Engine("new")
    group = SimpleNamespace(all_engines=[old], nodes_per_engine=1)
    target = health.SGLangEngineHealthChecks(group).targets()[0]

    def finish_shutdown(*args, **kwargs):
        group.all_engines[0] = replacement

    monkeypatch.setattr(health.ray, "get", finish_shutdown)
    target.retire(1)
    assert kills == [old]
    assert group.all_engines == [replacement]


def test_shutdown_timeout_still_retires_captured_ray_handles(transport, monkeypatch):
    _, kills = transport
    engine = Engine("unresponsive")
    group = SimpleNamespace(all_engines=[engine], nodes_per_engine=1)

    def timeout(*args, **kwargs):
        raise TimeoutError("shutdown")

    monkeypatch.setattr(health.ray, "get", timeout)
    health.SGLangEngineHealthChecks(group).targets()[0].retire(1)
    assert kills == [engine]
    assert group.all_engines == [None]


def test_snapshot_skips_missing_leaders_and_captures_each_replica(transport):
    group = SimpleNamespace(all_engines=[None, Engine("orphan"), Engine("leader"), None], nodes_per_engine=2)
    targets = health.SGLangEngineHealthChecks(group).targets()
    assert len(targets) == 1
    targets[0].check(1)
    assert transport[0][0][0][0] == "leader"


def test_retirement_attempts_all_nodes_and_reports_kill_failure(transport, monkeypatch):
    engines = [Engine("leader"), Engine("follower")]
    group = SimpleNamespace(all_engines=list(engines), nodes_per_engine=2)
    attempted = []

    def kill(engine, **kwargs):
        attempted.append(engine)
        if engine is engines[0]:
            raise RuntimeError("kill failed")

    monkeypatch.setattr(health.ray, "kill", kill)
    with pytest.raises(RuntimeError, match="kill failed"):
        health.SGLangEngineHealthChecks(group).targets()[0].retire(1)
    assert attempted == engines
    assert group.all_engines == [engines[0], None]


def test_false_probe_result_is_a_failure(transport, monkeypatch):
    monkeypatch.setattr(health.ray, "get", lambda *args, **kwargs: False)
    group = SimpleNamespace(all_engines=[Engine("failed")], nodes_per_engine=1)
    with pytest.raises(RuntimeError, match="did not report success"):
        health.SGLangEngineHealthChecks(group).targets()[0].check(1)
