"""The bridge time-slices one adapter slot between scenarios (one LoRA slot, several scenarios)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ray")

from reef_service.test_sao_bridge import _RecordingGroup

from reef.runtime.adapter_residency import AdapterCapacityExhausted
from reef.train.slime_backend.reef_adapters import bridge
from reef.train.slime_backend.reef_adapters.megatron.lora import scenario_adapter_name
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioLedger, ledger_path

from .test_sao_bridge import _FakeRank, _FakeRolloutManager, _payload, _RemoteMethod, _sao_row

INCARNATION = "inc"


class _EngineVersion:
    """One engine-global weight version shared by the group and the rollout manager."""

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence

    def __str__(self) -> str:
        return f"{INCARNATION}:{self.sequence}"


class _SlottedGroup:
    """Actor group with one adapter slot; records which scenario occupies it."""

    def __init__(self, template: str, version: _EngineVersion) -> None:
        self.template = template
        self.version = version
        self.active: str | None = None
        self.activations: list[str] = []
        self.published: list[tuple[str, str]] = []
        self.publications: list[tuple[str, str]] = []
        self.train_calls: list[tuple[int, str | None]] = []
        self._actor_handlers = [_FakeRank(version=str(version))]
        self._actor_handlers[0].get_weight_version = _RemoteMethod(lambda: str(self.version))

    def activate_scenario(self, scenario: str) -> bool:
        existed = scenario in set(self.activations)
        self.active = scenario
        self.activations.append(scenario)
        return existed

    def publish_adapter(self, scenario: str, lora_name: str) -> None:
        self.published.append((scenario, lora_name))

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append((rollout_id, self.active))
        return [{"values": [0.25]}]

    def async_pop_rank0_metrics(self):
        return self._actor_handlers[0].pop_metrics.remote()

    def async_get_rank0_weight_version(self):
        return self._actor_handlers[0].get_weight_version.remote()

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
        del manage_generation, force_full
        self.version.sequence += 1
        self.publications.append((self.active or "?", str(self.version)))

    def sync_serving_weight_version(self) -> str:
        return str(self.version)

    def restore_weight_version_for_republication(self, weight_version):
        self.version.sequence = int(weight_version.rsplit(":", 1)[1]) - 1

    def save_model(self, rollout_id, force_sync=False):
        checkpoint = Path(self.template.format(rollout_id=rollout_id))
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("hf", encoding="utf-8")


class _Engine:
    """One SGLang engine actor: records the adapter names it is told to drop."""

    def __init__(self) -> None:
        self.unloaded: list[str] = []
        self.refuse: set[str] = set()
        self.unload_lora_adapter = _RemoteMethod(self._unload)

    def _unload(self, *, lora_name: str) -> dict:
        if lora_name in self.refuse:
            return {"success": False, "message": f"engine keeps {lora_name}"}
        self.unloaded.append(lora_name)
        return {"success": True}


class _Manager(_FakeRolloutManager):
    def __init__(self, version: _EngineVersion) -> None:
        super().__init__(["packed"])
        self.inference_url = _RemoteMethod(lambda: "http://10.0.0.7:30000")
        self.get_weight_versions = _RemoteMethod(lambda: [str(version)])
        self.paused: list[str] = []
        self.pause_generation_for_update = _RemoteMethod(lambda: self.paused.append("pause"))
        self.continue_generation_after_update = _RemoteMethod(lambda: self.paused.append("continue"))
        self.engine = _Engine()
        self.get_updatable_engines_and_lock = _RemoteMethod(lambda: ([self.engine], None, 0, [], [], []))


def _actor(
    tmp_path: Path,
    version: _EngineVersion,
    *,
    start_rollout_id: int = 0,
    adapter_capacity: int | None = None,
):
    template = str(tmp_path / "hf" / "checkpoint-{rollout_id}")
    group = _SlottedGroup(template, version)
    manager = _Manager(version)
    actor = bridge.TrainBridgeActorImpl(
        group,
        manager,
        save_hf_template=template,
        start_rollout_id=start_rollout_id,
        lora=True,
        adapter_capacity=adapter_capacity,
        critic_group=_RecordingGroup(template, critic=True),
        loss_family="sao",
    )
    return actor, group, manager, template


def _job(scenario: str, step: int, producing: str, *, max_staleness: int | None = None) -> dict:
    payload = _payload([_sao_row(f"{scenario}-{step}", producing_weight_version=producing)])
    payload.update(scenario=scenario, rollout_id=step, expected_weight_version=producing)
    if max_staleness is not None:
        payload.update(max_staleness=max_staleness, producing_weight_versions=[producing])
    return payload


def _run(actor, payload):
    result = actor.execute_training_job(payload)
    if result.outcome != "checkpoint":
        return result
    result = actor.update_serving_weights(result.training_job_id)
    actor.acknowledge_training_commit(result.training_job_id)
    return result


@pytest.fixture
def _local_ray_get(monkeypatch):
    monkeypatch.setattr(bridge.ray, "get", lambda value, **kwargs: value)


@pytest.mark.unit
def test_scenarios_take_turns_in_the_slot_and_publish_versioned_names(tmp_path, _local_ray_get) -> None:
    # A LoRA bridge that never trained publishes nothing at startup; every
    # training publication advances the engine version by one.
    version = _EngineVersion(0)
    actor, group, _, template = _actor(tmp_path, version)

    a1 = _run(actor, _job("a", 0, "inc:0"))
    assert a1.outcome == "complete" and a1.weight_version == "inc:1"
    b1 = _run(actor, _job("b", 0, "inc:1"))  # b's rollouts came from the engine after a published
    assert b1.outcome == "complete" and b1.weight_version == "inc:2"
    a2 = _run(actor, _job("a", 1, "inc:2"))
    assert a2.outcome == "complete" and a2.weight_version == "inc:3"

    # Each job activated its own scenario before training; the bridge's
    # checkpoint index stays one sequence while scenario steps are per scenario.
    assert group.train_calls == [(0, "a"), (1, "b"), (2, "a")]
    assert group.publications == [("a", "inc:1"), ("b", "inc:2"), ("a", "inc:3")]
    assert Path(template.format(rollout_id=2)).is_dir()

    ledger = ScenarioLedger(ledger_path(template))
    assert ledger.status()["a"] == {
        "weight_version": "inc:3",
        "adapter": scenario_adapter_name("a", "inc:3"),
        "publications": 2,
        "rollout_id": 2,
        "steps": 2,
    }
    assert ledger.status()["b"]["adapter"] == scenario_adapter_name("b", "inc:2")
    health = actor.health()
    assert health["lora_mode"] == "scenario" and health["lora_adapter"] is None
    assert health["lora_adapters"]["b"]["weight_version"] == "inc:2"
    assert health["training_job"]["scenario"] == "a" and health["training_job"]["rollout_id"] == 1


@pytest.mark.unit
def test_staleness_counts_only_the_scenarios_own_publications(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, _, _, _ = _actor(tmp_path, version)
    assert _run(actor, _job("a", 0, "inc:0")).outcome == "complete"  # engine now inc:1
    assert _run(actor, _job("b", 0, "inc:1")).outcome == "complete"  # engine now inc:2

    # a's rollout produced at inc:1 (right after a published) is still fresh
    # for a even though b moved the engine to inc:2.
    assert _run(actor, _job("a", 1, "inc:1")).outcome == "complete"  # engine inc:3
    # A rollout produced before a's latest publication is stale for a ...
    stale = _run(actor, _job("a", 2, "inc:2"))
    assert stale.outcome == "stale"
    assert stale.metrics["staleness/drop_reason"] == "policy_lag_exceeded"
    # ... unless the policy admits one publication of lag.
    assert _run(actor, _job("a", 2, "inc:2", max_staleness=1)).outcome == "complete"
    # b has published once; a's two later publications do not age b's rollouts.
    assert _run(actor, _job("b", 1, "inc:2")).outcome == "complete"
    assert _run(actor, _job("b", 2, "old:1")).outcome == "stale"


@pytest.mark.unit
def test_per_scenario_jobs_must_name_their_scenario(tmp_path, _local_ray_get) -> None:
    actor, _, _, _ = _actor(tmp_path, _EngineVersion(0))
    payload = _job("a", 0, "inc:0")
    del payload["scenario"]
    with pytest.raises(ValueError, match="name their scenario"):
        actor.execute_training_job(payload)


@pytest.mark.unit
def test_restart_re_registers_every_scenario_before_serving(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, _, _, _ = _actor(tmp_path, version)
    _run(actor, _job("a", 0, "inc:0"))
    _run(actor, _job("b", 0, "inc:1"))
    _run(actor, _job("c", 0, "inc:2"))
    _run(actor, _job("a", 1, "inc:3"))  # marker: a, published inc:4

    # A new bridge over the same checkpoints (engines restarted at inc:0).
    restarted_version = _EngineVersion(4)
    restarted, group, manager, _ = _actor(tmp_path, restarted_version, start_rollout_id=4)
    # Other scenarios came back under their recorded names; the marker's
    # scenario was activated last and republished under its recorded version.
    assert group.published == [
        ("b", scenario_adapter_name("b", "inc:2")),
        ("c", scenario_adapter_name("c", "inc:3")),
    ]
    assert group.activations[-1] == "a"
    assert group.publications == [("a", "inc:4")]
    assert manager.paused == ["pause", "continue"]
    assert restarted.health()["phase"] == "serving"
    assert restarted.serving_weight_version() == "inc:4"
    # The residency manager starts from the reloaded set, not from empty.
    residency = restarted.health()["adapter_residency"]
    assert {name: block["current"]["version"] for name, block in residency["scenarios"].items()} == {
        "a": "inc:4",
        "b": "inc:2",
        "c": "inc:3",
    }
    assert residency["counters"]["loads"] == 3 and manager.engine.unloaded == []
    # Training resumes with the next global checkpoint index and per-scenario steps.
    result = _run(restarted, _job("b", 1, "inc:2"))
    assert result.outcome == "complete" and group.train_calls == [(4, "b")]


@pytest.mark.unit
def test_publication_evicts_the_oldest_superseded_revision_never_a_peers_current(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, _, manager, _ = _actor(tmp_path, version, adapter_capacity=3)
    _run(actor, _job("a", 0, "inc:0"))  # a: inc:1
    _run(actor, _job("b", 0, "inc:1"))  # b: inc:2
    _run(actor, _job("a", 1, "inc:2"))  # a: inc:3; inc:1 is superseded but still fits
    assert manager.engine.unloaded == []
    residency = actor.health()["adapter_residency"]
    assert residency == {**residency, "capacity": 3, "resident": 3, "leaked": 0}
    assert residency["scenarios"]["a"]["resident"] == ["inc:1", "inc:3"]

    _run(actor, _job("b", 1, "inc:3"))  # b: inc:4 needs a slot: a's inc:1 goes, never b's inc:2
    assert manager.engine.unloaded == [scenario_adapter_name("a", "inc:1")]
    residency = actor.health()["adapter_residency"]
    assert residency["scenarios"]["a"]["resident"] == ["inc:3"]
    assert residency["scenarios"]["b"]["resident"] == ["inc:2", "inc:4"]
    assert residency["scenarios"]["b"]["current"] == {
        "version": "inc:4",
        "adapter": scenario_adapter_name("b", "inc:4"),
    }
    assert residency["counters"]["evictions"] == 1
    assert [entry["action"] for entry in residency["recent_actions"]] == ["evicted"]


@pytest.mark.unit
def test_a_single_slot_supersedes_the_publishing_scenarios_own_revision(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, group, manager, _ = _actor(tmp_path, version, adapter_capacity=1)
    _run(actor, _job("a", 0, "inc:0"))
    _run(actor, _job("a", 1, "inc:1"))
    # Generation is paused for the publication, so the incumbent may leave
    # before its successor loads; the unload precedes the publication.
    assert manager.engine.unloaded == [scenario_adapter_name("a", "inc:1")]
    assert group.publications == [("a", "inc:1"), ("a", "inc:2")]
    assert actor.health()["adapter_residency"]["scenarios"]["a"]["resident"] == ["inc:2"]


@pytest.mark.unit
def test_a_full_engine_refuses_to_evict_another_scenarios_current_revision(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, _, manager, _ = _actor(tmp_path, version, adapter_capacity=1)
    _run(actor, _job("a", 0, "inc:0"))
    result = actor.execute_training_job(_job("b", 0, "inc:1"))
    assert result.outcome == "checkpoint"
    with pytest.raises(AdapterCapacityExhausted, match="exhausted"):
        actor.update_serving_weights(result.training_job_id)
    assert manager.engine.unloaded == []
    assert actor.health()["phase"] == "weight_sync_failed"
    assert actor.health()["adapter_residency"]["counters"]["capacity_rejections"] == 1


@pytest.mark.unit
def test_an_unload_the_engine_refuses_leaks_visibly(tmp_path, _local_ray_get) -> None:
    version = _EngineVersion(0)
    actor, _, manager, _ = _actor(tmp_path, version, adapter_capacity=2)
    _run(actor, _job("a", 0, "inc:0"))
    _run(actor, _job("a", 1, "inc:1"))
    manager.engine.refuse.add(scenario_adapter_name("a", "inc:1"))
    result = actor.execute_training_job(_job("a", 2, "inc:2"))
    with pytest.raises(AdapterCapacityExhausted, match="leaked"):
        actor.update_serving_weights(result.training_job_id)
    residency = actor.health()["adapter_residency"]
    assert residency["leaked"] == 1 and residency["counters"]["unload_failures"] == 1
    assert residency["recent_actions"][-1]["action"] == "leaked"
