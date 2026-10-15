"""Static quota settings and opt-in deployment wiring."""

from dataclasses import dataclass
from http.client import BadStatusLine, IncompleteRead
from unittest.mock import patch

import pytest

from reef.harness.episodes.executor import LocalExecutor
from reef.recipe import Recipe
from reef.recipe.composite import CompositeRecipe
from reef.recipe.cordis import CordisRecipe
from reef.runtime.executor.config import ExecutorSettings
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.service.scheduler import build_scheduler
from reef.service.scheduler_config import parse_scheduler_config
from reef.service.scheduling import SchedulerHTTPClient, SchedulerUnavailable, ServiceSchedulingConfig
from reef.train.cordis_backend.strategies import EpisodeScorer, Proposer


def test_versioned_service_configuration_requires_complete_scheduler_connection():
    config = {
        "schema-version": 2,
        "recipe": {"implementation": "recipe"},
        "scheduling": {
            "scheduler_url": "http://127.0.0.1:8902",
            "service_id": 1,
            "service_name": "development",
            "control_token": "test-control",
        },
    }
    settings = service_config_from_mapping(config)
    assert settings.scheduler_service_id == 1
    assert settings.scheduler_service_name == "development"
    assert settings.scheduler_control_token == "test-control"
    assert "test-control" not in repr(settings)
    config["scheduling"].pop("service_id")
    with pytest.raises(ValueError, match="scheduling requires"):
        service_config_from_mapping(config)


def test_scheduler_disabled_by_default():
    assert ServiceConfig(recipe="recipe").scheduler_url is None
    with pytest.raises(ValueError, match="scheduler_url"):
        ServiceConfig(recipe="recipe", scheduler_service_id=1)


def test_scheduler_configuration_uses_host_budget_and_static_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("REEF_TEST_CONTROL", "control-secret")
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 8)
    payload = {
        "database": str(tmp_path / "scheduler.sqlite"),
        "max_logical_thread_percentage": 50,
        "tenant_quotas": {1: 4},
        "recipe_execution_units": {"harness": 2},
        "services": [
            {
                "service_id": 1,
                "service_name": "development",
                "recipe": "harness",
                "service_url": "http://127.0.0.1:8901",
                "control_token_env": "REEF_TEST_CONTROL",
            },
        ],
    }
    controller = build_scheduler(payload)
    try:
        assert controller.scheduler.config.global_execution_units == 4
        assert controller.scheduler.config.max_logical_thread_percentage == 50
        assert controller.scheduler.config.idle_ttl_seconds is None
        assert controller.scheduler.services[1].standard_execution_units == 2
        assert controller.scheduler.tenant_allocations() == ()
        assert not controller.scheduler.availability[1]
    finally:
        controller.scheduler.close()
    payload["max_logical_thread_percentage"] = 25
    with pytest.raises(ValueError, match="tenant quota cannot exceed"):
        build_scheduler(payload)


@pytest.mark.parametrize(
    ("logical_threads", "percentage", "expected"),
    [(8, None, 5), (32, None, 22), (30, 70, 21), (8, 50, 4), (8, 12.5, 1), (1, 100, 1)],
)
def test_startup_budget_uses_logical_threads_and_percentage(
    tmp_path, monkeypatch, logical_threads, percentage, expected
):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: logical_threads)
    settings = {"database": str(tmp_path / "scheduler.sqlite"), "tenant_quotas": {1: 1}}
    if percentage is not None:
        settings["max_logical_thread_percentage"] = percentage
    config = parse_scheduler_config(settings)
    assert config.global_execution_units == expected
    assert config.max_logical_thread_percentage == (70 if percentage is None else percentage)
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 64)
    assert config.global_execution_units == expected


@pytest.mark.parametrize("percentage", [True, False, None, "70", 0, -1, 100.1, float("nan"), float("inf")])
def test_startup_rejects_invalid_logical_thread_percentage(tmp_path, monkeypatch, percentage):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 8)
    settings = {
        "database": str(tmp_path / "scheduler.sqlite"),
        "tenant_quotas": {1: 1},
        "max_logical_thread_percentage": percentage,
    }
    with pytest.raises(ValueError, match="max_logical_thread_percentage"):
        parse_scheduler_config(settings)


def test_startup_rejects_unknown_logical_thread_count(tmp_path, monkeypatch):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: None)
    with pytest.raises(ValueError, match="logical thread count"):
        parse_scheduler_config({"database": str(tmp_path / "scheduler.sqlite"), "tenant_quotas": {1: 1}})


def test_startup_does_not_round_zero_budget_above_percentage_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 1)
    with pytest.raises(ValueError, match="less than one execution unit"):
        parse_scheduler_config({"database": str(tmp_path / "scheduler.sqlite"), "tenant_quotas": {1: 1}})


@pytest.mark.parametrize("legacy_budget", [1, 1000])
def test_legacy_global_budget_cannot_override_host_percentage(tmp_path, monkeypatch, caplog, legacy_budget):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 8)
    config = parse_scheduler_config(
        {
            "database": str(tmp_path / "scheduler.sqlite"),
            "tenant_quotas": {1: 1},
            "global_execution_units": legacy_budget,
        }
    )
    assert config.global_execution_units == 5
    assert "global_execution_units" in caplog.text
    assert "ignored" in caplog.text


def test_connection_requires_valid_endpoint_and_static_units():
    with pytest.raises(ValueError, match="HTTP"):
        ServiceSchedulingConfig("file:///tmp/scheduler", 1, "development", "harness", 2, "control")
    with pytest.raises(ValueError, match="positive integer"):
        ServiceSchedulingConfig("http://127.0.0.1:8902", 1, "development", "harness", True, "control")


@dataclass(frozen=True, kw_only=True)
class LocalHarnessRecipe(Recipe):
    workers: int = 2

    def scenario_execution_units(self) -> int:
        return self.workers


def test_composite_static_estimate_sums_only_local_evaluation_workers():
    recipe = CompositeRecipe(components={"files": LocalHarnessRecipe(), "records": Recipe()})
    assert recipe.scenario_execution_units() == 2
    assert Recipe().scenario_execution_units() is None


class NoProposal(Proposer):
    def __call__(self, **kwargs):
        return None


class FourWorkerScorer(EpisodeScorer):
    def __call__(self, task, result):
        return 1.0

    def execution_requirements(self):
        return ExecutionRequirements(workers=4)


@pytest.mark.parametrize(("workers", "expected"), [(None, 4), (2, 2)])
def test_recipe_static_cost_uses_effective_evaluation_worker_configuration(workers, expected):
    recipe = CordisRecipe(
        propose=NoProposal(),
        score_episode=FourWorkerScorer(),
        tasks=("task",),
        executor=LocalExecutor(),
        worker_executor=ExecutorSettings(workers=workers),
    )
    assert recipe.scenario_execution_units() == expected


@pytest.mark.parametrize("failure", [IncompleteRead(b"partial", 20), BadStatusLine("invalid status")])
def test_incomplete_control_responses_remain_retryable(failure):
    client = SchedulerHTTPClient("http://localhost:8902")
    with (
        patch("reef.service.scheduling.urlopen", side_effect=failure),
        pytest.raises(SchedulerUnavailable, match="outcome is unknown"),
    ):
        client.request("POST", "/reef/scheduler/control", payload={"operation": "status"})


@pytest.mark.parametrize(
    "mapping", [{"alice": True}, {"alice": "1"}, {"alice": -1}, {"alice": 2**63}, {"alice": 1, "bob": 1}]
)
def test_service_migration_map_rejects_invalid_or_colliding_targets(mapping):
    with pytest.raises(ValueError, match="tenant_id_map"):
        service_config_from_mapping(
            {
                "schema-version": 2,
                "recipe": {"implementation": "recipe"},
                "scheduling": {
                    "scheduler_url": "http://localhost:8902",
                    "service_id": 1,
                    "service_name": "local",
                    "control_token": "test-control",
                    "tenant_id_map": mapping,
                },
            }
        )


def test_service_migration_map_preserves_explicit_numeric_targets():
    config = service_config_from_mapping(
        {
            "schema-version": 2,
            "recipe": {"implementation": "recipe"},
            "scheduling": {
                "scheduler_url": "http://localhost:8902",
                "service_id": 1,
                "service_name": "local",
                "control_token": "test-control",
                "tenant_id_map": {"alice": 7},
            },
        }
    )
    assert config.scheduler_tenant_id_map == {"alice": 7}


def test_removed_queue_ttl_configuration_is_rejected():
    with pytest.raises(ValueError, match="queue_ttl_seconds is no longer supported"):
        build_scheduler({"queue_ttl_seconds": None})
