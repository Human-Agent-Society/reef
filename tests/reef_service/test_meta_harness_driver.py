"""Real scenario commits protect the benchmark's search and evidence."""

from __future__ import annotations

import dataclasses
import json
from argparse import Namespace

import pytest

from recipes.meta_harness.backend import POPULATION_STATE_KEY
from recipes.meta_harness.examples.terminal_bench.campaign import CAMPAIGN_STATE_KEY, TerminalBenchBackend
from recipes.meta_harness.examples.terminal_bench.run import SCENARIO, advance, baseline_ready, make_recipe
from recipes.meta_harness.population import content_id
from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.model_binding import ModelBinding

IMPROVED = [{"id": "check", "name": "rules", "config": {"text": "verify"}}]


def arguments(tmp_path):
    return Namespace(
        tasks="terminal-bench/a",
        tasks_file=None,
        trials=1,
        iterations=1,
        concurrency=2,
        max_attempts=2,
        episode_timeout_s=28800,
        max_observed_cost_usd=10,
        mode="full_history",
        target_model="target",
        target_url="http://localhost:9",
        proposer_model="proposer",
        proposer_url="http://localhost:9",
        proposer_effort="max",
        output_dir=str(tmp_path),
    )


def test_cli_freezes_documented_capacity_and_scoring_options(tmp_path, monkeypatch, capsys):
    from recipes.meta_harness.examples.terminal_bench import run

    monkeypatch.setattr(run, "runtime_fingerprint", lambda: {"test": True})
    monkeypatch.setattr(
        "recipes.meta_harness.examples.terminal_bench.proposer_usage.freeze_pricing", lambda model: {"test": True}
    )
    assert (
        run.main(
            [
                "--tasks",
                "terminal-bench/password-recovery",
                "--iterations",
                "4",
                "--concurrency",
                "12",
                "--benchmark-sandbox-limit",
                "64",
                "--agent-failure-policy",
                "upstream-completed-terminal-error-zero-v5",
                "--max-observed-cost-usd",
                "100",
                "--max-proposer-cost-usd",
                "30",
                "--output-dir",
                str(tmp_path / "unused"),
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["benchmark_sandbox_limit"] == 64 and plan["concurrency"] == 12
    assert plan["agent_failure_policy"] == "upstream-completed-terminal-error-zero-v5"
    assert plan["max_observed_cost_usd"] == 100 and plan["max_proposer_cost_usd"] == 30
    assert not (tmp_path / "unused").exists()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    calls = []
    prompts = []

    def episode(self, files, identity):
        calls.append(identity["id"])
        improved = "verify" in files.get("terminus/AGENTS.md", "")
        return {
            **identity,
            "valid": True,
            "phase": "verified",
            "reward": float(improved),
            "cost_usd": 0.02,
            "trajectory": [{"type": "observation", "text": "diagnostic evidence"}],
        }

    def completion(self, body, **kwargs):
        prompts.append(body)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "parent_id": content_id(()),
                                "entries": IMPROVED,
                                "hypothesis": "verify the artifact",
                                "changes": "add a check",
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(TerminalBenchBackend, "_episode", episode)
    monkeypatch.setattr("recipes.meta_harness.examples.terminal_bench.campaign.check_capacity", lambda workers: None)
    monkeypatch.setattr(ModelBinding, "complete", completion)
    recipe = make_recipe(arguments(tmp_path), fingerprint={"test": True})
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    return recipe, factory, calls, prompts


def open_campaign(recipe, factory):
    dispatcher = Dispatcher(recipe, factory, agent_record_dir=recipe.output / "records")
    return dispatcher, dispatcher.get_or_create_scenario(SCENARIO)


def finish(scenario):
    for _ in range(40):
        data = scenario.trainer.state[CAMPAIGN_STATE_KEY]
        if data["status"] != "running":
            return data
        advance(scenario)
    raise AssertionError("campaign did not terminate")


def test_baseline_precedes_proposal_and_candidate_only_scores(campaign):
    recipe, factory, calls, prompts = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        population = scenario.trainer.state[POPULATION_STATE_KEY]
        assert data["status"] == "complete"
        assert len(calls) == 2  # one baseline + one candidate; no paid incumbent leg
        assert population["episode_calls"] == 2
        assert population["served_id"] == content_id(IMPROVED)
        assert len(prompts) == 1
        assert '"side": "seed"' in prompts[0]["messages"][-1]["content"]
        assert '"scores": [\n        0.0' in prompts[0]["messages"][1]["content"]
        assert data["observed_cost_usd"] == 0.04
        assert recipe.output.joinpath("final-population.json").exists()
        assert scenario.repository.require_current_artifact().release_id
    finally:
        dispatcher.close()


def test_proposer_usage_unknown_is_committed_and_never_evaluated(campaign):
    recipe, factory, calls, prompts = campaign
    recipe = dataclasses.replace(
        recipe,
        plan={
            **recipe.plan,
            "proposer_pricing": {"rates": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}},
            "max_proposer_cost_usd": 1,
        },
    )
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        assert data["status"] == "usage_unknown"
        assert data["unknown_usage"]
        assert len(calls) == len(prompts) == 1
        assert data["observed_cost_usd"] == 0.02
        assert data["proposals"][0]["unknown_usage"]
    finally:
        dispatcher.close()
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        assert finish(scenario)["status"] == "usage_unknown"
        assert len(prompts) == 1
    finally:
        dispatcher.close()


def test_failed_proposal_commit_retries_without_rebilling(campaign, monkeypatch):
    recipe, factory, calls, prompts = campaign
    recipe = dataclasses.replace(
        recipe,
        plan={
            **recipe.plan,
            "proposer_pricing": {"rates": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}},
            "max_proposer_cost_usd": 0.03,
        },
    )
    original_completion = ModelBinding.complete

    def metered(self, *args, **kwargs):
        return {**original_completion(self, *args, **kwargs), "usage": {"prompt_tokens": 50, "completion_tokens": 10}}

    monkeypatch.setattr(ModelBinding, "complete", metered)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        for _ in range(10):
            reservation = scenario.trainer.state[CAMPAIGN_STATE_KEY]["reservation"]
            if reservation and reservation["kind"] == "proposal":
                break
            advance(scenario)
        previous = json.loads(json.dumps(scenario.trainer.state))
        mirror = recipe.output.joinpath("run.json").read_text()
        original_append = scenario.commit_log.append
        monkeypatch.setattr(
            scenario.commit_log, "append", lambda record: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        with pytest.raises(RuntimeError, match="offline"):
            advance(scenario)
        assert scenario.trainer.state == previous
        assert recipe.output.joinpath("run.json").read_text() == mirror
        assert len(prompts) == 1
        monkeypatch.setattr(scenario.commit_log, "append", original_append)
        data = finish(scenario)
        assert len(prompts) == 1
        assert data["proposer_cost_usd"] == pytest.approx(0.07)
        assert data["observed_cost_usd"] == pytest.approx(0.11)
        # One in-flight proposal can cross its sub-cap; it is retained and
        # evaluated within the remaining total budget, then search stops.
        assert len(calls) == 2
    finally:
        dispatcher.close()


def test_baseline_stop_and_restart_do_not_propose_or_repeat_paid_work(campaign):
    recipe, factory, calls, prompts = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        for _ in range(10):
            if baseline_ready(scenario.trainer.state[CAMPAIGN_STATE_KEY]):
                break
            advance(scenario)
        else:
            raise AssertionError("baseline boundary was not reached")
        assert len(calls) == 1
        assert not prompts
        assert not recipe.output.joinpath("final-population.json").exists()
    finally:
        dispatcher.close()

    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        assert baseline_ready(scenario.trainer.state[CAMPAIGN_STATE_KEY])
        assert len(calls) == 1
        assert not prompts
        assert finish(scenario)["status"] == "complete"
        assert len(calls) == 2
        assert len(prompts) == 1
    finally:
        dispatcher.close()


def test_rolling_schedule_refills_only_after_committed_reservation(campaign, monkeypatch):
    import threading

    recipe, factory, calls, prompts = campaign
    tasks = ("terminal-bench/slow", "terminal-bench/fast", "terminal-bench/next")
    recipe = dataclasses.replace(
        recipe, tasks=tasks, plan={**recipe.plan, "tasks": list(tasks), "episode_schedule": "rolling"}
    )
    release = threading.Event()
    started = threading.Event()
    original_episode = TerminalBenchBackend._episode

    def episode(self, files, identity):
        if identity["task"].endswith("/slow"):
            started.set()
            assert release.wait(10)
        return original_episode(self, files, identity)

    monkeypatch.setattr(TerminalBenchBackend, "_episode", episode)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        advance(scenario)  # evaluation plan
        advance(scenario)  # reserve the first two slots
        data = advance(scenario)  # commit the fast result while slow remains active
        assert started.is_set() and not release.is_set()
        assert len(data["records"]) == 1
        assert len(calls) == 1
        previous = json.loads(json.dumps(scenario.trainer.state))
        original_append = scenario.commit_log.append
        monkeypatch.setattr(
            scenario.commit_log, "append", lambda record: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        with pytest.raises(RuntimeError, match="offline"):
            advance(scenario)  # failed refill commit must not launch the third trial
        assert scenario.trainer.state == previous
        assert len(calls) == 1
        monkeypatch.setattr(scenario.commit_log, "append", original_append)
        data = advance(scenario)  # now refill is durable, but has not executed
        assert len(data["reservation"]["wave"]) == 2
        assert len(calls) == 1
        data = advance(scenario)
        assert len(data["records"]) == 2 and len(calls) == 2
        assert not release.is_set()
        release.set()
        while not baseline_ready(data):
            data = advance(scenario)
        assert len(calls) == 3 and len(set(calls)) == 3
        assert data["observed_cost_usd"] == pytest.approx(0.06)
        assert not prompts
    finally:
        release.set()
        dispatcher.close()


def test_rolling_unknown_usage_drains_reserved_work_without_refill(campaign, monkeypatch):
    import threading

    recipe, factory, calls, _ = campaign
    tasks = ("terminal-bench/slow", "terminal-bench/unknown", "terminal-bench/never")
    recipe = dataclasses.replace(
        recipe, tasks=tasks, plan={**recipe.plan, "tasks": list(tasks), "episode_schedule": "rolling"}
    )
    release = threading.Event()
    original_episode = TerminalBenchBackend._episode

    def episode(self, files, identity):
        if identity["task"].endswith("/slow"):
            assert release.wait(10)
        result = original_episode(self, files, identity)
        if identity["task"].endswith("/unknown"):
            result["cost_usd"] = None
        return result

    monkeypatch.setattr(TerminalBenchBackend, "_episode", episode)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        advance(scenario)
        advance(scenario)
        data = advance(scenario)
        assert data["unknown_usage"] and len(data["records"]) == 1
        release.set()
        data = finish(scenario)
        assert data["status"] == "usage_unknown"
        assert len(data["records"]) == len(calls) == 2
        assert data["observed_cost_usd"] == 0.02
        assert data["reservation"] is None
    finally:
        release.set()
        dispatcher.close()


def test_invalid_evaluation_never_enters_frontier(campaign, monkeypatch):
    recipe, factory, _, _ = campaign
    original = TerminalBenchBackend._episode

    def failed(self, files, identity):
        result = original(self, files, identity)
        if "verify" in files.get("terminus/AGENTS.md", ""):
            result.update(valid=False, phase="setup_failure", reward=None, cost_usd=0)
        return result

    monkeypatch.setattr(TerminalBenchBackend, "_episode", failed)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        population = scenario.trainer.state[POPULATION_STATE_KEY]
        assert data["status"] == "evaluation_failed"
        assert population["served_id"] == content_id(())
        assert population["candidates"][1]["scores"] is None
        assert population["episode_calls"] == 3  # failures remain counted
        assert len(data["rounds"]) == 1
        assert not recipe.output.joinpath("final-population.json").exists()
    finally:
        dispatcher.close()


def test_failed_commit_does_not_advance_records_cost_or_population(campaign, monkeypatch):
    recipe, factory, calls, _ = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        advance(scenario)  # evaluation plan
        advance(scenario)  # durable reservation
        previous = json.loads(json.dumps(scenario.trainer.state))
        mirror = recipe.output.joinpath("run.json").read_text()
        original = scenario.commit_log.append
        monkeypatch.setattr(
            scenario.commit_log, "append", lambda record: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        with pytest.raises(RuntimeError, match="offline"):
            advance(scenario)
        assert scenario.trainer.state == previous
        assert recipe.output.joinpath("run.json").read_text() == mirror
        assert len(calls) == 1
        monkeypatch.setattr(scenario.commit_log, "append", original)
    finally:
        dispatcher.close()
    # No replay of a possibly billed, uncommitted episode after a restart.
    dispatcher, recovered = open_campaign(recipe, factory)
    try:
        data = finish(recovered)
        assert data["status"] == "interrupted"
        assert data["unknown_usage"] is True
        assert len(calls) == 1
        assert not data["records"]
    finally:
        dispatcher.close()


def test_restart_heals_stale_mirrors_from_committed_state(campaign):
    recipe, factory, calls, _ = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        finish(scenario)
        committed = json.loads(json.dumps(scenario.trainer.state))
    finally:
        dispatcher.close()
    recipe.output.joinpath("population.json").write_text('{"corrupt": true}')
    recipe.output.joinpath("run.json").write_text('{"corrupt": true}')
    dispatcher, recovered = open_campaign(recipe, factory)
    try:
        assert recovered.trainer.state == committed
        assert json.loads(recipe.output.joinpath("population.json").read_text()) == committed[POPULATION_STATE_KEY]
        assert json.loads(recipe.output.joinpath("run.json").read_text()) == committed[CAMPAIGN_STATE_KEY]
        assert len(calls) == 2
    finally:
        dispatcher.close()


def test_plan_change_is_refused_before_new_work(campaign):
    recipe, factory, calls, _ = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    advance(scenario)
    dispatcher.close()
    changed = dataclasses.replace(recipe, plan={**recipe.plan, "trials": 3})
    dispatcher = Dispatcher(changed, factory, agent_record_dir=recipe.output / "records")
    try:
        with pytest.raises(ValueError, match="plan differs"):
            dispatcher.get_or_create_scenario(SCENARIO)
        assert not calls
    finally:
        dispatcher.close()


def test_cost_cap_stops_before_another_wave(campaign):
    recipe, factory, calls, _ = campaign
    recipe = dataclasses.replace(recipe, plan={**recipe.plan, "max_observed_cost_usd": 0.01})
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        assert data["status"] == "budget_exhausted"
        assert len(calls) == 1
        assert data["observed_cost_usd"] == 0.02
    finally:
        dispatcher.close()


def test_unknown_usage_does_not_become_free_work(campaign, monkeypatch):
    recipe, factory, calls, _ = campaign
    original = TerminalBenchBackend._episode

    def unknown(self, files, identity):
        record = original(self, files, identity)
        return {**record, "cost_usd": None}

    monkeypatch.setattr(TerminalBenchBackend, "_episode", unknown)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        assert data["status"] == "usage_unknown"
        assert len(calls) == 1
        assert next(iter(data["records"].values()))["cost_usd"] is None
    finally:
        dispatcher.close()


def test_restart_with_new_git_repository_factory(campaign):
    recipe, _, calls, _ = campaign

    def factory():
        return GitLFSRepositoryBackend.factory(
            recipe.output / "artifacts.git", work_dir=recipe.output / "git-work", cache_dir=recipe.output / "git-cache"
        )

    dispatcher, scenario = open_campaign(recipe, factory())
    try:
        finish(scenario)
        served = scenario.trainer.state[POPULATION_STATE_KEY]["served_id"]
    finally:
        dispatcher.close()
    dispatcher, recovered = open_campaign(recipe, factory())
    try:
        assert finish(recovered)["status"] == "complete"
        assert recovered.trainer.state[POPULATION_STATE_KEY]["served_id"] == served
        assert len(calls) == 2
    finally:
        dispatcher.close()


def test_failed_selection_commit_keeps_served_state_and_retries_without_work(campaign, monkeypatch):
    recipe, factory, calls, _ = campaign
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        for _ in range(8):
            advance(scenario)
        previous = json.loads(json.dumps(scenario.trainer.state))
        population = previous[POPULATION_STATE_KEY]
        assert population["pending_id"] == content_id(IMPROVED)
        assert population["served_id"] == content_id(())
        backend = scenario.trainer.training_backend
        mirror = recipe.output.joinpath("population.json").read_text()
        head = scenario.current_artifact_ref()
        original = scenario.commit_log.append
        monkeypatch.setattr(
            scenario.commit_log, "append", lambda record: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        with pytest.raises(RuntimeError, match="offline"):
            advance(scenario)
        assert scenario.trainer.state == previous
        assert backend._population_store.committed.to_dict() == population
        assert recipe.output.joinpath("population.json").read_text() == mirror
        assert scenario.current_artifact_ref() == head
        assert backend._entries() == []
        monkeypatch.setattr(scenario.commit_log, "append", original)
        assert finish(scenario)["status"] == "complete"
        assert len(calls) == 2
        assert scenario.trainer.state[POPULATION_STATE_KEY]["served_id"] == content_id(IMPROVED)
    finally:
        dispatcher.close()


def test_parallel_wave_accounts_every_completion_once(campaign):
    recipe, factory, calls, _ = campaign
    recipe = dataclasses.replace(
        recipe, episode_repeats=5, plan={**recipe.plan, "trials": 5, "max_observed_cost_usd": 0.01}
    )
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        assert data["status"] == "budget_exhausted"
        assert len(calls) == len(set(calls)) == len(data["records"]) == 2
        assert data["observed_cost_usd"] == 0.04
        assert scenario.trainer.state[POPULATION_STATE_KEY]["episode_calls"] == 2
    finally:
        dispatcher.close()


def test_invalid_proposal_cannot_be_reported_as_completed_search(campaign, monkeypatch):
    recipe, factory, calls, _ = campaign
    monkeypatch.setattr(
        ModelBinding,
        "complete",
        lambda *args, **kwargs: {"choices": [{"message": {"content": "invalid composition"}}]},
    )
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        assert finish(scenario)["status"] == "proposal_failed"
        assert len(calls) == 1
        assert not recipe.output.joinpath("final-population.json").exists()
    finally:
        dispatcher.close()


@pytest.mark.parametrize("mode, visible", [("full_history", 2), ("incumbent_only", 1)])
def test_history_visibility_includes_retained_candidates_only_in_full_history(campaign, monkeypatch, mode, visible):
    recipe, factory, calls, _ = campaign
    recipe = dataclasses.replace(recipe, mode=mode, plan={**recipe.plan, "mode": mode, "iterations": 2})
    original = TerminalBenchBackend._episode

    def tied(self, files, identity):
        return {**original(self, files, identity), "reward": 0.0}

    monkeypatch.setattr(TerminalBenchBackend, "_episode", tied)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        data = finish(scenario)
        assert data["status"] == "complete"
        assert len(data["proposals"][1]["visible_record_ids"]) == visible
        assert scenario.trainer.state[POPULATION_STATE_KEY]["served_id"] == content_id(())
        assert len(calls) == 2  # second proposal duplicates an already measured candidate
    finally:
        dispatcher.close()


def test_restart_after_committed_partial_wave_reuses_completed_slots(campaign):
    recipe, factory, calls, _ = campaign
    recipe = dataclasses.replace(recipe, episode_repeats=3, plan={**recipe.plan, "trials": 3})
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        for _ in range(3):
            advance(scenario)
        assert len(calls) == 2
        assert len(scenario.trainer.state[CAMPAIGN_STATE_KEY]["records"]) == 2
    finally:
        dispatcher.close()
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        assert finish(scenario)["status"] == "complete"
        assert len(calls) == len(set(calls)) == 6
    finally:
        dispatcher.close()
