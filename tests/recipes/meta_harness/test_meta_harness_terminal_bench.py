"""The Terminal-Bench campaign uses real Reef commits, without paid rollouts."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reef.execution.sandbox.process import SandboxExecutor
from reef.harness import RolloutResult
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.tree import HarnessTree

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/meta_harness/examples/terminal_bench"


def purge_modules():
    for name in list(sys.modules):
        if name in {"harness", "run"} or name.startswith("harness."):
            sys.modules.pop(name)


@pytest.fixture
def driver(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for key, value in {
        "REEF_WORK": str(tmp_path),
        "REEF_MODEL": "openai/target",
        "REEF_UPSTREAM_URL": "http://127.0.0.1:9",
        "REEF_UPSTREAM_API_KEY": "target-test-key",
        "REEF_PROPOSER_URL": "http://127.0.0.1:9",
        "REEF_PROPOSER_MODEL": "proposer",
        "REEF_PROPOSER_API_KEY": "proposer-test-key",
        "REEF_META_HARNESS_WORKERS": "1",
        "REEF_TERMINUS_ENVIRONMENT": "e2b",
        "E2B_API_KEY": "e2b-test-key",
    }.items():
        monkeypatch.setenv(key, value)
    purge_modules()
    yield importlib.import_module("run")
    purge_modules()


def rollout(task, score=1, **kwargs):
    return RolloutResult(
        exit_code=kwargs.get("exit_code", 0),
        stdout="",
        stderr="",
        trajectory=({"type": "verifier", "task": task, "reward": score, "failed": False},),
        residue=kwargs.get("residue", ()),
    )


@pytest.mark.parametrize("score", [0, 1])
def test_binary_verifier_scores(driver, score):
    assert driver.benchmark.evaluate("task", rollout("task", score)) == score


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "1", True, 0.5])
def test_invalid_verifier_rewards_refuse(driver, score):
    with pytest.raises(ValueError, match="reward"):
        driver.benchmark.evaluate("task", rollout("task", score))


def test_missing_failed_and_mismatched_verifier(driver):
    assert driver.benchmark.evaluate("task", rollout("task", None)) == 0
    assert driver.benchmark.evaluate("task", rollout("task", 1, exit_code=1)) == 0
    with pytest.raises(ValueError, match="verifier"):
        driver.benchmark.evaluate("another", rollout("task"))


def test_task_pin_and_drift_checks(driver, tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    root.mkdir()
    for name in driver.benchmark.MANIFEST["tasks"]:
        task = root / name
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n')
    task = root / "extract-elf"  # A task outside the historical hard subset.
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "tasks"],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ValueError, match="revision"):
        driver.benchmark.task_paths(root, [task.name])
    monkeypatch.setitem(driver.benchmark.MANIFEST, "revision", revision)
    assert driver.benchmark.task_paths(root) == tuple(str(root / name) for name in driver.benchmark.MANIFEST["tasks"])
    assert driver.benchmark.task_paths(root, [task.name]) == (str(task.resolve()),)
    for names in [[], [task.name, task.name], ["../escape"]]:
        with pytest.raises(ValueError, match="unique names"):
            driver.benchmark.task_paths(root, names)
    (task / "extra.txt").write_text("untracked drift")
    with pytest.raises(ValueError, match="clean"):
        driver.benchmark.task_paths(root, [task.name])


def test_configuration_scales_budgets_and_isolates_campaigns(driver, monkeypatch):
    full = driver.configuration(tuple(driver.benchmark.MANIFEST["tasks"]), None, None)
    tasks = full["evolution"]["evaluation"]["tasks"]
    assert len(tasks) == len(set(tasks)) == 89
    # The seed and four candidates, each evaluated once on 89 tasks with two repeats.
    assert full["evolution"]["method"]["config"]["max_target_rollouts"] == 890
    short = driver.configuration(("task",), 1, 1)
    assert short["evolution"]["max_steps"] == 2
    assert short["evolution"]["method"]["config"]["max_target_rollouts"] == 2
    original = driver.campaign_id(short)
    assert original != driver.campaign_id(full)
    monkeypatch.setenv("REEF_MODEL", "another-model")
    assert original != driver.campaign_id(short)


class Proposals(ModelBinding):
    """The proposer model: it answers every prompt with the seed's agent improved, the seed as the parent."""

    def __init__(self, seed: HarnessTree):
        super().__init__(base_url="http://127.0.0.1:9", model="proposer")
        improved = {**seed.files, "extensions/agent.py": seed.files["extensions/agent.py"] + "\n# improved\n"}
        object.__setattr__(self, "reply", json.dumps({"parent_id": seed.checksum(), "files": improved}))
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[-1]["content"])
        return self.reply


@pytest.fixture
def campaign(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(SandboxExecutor, "preflight", lambda self: None)
    launches = []

    def launch(descriptor, files, task, **kwargs):
        launches.append(kwargs)
        assert isinstance(kwargs["executor"], SandboxExecutor)
        assert kwargs["timeout"] == 9000
        return rollout(task, int("# improved" in files.get("terminus/context/agent.py", "")))

    monkeypatch.setattr(driver, "run_rollout", launch)
    monkeypatch.setattr("reef.evaluation.harness.runner.run_rollout", launch)
    config = driver.configuration(("task",), 1, 1)
    recipe = driver.load_recipe(config, tmp_path)
    chat = Proposals(HarnessTree.read(driver.SEED))
    recipe = dataclasses.replace(recipe, models={"proposer": chat})
    return recipe, chat, launches, tmp_path


def test_the_campaign_starts_from_the_seed_tree(driver, campaign):
    recipe, _, _, directory = campaign
    dispatcher = driver.open_campaign(recipe, directory)
    try:
        assert driver.served_tree(dispatcher.get_or_create_scenario(driver.SCENARIO)) == recipe.seed
    finally:
        dispatcher.close()


def test_search_commits_and_restart_resumes_from_the_commit_without_more_rollouts(driver, campaign):
    recipe, chat, launches, directory = campaign
    dispatcher = driver.open_campaign(recipe, directory)
    try:
        scenario = dispatcher.get_or_create_scenario(driver.SCENARIO)
        summary = driver.search(scenario, recipe, directory)
        assert summary["generated_candidates"] == 1
        assert summary["evaluation_rollouts"] == 2
        assert len(launches) == 3  # One recorded rollout and one paired evaluation.
        assert len(chat.prompts) == 1 and "terminus_rollout" in chat.prompts[0]
        assert "target-test-key" not in chat.prompts[0]
        served = driver.served_tree(scenario).checksum()
        assert summary["served_id"] == served != recipe.seed.checksum()
        assert [(candidate["outcome"], candidate["scores"]) for candidate in summary["candidates"]] == [
            ("undecided", [0.0]),
            ("served", [1.0]),
        ]
    finally:
        dispatcher.close()
    # The summary is disposable: a restart rebuilds it from the latest durable commit.
    (directory / "summary.json").write_text("stale")
    restarted = driver.open_campaign(recipe, directory)
    try:
        scenario = restarted.get_or_create_scenario(driver.SCENARIO)
        assert driver.search(scenario, recipe, directory) == summary
        assert json.loads((directory / "summary.json").read_text())["served_id"] == summary["served_id"]
        assert len(launches) == 3 and len(chat.prompts) == 1
    finally:
        restarted.close()


def test_commit_failure_writes_no_summary_and_restart_replays_pending_report(driver, campaign, monkeypatch):
    recipe, chat, launches, directory = campaign
    dispatcher = driver.open_campaign(recipe, directory)
    try:
        scenario = dispatcher.get_or_create_scenario(driver.SCENARIO)
        before = copy.deepcopy(scenario.trainer.state)

        def fail(record):
            raise OSError("commit failed")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.store.commit_history, "append", fail)
            with pytest.raises(OSError, match="commit failed"):
                driver.search(scenario, recipe, directory)
        assert scenario.trainer.state == before
        assert not (directory / "summary.json").exists()
    finally:
        dispatcher.close()
    restarted = driver.open_campaign(recipe, directory)
    try:
        scenario = restarted.get_or_create_scenario(driver.SCENARIO)
        summary = driver.search(scenario, recipe, directory)
        # The replayed step proposes the same tree again: the failed attempt neither spent its budget nor made
        # it a known tree, and the history's rollouts score it without another launch.
        assert (summary["generated_candidates"], summary["evaluation_rollouts"]) == (1, 2)
        assert summary["served_id"] != recipe.seed.checksum()
        assert len(launches) == 3 and len(chat.prompts) == 2
    finally:
        restarted.close()


def test_rollout_record_survives_interrupted_report(driver, campaign, monkeypatch):
    recipe, _, launches, directory = campaign
    dispatcher = driver.open_campaign(recipe, directory)
    try:
        scenario = dispatcher.get_or_create_scenario(driver.SCENARIO)
        append = scenario.records.append

        def fail_report(record):
            if record.request_type is driver.RequestType.REPORT:
                raise OSError("report interrupted")
            return append(record)

        with monkeypatch.context() as patch:
            patch.setattr(scenario.records, "append", fail_report)
            with pytest.raises(OSError, match="report interrupted"):
                driver.record_rollout(scenario, recipe)
        driver.record_rollout(scenario, recipe)
        assert len(launches) == 1
        assert scenario.records.count(scenario.name) == 2
    finally:
        dispatcher.close()
