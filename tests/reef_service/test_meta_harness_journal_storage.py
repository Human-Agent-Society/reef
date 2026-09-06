import copy
import json

import pytest

from recipes.meta_harness.examples.terminal_bench.campaign import CAMPAIGN_STATE_KEY
from recipes.meta_harness.examples.terminal_bench.journal_storage import (
    CompressedCommitLog,
    CompressedDispatcher,
    committed_state,
)
from recipes.meta_harness.examples.terminal_bench.run import SCENARIO, advance, baseline_ready
from reef.artifact import GitLFSRepositoryBackend
from reef.scenario.commit_log import CommitLogError

from .test_commit_log import sample_record
from .test_meta_harness_driver import campaign, finish  # noqa: F401


def test_compressed_frames_preserve_exact_state_and_lazy_records(tmp_path, monkeypatch):
    log = CompressedCommitLog(tmp_path / "records.zjsonl")
    original = [sample_record(step=i) for i in (1, 2)]
    for record in original:
        record.algorithm_state["history"] = "repeated full history " * 10000
        log.append(record)
    loaded = log.records()
    assert loaded == tuple(original)
    assert not any(isinstance(value, dict) and "history" in value for r in loaded for value in r.__dict__.values())
    assert log.path.stat().st_size < len(original[-1].algorithm_state["history"]) / 10
    assert loaded[-1].algorithm_state == original[-1].algorithm_state
    # Metadata can be read without materializing its historical state again.
    monkeypatch.setattr(
        "recipes.meta_harness.examples.terminal_bench.journal_storage._decode",
        lambda frame: pytest.fail("metadata read decompressed history"),
    )
    assert [r.step for r in loaded] == [1, 2]
    assert loaded[-1].consumed_ids == original[-1].consumed_ids


def test_compressed_reader_tolerates_torn_tail_but_refuses_to_append_past_it(tmp_path):
    log = CompressedCommitLog(tmp_path / "records.zjsonl")
    record = sample_record()
    log.append(record)
    with log.path.open("ab") as f:
        f.write(b'{"storage_record":')
    assert log.records() == (record,)
    with pytest.raises(CommitLogError, match="torn"):
        log.append(sample_record(step=2))


@pytest.mark.parametrize("problem", ["interior_json", "digest", "payload", "metadata", "indexed_mutation"])
def test_compressed_reader_rejects_corruption(tmp_path, problem):
    log = CompressedCommitLog(tmp_path / "records.zjsonl")
    log.append(sample_record())
    log.append(sample_record(step=2))
    indexed = log.records()
    lines = log.path.read_text().splitlines()
    if problem == "interior_json":
        lines[0] = "{"
    else:
        value = json.loads(lines[0])
        if problem == "digest":
            value["state_sha256"] = "different"
        elif problem == "payload":
            value["state_zlib_base64"] = "wrong"
        elif problem == "metadata":
            value["commit"]["step"] = -1
        else:
            value["commit"]["recorded_at"] += 1
        lines[0] = json.dumps(value)
    log.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CommitLogError):
        if problem == "indexed_mutation":
            _ = indexed[0].algorithm_state
        else:
            log.records()


def test_compressed_campaign_commit_failure_restart_and_stale_mirrors(campaign, monkeypatch):  # noqa: F811
    recipe, factory, calls, prompts = campaign
    dispatcher = CompressedDispatcher(recipe, factory, agent_record_dir=recipe.output / "records")
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    try:
        while not baseline_ready(scenario.trainer.state[CAMPAIGN_STATE_KEY]):
            advance(scenario)
        before = copy.deepcopy(scenario.trainer.state)
        append = scenario.commit_log.append
        monkeypatch.setattr(scenario.commit_log, "append", lambda r: (_ for _ in ()).throw(RuntimeError("offline")))
        with pytest.raises(RuntimeError, match="offline"):
            advance(scenario)  # A failed proposal reservation must not invoke the model.
        assert scenario.trainer.state == before and len(calls) == 1 and not prompts
        assert committed_state(recipe.output)[0] == before
        monkeypatch.setattr(scenario.commit_log, "append", append)
    finally:
        dispatcher.close()
    (recipe.output / "run.json").write_text("{}")
    dispatcher = CompressedDispatcher(recipe, factory, agent_record_dir=recipe.output / "records")
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        assert scenario.trainer.state == before
        data = finish(scenario)
        assert data["status"] == "complete" and len(calls) == 2 and len(prompts) == 1
        assert committed_state(recipe.output)[0] == scenario.trainer.state
        assert json.loads((recipe.output / "run.json").read_text()) == data
        assert not list((recipe.output / "records").glob("*.commits.jsonl"))
    finally:
        dispatcher.close()


def test_compressed_journal_names_are_path_safe(campaign):  # noqa: F811
    recipe, factory, _, _ = campaign
    dispatcher = CompressedDispatcher(recipe, factory, agent_record_dir=recipe.output / "records")
    try:
        log = dispatcher._registry._scenario_factory._commit_log_for("../../outside/escape")
        assert log.path.parent == recipe.output / "records"
        assert log.path.name.endswith(".commits.zjsonl") and "/" not in log.path.name
    finally:
        dispatcher.close()


def test_compressed_campaign_reopens_with_new_git_factory_without_plain_preload(campaign):  # noqa: F811
    recipe, _, calls, _ = campaign

    def factory():
        return GitLFSRepositoryBackend.factory(
            recipe.output / "artifacts.git", work_dir=recipe.output / "git-work", cache_dir=recipe.output / "git-cache"
        )

    dispatcher = CompressedDispatcher(recipe, factory(), agent_record_dir=recipe.output / "records")
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        assert finish(scenario)["status"] == "complete"
        original = copy.deepcopy(scenario.trainer.state)
    finally:
        dispatcher.close()
    dispatcher = CompressedDispatcher(recipe, factory(), agent_record_dir=recipe.output / "records")
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        assert scenario.trainer.state == original and len(calls) == 2
        assert not list((recipe.output / "records").glob("*.commits.jsonl"))
    finally:
        dispatcher.close()
