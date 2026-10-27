"""Official Retail environment, scripted inference, and the real SAO assembly.

Install recipes/sao/examples/servelearnbench to run this optional benchmark suite.
Scripted responses test the harness contract, not model quality or GPU execution.
"""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("servelearnbench")

from reef_client import ReefClient, ReefClientError
from servelearnbench import agent, llm, protocol
from servelearnbench.scenarios import get_bundle, n_windows
from servelearnbench.scoring import read_rows, setting_scores
from tests.support.config_helpers import load_deployment

from recipes.sao.examples.servelearnbench import harness as reef_harness
from recipes.sao.examples.servelearnbench.batching import BatchGrouping, RetailBatchReports
from recipes.sao.examples.servelearnbench.harness import agent as reef_agent
from recipes.sao.examples.servelearnbench.harness import campaign as campaign_results
from recipes.sao.examples.servelearnbench.harness import config as campaign_config
from recipes.sao.examples.servelearnbench.harness import report as reef_report
from recipes.sao.examples.servelearnbench.recipe import RetailJobPlan, RetailSAORecipe
from recipes.sao.processor import SAOJobData, SAOTasks
from reef.core import AgentRecord, RequestType
from reef.core.job_data import JobOutcome
from reef.core.reports import ScoredRolloutReport
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2record import ReportContext
from reef.record2task.assembly import SampleAssemblySettings
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import job_training
from reef.train.algos.schedule import StepScheduling
from reef.train.jobs import JobPlanContext
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/sao/examples/servelearnbench"


def load_campaign(monkeypatch):
    """Load run.py as the campaign module, with the example's harness package importable as ``harness``."""
    monkeypatch.setitem(sys.modules, "harness", reef_harness)
    monkeypatch.setitem(sys.modules, "harness.agent", reef_agent)
    monkeypatch.setitem(sys.modules, "harness.report", reef_report)
    monkeypatch.setitem(sys.modules, "harness.campaign", campaign_results)
    monkeypatch.setitem(sys.modules, "harness.config", campaign_config)
    spec = importlib.util.spec_from_file_location("retail_campaign", EXAMPLE / "run.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    return campaign


@pytest.fixture(autouse=True)
def campaign_environment(monkeypatch):
    monkeypatch.setenv("SAO_BATCH", "64")
    monkeypatch.setenv("SAO_MICRO_BATCH", "1")
    monkeypatch.setenv("SAO_WARMUP", "1")
    monkeypatch.setenv("SAO_CONTEXT", "40960")


@pytest.fixture
def bundle():
    result = get_bundle("retail_l1", fresh=True)
    protocol.stamp_serving(result["serving"])
    return result


def completion(name, arguments, index, finish_reason="tool_calls"):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": None,
                    "reasoning_content": "Scripted test response.",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }


def ground_truth_responses(task):
    # Hidden ground truth is used only by this test fixture, never by the harness.
    responses = [
        completion(a.name, a.kwargs, i) for i, a in enumerate(task.actions) if a.name not in ("respond", "finish")
    ]
    answer = task.answer
    if answer and answer["kind"] == "value":
        finish = {"answer": answer["expect"]}
    elif answer and answer["kind"] == "refuse":
        finish = {"decision": "refuse", "reason": answer["reasons"][0]}
    else:
        finish = {}
    responses.append(completion("finish", finish, len(responses)))
    return responses


def model_with_responses(responses):
    client = Mock(spec=ReefClient)
    client.service_url = "http://reef.invalid"
    client.inference_with_record.side_effect = [(response, f"receipt-{i}") for i, response in enumerate(responses)]
    return reef_agent.ReefLLM(client, "retail-test", llm.ModelConfig(model="reef", max_tokens=512), 42)


def test_split_is_fixed_ordered_and_disjoint(bundle):
    serving = [t for t in bundle["serving"] if t.z["window"] == 1]
    tests = bundle["test_set"](1)
    protocol.stamp_test(tests, bundle["serving"], 1)
    assert (len(serving), len(tests)) == (140, 64)
    assert [t.z["stream_pos"] for t in serving] == list(range(80, 220))
    assert {t.task_id for t in serving}.isdisjoint(t.task_id for t in tests)
    assert protocol.window_end(bundle["serving"], 1) == 219


@pytest.mark.parametrize("correct", [True, False])
def test_configurable_episode_preserves_official_execution_and_scoring(bundle, correct):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1 and t.answer is None and t.actions)
    responses = ground_truth_responses(task) if correct else [completion("finish", {}, 0)]
    official = agent.run_episode(model_with_responses(copy.deepcopy(responses)), bundle, task)
    configured = reef_agent.run_episode(model_with_responses(responses), bundle, task, max_steps=20)
    assert configured == official


def test_configured_turn_cap_keeps_official_reward_and_all_receipts(bundle):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1 and t.answer is None and t.actions)
    model = model_with_responses([completion("finish", {}, i, finish_reason="length") for i in range(20)])
    reward, trace, usage, _ = reef_agent.run_episode(model, bundle, task, max_steps=20)
    assert reward.reward == 0
    assert usage["calls"] == len(model.receipts) == 20
    assert reef_report.episode_end_reason(trace) == "step_cap"
    reef_report.report_episode(model, "capped-run", task.task_id, reward.reward, "step_cap")
    assert model.reef.report.call_args.args[1]["references"] == model.receipts
    assert protocol.MAX_STEPS == 100


@pytest.mark.parametrize("correct", [True, False])
def test_official_reward_reaches_one_multi_turn_sao_sample(bundle, correct):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1 and t.answer is None and t.actions)
    responses = ground_truth_responses(task)
    if not correct:
        responses = [completion("finish", {"decision": "refuse", "reason": "deliberately wrong"}, 0)]
    # A length-limited turn is not an episode ending. The official loop must
    # reject its partial tool call, continue, then grade the complete task.
    truncated = completion("finish", {}, -1)
    truncated["choices"][0]["meta_info"] = {"finish_reason": {"type": "length", "length": 512}}
    responses.insert(0, truncated)
    model = model_with_responses(responses)
    reward, trace, usage, _ = agent.run_episode(model, bundle, task)
    assert reward.reward == float(correct)
    assert trace[0]["invalid_reason"] == "length"
    assert reef_report.episode_end_reason(trace) == "finish"
    assert usage["calls"] == len(responses)
    assert usage["output"] == 4 * len(responses)
    reef_report.report_episode(model, "run", task.task_id, reward.reward, "finish")
    payload = model.reef.report.call_args.args[1]
    assert payload["score"] == reward.reward
    assert payload["feedback"] == protocol.score_line(reward.reward)
    assert payload["references"] == model.receipts
    assert set(payload["metadata"]) == {"task", "end_reason"}

    records = []
    tokens = [10]
    for i, receipt in enumerate(model.receipts):
        if i:
            tokens.append(200 + i)  # An observation is context, not an action.
        tokens.append(20 + i)
        records.append(
            AgentRecord.create(
                scenario=model.scenario,
                request_type=RequestType.INFERENCE,
                agent_record_id=receipt,
                payload={
                    "tokens": list(tokens),
                    "loss_mask": [1],
                    "rollout_log_probs": [-0.2],
                    "runtime_load_id": "weights-0",
                },
            )
        )
    report = AgentRecord.create(
        scenario=model.scenario,
        request_type=RequestType.REPORT,
        payload=payload,
        references=tuple(model.receipts),
    )
    tasks = SAOTasks(SampleAssemblySettings(accept_multi_turn_policy_samples=True))
    sample = tasks.make_sample(ReportContext(report=report, score=reward.reward, inferences=tuple(records)))
    assert trajectory_reward(sample) == reward.reward
    assert sample.training["action_mask"] == [1] + [x for _ in responses[1:] for x in (0, 1)]
    assert sample.training["turn_count"] == len(responses)


@pytest.mark.parametrize(
    "status,body,scored", [(400, "maximum context length exceeded", True), (500, "engine unavailable", False)]
)
def test_context_overflow_is_scored_but_service_failure_aborts(bundle, status, body, scored):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1)
    model = model_with_responses([])
    model.reef.inference_with_record.side_effect = ReefClientError(status, body)
    if scored:
        reward, trace, _, _ = agent.run_episode(model, bundle, task)
        assert reward.reward == 0
        assert reef_report.episode_end_reason(trace) == "context_overflow"
    else:
        with pytest.raises(llm.InfraError, match="engine unavailable"):
            agent.run_episode(model, bundle, task)
    model.reef.report.assert_not_called()


def test_training_barrier_never_succeeds_on_timeout(monkeypatch):
    campaign = load_campaign(monkeypatch)
    client = Mock(spec=ReefClient)
    client.get.return_value = {"releases": [], "error": None}
    clock = iter([0, 0, 2, 2])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="did not commit"):
        campaign.wait_for_training(client, "retail", 1, 1)


def test_saved_releases_wait_for_published_version_and_use_newest_commit(monkeypatch):
    campaign = load_campaign(monkeypatch)
    client = Mock(spec=ReefClient)
    checkpoint = {
        "operation": "training",
        "metrics": {"training_job": {"event": "checkpoint", "job_id": "first", "step": 2}, "selected": True},
    }
    newest = {
        **checkpoint,
        "release_id": "second",
        "metrics": {"selected": True, "training_job": {"event": "checkpoint", "job_id": "second", "step": 1}},
    }
    older = {**checkpoint, "release_id": "first"}
    pending = {**checkpoint, "release_id": "pending", "pending": True}
    submitted = {"operation": "training", "metrics": {"training_job": {"event": "submitted"}}}
    finished = {"operation": "training", "metrics": {"training_job": {"event": "finished"}}}
    # Recovered checkpoints may lack worker metrics, including actor_trained.
    warmup = {
        **checkpoint,
        "metrics": {"selected": True, "training_job": {"event": "checkpoint", "job_id": "first", "step": 1}},
    }
    # Open admission alone is insufficient while an asynchronous job still runs.
    states = iter([(True, 1), (False, 2), (True, 2)])

    def get(path):
        if path.endswith("/releases"):
            return {"releases": [finished, pending, newest, submitted, older, warmup, {"operation": "creation"}]}
        open_for_inference, finished_jobs = next(states)
        return {
            "error": None,
            "scenarios": {
                "retail": {
                    "current_runtime_load_id": "v2" if open_for_inference else "v1",
                    "inference_admission": {"open": open_for_inference},
                    "finished_jobs": finished_jobs,
                    "training_job": None if finished_jobs == 2 else {"job_id": "running"},
                }
            },
        }

    client.get.side_effect = get
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    assert campaign.training_releases(client, "retail") == [older, newest]
    release = campaign.wait_for_training(client, "retail", 2, 10)
    assert release == {**newest, "runtime_load_id": "v2"}
    client.get.side_effect = None
    client.get.return_value = {"releases": [warmup]}
    assert campaign.training_releases(client, "retail") == []
    monkeypatch.setenv("SAO_WARMUP", "0")
    assert campaign.training_releases(client, "retail") == [warmup]


def test_training_failure_does_not_wait_for_timeout(monkeypatch):
    campaign = load_campaign(monkeypatch)
    client = Mock(spec=ReefClient)
    client.get.side_effect = [{"releases": []}, {"error": "optimizer failed"}]
    with pytest.raises(RuntimeError, match="optimizer failed"):
        campaign.wait_for_training(client, "retail", 1, 10)


def test_mixed_weight_versions_fail_before_reporting():
    model = model_with_responses([])
    model.receipts = ["a", "b"]
    model.reef.get.side_effect = [{"payload": {"runtime_load_id": version}} for version in ("v0", "v1")]
    with pytest.raises(RuntimeError, match="exactly one"):
        reef_report.episode_runtime_load_id(model)
    model.reef.report.assert_not_called()


@pytest.mark.parametrize("method", ["blind", "oracle", "sao"])
@pytest.mark.parametrize(
    "batch_size,smoke,windows_arg", [(1, True, ""), (8, True, ""), (8, False, ""), (8, True, "1")]
)
def test_campaign_never_reports_test_tasks(bundle, monkeypatch, tmp_path, method, batch_size, smoke, windows_arg):
    monkeypatch.delenv("SAO_BATCH", raising=False)
    campaign = load_campaign(monkeypatch)
    client = Mock(spec=ReefClient)
    client.service_url, client.token = "http://reef.invalid", None
    monkeypatch.setattr(campaign, "ReefClient", lambda *args, **kwargs: client)
    releases, reports, phases, batches = [], [], [], []
    monkeypatch.setattr(campaign, "training_releases", lambda *args: copy.deepcopy(releases))
    monkeypatch.setattr(campaign, "episode_runtime_load_id", lambda model: f"v{len(releases)}")

    def run_episode(model, bundle, task, extra_system, max_steps):
        assert max_steps == 20
        phases.append((task.task_id, bool(extra_system), len(releases)))
        model.receipts.append(f"receipt-{task.task_id}")
        return (
            type("Reward", (), {"reward": 1.0, "answer_detail": {}})(),
            [{"kind": "finish", "name": "finish"}],
            {},
            {},
        )

    def report_episode(model, run_id, task_id, reward, end_reason, *, batch_task_ids):
        if task_id == batch_task_ids[0]:
            batches.append(batch_task_ids)
        reports.append(task_id)
        return {"status": "accepted"}

    def wait_for_training(client, scenario, count, timeout):
        assert len(batches) == count
        assert len(reports) == sum(len(batch) for batch in batches)
        releases.append({"operation": "training", "runtime_load_id": f"v{count}", "release_id": f"release-{count}"})
        return releases[-1]

    monkeypatch.setattr(campaign, "run_episode", run_episode)
    monkeypatch.setattr(campaign, "report_episode", report_episode)
    monkeypatch.setattr(campaign, "wait_for_training", wait_for_training)
    monkeypatch.setenv("REEF_SERVICE_URL", client.service_url)
    path = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--method",
            method,
            "--max-steps",
            "20",
            "--batch-size",
            str(batch_size),
            "--output",
            str(path),
            *(["--smoke"] if smoke else []),
            *(["--windows", windows_arg] if windows_arg else []),
        ],
    )
    campaign.main()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    episodes = [row for row in rows if row.get("type") == "episode"]
    # The stream the driver must have run: every window in order (or the
    # requested ones), the first window only under --smoke.
    windows = [int(w) for w in windows_arg.split(",") if w] or list(range(n_windows(bundle)))
    if smoke:
        windows = windows[:1]
    plan = []
    for window in windows:
        serving = [t for t in bundle["serving"] if t.z["window"] == window]
        tests = bundle["test_set"](window)
        if smoke:
            serving, tests = serving[: max(2, 2 * batch_size - 1)], tests[:2]
        plan.append((window, len(serving), len(tests)))
    serving_count = sum(n for _, n, _ in plan)
    test_count = sum(m for _, _, m in plan)
    trained = sum(n for _, n, _ in plan) if method == "sao" else 0
    assert len(reports) == trained
    assert set(reports).isdisjoint(row["task_id"] for row in episodes if row["phase"] == "test")
    # Each episode sees the releases committed so far; a batch completes a
    # release only within its window, and tests see the window's final count.
    expected_versions: list[int] = []
    released = 0
    for _window, n, m in plan:
        for i in range(n):
            expected_versions.append(released)
            if method == "sao" and ((i + 1) % batch_size == 0 or i == n - 1):
                released += 1
        expected_versions += [released] * m
    assert [row["training_releases_before"] for row in episodes] == expected_versions
    assert [row["window"] for row in episodes] == [w for w, n, m in plan for _ in range(n + m)]
    tails = [row for row in rows if row.get("type") == "untrained_tail"]
    assert not tails
    assert rows[-1]["partial_batch_tasks"] == (sum(n % batch_size for _, n, _ in plan) if method == "sao" else 0)
    assert rows[-1]["training_releases"] == released
    assert rows[-1]["trained_tasks"] == trained
    assert rows[-1]["windows"] == windows
    assert rows[-1]["window_scores"] == {str(w): 100.0 for w in windows}
    assert rows[-1]["ending_counts"] == {"serving": {"finish": serving_count}, "test": {"finish": test_count}}
    assert all(oracle == (method == "oracle") for _, oracle, _ in phases)
    assert rows[-1]["test_score"] == 100
    config, official_rows = read_rows(str(path))
    assert config["method"] == method
    assert config["max_steps"] == 20
    assert config["windows"] == windows
    assert setting_scores(official_rows)["n_test"] == test_count


@pytest.mark.parametrize("warmup", [0, 1, 3, 14])
def test_warmup_adds_passes_without_consuming_actor_batches(monkeypatch, capsys, tmp_path, warmup):
    campaign = load_campaign(monkeypatch)
    monkeypatch.setenv("SAO_MICRO_BATCH", "16")
    monkeypatch.setenv("SAO_WARMUP", str(warmup))
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--method", "sao", "--output", str(tmp_path / "results.jsonl"), "--check-only"]
    )
    campaign.main()
    plan = json.loads(capsys.readouterr().out)["training_plan"]
    assert (plan["training_batches"], plan["critic_only_batches"], plan["actor_updates"]) == (14 + warmup, warmup, 14)
    assert plan["training_jobs"] == 14
    assert sum(row["training_batches"] for row in plan["windows"]) == 14 + warmup
    assert [row["partial_batch_tasks"] for row in plan["windows"]] == [15, 12, 12, 56, 56, 57]
    assert not (tmp_path / "results.jsonl").exists()


@pytest.mark.parametrize("windows", ["1,1", "2,1", "-1", "6", "1,", "a"])
def test_invalid_windows_fail_before_connecting(monkeypatch, tmp_path, windows):
    campaign = load_campaign(monkeypatch)
    monkeypatch.setattr(campaign, "ReefClient", Mock(side_effect=AssertionError("must not connect")))
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--method", "blind", "--windows", windows, "--output", str(tmp_path / "out")]
    )
    with pytest.raises(SystemExit, match="2"):
        campaign.main()


def test_category_scores_match_official_weighting():
    rows = [
        {"phase": "test", "category": "hidden", "reward": 1.0},
        {"phase": "test", "category": "hidden", "reward": 0.0},
        {"phase": "test", "category": "fully_specified", "reward": 1.0},
        {"phase": "serving", "category": "hidden", "reward": 1.0},
    ]
    assert campaign_results.category_scores(rows) == {
        "hidden": {"count": 2, "score": 50.0},
        "fully_specified": {"count": 1, "score": 100.0},
    }
    assert campaign_results.category_scores(rows[:2])["fully_specified"] == {"count": 0, "score": None}


def test_config_single_source_and_micro_divisibility(monkeypatch):
    for name in campaign_config.DEFAULTS:
        monkeypatch.delenv(name)
    assert campaign_config.training_config() == {
        "SAO_BATCH": 64,
        "SAO_MICRO_BATCH": 16,
        "SAO_WARMUP": 1,
        "SAO_CONTEXT": 40960,
    }
    monkeypatch.setenv("SAO_BATCH", "63")
    with pytest.raises(ValueError, match="divisible"):
        campaign_config.training_config()


def test_inference_options_keep_native_numeric_types(monkeypatch):
    monkeypatch.setenv("SAO_CONTEXT", "32768")
    options = campaign_config.inference_options()
    assert options["context-length"] == 32768
    assert type(options["context-length"]) is int
    assert type(options["max-running-requests"]) is int
    assert type(options["chunked-prefill-size"]) is int
    assert type(options["mem-fraction-static"]) is float


def test_deployment_builds_declared_batch_pipeline(monkeypatch, tmp_path):
    for name in ("MODEL_DIR", "CHECKPOINT_DIR", "REEF_WORK"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("REEF_PORT", "8900")
    monkeypatch.setenv("REEF_TOKEN", "test-token")
    monkeypatch.setenv("SAO_GPUS", "1")
    deployment = load_deployment(EXAMPLE / "serve.yaml")
    stages = PipelineConfig.from_setting(
        deployment["pipeline"], spec=RetailSAORecipe.pipeline_spec(), environ=os.environ
    )
    pipeline = stages.build(PipelineContext("retail", report_type=ScoredRolloutReport))
    assert isinstance(pipeline.record2record, RetailBatchReports)
    assert pipeline.record2record.grouping.max_batch_size == 64
    assert pipeline.task2job.job_samples == 1
    assert pipeline.task2job.buffer_size == 64
    assert deployment["reef"]["warmup_steps"] == 1
    assert str(deployment["reef"]["training_backend_options"]["num-critic-only-steps"]) == "1"


@pytest.mark.parametrize(
    "interruption",
    [
        "batch_boundary",
        "partial_collection",
        "partial_reports",
        "committed_without_log",
        "test",
        "tail_report",
        "tail_committed",
    ],
)
def test_resume_matches_uninterrupted_campaign(bundle, monkeypatch, tmp_path, interruption):
    campaign = load_campaign(monkeypatch)
    monkeypatch.setenv("SAO_BATCH", "2")
    client = Mock(spec=ReefClient)
    client.service_url, client.token = "http://reef.invalid", None
    monkeypatch.setattr(campaign, "ReefClient", lambda *args, **kwargs: client)
    releases, accepted, sampled = [], {}, []
    monkeypatch.setattr(campaign, "training_releases", lambda *args: copy.deepcopy(releases))
    monkeypatch.setattr(campaign, "episode_runtime_load_id", lambda model: f"session-{session[0]}-v{len(releases)}")
    session = [0]
    interrupted = [False]

    def run_episode(model, bundle, task, **kwargs):
        # Stop before the second collection or test task. Its predecessor is already durable.
        threshold = 1 if interruption == "partial_collection" else 4
        if interruption in ("partial_collection", "test") and not interrupted[0] and len(sampled) == threshold:
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        sampled.append(task.task_id)
        model.receipts.append(f"receipt-{len(sampled)}")
        return (
            type("Reward", (), {"reward": 1.0, "answer_detail": {}})(),
            [{"kind": "finish", "name": "finish"}],
            {},
            {},
        )

    def report_episode(model, run_id, task_id, reward, end_reason, *, batch_task_ids):
        report = (tuple(model.receipts), reward, end_reason, tuple(batch_task_ids))
        if task_id in accepted:
            assert accepted[task_id] == report  # Idempotent retry must keep the original receipts.
        else:
            accepted[task_id] = report
        if (
            interruption == "partial_reports" or (interruption == "tail_report" and len(accepted) == 3)
        ) and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        return {"status": "accepted"}

    def wait_for_training(client, scenario, count, timeout):
        if len(releases) < count:
            assert len(accepted) == min(count * 2, 3)
            releases.append({"operation": "training", "release_id": f"release-{count}"})
        if (
            interruption == "committed_without_log" or (interruption == "tail_committed" and count == 2)
        ) and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        return {**releases[-1], "runtime_load_id": f"session-{session[0]}-v{count}"}

    monkeypatch.setattr(campaign, "run_episode", run_episode)
    monkeypatch.setattr(campaign, "report_episode", report_episode)
    monkeypatch.setattr(campaign, "wait_for_training", wait_for_training)
    monkeypatch.setenv("REEF_SERVICE_URL", client.service_url)
    output = tmp_path / "results.jsonl"
    command = ["run.py", "--method", "sao", "--smoke", "--windows", "1", "--output", str(output)]
    monkeypatch.setattr(
        sys, "argv", command + (["--stop-after-batches", "1"] if interruption == "batch_boundary" else [])
    )
    if interruption == "batch_boundary":
        campaign.main()
    else:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            campaign.main()
    session[0] += 1
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    campaign.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    episodes = [row for row in rows if row.get("type") == "episode"]
    assert len(episodes) == len({row["task_id"] for row in episodes}) == 5
    assert len(accepted) == 3 and len(releases) == 2
    assert all(row["task_id"] not in accepted for row in episodes if row["phase"] == "test")
    assert rows[-1]["test_score"] == 100.0
    assert rows[-1]["training_releases"] == 2
    assert rows[-1]["trained_tasks"] == 3
    # Resuming a completed run neither samples nor reports again.
    counts = (len(sampled), len(accepted), len(releases))
    campaign.main()
    assert (len(sampled), len(accepted), len(releases)) == counts


def test_resume_rejects_changed_config_and_mismatched_checkpoints(monkeypatch, tmp_path):
    campaign = load_campaign(monkeypatch)
    path = tmp_path / "results.jsonl"
    log = campaign_results.ResultLog(path, resume=False)
    log.append({"config": {"seed": 42}})
    client = Mock(spec=ReefClient)
    model = model_with_responses([])
    try:
        with pytest.raises(RuntimeError, match="configuration differs"):
            campaign.recover_campaign(log, client, model, {"seed": 43}, 1)
        log.append({"type": "batch_ready", "task_ids": ["a"]})
        log.append({"type": "training", "task_ids": ["a"], "release": {"release_id": "original"}})
        monkeypatch.setattr(campaign, "training_releases", lambda *args: [{"release_id": "different"}])
        with pytest.raises(RuntimeError, match="history disagree"):
            campaign.recover_campaign(log, client, model, {"seed": 42}, 1)
    finally:
        log.close()


def test_resume_discards_only_incomplete_final_json_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"config": {}}\n{"type": "episode"')
    log = campaign_results.ResultLog(path, resume=True)
    try:
        assert log.rows == [{"config": {}}]
        assert path.read_bytes() == b'{"config": {}}\n'
    finally:
        log.close()


@pytest.mark.parametrize("count", [64, 15, 12, 56, 57])
def test_declared_batch_waits_for_every_task_and_keeps_tail_rewards(count):
    context = PipelineContext("retail", report_type=ScoredRolloutReport)

    def pipeline():
        reports = RetailBatchReports.from_config({"max_batch_size": 64}, context=context, environ={})
        tasks = SAOTasks(SampleAssemblySettings(accept_multi_turn_policy_samples=True))
        jobs = SAOJobData(context, buffer_size=64, job_samples=1)
        return TrainingPipeline(context, record2record=reports, record2task=tasks, task2job=jobs)

    task_ids = [f"task-{index}" for index in range(count)]
    records = []
    active = pipeline()
    for index, task_id in enumerate(task_ids):
        model = model_with_responses([])
        model.scenario = "retail"
        model.receipts = [f"inference-{index}"]
        reef_report.report_episode(model, "retail", task_id, float(index % 2), "finish", batch_task_ids=task_ids)
        payload = model.reef.report.call_args.args[1]
        inference = AgentRecord.create(
            scenario="retail",
            request_type=RequestType.INFERENCE,
            agent_record_id=model.receipts[0],
            payload={"tokens": [10, 20], "loss_mask": [1], "rollout_log_probs": [-0.2], "runtime_load_id": "v0"},
        )
        report = AgentRecord.create(
            scenario="retail",
            request_type=RequestType.REPORT,
            agent_record_id=payload["agent_record_id"],
            payload=payload,
            references=tuple(model.receipts),
        )
        records.extend((inference, report))
        active.push(inference)
        active.push(report)
        assert active.ready(1) == (index == count - 1)
        if index == count // 2:
            # Rebuild from persisted records halfway through collection.
            active = pipeline()
            for record in records:
                active.push(record)
            assert not active.ready(1)
    data = active.build_job_data(1)
    assert len(data.items) == count
    assert [trajectory_reward(item) for item in data.items] == [float(index % 2) for index in range(count)]
    prepared = prepare_slime_job(data, TrainingMethod("sao"), {}, StepScheduling(), step_size=64)
    assert prepared.payload["external_step_sizes"] == [count]
    assert prepared.metrics["optimizer_steps"] == 1
    # First-job warmup repeats the exact rows, with a separate saved step per pass.
    for warmup_steps in (0, 1, 3):
        job_plan = RetailJobPlan(warmup_steps)
        context = JobPlanContext(data, 0, None)
        request = job_plan.next_job(context)
        assert request == RetailJobPlan(warmup_steps).next_job(context)
        method, scheduling = job_training(request.parameters)
        repeated = prepare_slime_job(data, method, {}, scheduling, step_size=64).payload
        assert repeated["external_step_sizes"] == [count] * (warmup_steps + 1)
        assert repeated["samples"] == prepared.payload["samples"] * (warmup_steps + 1)
        assert "advantages" not in repeated
        assert request.parameters["save_interval"] == 1
        later = job_plan.next_job(JobPlanContext(data, 1, JobOutcome("first", "slime", "finished")))
        assert later.parameters["scheduling"]["epochs"] == 1
    assert active.acknowledge(data.data_id)
    assert not active.ready(1)


@pytest.mark.parametrize("task_ids", [[], ["a", "a"], ["b"], list(map(str, range(65)))])
def test_batch_membership_rejects_invalid_task_lists(task_ids):
    report = AgentRecord.create(
        scenario="retail",
        request_type=RequestType.REPORT,
        payload={"score": 1, "metadata": {"task": {"name": "a"}, "batch_task_ids": task_ids}},
    )
    with pytest.raises(ValueError, match="batch_task_ids"):
        BatchGrouping(64).placement(ReportContext(report=report, score=1, inferences=()))
