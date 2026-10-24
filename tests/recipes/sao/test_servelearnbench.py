"""Official Retail environment, scripted inference, and the real SAO assembly.

Install recipes/sao/examples/servelearnbench to run this optional benchmark suite.
Scripted responses test the adapter contract, not model quality or GPU execution.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("servelearnbench")

from reef_client import ReefClient, ReefClientError
from servelearnbench import agent, llm, protocol
from servelearnbench.scenarios import get_bundle
from servelearnbench.scoring import read_rows, setting_scores

from recipes.sao.examples.servelearnbench.harness import adapter
from recipes.sao.processor import SAOTasks
from reef.core import AgentRecord, RequestType
from reef.core.trajectories import trajectory_reward
from reef.record2record import ReportContext
from reef.record2task.assembly import SampleAssemblySettings

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/sao/examples/servelearnbench"


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
    return adapter.ReefLLM(client, "retail-test", llm.ModelConfig(model="reef", max_tokens=512), 42)


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
    configured = adapter.run_episode(model_with_responses(responses), bundle, task, max_steps=20)
    assert configured == official


def test_configured_turn_cap_keeps_official_reward_and_all_receipts(bundle):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1 and t.answer is None and t.actions)
    model = model_with_responses([completion("finish", {}, i, finish_reason="length") for i in range(20)])
    reward, trace, usage, _ = adapter.run_episode(model, bundle, task, max_steps=20)
    assert reward.reward == 0
    assert usage["calls"] == len(model.receipts) == 20
    assert adapter.episode_end_reason(trace) == "step_cap"
    adapter.report_episode(model, "capped-run", task.task_id, reward.reward, "step_cap")
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
    assert adapter.episode_end_reason(trace) == "finish"
    assert usage["calls"] == len(responses)
    assert usage["output"] == 4 * len(responses)
    adapter.report_episode(model, "run", task.task_id, reward.reward, "finish")
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
        assert adapter.episode_end_reason(trace) == "context_overflow"
    else:
        with pytest.raises(llm.InfraError, match="engine unavailable"):
            agent.run_episode(model, bundle, task)
    model.reef.report.assert_not_called()


def test_training_barrier_never_succeeds_on_timeout(monkeypatch):
    client = Mock(spec=ReefClient)
    client.get.return_value = {"releases": [], "error": None}
    clock = iter([0, 0, 2, 2])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(adapter.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="did not commit"):
        adapter.wait_for_training(client, "retail", 1, 1)


def test_saved_releases_wait_for_published_version_and_use_newest_commit(monkeypatch):
    client = Mock(spec=ReefClient)
    newest = {"operation": "training", "release_id": "second", "content_kind": "saved_artifact"}
    older = {"operation": "training", "release_id": "first", "content_kind": "saved_artifact"}
    pending = {"operation": "training", "release_id": "pending", "pending": True}
    admissions = iter([False, True])

    def get(path):
        if path.endswith("/releases"):
            return {"releases": [pending, newest, older, {"operation": "creation"}]}
        open_for_inference = next(admissions)
        return {
            "error": None,
            "scenarios": {
                "retail": {
                    "current_runtime_load_id": "v2" if open_for_inference else "v1",
                    "inference_admission": {"open": open_for_inference},
                }
            },
        }

    client.get.side_effect = get
    monkeypatch.setattr(adapter.time, "sleep", lambda _: None)
    assert adapter.training_releases(client, "retail") == [older, newest]
    release = adapter.wait_for_training(client, "retail", 2, 10)
    assert release == {**newest, "runtime_load_id": "v2"}


def test_training_failure_does_not_wait_for_timeout():
    client = Mock(spec=ReefClient)
    client.get.side_effect = [{"releases": []}, {"error": "optimizer failed"}]
    with pytest.raises(RuntimeError, match="optimizer failed"):
        adapter.wait_for_training(client, "retail", 1, 10)


def test_mixed_weight_versions_fail_before_reporting():
    model = model_with_responses([])
    model.receipts = ["a", "b"]
    model.reef.get.side_effect = [{"payload": {"runtime_load_id": version}} for version in ("v0", "v1")]
    with pytest.raises(RuntimeError, match="exactly one"):
        adapter.episode_runtime_load_id(model)
    model.reef.report.assert_not_called()


@pytest.mark.parametrize("method", ["blind", "oracle", "sao"])
@pytest.mark.parametrize("batch_size,smoke", [(1, True), (8, True), (8, False)])
def test_campaign_never_reports_test_tasks(bundle, monkeypatch, tmp_path, method, batch_size, smoke):
    monkeypatch.delenv("SAO_BATCH", raising=False)
    monkeypatch.setitem(sys.modules, "harness.adapter", adapter)
    spec = importlib.util.spec_from_file_location("retail_campaign", EXAMPLE / "run.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    client = Mock(spec=ReefClient)
    client.service_url, client.token = "http://reef.invalid", None
    monkeypatch.setattr(campaign, "ReefClient", lambda *args, **kwargs: client)
    releases, reports, phases = [], [], []
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

    def report_episode(model, run_id, task_id, reward, end_reason):
        reports.append(task_id)
        return {"status": "accepted"}

    def wait_for_training(client, scenario, count, timeout):
        assert len(reports) == count * batch_size
        releases.append({"operation": "training", "runtime_load_id": f"v{count}"})
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
        ],
    )
    campaign.main()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    episodes = [row for row in rows if row.get("type") == "episode"]
    serving_count = 2 * batch_size if smoke else 140
    test_count = 2 if smoke else 64
    expected = serving_count // batch_size * batch_size if method == "sao" else 0
    assert len(reports) == expected
    assert set(reports).isdisjoint(row["task_id"] for row in episodes if row["phase"] == "test")
    expected_versions = (
        ([i // batch_size for i in range(serving_count)] + [expected // batch_size] * test_count)
        if method == "sao"
        else [0] * (serving_count + test_count)
    )
    assert [row["training_releases_before"] for row in episodes] == expected_versions
    tails = [row for row in rows if row.get("type") == "untrained_tail"]
    assert len(tails) == (serving_count % batch_size if method == "sao" else 0)
    assert rows[-1]["training_releases"] == expected // batch_size
    assert rows[-1]["trained_tasks"] == expected
    assert rows[-1]["ending_counts"] == {"serving": {"finish": serving_count}, "test": {"finish": test_count}}
    assert all(oracle == (method == "oracle") for _, oracle, _ in phases)
    assert rows[-1]["test_score"] == 100
    config, official_rows = read_rows(str(path))
    assert config["method"] == method
    assert config["max_steps"] == 20
    assert setting_scores(official_rows)["n_test"] == test_count
