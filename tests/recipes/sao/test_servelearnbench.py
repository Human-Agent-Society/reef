"""Official Retail environment, scripted inference, and the real SAO sample assembly.

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
from tests.support._trajectories import policy_trajectory
from tests.support.config_helpers import load_deployment
from tests.support.conversations import conversation

from recipes.sao.examples.servelearnbench import harness as reef_harness
from recipes.sao.examples.servelearnbench.harness import agent as reef_agent
from recipes.sao.examples.servelearnbench.harness import campaign as campaign_results
from recipes.sao.examples.servelearnbench.harness import config as campaign_config
from recipes.sao.examples.servelearnbench.harness import report as reef_report
from recipes.sao.processor import SAOJobData, SAOSamples
from recipes.sao.recipe import SAORecipe
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import TokenizerChatTemplate, render_trajectory
from reef.core.reports import ScoredRolloutReport
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2sample.assembly import SampleAssemblySettings
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import job_training
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/sao/examples/servelearnbench"
SCENARIO = "retail-test"


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
    """A chat completion as Reef's handler answers it: the reply message carries its role."""
    return {
        "id": f"chatcmpl-{index}",
        "object": "chat.completion",
        "model": "reef",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Scripted test response.",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "meta_info": {},
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


def model_with_responses(responses, artifact_id="artifact-0"):
    """A ReefLLM over a client that answers ``responses`` in order and keeps a copy of each request body."""
    client = Mock(spec=ReefClient)
    client.service_url, client.token = "http://reef.invalid", None
    client.requests = []

    def post(path, scenario, payload, **_kwargs):
        client.requests.append(copy.deepcopy(payload))
        index = len(client.requests) - 1
        return copy.deepcopy(responses[index]), {
            "x-reef-agent-record-id": f"receipt-{index}",
            "x-reef-release-id": artifact_id,
        }

    client.post.side_effect = post
    return reef_agent.ReefLLM(client, SCENARIO, llm.ModelConfig(model="reef", max_tokens=512), 42)


def manual_sao_pipeline(scenario=SCENARIO):
    context = PipelineContext(scenario, report_type=ScoredRolloutReport, training_mode="manual")
    return TrainingPipeline(
        context,
        record2sample=SAOSamples(context, SampleAssemblySettings(accept_multi_turn_policy_samples=True)),
        sample2job=SAOJobData(context, None, 1),
    )


def instruction_request(campaign, instruction_id, epochs, artifact_id="artifact-0"):
    """The queued instruction that the driver's ``POST /reef/train`` body makes."""
    body = campaign.training_instruction(instruction_id, "train the batch", epochs, artifact_id)
    return TrainingRequest.from_dict(
        {key: value for key, value in body.items() if key != "agent_record_id"}, request_id=instruction_id
    )


def episode_records(model, responses):
    """The inference records of the episode that ``model`` served, as the inference service stores them."""
    return [
        AgentRecord.create(
            scenario=model.scenario,
            request_type=RequestType.INFERENCE,
            agent_record_id=receipt,
            payload={**request, "artifact_id": artifact_id, "response": response},
        )
        for receipt, artifact_id, request, response in zip(
            model.receipts, model.artifact_ids, model.reef.requests, responses, strict=True
        )
    ]


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


def test_configured_turn_cap_keeps_official_reward_and_reports_the_last_call(bundle):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1 and t.answer is None and t.actions)
    model = model_with_responses([completion("finish", {}, i, finish_reason="length") for i in range(20)])
    reward, trace, usage, _ = reef_agent.run_episode(model, bundle, task, max_steps=20)
    assert reward.reward == 0
    assert usage["calls"] == len(model.receipts) == len(model.artifact_ids) == 20
    assert reef_report.episode_end_reason(trace) == "step_cap"
    assert reef_report.episode_artifact_id(model) == "artifact-0"
    reef_report.report_episode(model, "capped-run", task.task_id, reward.reward, "step_cap")
    payload = model.reef.report.call_args.args[1]
    assert payload["references"] == [model.receipts[-1]]
    assert payload["metadata"] == {"task": {"name": task.task_id}, "end_reason": "step_cap", "calls": 20}
    assert protocol.MAX_STEPS == 100


def sao_sample_of_episode(campaign, bundle, correct):
    """Run one scripted episode, report it, and make its SAO sample through the manual pipeline."""
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
    assert payload["references"] == [model.receipts[-1]]

    pipeline = manual_sao_pipeline()
    for record in episode_records(model, responses):
        pipeline.push(record)
    pipeline.push(
        AgentRecord.create(
            scenario=model.scenario,
            request_type=RequestType.REPORT,
            agent_record_id=payload["agent_record_id"],
            payload=payload,
        )
    )
    pipeline.store_samples()
    sample_ids = tuple(stored.sample_id for stored in pipeline.store.page())
    assert sample_ids == (f"{payload['agent_record_id']}:0",)
    pipeline.insert_into_buffer(sample_ids)
    pipeline.queue_instructions([instruction_request(campaign, "instruction-1", 2)])
    data = pipeline.build_job_data(1)
    (sample,) = data.samples
    assert trajectory_reward(sample) == reward.reward
    assert sample.source_agent_record_ids == (*model.receipts, payload["agent_record_id"])
    return data, len(responses)


@pytest.mark.parametrize("correct", [True, False])
def test_official_reward_reaches_one_multi_turn_sao_sample(bundle, monkeypatch, correct):
    campaign = load_campaign(monkeypatch)
    data, calls = sao_sample_of_episode(campaign, bundle, correct)
    (sample,) = data.samples
    # One trained agent step per model call: the benchmark resends each reply as Reef recorded it, so no
    # reply comes back as copied context, and the tool results fold into the steps that called them.
    agent_steps = [step for step in sample.trajectory["steps"] if step["source"] == "agent"]
    assert len(agent_steps) == calls
    assert not any(step.get("extra", {}).get("reef", {}).get("is_copied_context") for step in agent_steps)
    assert sample.artifact_id == "artifact-0"
    assert isinstance(sample.training["rollout_created_at"], float)
    assert "tokens" not in sample.training  # The training component renders the sample.
    method, scheduling = job_training({**data.request.parameters, "method": TrainingMethod("sao").to_dict()})
    assert (method.objective, scheduling.epochs, scheduling.remainder) == ("sao", 2, "partial")


def test_qwen3_template_renders_every_reply_of_a_tool_episode(bundle, monkeypatch):
    path = os.environ.get("REEF_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set REEF_TEST_QWEN_TOKENIZER to a local Qwen3 tokenizer for this check")
    transformers = pytest.importorskip("transformers")
    campaign = load_campaign(monkeypatch)
    data, calls = sao_sample_of_episode(campaign, bundle, True)
    (sample,) = data.samples
    template = TokenizerChatTemplate(transformers.AutoTokenizer.from_pretrained(path), {"enable_thinking": True})
    rendered = render_trajectory(template, sample.trajectory)
    assert 0 < sum(rendered.loss_mask) < len(rendered.loss_mask) < len(rendered.tokens)
    # Each reply is one trained run of tokens; the observations between them are context.
    runs = sum(
        1
        for index, flag in enumerate(rendered.loss_mask)
        if flag and (index == 0 or not rendered.loss_mask[index - 1])
    )
    assert runs == calls


@pytest.mark.parametrize(
    "status,body,scored", [(400, "maximum context length exceeded", True), (500, "engine unavailable", False)]
)
def test_context_overflow_is_scored_but_service_failure_aborts(bundle, status, body, scored):
    task = next(t for t in bundle["serving"] if t.z["window"] == 1)
    model = model_with_responses([])
    model.reef.post.side_effect = ReefClientError(status, body)
    if scored:
        reward, trace, _, _ = agent.run_episode(model, bundle, task)
        assert reward.reward == 0
        assert reef_report.episode_end_reason(trace) == "context_overflow"
    else:
        with pytest.raises(llm.InfraError, match="engine unavailable"):
            agent.run_episode(model, bundle, task)
    model.reef.report.assert_not_called()


def test_mixed_served_artifacts_fail_before_reporting():
    model = model_with_responses([])
    model.receipts, model.artifact_ids = ["a", "b"], ["artifact-0", "artifact-1"]
    with pytest.raises(RuntimeError, match="exactly one"):
        reef_report.episode_artifact_id(model)
    model.reef.report.assert_not_called()


def status_block(finished_jobs, *, running=False, open_admission=True, served=None, start=None):
    """One scenario's block of GET /reef/status after ``finished_jobs`` jobs."""
    served = f"artifact-{finished_jobs}" if served is None else served
    return {
        "scenario_step": 3 * finished_jobs,
        "finished_jobs": finished_jobs,
        "training_job": {"job_id": "job-running"} if running else None,
        "served_artifact_id": served,
        "training_mode": "manual",
        "inference_admission": {"open": open_admission},
        "artifacts": {"": {"active": served, "training_start": served if start is None else start}},
    }


def job_row(instruction_id, state="finished", error=None):
    return {
        "id": f"job-{instruction_id}",
        "state": state,
        "error": error,
        "checkpoints": [],
        "instruction": {"id": instruction_id},
    }


def reef_reads(statuses, jobs, sample_pages=()):
    """A ReefClient mock whose GET answers the status blocks in turn, the job list, and the sample pages."""
    client = Mock(spec=ReefClient)
    client.service_url, client.token = "http://reef.invalid", None
    remaining = list(statuses)
    pages = list(sample_pages)

    def get(path, **_kwargs):
        if path == "/reef/status":
            block = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return {"error": block.get("error"), "scenarios": {SCENARIO: block}}
        if path.endswith("/jobs"):
            return {"scenario": SCENARIO, "jobs": list(jobs)}
        if "/samples?" in path:
            return pages.pop(0) if len(pages) > 1 else pages[0]
        raise AssertionError(f"unexpected read {path}")

    client.get.side_effect = get
    return client


def test_training_barrier_never_succeeds_on_timeout(monkeypatch):
    campaign = load_campaign(monkeypatch)
    client = reef_reads([status_block(0, running=True)], [])
    clock = iter([0, 0, 2, 2])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="did not serve"):
        campaign.wait_for_training(client, SCENARIO, "i1", 1, "artifact-0", 1)


def test_training_barrier_needs_the_finished_job_open_admission_and_the_served_checkpoint(monkeypatch):
    campaign = load_campaign(monkeypatch)
    statuses = [
        status_block(0, running=True, served="artifact-0"),
        status_block(1, open_admission=False),
        status_block(1, served="artifact-0"),  # The job finished; serving has not moved to its checkpoint yet.
        status_block(1, start="artifact-0"),  # Serving moved, the trainer's start not yet.
        status_block(1),
    ]
    client = reef_reads(statuses, [job_row("i1")])
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    result = campaign.wait_for_training(client, SCENARIO, "i1", 1, "artifact-0", 10)
    assert result == {"job_id": "job-i1", "artifact_id": "artifact-1", "scenario_step": 3}
    assert client.get.call_count == 10


@pytest.mark.parametrize(
    "statuses,jobs,match",
    [
        ([{**status_block(0), "error": "optimizer failed"}], [], "optimizer failed"),
        ([status_block(0, running=True)], [job_row("i1", "failed", "cuda out of memory")], "cuda out of memory"),
        ([status_block(2)], [job_row("i1")], "Unexpected training job"),
    ],
)
def test_training_failures_do_not_wait_for_timeout(monkeypatch, statuses, jobs, match):
    campaign = load_campaign(monkeypatch)
    client = reef_reads(statuses, jobs)
    with pytest.raises(RuntimeError, match=match):
        campaign.wait_for_training(client, SCENARIO, "i1", 1, "artifact-0", 10)


def test_wait_for_samples_pages_the_store_until_every_report_has_a_sample(monkeypatch):
    campaign = load_campaign(monkeypatch)
    first = {"sample_id": "r1:0", "sequence": 1, "train_counts": {}}
    second = {"sample_id": "r2:0", "sequence": 7, "train_counts": {"": 1}}
    pages = [
        {"samples": [first, {"sample_id": "other:0", "sequence": 2, "train_counts": {}}], "next": 2},
        {"samples": [], "next": None},
        {"samples": [first], "next": 2},
        {"samples": [second], "next": None},
    ]
    client = reef_reads([status_block(0)], [], pages)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    assert campaign.wait_for_samples(client, SCENARIO, ["r2", "r1"], 10) == [second, first]
    assert [call.args[0] for call in client.get.call_args_list if "/samples?" in call.args[0]] == [
        f"/reef/scenarios/{SCENARIO}/samples?after=0&limit=1000",
        f"/reef/scenarios/{SCENARIO}/samples?after=2&limit=1000",
        f"/reef/scenarios/{SCENARIO}/samples?after=0&limit=1000",
        f"/reef/scenarios/{SCENARIO}/samples?after=2&limit=1000",
    ]


def test_train_batch_queues_a_batch_once_and_repeats_safely(monkeypatch):
    campaign = load_campaign(monkeypatch)
    model = model_with_responses([])
    batch = [
        {"task_id": "t1", "receipts": ["a1", "a2"], "reward": 1.0, "end_reason": "finish"},
        {"task_id": "t2", "receipts": ["b1"], "reward": 0.0, "end_reason": "step_cap"},
    ]
    report_ids = [reef_report.report_id(SCENARIO, row["task_id"]) for row in batch]
    instruction_id = campaign.training_instruction_id(SCENARIO, ["t1", "t2"])
    waited = []
    monkeypatch.setattr(campaign, "wait_for_training", lambda *args: waited.append(args) or {"job_id": "job"})

    def samples(trained):
        return {
            "samples": [
                {"sample_id": f"{r}:0", "sequence": i, "train_counts": trained} for i, r in enumerate(report_ids)
            ],
            "next": None,
        }

    def run(jobs, trained):
        client = reef_reads([status_block(0)], jobs, [samples(trained)])
        model.reef.report.reset_mock()
        return client, campaign.train_batch(
            client,
            model,
            SCENARIO,
            batch,
            instruction_id=instruction_id,
            text="train",
            epochs=2,
            expected_jobs=1,
            served_artifact_id="artifact-0",
            timeout_seconds=10,
        )

    client, result = run([], {})
    assert result == {"job_id": "job"}
    assert [call.args[1]["references"] for call in model.reef.report.call_args_list] == [["a2"], ["b1"]]
    assert [call.args[0] for call in client.post.call_args_list] == [
        f"/reef/scenarios/{SCENARIO}/buffer/insert",
        "/reef/train",
    ]
    assert client.post.call_args_list[0].args[2] == {"sample_ids": [f"{r}:0" for r in report_ids]}
    assert client.post.call_args_list[1].args[2] == campaign.training_instruction(
        instruction_id, "train", 2, "artifact-0"
    )
    assert waited[-1][2:] == (instruction_id, 1, "artifact-0", 10)
    # After an interruption, the reports repeat with the same IDs and a batch whose job exists is not queued again.
    client, _ = run([job_row(instruction_id)], {"": 1})
    assert model.reef.report.call_count == 2
    client.post.assert_not_called()
    with pytest.raises(RuntimeError, match="trained outside"):
        run([], {"": 1})


class FakeTraining:
    """The training state that the driver reads: finished jobs by instruction, and the artifact each one serves."""

    def __init__(self):
        self.jobs: list[dict] = []
        self.reports: list[str] = []
        self.batches: list[list[str]] = []

    def status(self, *_args):
        return status_block(len(self.jobs))

    def listing(self, *_args):
        return [job_row(job["instruction_id"]) for job in self.jobs]

    def train_batch(
        self,
        client,
        model,
        scenario,
        batch,
        *,
        instruction_id,
        text,
        epochs,
        expected_jobs,
        served_artifact_id,
        timeout_seconds,
    ):
        assert expected_jobs == len(self.jobs) + 1
        assert served_artifact_id == f"artifact-{len(self.jobs)}"
        assert epochs == (2 if not self.jobs else 1)
        for row in batch:
            assert row["receipts"]
            self.reports.append(row["task_id"])
        self.batches.append([row["task_id"] for row in batch])
        self.jobs.append({"instruction_id": instruction_id, "task_ids": self.batches[-1]})
        return {
            "job_id": f"job-{len(self.jobs)}",
            "artifact_id": f"artifact-{len(self.jobs)}",
            "scenario_step": 3 * len(self.jobs),
        }

    def install(self, monkeypatch, campaign):
        client = Mock(spec=ReefClient)
        client.service_url, client.token = "http://reef.invalid", None
        monkeypatch.setattr(campaign, "ReefClient", lambda *args, **kwargs: client)
        monkeypatch.setattr(campaign, "ensure_scenario", lambda client, scenario, *, trains: self.status())
        monkeypatch.setattr(campaign, "scenario_status", self.status)
        monkeypatch.setattr(campaign, "training_jobs", self.listing)
        monkeypatch.setattr(campaign, "train_batch", self.train_batch)
        monkeypatch.setattr(campaign, "episode_artifact_id", lambda model: f"artifact-{len(self.jobs)}")
        monkeypatch.setenv("REEF_SERVICE_URL", client.service_url)
        return client


@pytest.mark.parametrize("method", ["blind", "oracle", "sao"])
@pytest.mark.parametrize(
    "batch_size,smoke,windows_arg", [(1, True, ""), (8, True, ""), (8, False, ""), (8, True, "1")]
)
def test_campaign_never_reports_test_tasks(bundle, monkeypatch, tmp_path, method, batch_size, smoke, windows_arg):
    monkeypatch.delenv("SAO_BATCH", raising=False)
    campaign = load_campaign(monkeypatch)
    training = FakeTraining()
    training.install(monkeypatch, campaign)
    phases = []

    def run_episode(model, bundle, task, extra_system, max_steps):
        assert max_steps == 20
        phases.append((task.task_id, bool(extra_system), len(training.jobs)))
        model.receipts.append(f"receipt-{task.task_id}")
        return (
            type("Reward", (), {"reward": 1.0, "answer_detail": {}})(),
            [{"kind": "finish", "name": "finish"}],
            {},
            {},
        )

    monkeypatch.setattr(campaign, "run_episode", run_episode)
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
    assert len(training.reports) == trained
    assert set(training.reports).isdisjoint(row["task_id"] for row in episodes if row["phase"] == "test")
    # Each episode sees the jobs finished so far; a batch completes a job only
    # within its window, and tests see the window's final count.
    expected_jobs: list[int] = []
    finished = 0
    for _window, n, m in plan:
        for i in range(n):
            expected_jobs.append(finished)
            if method == "sao" and ((i + 1) % batch_size == 0 or i == n - 1):
                finished += 1
        expected_jobs += [finished] * m
    assert [row["training_jobs_before"] for row in episodes] == expected_jobs
    assert [row["artifact_id"] for row in episodes] == [f"artifact-{count}" for count in expected_jobs]
    assert [row["window"] for row in episodes] == [w for w, n, m in plan for _ in range(n + m)]
    declared = [row for row in rows if row.get("type") == "batch_ready"]
    assert [row["task_ids"] for row in declared] == training.batches
    assert [row["epochs"] for row in declared] == ([2] + [1] * (len(declared) - 1) if declared else [])
    assert rows[-1]["partial_batch_tasks"] == (sum(n % batch_size for _, n, _ in plan) if method == "sao" else 0)
    assert rows[-1]["training_jobs"] == finished
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


def test_deployment_builds_the_manual_sao_pipeline(monkeypatch, tmp_path):
    for name in ("MODEL_DIR", "CHECKPOINT_DIR", "REEF_WORK"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("REEF_PORT", "8900")
    monkeypatch.setenv("REEF_TOKEN", "test-token")
    monkeypatch.setenv("SAO_GPUS", "1")
    deployment = load_deployment(EXAMPLE / "serve.yaml")
    assert deployment["reef"]["training_mode"] == "manual"
    assert deployment["reef"]["inference_handler_config"]["rollout_continuation"] is True
    options = deployment["reef"]["training_backend_options"]
    assert str(options["num-critic-only-steps"]) == "1"
    # The actor computes the old-policy log-probabilities of each batch with the weights of the job's start.
    assert "use-rollout-logprobs" not in options
    stages = PipelineConfig.from_setting(deployment["pipeline"], spec=SAORecipe.pipeline_spec(), environ=os.environ)
    pipeline = stages.build(PipelineContext("retail", report_type=ScoredRolloutReport, training_mode="manual"))
    assert isinstance(pipeline.record2sample, SAOSamples)
    assert pipeline.record2sample.assembly.settings.accept_multi_turn_policy_samples
    assert isinstance(pipeline.sample2job, SAOJobData)
    assert pipeline.sample2job.job_samples == 1


@pytest.mark.parametrize(
    "interruption",
    [
        "batch_boundary",
        "partial_collection",
        "before_training",
        "trained_without_log",
        "test",
        "tail_before_training",
        "tail_trained_without_log",
    ],
)
def test_resume_matches_uninterrupted_campaign(bundle, monkeypatch, tmp_path, interruption):
    campaign = load_campaign(monkeypatch)
    monkeypatch.setenv("SAO_BATCH", "2")
    training = FakeTraining()
    training.install(monkeypatch, campaign)
    sampled = []
    interrupted = [False]
    accepted: dict[str, tuple] = {}
    finished_train_batch = training.train_batch

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

    def train_batch(client, model, scenario, batch, *, instruction_id, **kwargs):
        tail = len(batch) == 1
        for row in batch:
            # A repeated batch keeps its reports: the same receipts, reward, ending and instruction.
            report = (tuple(row["receipts"]), row["reward"], row["end_reason"], instruction_id)
            assert accepted.setdefault(row["task_id"], report) == report
        existing = next((job for job in training.jobs if job["instruction_id"] == instruction_id), None)
        if existing is not None:
            # The job of an interrupted batch finished: nothing trains again.
            index = training.jobs.index(existing) + 1
            return {"job_id": f"job-{index}", "artifact_id": f"artifact-{index}", "scenario_step": 3 * index}
        if (
            interruption in ("before_training", "tail_before_training")
            and not interrupted[0]
            and tail == interruption.startswith("tail")
        ):
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        result = finished_train_batch(client, model, scenario, batch, instruction_id=instruction_id, **kwargs)
        if (
            interruption in ("trained_without_log", "tail_trained_without_log")
            and not interrupted[0]
            and tail == interruption.startswith("tail")
        ):
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(campaign, "run_episode", run_episode)
    monkeypatch.setattr(campaign, "train_batch", train_batch)
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
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    campaign.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    episodes = [row for row in rows if row.get("type") == "episode"]
    assert len(episodes) == len({row["task_id"] for row in episodes}) == 5
    assert len(accepted) == 3 and len(training.jobs) == 2
    assert all(row["task_id"] not in accepted for row in episodes if row["phase"] == "test")
    assert [row["instruction_id"] for row in rows if row.get("type") == "training"] == [
        job["instruction_id"] for job in training.jobs
    ]
    assert rows[-1]["test_score"] == 100.0
    assert rows[-1]["training_jobs"] == 2
    assert rows[-1]["trained_tasks"] == 3
    # Resuming a completed run neither samples nor reports again.
    counts = (len(sampled), len(accepted), len(training.jobs))
    campaign.main()
    assert (len(sampled), len(accepted), len(training.jobs)) == counts


def test_resume_rejects_changed_config_and_mismatched_jobs(monkeypatch, tmp_path):
    campaign = load_campaign(monkeypatch)
    path = tmp_path / "results.jsonl"
    log = campaign_results.ResultLog(path, resume=False)
    log.append({"config": {"seed": 42}})
    client = Mock(spec=ReefClient)
    model = model_with_responses([])
    try:
        with pytest.raises(RuntimeError, match="configuration differs"):
            campaign.recover_campaign(log, client, model, {"seed": 43}, 1)
        log.append({"type": "batch_ready", "task_ids": ["a"], "instruction_id": "original"})
        log.append({"type": "training", "task_ids": ["a"], "instruction_id": "original"})
        monkeypatch.setattr(campaign, "training_jobs", lambda *args: [job_row("different")])
        with pytest.raises(RuntimeError, match="history disagree"):
            campaign.recover_campaign(log, client, model, {"seed": 42}, 1)
        monkeypatch.setattr(campaign, "training_jobs", lambda *args: [job_row("original"), job_row("extra")])
        with pytest.raises(RuntimeError, match="does not match this campaign"):
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
def test_manual_batch_trains_every_declared_task_including_a_window_tail(monkeypatch, count):
    campaign = load_campaign(monkeypatch)
    pipeline = manual_sao_pipeline("retail")
    task_ids = [f"task-{index}" for index in range(count)]
    for index, task_id in enumerate(task_ids):
        model = model_with_responses([])
        model.scenario, model.receipts = "retail", [f"inference-{index}"]
        reef_report.report_episode(model, "retail", task_id, float(index % 2), "finish")
        payload = model.reef.report.call_args.args[1]
        pipeline.push(conversation(model.receipts[0], 1, scenario="retail"))
        pipeline.push(
            AgentRecord.create(
                scenario="retail",
                request_type=RequestType.REPORT,
                agent_record_id=payload["agent_record_id"],
                payload=payload,
            )
        )
    pipeline.store_samples()
    sample_ids = tuple(stored.sample_id for stored in pipeline.store.page())
    assert len(sample_ids) == count
    # The driver declares the whole batch: without an instruction nothing trains, with one the whole buffer does.
    pipeline.insert_into_buffer(sample_ids)
    assert not pipeline.ready(1)
    instruction_id = campaign.training_instruction_id("retail", task_ids)
    pipeline.queue_instructions([instruction_request(campaign, instruction_id, 1)])
    data = pipeline.build_job_data(1)
    assert data.data_id == f"retail:instruction:{instruction_id}"
    assert [trajectory_reward(item) for item in data.samples] == [float(index % 2) for index in range(count)]
    assert pipeline.acknowledge(data.data_id) >= {instruction_id}
    assert pipeline.buffer() == ()
    # The first job's warmup repeats the exact rows, one optimizer step of the whole batch per pass.
    rows = tuple(
        policy_trajectory(f"inference-{index}", (5, 1, 2), (1, 1), (-0.1, -0.2), float(index % 2), action_mask=(1, 1))
        for index in range(count)
    )
    for warmup_steps in (0, 1, 3):
        parameters = campaign.training_instruction("i", "train", warmup_steps + 1, "artifact-0")["parameters"]
        method, scheduling = job_training({**parameters, "method": TrainingMethod("sao").to_dict()})
        prepared = prepare_slime_job(JobData("retail:instruction:i", rows), method, {}, scheduling, step_size=64)
        assert prepared.payload["external_step_sizes"] == [count] * (warmup_steps + 1)
        assert len(prepared.payload["samples"]) == count * (warmup_steps + 1)
        assert "advantages" not in prepared.payload
