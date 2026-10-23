"""Evaluation and batching safeguards for the OPD math campaign."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from recipes.opd.examples.math.prepare import OPENTHOUGHTS3_DUPLICATE_IDS, select_prompts
from recipes.opd.examples.math.run import Campaign, boxed_answer

EXAMPLE_CONFIG = Path(__file__).resolve().parents[3] / "recipes" / "opd" / "examples" / "math" / "serve.yaml"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"The answer is \boxed{007}.", 7),
        (r"\boxed{12} is intermediate; finally \boxed{42}.", 42),
        (r"\boxed{\text{42}}", 42),
        (r"\boxed{42}, revised to \boxed{x}.", None),
        (r"\boxed{1234}", None),
        ("My intermediate calculation is 42.", None),
    ],
)
def test_only_final_boxed_integer_is_scored(response, expected) -> None:
    assert boxed_answer(response) == expected


def test_driver_rejects_batch_mismatch_before_creating_output(tmp_path) -> None:
    config = tmp_path / "serve.yaml"
    config.write_text("training:\n  config:\n    global_batch_size: 32\n")
    output = tmp_path / "results"
    args = Namespace(config=config, steps=1, prompts_per_step=512, samples_per_prompt=4, output=output)
    with pytest.raises(ValueError, match="must equal the global batch size"):
        Campaign(args)
    assert not output.exists()


def test_evaluation_never_submits_training_reports(tmp_path) -> None:
    class Client:
        def __init__(self):
            self.calls = []

        def post(self, path, scenario, payload):
            self.calls.append((path, scenario, payload))
            return {
                "choices": [{"message": {"content": r"\boxed{42}"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 8},
            }, {"x-reef-agent-record-id": "receipt"}

        def get(self, path):
            assert path.endswith("/records/receipt")
            return {"agent_record_id": "receipt", "artifact_ref": {"release_id": "sft-baseline"}}

        def report(self, *args, **kwargs):
            raise AssertionError("Held-out evaluation must not train")

    from recipes.opd.examples.math.run import Question

    campaign = object.__new__(Campaign)
    campaign.output = tmp_path
    campaign.client = Client()
    campaign.args = Namespace(scenario="opd", model="qwen35", seed=0, eval_repeats=2, concurrency=1, eval_tokens=64000)
    campaign.evaluate([Question("q0", "A problem", "42")], 0)
    records = [json.loads(row) for row in (tmp_path / "eval-0000.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {row["seed"] for row in records} == {0, 1}
    assert all(row["evaluation"] and row["correct"] for row in records)
    assert {row["release_id"] for row in records} == {"sft-baseline"}
    assert [call[2]["max_tokens"] for call in campaign.client.calls] == [64000, 64000]


class CampaignService:
    """A small service double with report deduplication and observable commits."""

    def __init__(self, failure=None):
        self.failure = failure
        self.initialized = False
        self.records = {}
        self.reports = {}
        self.pending = []
        self.commits = []
        self.inference_calls = 0

    def fail_once(self, point):
        if self.failure == point:
            self.failure = None
            raise ConnectionError(f"Injected disconnect at {point}")

    def post(self, path, scenario, payload):
        self.initialized = True
        evaluation = payload["max_tokens"] == 4
        if evaluation and len(self.commits) == 1:
            self.fail_once("eval-1")
        self.inference_calls += 1
        receipt = f"receipt-{self.inference_calls}"
        self.records[receipt] = {"agent_record_id": receipt, "artifact_ref": {"release_id": f"r{len(self.commits)}"}}
        return {
            "choices": [{"message": {"content": r"\boxed{42}"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }, {"x-reef-agent-record-id": receipt}

    def get(self, path):
        from urllib.parse import parse_qs, urlsplit

        from reef_client import ReefClientError

        if path.endswith("/releases"):
            if not self.initialized:
                raise ReefClientError(404, "No scenario")
            rows = [
                {
                    "release_id": f"r{i}",
                    "operation": "training" if i else "creation",
                    "current": i == len(self.commits),
                }
                for i in range(len(self.commits) + 1)
            ]
            return {"releases": list(reversed(rows))}
        if "/records/" in path:
            return self.records[path.rsplit("/", 1)[-1]]
        if "/commits?" in path:
            self.fail_once("verify-1")
            after = int(parse_qs(urlsplit(path).query)["after_step"][0])
            return {"commits": self.commits[after : after + 1]}
        assert path == "/reef/status"
        return {}

    def report(self, scenario, payload, *, references):
        identifier = payload["agent_record_id"]
        saved = {**payload, "references": references}
        if identifier in self.reports:
            assert self.reports[identifier] == saved
        else:
            self.reports[identifier] = saved
            self.pending.extend([identifier, *references])
            if len(self.pending) == 4:
                step = len(self.commits) + 1
                self.commits.append(
                    {
                        "step": step,
                        "operation": "training",
                        "pending": False,
                        "artifact_ref": {"release_id": f"r{step}"},
                        "consumed_ids": self.pending,
                    }
                )
                self.pending = []
        # The service accepted the report even when the HTTP response is lost.
        self.fail_once(f"report-{len(self.reports)}")
        return {}


@pytest.fixture
def campaign_args(tmp_path):
    config = tmp_path / "serve.yaml"
    config.write_text(
        "training:\n  config:\n    global_batch_size: 2\ninference:\n  options:\n    context-length: 16\n"
    )
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("\n".join(json.dumps({"id": f"t{i}", "prompt": "Training problem"}) for i in range(2)))
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text(json.dumps({"id": "e0", "prompt": "Held-out problem", "answer": "42"}) + "\n")
    return Namespace(
        config=config,
        url="http://test",
        scenario="test",
        model="qwen35",
        output=tmp_path / "results",
        train_data=train_data,
        eval_data=eval_data,
        steps=2,
        prompts_per_step=1,
        samples_per_prompt=2,
        train_tokens=3,
        eval_tokens=4,
        eval_repeats=2,
        eval_every=1,
        seed=0,
        concurrency=1,
        timeout=1,
        resume=False,
        baseline_eval=None,
    )


def test_the_shipped_stack_takes_run_sh_batches_of_64_prompts_by_four_samples(campaign_args) -> None:
    # run.sh drives 64 prompts by four samples per update; the global batch size must hold all of them.
    campaign_args.config = EXAMPLE_CONFIG
    campaign_args.prompts_per_step, campaign_args.samples_per_prompt = 64, 4
    Campaign(campaign_args)
    assert (campaign_args.output / "config.json").is_file()


@pytest.mark.parametrize("failure", ["report-1", "report-2", "verify-1", "eval-1"])
def test_campaign_resumes_without_retraining_or_resampling_finished_batches(monkeypatch, campaign_args, failure):
    from recipes.opd.examples.math import run

    service = CampaignService(failure)
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    with pytest.raises(ConnectionError, match="Injected disconnect"):
        Campaign(campaign_args).run()
    campaign_args.resume = True
    Campaign(campaign_args).run()
    assert len(service.commits) == 2
    assert len(service.reports) == 4
    assert service.inference_calls == 10
    metrics = [json.loads(line) for line in (campaign_args.output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [0, 1, 2]
    assert [row["release_id"] for row in metrics] == ["r0", "r1", "r2"]
    assert all(row["accuracy"] == 1 for row in metrics)
    snapshots = {path: path.read_bytes() for path in campaign_args.output.glob("releases-*.json")}
    # Repeating a completed resume is read-only with respect to the service.
    Campaign(campaign_args).run()
    assert service.inference_calls == 10
    assert len(service.commits) == 2
    assert len((campaign_args.output / "metrics.jsonl").read_text().splitlines()) == 3
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())


def test_resume_rejects_modified_data_before_any_service_call(monkeypatch, campaign_args):
    from recipes.opd.examples.math import run

    service = CampaignService()
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    Campaign(campaign_args)
    campaign_args.resume = True
    campaign_args.eval_data.write_text('{"id":"different","prompt":"Changed","answer":"42"}\n')
    with pytest.raises(ValueError, match="same protocol"):
        Campaign(campaign_args)
    assert service.inference_calls == 0


def test_partial_collection_recovers_only_missing_samples_and_torn_last_line(monkeypatch, campaign_args):
    from recipes.opd.examples.math import run

    service = CampaignService()
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    campaign = Campaign(campaign_args)
    questions = run.load_questions(campaign_args.eval_data)
    first = campaign.sample((questions[0], 0, True))
    path = campaign.output / "eval-0000.jsonl"
    path.write_bytes((json.dumps(first) + '\n{"question_id":').encode())
    campaign.evaluate(questions, 0, release_id="r0")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0] == first
    assert service.inference_calls == 2


def test_resume_rejects_incomplete_historical_evaluation(monkeypatch, campaign_args):
    from recipes.opd.examples.math import run

    service = CampaignService("report-2")
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    with pytest.raises(ConnectionError):
        Campaign(campaign_args).run()
    path = campaign_args.output / "eval-0000.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    campaign_args.resume = True
    with pytest.raises(RuntimeError, match="historical release"):
        Campaign(campaign_args).run()
    assert len(service.commits) == 1
    assert service.inference_calls == 4


def test_resume_rejects_unrelated_committed_reports(monkeypatch, campaign_args):
    from recipes.opd.examples.math import run

    service = CampaignService("report-2")
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    with pytest.raises(ConnectionError):
        Campaign(campaign_args).run()
    service.commits[0]["consumed_ids"][-1] = "unrelated-receipt"
    campaign_args.resume = True
    with pytest.raises(RuntimeError, match="exact batch"):
        Campaign(campaign_args).run()
    assert len(service.commits) == 1


def test_campaign_prevents_two_drivers_for_one_output(monkeypatch, campaign_args):
    import fcntl

    from recipes.opd.examples.math import run

    service = CampaignService()
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    campaign = Campaign(campaign_args)
    with (campaign.output / ".driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another driver"):
            campaign.run()
    assert service.inference_calls == 0


def test_prompt_pool_is_seeded_and_disjoint_from_evaluation_and_openthoughts3():
    duplicate = sorted(OPENTHOUGHTS3_DUPLICATE_IDS, key=int)[0]
    prompts = [{"id": str(index), "prompt": f"Question {index}?"} for index in range(40)]
    prompts += [
        {"id": duplicate, "prompt": "Also in OpenThoughts3"},
        {"id": "aime", "prompt": "Find the AIME answer."},
        {"id": "repeat", "prompt": "question 3?"},
    ]
    selected = select_prompts(prompts, ["find the AIME answer"], 30)
    ids = [row["id"] for row in selected]
    assert len(ids) == 30 and len(set(ids)) == 30
    assert not {duplicate, "aime", "repeat"} & set(ids)
    assert ids == [row["id"] for row in select_prompts(prompts, ["find the AIME answer"], 30)]
    assert ids != [str(index) for index in range(30)]
    with pytest.raises(ValueError, match="disjoint prompts"):
        select_prompts(prompts, [], 42)


def test_baseline_eval_reuses_an_earlier_evaluation_and_opens_the_scenario_with_the_first_batch(
    monkeypatch, campaign_args, tmp_path
):
    from recipes.opd.examples.math import run

    service = CampaignService()
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: service)
    earlier = tmp_path / "earlier-eval-0000.jsonl"
    rows = [
        {
            "question_id": "e0",
            "seed": seed,
            "receipt": f"old-receipt-{seed}",
            "release_id": "old-r0",
            "correct": seed == 0,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 5},
            # A real response can hold a Unicode line separator; only newlines end a record.
            "response": "step one\u2028step two \\boxed{42}",
        }
        for seed in range(2)
    ]
    earlier.write_text("".join(json.dumps(row) + "\n" for row in rows))
    campaign_args.baseline_eval = earlier
    campaign_args.steps = 1
    Campaign(campaign_args).run()
    # Two training samples and the step-1 evaluation; nothing for step 0.
    assert service.inference_calls == 4
    assert len(service.commits) == 1
    metrics = [json.loads(line) for line in (campaign_args.output / "metrics.jsonl").read_text().splitlines()]
    assert [(row["step"], row["release_id"], row["accuracy"]) for row in metrics] == [
        (0, "old-r0", 0.5),
        (1, "r1", 1.0),
    ]
    assert metrics[0]["elapsed_scope"] == "reused" and metrics[0]["reused_from"] == str(earlier)
    assert (campaign_args.output / "eval-0000.jsonl").read_text() == earlier.read_text()
    # Repeating the finished campaign is read-only with respect to the service.
    campaign_args.resume = True
    Campaign(campaign_args).run()
    assert service.inference_calls == 4
    # A baseline for other seeds is refused before any request reaches a fresh scenario.
    fresh = CampaignService()
    monkeypatch.setattr(run, "ReefClient", lambda *args, **kwargs: fresh)
    earlier.write_text(json.dumps({**rows[0], "seed": 7}) + "\n" + json.dumps(rows[1]) + "\n")
    campaign_args.resume = False
    campaign_args.output = tmp_path / "results-2"
    with pytest.raises(ValueError, match="questions and seeds"):
        Campaign(campaign_args).run()
    assert fresh.inference_calls == 0
