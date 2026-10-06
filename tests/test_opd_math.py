"""Evaluation and batching safeguards for the OPD math campaign."""

import json
from argparse import Namespace

import pytest

from recipes.opd.examples.math.run import Campaign, boxed_answer


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
    config.write_text("recipe:\n  config:\n    batch-size: 32\ntraining:\n  config:\n    global_batch_size: 32\n")
    output = tmp_path / "results"
    args = Namespace(config=config, steps=1, prompts_per_step=512, samples_per_prompt=4, output=output)
    with pytest.raises(ValueError, match="recipe's batch-size"):
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


def test_acceptance_uses_final_checkpoint_and_paired_questions(tmp_path) -> None:
    from recipes.opd.examples.math.analyze import analyze

    for step in range(0, 201, 20):
        values = [False, False] if step == 0 else ([True, False] if step == 200 else [True, True])
        rows = [
            {"question_id": str(i), "seed": 0, "evaluation": True, "correct": value, "release_id": str(step)}
            for i, value in enumerate(values)
        ]
        (tmp_path / f"eval-{step:04d}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path, questions=2, repeats=1)
    assert result["absolute_improvement"] == 0.5
    assert result["target_met"] is True
    assert result["paired_question_bootstrap_95_interval"] == [0.0, 1.0]
    assert [point["step"] for point in result["curve"]] == list(range(0, 201, 20))
    with pytest.raises(ValueError, match="acceptance is pending"):
        analyze(tmp_path, final_step=220, questions=2, repeats=1)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([0, 200], "missing="),
        ([*range(0, 201, 20), 10], "unexpected="),
    ],
)
def test_acceptance_requires_the_complete_declared_schedule(tmp_path, steps, message) -> None:
    from recipes.opd.examples.math.analyze import analyze

    for step in steps:
        row = {"question_id": "q", "seed": 0, "evaluation": True, "correct": step > 0, "release_id": str(step)}
        (tmp_path / f"eval-{step:04d}.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        analyze(tmp_path, questions=1, repeats=1)


@pytest.mark.parametrize(("eval_every", "steps"), [(20, [0, 20, 25]), (10, [0, 10, 20, 25])])
def test_acceptance_includes_final_step_outside_regular_interval(tmp_path, eval_every, steps) -> None:
    from recipes.opd.examples.math.analyze import analyze

    for step in steps:
        row = {"question_id": "q", "seed": 0, "evaluation": True, "correct": step > 0, "release_id": str(step)}
        (tmp_path / f"eval-{step:04d}.jsonl").write_text(json.dumps(row) + "\n")
    result = analyze(tmp_path, final_step=25, eval_every=eval_every, questions=1, repeats=1)
    assert [point["step"] for point in result["curve"]] == steps
    assert result["target_met"] is True


def test_acceptance_rejects_duplicate_step_files(tmp_path) -> None:
    from recipes.opd.examples.math.analyze import analyze

    for filename, step in [("eval-0000.jsonl", 0), ("eval-0200.jsonl", 200), ("eval-200.jsonl", 200)]:
        row = {"question_id": "q", "seed": 0, "evaluation": True, "correct": step > 0, "release_id": str(step)}
        (tmp_path / filename).write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate evaluation steps"):
        analyze(tmp_path, questions=1, repeats=1)


@pytest.mark.parametrize(("final_step", "eval_every"), [(-1, 20), (200, 0)])
def test_acceptance_rejects_invalid_schedule(tmp_path, final_step, eval_every) -> None:
    from recipes.opd.examples.math.analyze import analyze

    with pytest.raises(ValueError, match=r"nonnegative.*positive"):
        analyze(tmp_path, final_step=final_step, eval_every=eval_every)


def test_paired_interval_rejects_different_eval_questions() -> None:
    from recipes.opd.examples.math.analyze import paired_interval

    with pytest.raises(ValueError, match="question/seed sets differ"):
        paired_interval({("q0", 0): False}, {("q1", 0): True})


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
        "recipe:\n  config:\n    batch-size: 2\ntraining:\n  config:\n    global_batch_size: 2\n"
        "inference:\n  options:\n    context-length: 16\n"
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
    )


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


@pytest.mark.parametrize("target", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_acceptance_rejects_invalid_improvement_target(tmp_path, target):
    from recipes.opd.examples.math.analyze import analyze

    with pytest.raises(ValueError, match="Target improvement"):
        analyze(tmp_path, target_improvement=target)


def test_small_experiment_reports_positive_gain_separately_from_target(tmp_path):
    from recipes.opd.examples.math.analyze import analyze

    for step in [0, 30]:
        rows = [
            {
                "question_id": str(i),
                "seed": 0,
                "evaluation": True,
                "correct": step > 0 and i == 0,
                "release_id": str(step),
            }
            for i in range(30)
        ]
        (tmp_path / f"eval-{step:04d}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path, final_step=30, eval_every=30, questions=30, repeats=1, target_improvement=0.05)
    assert result["observed_positive_improvement"] is True
    assert result["target_met"] is False
    assert result["target_absolute_improvement"] == 0.05
    assert result["absolute_improvement"] == pytest.approx(1 / 30)
