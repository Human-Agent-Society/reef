"""Formal Reefine checks decide publication independently of proposal notes."""

import io
import json
import time
from dataclasses import replace

import pytest
from rich.console import Console

from reef.core.evaluation import EvaluationResult
from reef.core.training_request import TrainingRequest
from reef.harness.client.check_display import CheckDisplay
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.evaluation import evaluation_factory
from reef.train.evaluation.reefine import (
    CheckResult,
    EvaluationInstructions,
    ReefineEvaluation,
    ReefineEvaluationSettings,
    validate_review,
)
from reef.train.reefine.backend import ReefineBackend, ReefineCandidate
from reef.train.reefine.context import EvaluationContext, ServedComposition
from reef.train.reefine.proposer import request_with_feedback


class ReviewModel(ModelBinding):
    responses: list[dict[str, object]] = []
    messages_seen: list[list[dict[str, object]]] = []

    def chat(self, messages, **params):
        self.messages_seen.append(messages)
        return json.dumps(self.responses.pop(0))


class EvaluationBackend(ReefineBackend):
    def __init__(self, model, results):
        self.models = ModelBindings(model)
        self.results = results
        self.progress = []

    @property
    def evaluation_models(self):
        return self.models

    @property
    def episode_descriptor(self):
        from reef.harness.adapters import get_adapter

        return get_adapter("pi")

    def show_evaluation_checks(self, checks):
        self.progress.append(checks)

    def score_health(self, task, result):
        from reef.train.evaluation.reefine_health import evaluate

        return evaluate(task, result)


class ScriptedEvaluation(ReefineEvaluation):
    def episode(self, candidate, side, prompt, files, initialize_git, record_dir, deadline):
        return self.backend.results.pop(0)


def episode(text="reef-ok", exit_code=0):
    return EpisodeResult(
        exit_code, "", "", ({"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}},), ()
    )


def candidate(tmp_path):
    request = TrainingRequest("print the command before its output", "session", "base", "request-id")
    return ReefineCandidate(
        candidate_id="candidate",
        candidate_files={"AGENTS.md": "new"},
        requires=({"name": "curl", "kind": "binary"},),
        current_files={"AGENTS.md": "old"},
        candidate_entries=(),
        current_entries=(),
        mutations=(),
        evaluation_tasks=("[health] echo reef-ok",),
        record_dir=tmp_path,
        context=EvaluationContext(request, ServedComposition("served-release", ())),
    )


def evaluation(tmp_path, request_pass=True, current_pass=True, candidate_pass=True, review_pass=True):
    model = ReviewModel("http://unused", "demo-model")
    model.messages_seen = []
    model.responses = [
        {"prompt": "run a command", "checks": ["command before output"]},
        {"passed": request_pass, "reason": "observed request"},
        {"passed": current_pass, "reason": "current result"},
        {"passed": candidate_pass, "reason": "candidate result"},
        {"passed": review_pass, "reason": "review result"},
    ]
    backend = EvaluationBackend(model, [episode(), episode("request"), episode("protected"), episode("protected")])
    plugin = ScriptedEvaluation(
        backend,
        ReefineEvaluationSettings(protected_tasks=("protected task",)),
        EvaluationInstructions("plan", "behavior", "review"),
    )
    return plugin, model, backend


@pytest.mark.parametrize("field", ["request_pass", "candidate_pass", "review_pass"])
def test_failed_required_check_rejects(tmp_path, field):
    plugin = evaluation(tmp_path, **{field: False})[0]
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    assert not decision.selected
    assert any(check["status"] == "fail" for check in measured.metrics["reefine_evaluation"]["checks"])
    assert measured.metrics["reefine_evaluation"]["current_release_id"] == "served-release"


def test_success_uses_fresh_review_context_and_writes_progress_and_record(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    assert plugin.decide(item, measured).selected
    assert all(len(messages) == 1 and messages[0]["role"] == "user" for messages in model.messages_seen)
    assert all("proposal_notes" not in messages[0]["content"] for messages in model.messages_seen)
    assert '"harness": {"AGENTS.md": "new"}' in model.messages_seen[1][0]["content"]
    assert '"harness": {"AGENTS.md": "old"}' in model.messages_seen[2][0]["content"]
    assert '"harness": {"AGENTS.md": "new"}' in model.messages_seen[3][0]["content"]
    assert '"harness"' not in model.messages_seen[0][0]["content"]
    assert '"requires": [{"name": "curl", "kind": "binary"}]' in model.messages_seen[1][0]["content"]
    assert '"requires": [{"name": "curl", "kind": "binary"}]' in model.messages_seen[-1][0]["content"]
    assert '"requires"' not in model.messages_seen[0][0]["content"]
    assert any(check["status"] == "running" for checks in backend.progress for check in checks)
    assert (tmp_path / "reefine-evaluation" / "result.json").is_file()
    assert measured.metrics["reefine_evaluation"]["same_model"] is True
    assert measured.metrics["reefine_evaluation"]["requires"] == [{"name": "curl", "kind": "binary"}]
    behavior = next(
        check for check in measured.metrics["reefine_evaluation"]["checks"] if check["id"] == "request-behavior"
    )
    assert behavior["expected"] == "command before output"


def test_invalid_plan_and_missing_request_never_approve(tmp_path):
    for missing in (False, True):
        plugin, model, backend = evaluation(tmp_path)
        item = candidate(tmp_path / str(missing))
        if missing:
            item = replace(item, context=replace(item.context, request=None))
            model.responses.pop(0)
        else:
            model.responses[0] = {"prompt": "", "checks": []}
        # Without a request episode there is no request judgment call.
        model.responses.pop(0 if missing else 1)
        backend.results.pop(1)
        measured = plugin.evaluate(item)
        assert not plugin.decide(item, measured).selected
        statuses = {check["id"]: check["status"] for check in measured.metrics["reefine_evaluation"]["checks"]}
        assert statuses["request-plan"] == "invalid" and statuses["request-behavior"] == "not_run"


def test_valid_baseline_failure_can_improve(tmp_path):
    plugin = evaluation(tmp_path, current_pass=False)[0]
    item = candidate(tmp_path)
    assert plugin.decide(item, plugin.evaluate(item)).selected


def test_empty_or_partial_evaluation_never_approves(tmp_path):
    plugin = evaluation(tmp_path)[0]
    for checks in ([], [CheckResult("only-health", "health", "candidate", "pass").to_dict()]):
        measured = EvaluationResult("reefine", "1", {"reefine_evaluation": {"checks": checks}})
        assert not plugin.decide(candidate(tmp_path), measured).selected


@pytest.mark.parametrize("payload", [{"passed": "true", "reason": "x"}, {"passed": True}, [], None])
def test_malformed_review_is_invalid(payload):
    with pytest.raises((ValueError, AttributeError)):
        validate_review(payload)


def test_configured_auxiliary_model_is_kept():
    from reef.inference.http import InferenceProxyRuntime
    from reef.inference.model_config import ModelConfig
    from reef.recipe.cordis import CordisRecipe, _ScenarioModels

    recipe = CordisRecipe.from_environment(
        {},
        config={
            "evolution": {
                "propose": "reef.recipe.reefine.evolution:propose",
                "evaluate": "reef.recipe.reefine.evolution:evaluate",
                "tasks": ["[health] test"],
                "models": {"reviewer": {"url": "http://reviewer", "model": "independent"}},
            }
        },
    )
    teacher = recipe.models["reviewer"]
    config = ModelConfig(runtime=InferenceProxyRuntime(base_url="http://served", model_path="served"))
    assert _ScenarioModels(config, recipe, auxiliary_models=recipe.models).resolve().named["reviewer"] is teacher


def test_named_reviewer_receives_all_fresh_checks_instead_of_served_model(tmp_path):
    plugin, served, backend = evaluation(tmp_path)
    reviewer = ReviewModel("http://reviewer", "independent-model")
    reviewer.messages_seen = []
    reviewer.responses = served.responses
    served.responses = []
    backend.models = ModelBindings(served, named={"reviewer": reviewer})
    plugin.settings = replace(plugin.settings, reviewer_model="reviewer")
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    assert plugin.decide(item, measured).selected
    assert len(reviewer.messages_seen) == 5 and not served.messages_seen
    assert all(len(messages) == 1 for messages in reviewer.messages_seen)
    report = measured.metrics["reefine_evaluation"]
    assert report["reviewer_model"] == "independent-model"
    assert report["reviewer_binding"] == "reviewer" and report["same_model"] is False


def test_feedback_passes_public_request_findings_only():
    request = {"text": "original"}
    rejected = [{"reason": "protected held-out details", "feedback": [{"reason": "missing request action"}]}]
    handed = request_with_feedback(request, rejected)
    assert handed["text"] == "original"
    assert "held-out" not in json.dumps(handed)


def test_display_aligns_unicode_and_pipe_has_no_ansi():
    stream = io.StringIO()
    console = Console(file=stream, width=100, force_terminal=False)
    with CheckDisplay(console) as display:
        display.update(
            "evaluating",
            [CheckResult("request-中", "request", "candidate", "running", expected="two actions").to_dict()],
        )
        display.update(
            "selected",
            [CheckResult("request-中", "request", "candidate", "pass", score=1, observed="First\nSecond").to_dict()],
        )
    output = stream.getvalue()
    assert "Current" in output and "Candidate" in output and "pass" in output
    assert "First Second" in output
    assert "\x1b[" not in output


def test_recipe_config_rejects_unsupported_keys():
    with pytest.raises(RecipeConfigError):
        evaluation_factory({"unknown": 1}, ("health",))


def test_task_environment_is_explicit_and_kept_out_of_settings_display(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy")
    assert evaluation_factory({}, ("health",)).settings.environment == {}
    factory = evaluation_factory({"env_from": ["HTTPS_PROXY"]}, ("health",))
    assert factory.settings.environment == {"HTTPS_PROXY": "http://operator-proxy"}
    assert "operator-proxy" not in repr(factory.settings)


@pytest.mark.parametrize("status, expected_calls", [(503, 2), (401, 1)])
def test_reviewer_retries_transient_failure_once_without_reusing_history(
    tmp_path, monkeypatch, status, expected_calls
):
    plugin, model = evaluation(tmp_path)[:2]
    calls = []

    def chat(self, messages, **params):
        calls.append((messages, params))
        if len(calls) == 1:
            raise ModelBindingError("endpoint failure", status=status)
        return '{"passed":true,"reason":"completed"}'

    monkeypatch.setattr(ReviewModel, "chat", chat)
    if status == 401:
        with pytest.raises(ModelBindingError):
            plugin.ask_json(model, "review", {}, time.monotonic() + 10)
    else:
        assert plugin.ask_json(model, "review", {}, time.monotonic() + 10)["passed"]
    assert len(calls) == expected_calls and all(len(messages) == 1 for messages, params in calls)
    if len(calls) == 2:
        assert calls[1][1]["timeout_s"] <= calls[0][1]["timeout_s"]


def test_unavailable_episode_cannot_approve(tmp_path, monkeypatch):
    plugin = evaluation(tmp_path)[0]

    def unavailable(*args, **kwargs):
        raise EpisodeError("episode timed out")

    monkeypatch.setattr(plugin, "episode", unavailable)
    item = candidate(tmp_path)
    result = plugin.evaluate(item)
    assert not plugin.decide(item, result).selected
    assert any(check["status"] == "invalid" for check in result.metrics["reefine_evaluation"]["checks"])


def test_checks_from_another_release_or_duplicate_checks_cannot_approve(tmp_path):
    plugin = evaluation(tmp_path)[0]
    item = candidate(tmp_path)
    result = plugin.evaluate(item)
    report = result.metrics["reefine_evaluation"]
    report["current_release_id"] = "other-release"
    assert not plugin.decide(item, result).selected
    report["current_release_id"] = "served-release"
    report["checks"].append(dict(report["checks"][0]))
    assert not plugin.decide(item, result).selected
