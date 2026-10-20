"""Formal Reefine checks decide publication independently of proposal notes."""

import io
import json
import time
from dataclasses import replace

import pytest
from rich.console import Console

from reef.core.evaluation import EvaluationResult, ServedComposition
from reef.core.training_request import TrainingRequest
from reef.evaluation.reefine import (
    CheckResult,
    EvaluationContext,
    EvaluationInstructions,
    ReefineEvaluation,
    ReefineEvaluationSettings,
    shortened_trajectory,
    validate_plan,
    validate_review,
)
from reef.harness.client.check_display import CheckDisplay
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.evaluation import evaluation_settings
from reef.sandbox.process import EpisodeTimeout
from reef.train.reefine.backend import ReefineBackend, ReefineCandidate
from reef.train.reefine.proposer import request_with_feedback


class ReviewModel(ModelBinding):
    responses: list[dict[str, object]] = []
    messages_seen: list[list[dict[str, object]]] = []

    def chat(self, messages, **params):
        self.messages_seen.append(messages)
        return json.dumps(self.responses.pop(0))


class EvaluationBackend(ReefineBackend):
    def __init__(self, model, results, adapter="pi"):
        self.models = ModelBindings(model)
        self.results = results
        self.progress = []
        self.adapter = adapter

    @property
    def evaluation_models(self):
        return self.models

    @property
    def episode_descriptor(self):
        from reef.harness.adapters import get_adapter

        return get_adapter(self.adapter)

    @property
    def harness_description(self):
        return {"adapter": self.adapter, "entry_kinds": ["skill", "rules"], "facts": None}

    def show_evaluation_checks(self, checks):
        self.progress.append(checks)


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
        {"tasks": [{"messages": ["run a command"], "checks": ["command before output"]}]},
        {"passed": request_pass, "reason": "observed request"},
        {"passed": current_pass, "reason": "current result"},
        {"passed": candidate_pass, "reason": "candidate result"},
        {"passed": review_pass, "unsafe": True, "reason": "review result"},
    ]
    backend = EvaluationBackend(model, [episode("request"), episode("protected"), episode("protected")])
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
    assert any(check["status"] == "fail" for check in measured.metrics["evaluation_report"]["checks"])
    assert measured.metrics["evaluation_report"]["current_release_id"] == "served-release"


def test_the_behavior_review_is_told_whether_the_workspace_is_a_git_repository(tmp_path):
    plugin, model = evaluation(tmp_path)[:2]
    plugin.evaluate(candidate(tmp_path))
    assert '"workspace_is_git_repository": false' in model.messages_seen[1][0]["content"]


def test_a_change_review_problem_that_is_not_unsafe_holds_a_passing_release_for_review(tmp_path):
    plugin, model = evaluation(tmp_path)[:2]
    model.responses[4] = {"passed": False, "unsafe": False, "reason": "The note adds an exception the request lacks."}
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    review = next(c for c in measured.metrics["evaluation_report"]["checks"] if c["id"] == "change-review")
    assert review["status"] == "concern"
    assert decision.selected and decision.requires_review
    assert decision.reason == (
        "All checks that ran passed. The change review asks for your judgment: The note adds an exception the "
        "request lacks. Review the release before you install it."
    )
    # An unsafe finding still rejects, and a failed review must say whether it is unsafe.
    plugin, model = evaluation(tmp_path, review_pass=False)[:2]
    assert not plugin.decide(item, plugin.evaluate(item)).selected
    plugin, model = evaluation(tmp_path)[:2]
    model.responses[4] = {"passed": False, "reason": "no unsafe field"}
    measured = plugin.evaluate(item)
    review = next(c for c in measured.metrics["evaluation_report"]["checks"] if c["id"] == "change-review")
    assert review["status"] == "invalid" and not plugin.decide(item, measured).selected


def test_a_request_task_that_does_not_finish_fails_in_plain_words(tmp_path, monkeypatch):
    plugin = evaluation(tmp_path)[0]

    def unfinished(*args, **kwargs):
        raise EpisodeError("episode timed out after 600s: ['pi', '--mode', 'json']") from EpisodeTimeout("600s")

    monkeypatch.setattr(plugin, "episode", unfinished)
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    behavior = next(c for c in measured.metrics["evaluation_report"]["checks"] if c["id"] == "request-behavior-0")
    assert behavior["status"] == "fail"
    assert behavior["reason"].startswith("The task did not finish: the session was still running after")
    assert "--mode" not in behavior["reason"] and not plugin.decide(item, measured).selected
    # With a record kept before the stop, the review reads it and says why the session did not finish.
    plugin, model = evaluation(tmp_path)[:2]
    model.responses[1] = {"passed": False, "reason": "The reviewer agent kept starting reviewer agents."}

    def unfinished_with_record(candidate, side, messages, files, initialize_git, record_dir, deadline):
        record_dir.mkdir(parents=True, exist_ok=True)
        event = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]}}
        (record_dir / "session.jsonl").write_text(json.dumps(event) + "\n")
        raise EpisodeError("episode timed out after 600s") from EpisodeTimeout("600s")

    monkeypatch.setattr(plugin, "episode", unfinished_with_record)
    measured = plugin.evaluate(item)
    behavior = next(c for c in measured.metrics["evaluation_report"]["checks"] if c["id"] == "request-behavior-0")
    assert behavior["status"] == "fail" and behavior["reason"] == "The reviewer agent kept starting reviewer agents."
    assert '"unfinished_after_seconds"' in model.messages_seen[1][0]["content"]


def test_the_plan_reads_the_changes_how_to_use_note(tmp_path):
    plugin, model = evaluation(tmp_path)[:2]
    item = replace(candidate(tmp_path), usage="Type /chat off to leave the mode.")
    plugin.evaluate(item)
    assert '"how_to_use": "Type /chat off to leave the mode."' in model.messages_seen[0][0]["content"]


def test_a_rejection_names_the_failed_checks_and_shortens_the_first_reason(tmp_path):
    plugin, model = evaluation(tmp_path, request_pass=False, review_pass=False)[:2]
    model.responses[1]["reason"] = "The test ran after the fix. " + "More detail follows here. " * 30
    item = candidate(tmp_path)
    decision = plugin.decide(item, plugin.evaluate(item))
    assert decision.reason.startswith(
        "Failed: Requested behavior 1, Change review. Requested behavior 1: The test ran after the fix."
    )
    assert len(decision.reason) < 400
    # A reason with no sentence end in reach is cut at a word and points to the page.
    plugin, model = evaluation(tmp_path, request_pass=False)[:2]
    model.responses[1]["reason"] = "The agent searched first and " + "then read several sections " * 30
    decision = plugin.decide(item, plugin.evaluate(item))
    assert decision.reason.endswith("sections... (the page has the full reason).")


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
    # The plan sees the harness description and never the candidate's files.
    assert '"harness": {"adapter": "pi"' in model.messages_seen[0][0]["content"]
    assert "AGENTS.md" not in model.messages_seen[0][0]["content"]
    assert '"requires": [{"name": "curl", "kind": "binary"}]' in model.messages_seen[1][0]["content"]
    assert '"requires": [{"name": "curl", "kind": "binary"}]' in model.messages_seen[-1][0]["content"]
    assert '"requires"' not in model.messages_seen[0][0]["content"]
    assert any(check["status"] == "running" for checks in backend.progress for check in checks)
    assert (tmp_path / "reefine-evaluation" / "result.json").is_file()
    assert measured.metrics["evaluation_report"]["same_model"] is True
    assert measured.metrics["evaluation_report"]["requires"] == [{"name": "curl", "kind": "binary"}]
    behavior = next(
        check for check in measured.metrics["evaluation_report"]["checks"] if check["id"] == "request-behavior-0"
    )
    assert behavior["expected"] == "command before output"


def test_invalid_plan_and_missing_request_never_approve(tmp_path):
    for missing in (False, True):
        plugin, model, backend = evaluation(tmp_path)
        item = candidate(tmp_path / str(missing))
        if missing:
            item = replace(item, context=replace(item.context, request=None))
            # No plan call and, without a request episode, no request judgment call.
            del model.responses[0:2]
        else:
            # The invalid plan is asked for once more and stays invalid; no request episode is judged.
            model.responses[0:2] = [{"prompt": "", "checks": []}, {"prompt": "", "checks": []}]
        backend.results.pop(0)
        measured = plugin.evaluate(item)
        assert not plugin.decide(item, measured).selected
        statuses = {check["id"]: check["status"] for check in measured.metrics["evaluation_report"]["checks"]}
        assert statuses["request-plan"] == "invalid" and statuses["request-behavior"] == "not_run"


def test_valid_baseline_failure_can_improve(tmp_path):
    plugin = evaluation(tmp_path, current_pass=False)[0]
    item = candidate(tmp_path)
    assert plugin.decide(item, plugin.evaluate(item)).selected


def test_empty_or_partial_evaluation_never_approves(tmp_path):
    plugin = evaluation(tmp_path)[0]
    for checks in ([], [CheckResult("only-review", "review", "candidate", "pass").to_dict()]):
        measured = EvaluationResult("reefine", "1", {"evaluation_report": {"checks": checks}})
        assert not plugin.decide(candidate(tmp_path), measured).selected


def test_a_harness_limit_is_reported_not_run_and_holds_a_passing_release_for_review(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    backend.adapter = "claude"
    model.responses[0] = {
        "tasks": [{"messages": ["run a command"], "checks": ["command before output"]}],
        "limits": [{"requirement": "refuse every other tool", "reason": "a command cannot take a tool away"}],
    }
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    assert decision.selected and decision.requires_review
    assert "not possible on this harness: refuse every other tool" in decision.reason
    checks = {check["id"]: check for check in measured.metrics["evaluation_report"]["checks"]}
    assert checks["limit-0"]["status"] == "unsupported" and checks["limit-0"]["group"] == "limit"
    assert checks["request-behavior-0"]["status"] == "pass"
    # Any harness is planned and run; the request judge reads the limits so it does not fail for them.
    assert '"adapter": "claude"' in model.messages_seen[0][0]["content"]
    assert "a command cannot take a tool away" in model.messages_seen[1][0]["content"]
    assert decision.metrics["request_feedback"][0]["id"] == "limit-0"


def test_a_request_made_only_of_limits_runs_no_episode_and_is_rejected(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    model.responses[0] = {
        "tasks": [],
        "limits": [{"requirement": "refuse every other tool", "reason": "a command cannot take a tool away"}],
    }
    model.responses.pop(1)
    backend.results.pop(0)
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    assert not decision.selected and not decision.requires_review
    statuses = {check["id"]: check["status"] for check in measured.metrics["evaluation_report"]["checks"]}
    assert statuses["request-behavior"] == "not_run" and statuses["limit-0"] == "unsupported"


def test_each_task_runs_in_its_own_session_and_every_task_must_pass(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    model.responses[0] = {
        "tasks": [
            {"messages": ["/mode on", "what is new"], "checks": ["the mode answers with a search"]},
            {"messages": ["create a.txt"], "checks": ["a.txt is created outside the mode"]},
        ]
    }
    model.responses.insert(2, {"passed": False, "reason": "a.txt was not created"})
    backend.results.insert(1, episode("second task"))
    seen = []
    original = plugin.episode

    def recorded(candidate, side, messages, *args):
        seen.append(messages)
        return original(candidate, side, messages, *args)

    plugin.episode = recorded
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    statuses = {check["id"]: check["status"] for check in measured.metrics["evaluation_report"]["checks"]}
    assert statuses["request-behavior-0"] == "pass" and statuses["request-behavior-1"] == "fail"
    assert seen[:2] == [["/mode on", "what is new"], ["create a.txt"]]
    assert '"checks": ["a.txt is created outside the mode"]' in model.messages_seen[2][0]["content"]
    assert not plugin.decide(item, measured).selected


def test_untested_behavior_is_shown_and_does_not_hold_a_passing_release(tmp_path):
    plugin, model = evaluation(tmp_path)[:2]
    model.responses[0]["untested"] = [{"requirement": "the mode is off after a restart", "reason": "no restart here"}]
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    assert decision.selected and not decision.requires_review
    assert "Not tested here: the mode is off after a restart" in decision.reason
    checks = {check["id"]: check for check in measured.metrics["evaluation_report"]["checks"]}
    assert checks["untested-0"]["status"] == "untested" and checks["untested-0"]["group"] == "untested"


def test_a_reply_that_is_not_json_is_asked_once_more_in_a_fresh_conversation(tmp_path, monkeypatch):
    plugin, model = evaluation(tmp_path)[:2]
    replies = iter(['{"passed": true, "reason": "unescaped "quote""}', '{"passed": true, "reason": "fine"}'])
    calls = []

    def chat(self, messages, **params):
        calls.append(messages)
        return next(replies)

    monkeypatch.setattr(ReviewModel, "chat", chat)
    assert plugin.ask_json(model, "review", {}, time.monotonic() + 10)["reason"] == "fine"
    assert len(calls) == 2 and calls[0] == calls[1]


def test_a_trajectory_within_the_budget_reaches_the_review_whole_except_binary_data():
    image = "iVBORw0KGgo1" * 400
    events = [{"content": "task"}, {"tool": "P" * 50_000}, {"image": image}, {"content": "final answer " * 400}]
    kept = shortened_trajectory(events)
    assert kept[1] == {"tool": "P" * 50_000} and kept[3] == events[3]
    assert kept[2]["image"].startswith("[binary content, such as a page image, that the tool returned to the agent")


def test_a_long_trajectory_cuts_its_longest_texts_first_and_keeps_short_answers():
    paper = "A" * 300_000 + "MIDDLE" + "Z" * 300_000
    answer = "The answer cites [1] and [2]. " * 200
    kept = shortened_trajectory([{"content": "task"}, {"tool": paper}, {"tool": "B" * 60_000}, {"content": answer}])
    assert kept[0] == {"content": "task"} and kept[3] == {"content": answer}
    assert "MIDDLE" not in kept[1]["tool"] and "characters omitted" in kept[1]["tool"]
    assert len(json.dumps(kept)) <= 410_000


def test_a_plan_that_breaks_a_rule_is_asked_for_once_more(tmp_path):
    plugin, model = evaluation(tmp_path)[:2]
    model.responses.insert(0, {"tasks": [{"messages": ["one"], "checks": ["shown"]}] * 6})
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    assert plugin.decide(item, measured).selected
    assert "previous_plan_problem" in model.messages_seen[1][0]["content"]
    replies = json.loads((tmp_path / "reefine-evaluation" / "plan-replies.json").read_text())
    assert len(replies) == 2


def test_a_task_of_several_messages_waits_for_review_on_an_adapter_that_sends_one(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    backend.adapter = "dsh"
    model.responses[0] = {
        "tasks": [
            {"messages": ["/mode on", "what is new"], "checks": ["the mode answers"]},
            {"messages": ["run a command"], "checks": ["command before output"]},
        ]
    }
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    decision = plugin.decide(item, measured)
    statuses = {check["id"]: check["status"] for check in measured.metrics["evaluation_report"]["checks"]}
    assert statuses["request-behavior-0"] == "untested" and statuses["request-behavior-1"] == "pass"
    assert decision.selected and decision.requires_review
    assert decision.reason == (
        "All checks that ran passed. Requested behavior 1 was not run: this task sends 2 messages in one session, "
        "and the dsh adapter sends one message per session. Review the release before you install it."
    )


def test_files_written_outside_the_workspace_reach_the_review_without_invalidating_the_run(tmp_path):
    plugin, model, backend = evaluation(tmp_path)
    backend.results[0] = replace(episode("request"), residue=("Library/Caches/pip/entry",))
    item = candidate(tmp_path)
    measured = plugin.evaluate(item)
    assert plugin.decide(item, measured).selected
    assert '"files_outside_workspace": ["Library/Caches/pip/entry"]' in model.messages_seen[1][0]["content"]


@pytest.mark.parametrize(
    "plan",
    [
        {"tasks": [{"messages": ["x"], "checks": ["y"]}], "limits": [{"requirement": "y"}]},
        {"tasks": [{"messages": ["x"], "checks": ["y"]}], "untested": "none"},
        {"tasks": [{"messages": [], "checks": ["y"]}]},
        {"tasks": [{"messages": ["a", " "], "checks": ["y"]}]},
        {"tasks": [], "limits": []},
        {"prompt": "x", "checks": ["y"]},
    ],
)
def test_malformed_plan_is_invalid(plan):
    with pytest.raises(ValueError):
        validate_plan(plan)


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
                "propose": "reef.recipe.reefine.evolution:TextProposer",
                "evaluate": "reef.recipe.reefine.evolution:AnswerScorer",
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
    report = measured.metrics["evaluation_report"]
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
    assert "Current" in output and "Candidate" in output and "passed" in output
    assert "First Second" in output
    assert "\x1b[" not in output


def test_recipe_config_rejects_unsupported_keys():
    with pytest.raises(RecipeConfigError):
        evaluation_settings({"unknown": 1}, ("health",))


def test_task_environment_is_explicit_and_kept_out_of_settings_display(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy")
    assert evaluation_settings({}, ("health",))[0].environment == {}
    settings, _ = evaluation_settings({"env_from": ["HTTPS_PROXY"]}, ("health",))
    assert settings.environment == {"HTTPS_PROXY": "http://operator-proxy"}
    assert "operator-proxy" not in repr(settings)


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
    assert any(check["status"] == "invalid" for check in result.metrics["evaluation_report"]["checks"])


def test_checks_from_another_release_or_duplicate_checks_cannot_approve(tmp_path):
    plugin = evaluation(tmp_path)[0]
    item = candidate(tmp_path)
    result = plugin.evaluate(item)
    report = result.metrics["evaluation_report"]
    report["current_release_id"] = "other-release"
    assert not plugin.decide(item, result).selected
    report["current_release_id"] = "served-release"
    report["checks"].append(dict(report["checks"][0]))
    assert not plugin.decide(item, result).selected
