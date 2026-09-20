"""Guarantees of the tutorials/evolve-your-harness cookbook example, hermetic: the
model binding is stubbed, episodes never run, and the boot test drives the
same serve.yaml materialization run.sh performs."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from reef_service._trajectories import recorded_trajectory
from reef_service.config_helpers import load_harness_deployment as load_config

from reef.harness.episodes.model_binding import ModelBindingError
from reef.harness.episodes.run import EpisodeResult
from reef.recipe import load_recipe_config
from reef.recipe.config import recipe_config_from_mapping
from reef.recipe.cordis import CordisRecipe
from reef.service.deploy.service_config import service_config_from_mapping
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.cordis_backend import Mutation, StepProposal
from reef.train.trainer import Trainer

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "evolve-your-harness"

#: The composition the proposer sees: the starter skill, as (kind, config)
#: pairs exactly like the backend passes. No provider node: the model binding
#: reef hands the proposer is the only endpoint in play.
NODES = (("skill", {"name": "answer-style", "text": "# answer-style\n\nStarter skill."}),)

SAMPLES = (recorded_trajectory("a1", {"messages": [{"role": "user", "content": "[fib] compute fib(90)"}]}, 0.0),)

#: One queued instruction, as the backend forwards it to a proposer that names ``requests``.
REQUEST = {
    "id": "ask-1",
    "text": "Add a skill that runs the tests before answering",
    "session": "session-1",
    "release_id": "release-1",
    "untrusted": True,
}


def _method(monkeypatch: pytest.MonkeyPatch, module: str) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    # Every example names its package ``harness``: drop any sibling
    # example's cached import so this file's syspath entry wins.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module(f"harness.{module}")


@pytest.fixture(params=["tutorial", "builtin"])
def evolution(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> ModuleType:
    if request.param == "builtin":
        return importlib.import_module("reef.recipe.reefine.evolution")
    return _method(monkeypatch, "evolution")


@pytest.fixture
def native_evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "native_evolution")


#: What marks the plan call's prompt, the first of a request's calls.
PLAN_MARKER = '"needs_tool": true or false'


class Model:
    """A ModelBindings stand-in: ``served`` answers the canned replies in order, the last one again once they run
    out; a reply that is an exception is raised on its call, and ``failure`` is raised on every call. A request's
    plan call, which comes first, is answered with ``plan`` (no step by default) and consumes no canned reply."""

    def __init__(self, *replies: str | Exception, plan: str = "[]", failure: Exception | None = None) -> None:
        self.replies, self.plan, self.failure, self.calls = list(replies), plan, failure, 0
        self.answered = 0
        self.prompts: list[str] = []
        self.params_of: list[dict[str, object]] = []
        self.served = self

    @property
    def prompt(self) -> str | None:
        """The write prompt: the second call's on a request (the plan call comes first), the only one otherwise."""
        return self.prompts[1] if len(self.prompts) > 1 else (self.prompts[0] if self.prompts else None)

    @property
    def params(self) -> dict[str, object]:
        """The write call's parameters, chosen as ``prompt`` is."""
        return self.params_of[1] if len(self.params_of) > 1 else (self.params_of[0] if self.params_of else {})

    def chat(self, messages, **params):
        self.calls += 1
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.params_of.append(dict(params))
        if self.failure is not None:
            raise self.failure
        if PLAN_MARKER in prompt:
            return self.plan
        self.answered += 1
        reply = self.replies[min(self.answered, len(self.replies)) - 1] if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        return reply


def canned(reply: str) -> Model:
    return Model(reply)


#: The reasons a request step records when the reply gave nothing to apply.
NO_ENTRY = "the reply holds no usable entry"
DROPPED = "every entry in the reply was dropped: a reserved id, or an id another kind holds"


def failure_of(proposal) -> str:
    """The reason a request step recorded for producing nothing: a StepProposal without mutations carries it."""
    assert isinstance(proposal, StepProposal) and proposal.mutations == ()
    return proposal.notes["failure"]


def proposal(entry_id: str, name: str = "skill", *, config_name: str | None = None) -> str:
    return json.dumps(
        {"id": entry_id, "name": name, "config": {"name": config_name or entry_id, "text": "# improved\n\ntext"}}
    )


# -- propose: parsing, routing, refusal ----------------------------------


def test_propose_updates_an_existing_skill_from_a_fenced_reply(evolution, monkeypatch) -> None:
    model = canned(f"```json\n{proposal('answer-style')}\n```")
    mutation = evolution.propose(NODES, SAMPLES, model)
    assert isinstance(mutation, Mutation)
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    assert mutation.options == {"name": "skill", "config": {"name": "answer-style", "text": "# improved\n\ntext"}}


def test_propose_creates_a_new_skill_for_an_unknown_id(evolution, monkeypatch) -> None:
    model = canned(proposal("csv-median"))
    mutation = evolution.propose(NODES, SAMPLES, model)
    assert (mutation.op, mutation.id) == ("create", "csv-median")


def test_propose_refuses_non_skill_kinds(evolution) -> None:
    model = canned(proposal("answer-style", name="rules"))
    assert evolution.propose(NODES, SAMPLES, model) is None


def test_propose_returns_none_on_garbage(evolution, monkeypatch) -> None:
    for reply in (
        "no json here",
        '{"id": 1, "name": "skill", "config": {}}',
        proposal("bad id with spaces"),
        proposal("answer-style", config_name="other-name"),
        json.dumps({"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "  "}}),
    ):
        model = canned(reply)
        assert evolution.propose(NODES, SAMPLES, model) is None


def test_propose_skips_when_the_endpoint_is_down(evolution) -> None:
    down = Model(failure=ModelBindingError("model endpoint unreachable: connection refused"))
    assert evolution.propose(NODES, SAMPLES, down) is None


def test_propose_without_failures_skips_without_calling_the_model(evolution) -> None:
    never = Model(failure=AssertionError("no samples, no model call"))
    assert evolution.propose(NODES, (), never) is None
    assert never.calls == 0


def test_propose_answers_a_queued_request_with_the_failures_as_context(evolution) -> None:
    model = canned(proposal("run-tests"))
    (mutation,) = evolution.propose(NODES, SAMPLES, model, requests=(REQUEST,)).mutations
    assert (mutation.op, mutation.id) == ("create", "run-tests")
    assert model.prompt.index(REQUEST["text"]) < model.prompt.index("[fib] compute fib(90)")
    assert "Design the change before you write it" in model.prompt


def test_propose_answers_a_request_alone_without_failures(evolution) -> None:
    model = canned(proposal("answer-style"))
    (mutation,) = evolution.propose(NODES, (), model, requests=(REQUEST,)).mutations
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    # The plan call, the entries, then the review of what it wrote.
    assert model.calls == 3
    assert REQUEST["text"] in model.prompt and "Recent failing requests" not in model.prompt


#: A report's feedback beside its request: what the reporter said was wrong, which the payload alone cannot show.
REPORTED = (
    recorded_trajectory(
        "a2",
        {"messages": [{"role": "user", "content": "fix the failing test in auth.py"}]},
        0.0,
        feedback="missed the empty-token case",
    ),
)


def test_propose_shows_each_failure_with_its_report_score_and_feedback(evolution) -> None:
    """The step hands the proposer ATIF trajectory items whose ``feedback`` is the report's text verbatim; a proposer that
    serialized the payload alone would learn what the model answered but never why it was scored down."""
    model = canned(proposal("answer-style"))
    evolution.propose(NODES, REPORTED + SAMPLES, model)
    prompt = model.prompt
    assert "fix the failing test in auth.py" in prompt
    assert '"feedback": "missed the empty-token case"' in prompt and '"score": 0.0' in prompt
    assert '"feedback": null' in prompt  # SAMPLES' report carried none: the key stays, so the shape is one
    assert prompt.index("[BEGIN") < prompt.index("missed the empty-token case") < prompt.index("[END")
    assert "addressing what the feedback names" in prompt


def test_propose_shows_the_feedback_beside_the_failures_a_request_carries(evolution) -> None:
    model = canned(proposal("run-tests"))
    evolution.propose(NODES, REPORTED, model, requests=(REQUEST,))
    prompt = model.prompt
    assert "Recent failing requests, for context (each with its report's score and feedback" in prompt
    assert '"feedback": "missed the empty-token case"' in prompt and '"score": 0.0' in prompt


def test_native_propose_shows_each_failure_with_its_report_score_and_feedback(native_evolution) -> None:
    model = canned(
        json.dumps({"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "x"}})
    )
    native_evolution.propose(NODES, REPORTED, model)
    prompt = model.prompt
    assert '"feedback": "missed the empty-token case"' in prompt and '"score": 0.0' in prompt
    assert prompt.index("[BEGIN") < prompt.index("missed the empty-token case") < prompt.index("[END")
    assert "addressing what the feedback names" in prompt


API_SKILL = (
    "skill",
    {"name": "reef-pi-extension-api", "text": "---\nname: reef-pi-extension-api\n---\n# pi API\npi.registerTool"},
)


def request_reply(*entries: dict) -> str:
    return json.dumps(list(entries))


def test_propose_answers_a_request_with_the_entries_the_request_and_the_api_skill_in_its_prompt(evolution) -> None:
    skill = {"name": "test-first", "text": "---\nname: test-first\ndescription: run tests first\n---\n# test-first\n"}
    model = canned(request_reply({"id": "test-first", "name": "skill", "config": skill}))
    mutations = evolution.propose(
        (*NODES, ("rules", {"text": "Be brief."}), API_SKILL), (), model, requests=(REQUEST,)
    ).mutations
    assert model.calls == 3  # the plan call, the entries, then the review
    prompt = model.prompt
    assert REQUEST["text"] in prompt and "[BEGIN user request" in prompt
    assert '"id": "answer-style"' in prompt and '"body": "# answer-style' in prompt
    # Without the step's entries the listing comes from the nodes, where a rules entry has no id to show.
    assert '"kind": "rules"' in prompt and '"id": null' in prompt
    assert "pi.registerTool" in prompt and "reef-pi-extension-api" in prompt
    for reserved in ("reef-version-check", "reef-requests"):
        assert reserved in prompt
    for kind in ("skill", "rules", "agent_command", "code_extension"):
        assert f"- {kind}:" in prompt
    # The tool must run on every platform; a canned reply names no tool step, so the write prompt carries no plan.
    assert "macOS, Linux or Windows under WSL 2" in prompt and "process.platform" in prompt
    assert "need a tool the harness does not have" not in prompt
    assert [(m.op, m.id, m.options) for m in mutations] == [
        ("create", "test-first", {"name": "skill", "config": skill})
    ]
    # Without the API skill in the tree the prompt carries no reference section.
    model = canned(request_reply({"id": "test-first", "name": "skill", "config": skill}))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert "pi.registerTool" not in model.prompt and "code_extension" in model.prompt


def test_propose_asks_for_a_plan_first_and_demands_a_tool_for_a_step_the_harness_cannot_perform(evolution) -> None:
    """The first call lists the request's steps; the ones the harness cannot perform go into the second
    call's prompt with the demand for a code_extension beside the rule. A plan the model does not give
    in that shape, or a step it can perform, adds nothing."""
    plan = json.dumps(
        [
            {"step": "reproduce the bug with a failing test", "needs_tool": False},
            {"step": "have a second agent review the diff", "needs_tool": True},
        ]
    )
    entries = request_reply({"id": "bug-fix-workflow", "name": "rules", "config": {"text": "# Bug fix\n"}})
    model = Model(entries, plan=plan)
    mutations = evolution.propose(NODES, (), model, requests=(REQUEST,)).mutations
    assert model.calls == 3 and [m.id for m in mutations] == ["bug-fix-workflow"]
    plan_prompt, write_prompt, _review_prompt = model.prompts
    assert REQUEST["text"] in plan_prompt and '"needs_tool": true or false' in plan_prompt
    assert "need a tool the harness does not have" in write_prompt
    assert "- have a second agent review the diff" in write_prompt
    assert (
        "reproduce the bug with a failing test"
        not in write_prompt.split("need a tool the harness does not have")[1].split("\n\n")[0]
    )
    assert model.params == {"timeout_s": 600.0, "max_tokens": 65536}
    # No step needs a tool: the write prompt carries no plan section.
    model = Model(entries, plan=json.dumps([{"step": "run the tests", "needs_tool": False}]))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert "need a tool the harness does not have" not in model.prompts[1]
    # A plan call that fails, or answers prose, leaves the request answered as before.
    model = Model(entries, plan="I cannot list steps.")
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert [m.id for m in proposal.mutations] == ["bug-fix-workflow"]
    assert "need a tool the harness does not have" not in model.prompts[1]


def test_propose_parses_every_request_kind_from_one_reply(evolution) -> None:
    code = "export default function (pi) {}\n"
    reply = request_reply(
        {"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "# updated"}},
        {"id": "test-first", "name": "rules", "config": {"text": "Run the tests first."}},
        {"id": "notify", "name": "code_extension", "config": {"name": "notify", "code": code, "extra": 1}},
        {"id": "review", "name": "agent_command", "config": {"name": "review", "text": "Review the diff."}},
        {
            "id": "test-first-skill",
            "name": "skill",
            "config": {"text": "--- name: test-first-skill ---\nRun the tests."},
        },
    )
    mutations = evolution.propose(NODES, (), canned(reply), requests=(REQUEST,)).mutations
    assert [(m.op, m.id, m.options) for m in mutations] == [
        ("update", "answer-style", {"name": "skill", "config": {"name": "answer-style", "text": "# updated"}}),
        ("create", "test-first", {"name": "rules", "config": {"text": "Run the tests first."}}),
        ("create", "notify", {"name": "code_extension", "config": {"name": "notify", "code": code}}),
        ("create", "review", {"name": "agent_command", "config": {"name": "review", "text": "Review the diff."}}),
        # A config that omits the name takes the entry id; that is how the served model writes a skill.
        (
            "create",
            "test-first-skill",
            {
                "name": "skill",
                "config": {"name": "test-first-skill", "text": "--- name: test-first-skill ---\nRun the tests."},
            },
        ),
    ]
    # One fenced object is a proposal too, and a named kind already in the tree updates by its name.
    fenced = f"```json\n{json.dumps({'id': 'notify', 'name': 'code_extension', 'config': {'name': 'notify', 'code': code}})}\n```"
    tree = (*NODES, ("code_extension", {"name": "notify", "code": "old"}))
    (mutation,) = evolution.propose(tree, (), canned(fenced), requests=(REQUEST,)).mutations
    assert (mutation.op, mutation.id) == ("update", "notify")
    # The prompt calls the value a kind, and a served model wrote it under that key; both keys read.
    by_kind = request_reply({"id": "bug-fix-workflow", "kind": "rules", "config": {"text": "Reproduce first."}})
    (mutation,) = evolution.propose(NODES, (), canned(by_kind), requests=(REQUEST,)).mutations
    assert (mutation.op, mutation.id, mutation.options) == (
        "create",
        "bug-fix-workflow",
        {"name": "rules", "config": {"text": "Reproduce first."}},
    )
    # The config fields written beside the id instead of under "config" read the same; a config that is
    # present but not an object still fails.
    flat = request_reply({"id": "arithmetic-questions", "kind": "rules", "text": "Answer in one sentence."})
    (mutation,) = evolution.propose(NODES, (), canned(flat), requests=(REQUEST,)).mutations
    assert (mutation.id, mutation.options) == (
        "arithmetic-questions",
        {"name": "rules", "config": {"text": "Answer in one sentence."}},
    )
    broken = request_reply({"id": "x", "name": "rules", "config": "Answer in one sentence."})
    assert failure_of(evolution.propose(NODES, (), canned(broken), requests=(REQUEST,))) == NO_ENTRY
    # A flattened named kind carries both keys: "kind" is the kind and "name" the entry's own name.
    flat_skill = request_reply({"id": "plan-first", "kind": "skill", "name": "plan-first", "text": "# plan-first\n"})
    (mutation,) = evolution.propose(NODES, (), canned(flat_skill), requests=(REQUEST,)).mutations
    assert (mutation.id, mutation.options) == (
        "plan-first",
        {"name": "skill", "config": {"name": "plan-first", "text": "# plan-first\n"}},
    )
    # A model may leave a rules entry without an id, as the tree listing once showed one: the text gives the
    # entry its id. A named kind with a null id stays refused.
    null_rules = request_reply({"id": None, "name": "rules", "config": {"text": "Show the command first."}})
    (mutation,) = evolution.propose(NODES, (), canned(null_rules), requests=(REQUEST,)).mutations
    assert mutation.id == "rules-" + hashlib.sha256(b"Show the command first.").hexdigest()[:8]
    assert mutation.op == "create"
    assert mutation.options == {"name": "rules", "config": {"text": "Show the command first."}}
    null_skill = request_reply({"id": None, "name": "skill", "config": {"text": "# x\n"}})
    assert failure_of(evolution.propose(NODES, (), canned(null_skill), requests=(REQUEST,))) == NO_ENTRY
    # An id that is another kind's name would be refused at admission as an existing entry: a rules entry
    # takes its text's id instead, a named kind is dropped.
    reused = request_reply({"id": "answer-style", "kind": "rules", "config": {"text": "One sentence."}})
    (mutation,) = evolution.propose(NODES, (), canned(reused), requests=(REQUEST,)).mutations
    assert (mutation.op, mutation.id) == ("create", "rules-" + hashlib.sha256(b"One sentence.").hexdigest()[:8])
    reused_named = request_reply({"id": "answer-style", "kind": "agent_command", "config": {"text": "Review."}})
    assert failure_of(evolution.propose(NODES, (), canned(reused_named), requests=(REQUEST,))) == DROPPED


def test_propose_passes_the_budgets_of_the_environment_to_the_model_call(evolution, monkeypatch) -> None:
    """The request path asks with 600 s and 65536 tokens (its plan call before that with 60 s and 4096, its
    review after with 120 s and 16384), the failure path asks with 60 s and 8192 (a thinking model spends part
    of each budget on its reasoning before the JSON), unless REEF_PROPOSER_TIMEOUT_S and REEF_PROPOSER_MAX_TOKENS
    say otherwise; a value that is not a number is ignored rather than turning the step into an error."""
    monkeypatch.delenv("REEF_PROPOSER_TIMEOUT_S", raising=False)
    monkeypatch.delenv("REEF_PROPOSER_MAX_TOKENS", raising=False)
    model = canned(request_reply({"id": "t", "name": "rules", "config": {"text": "Test first."}}))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert model.params_of == [
        {"timeout_s": 60.0, "max_tokens": 4096},
        {"timeout_s": 600.0, "max_tokens": 65536},
        {"timeout_s": 120.0, "max_tokens": 16384},
    ]
    model = canned("no json here")
    evolution.propose(NODES, SAMPLES, model)
    assert model.params_of == [{"timeout_s": 60.0, "max_tokens": 8192}]
    monkeypatch.setenv("REEF_PROPOSER_TIMEOUT_S", "900")
    monkeypatch.setenv("REEF_PROPOSER_MAX_TOKENS", "32768")
    model = canned(request_reply({"id": "t", "name": "rules", "config": {"text": "Test first."}}))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert model.params_of == [{"timeout_s": 900.0, "max_tokens": 32768}] * 3
    monkeypatch.setenv("REEF_PROPOSER_MAX_TOKENS", "16k")
    model = canned(request_reply({"id": "t", "name": "rules", "config": {"text": "Test first."}}))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert model.params_of == [
        {"timeout_s": 900.0, "max_tokens": 4096},
        {"timeout_s": 900.0, "max_tokens": 65536},
        {"timeout_s": 900.0, "max_tokens": 16384},
    ]


def test_propose_drops_a_reserved_id_and_a_malformed_object_from_a_request_reply(evolution) -> None:
    reply = request_reply(
        {"id": "reef-requests", "name": "code_extension", "config": {"name": "reef-requests", "code": "x"}},
        {"id": "reef-version-check", "name": "rules", "config": {"text": "x"}},
        {"id": "notify", "name": "code_extension", "config": {"name": "other", "code": "x"}},
        {"id": "shout", "name": "native_tool", "config": {"name": "shout", "code": "x"}},
        {"id": "bad id", "name": "rules", "config": {"text": "x"}},
        {"id": "empty", "name": "rules", "config": {"text": "  "}},
        {"id": "ok", "name": "rules", "config": {"text": "ok"}},
    )
    mutations = evolution.propose(NODES, (), canned(reply), requests=(REQUEST,)).mutations
    assert [(m.op, m.id) for m in mutations] == [("create", "ok")]
    only_reserved = json.dumps(
        {"id": "reef-pi-extension-api", "name": "skill", "config": {"name": "reef-pi-extension-api", "text": "x"}}
    )
    # A reply with nothing to apply is a proposal without mutations, the reason in its notes.
    assert failure_of(evolution.propose(NODES, (), canned(only_reserved), requests=(REQUEST,))) == DROPPED
    assert failure_of(evolution.propose(NODES, (), canned("no json here"), requests=(REQUEST,))) == NO_ENTRY


def test_propose_without_a_request_keeps_the_failure_path(evolution) -> None:
    never = Model(failure=AssertionError("no samples, no request, no model call"))
    assert evolution.propose(NODES, (), never, requests=()) is None and never.calls == 0
    # The failure path still writes skills only, whatever kinds a request may name.
    assert evolution.propose(NODES, SAMPLES, canned(proposal("answer-style", name="rules"))) is None
    down = Model(failure=ModelBindingError("model endpoint unreachable: connection refused"))
    failure = failure_of(evolution.propose(NODES, (), down, requests=(REQUEST,)))
    assert failure.endswith("(max_tokens=65536): model endpoint unreachable: connection refused")
    # With the tree's entries in hand the failure path is what it was: one call, a bare Mutation, no design
    # and no review.
    model = canned(proposal("answer-style"))
    mutation = evolution.propose(NODES, SAMPLES, model, entries=ENTRIES)
    assert isinstance(mutation, Mutation) and (mutation.op, mutation.id) == ("update", "answer-style")
    assert model.calls == 1 and "Design the change" not in model.prompt


def test_propose_keeps_reefs_own_skill_out_of_the_failure_path(evolution) -> None:
    """The API reference skill is reserved: the failure prompt lists the skills the model may update, so it
    omits the reference, and a reply that names it anyway is dropped."""
    tree = (*NODES, API_SKILL)
    model = canned(proposal("answer-style"))
    mutation = evolution.propose(tree, SAMPLES, model)
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    assert '"name": "answer-style"' in model.prompt and "Current skills" in model.prompt
    assert "reef-pi-extension-api" not in model.prompt and "pi.registerTool" not in model.prompt
    assert evolution.propose(tree, SAMPLES, canned(proposal("reef-pi-extension-api"))) is None
    # In an array reply the reserved object is dropped and the next usable one is the proposal.
    both = f"[{proposal('reef-pi-extension-api')}, {proposal('csv-median')}]"
    mutation = evolution.propose(tree, SAMPLES, canned(both))
    assert (mutation.op, mutation.id) == ("create", "csv-median")


# -- propose: the design, the review and the notes a request records ------

#: The step's tree as the backend hands it to a proposer that names ``entries``: every entry with its id.
ENTRIES = (
    {"id": "answer-style", "name": "skill", "config": NODES[0][1]},
    {"id": "brevity", "name": "rules", "config": {"text": "Be brief."}},
)

DESIGN = "The user wants the tests run before every answer. Trigger: every task; no state. Nothing to set up."

#: A complete review, so a case answers once; the retry after a short review has cases of its own.
REVIEW = {"result": "complete", "covered": ["the tests run first"], "uncovered": []}


def designed(*entries: dict, design: str = DESIGN, requires: list | None = None) -> str:
    """A request reply as the prompt asks for it: the design object, the entries, the requires object last."""
    objects = [{"design": design}, *entries]
    if requires is not None:
        objects.append({"requires": requires})
    return json.dumps(objects)


def skill(entry_id: str, text: str = "# improved\n\ntext") -> dict:
    return {"id": entry_id, "name": "skill", "config": {"name": entry_id, "text": text}}


def rules(entry_id: str, text: str) -> dict:
    return {"id": entry_id, "name": "rules", "config": {"text": text}}


def extension(entry_id: str, code: str) -> dict:
    return {"id": entry_id, "name": "code_extension", "config": {"name": entry_id, "code": code}}


def test_propose_answers_a_request_with_the_design_and_the_review_in_the_notes(evolution) -> None:
    model = Model(designed(skill("run-tests")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert isinstance(proposal, StepProposal)
    assert [(m.op, m.id) for m in proposal.mutations] == [("create", "run-tests")]
    assert proposal.notes == {"design": DESIGN, "review": REVIEW}
    _plan_prompt, request_prompt, review_prompt = model.prompts
    # The prompt asks for the design first: the restatement, the triggers and states with their sources and
    # the explicit toggle rule, what the user must provide, then the entries, complete and nothing more.
    assert "1. Restate the request in one sentence." in request_prompt
    assert "turn it on and off" in request_prompt and "never a rule that assumes the state holds" in request_prompt
    assert "5. Then write the entries: complete for what the request implies" in request_prompt
    assert "nothing the request did not ask for" in request_prompt and "smallest change" not in request_prompt
    # What only the user can provide is declared, with a prompt for setup: the extension never asks for it,
    # stores it or hardcodes it, and reads an env item's value from the environment at run time.
    assert "3. List what only the user can provide (a phone number, a credential, a permission, an account)" in (
        request_prompt
    )
    assert "read at run time from process.env.NAME" in request_prompt
    assert "never asks the user for it, never stores it in a file of its own and never hardcodes it" in request_prompt
    assert "the extension itself never asks" in request_prompt and "once, at install time" in request_prompt
    assert (
        '{"name": "REEF_AWAY_PHONE", "kind": "env", "prompt": "The phone number to text, with the country code"}'
        in request_prompt
    )
    assert (
        '{"name": "messages-automation", "kind": "permission", "check": "osascript -e \'tell application '
        '\\"Messages\\" to get name\'", "prompt": "Allow the agent to control Messages when macOS asks"}'
        in request_prompt
    )
    assert (
        '{"name": "github-cli", "kind": "service", "check": "gh auth status", "prompt": "Sign in to the GitHub CLI"}'
        in request_prompt
    )
    assert '{"design": "<the design>"}' in request_prompt and "null" not in request_prompt
    # The review reads the request, fenced as data, the design and the entries as written.
    assert REQUEST["text"] in review_prompt and "[BEGIN user request" in review_prompt
    assert DESIGN in review_prompt and '"id": "run-tests"' in review_prompt and "# improved" in review_prompt
    assert '"result": "complete" or "partial"' in review_prompt
    assert "a value the user must provide that the extension asks for or stores itself instead of declaring" in (
        review_prompt
    )
    assert model.params_of == [
        {"timeout_s": 60.0, "max_tokens": 4096},
        {"timeout_s": 600.0, "max_tokens": 65536},
        {"timeout_s": 120.0, "max_tokens": 16384},
    ]
    # A design longer than the record keeps is cut, and a fenced review still reads.
    fenced = f"Here it is:\n```json\n{json.dumps(REVIEW)}\n```"
    model = Model(designed(skill("run-tests"), design="x" * 2000), fenced)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes["design"] == "x" * 1500 and proposal.notes["review"] == REVIEW
    # Without a design object the notes carry the review alone, and the review prompt says none was written.
    model = Model(request_reply(skill("run-tests")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes == {"review": REVIEW} and "(none written)" in model.prompts[2]


def test_a_review_that_fails_leaves_the_notes_without_one_and_the_mutations_stand(evolution) -> None:
    for review in (
        "no json here",
        json.dumps({"result": "done", "covered": []}),
        json.dumps(["complete"]),
        ModelBindingError("model endpoint unreachable: connection refused"),
    ):
        model = Model(designed(skill("run-tests")), review)
        proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
        assert [(m.op, m.id) for m in proposal.mutations] == [("create", "run-tests")]
        assert proposal.notes == {"design": DESIGN} and model.calls == 3
    # The verdict's case and the lists are read leniently: strings only, trimmed, anything else dropped.
    lenient = {"result": "Complete", "covered": ["a", 1, " b ", ""], "uncovered": "none"}
    model = Model(designed(skill("run-tests")), json.dumps(lenient))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes["review"] == {"result": "complete", "covered": ["a", "b"], "uncovered": []}


def test_legacy_review_key_is_normalized_to_result(evolution) -> None:
    review = {"verdict": "partial", "covered": ["tests"], "uncovered": ["review"]}
    model = Model(designed(skill("run-tests")), json.dumps(review))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes["review"] == {"result": "partial", "covered": ["tests"], "uncovered": ["review"]}


def test_a_failed_model_call_is_the_request_steps_failure_note_and_still_a_skip_on_the_failure_path(
    evolution, monkeypatch, caplog
) -> None:
    """A request whose model call fails gets a StepProposal without mutations whose notes say why, in one line:
    how long the call took, the reply budget and the endpoint's error, with the budget hint when the reply had no
    text; the reason is logged too. The failure path keeps returning None, the reason in the log alone."""
    monkeypatch.delenv("REEF_PROPOSER_MAX_TOKENS", raising=False)
    no_text = ModelBindingError("model endpoint returned non-text content")
    down = Model(failure=no_text)
    with caplog.at_level(logging.WARNING):
        proposal = evolution.propose(NODES, (), down, requests=(REQUEST,), entries=ENTRIES)
    assert isinstance(proposal, StepProposal) and proposal.mutations == () and list(proposal.notes) == ["failure"]
    assert re.fullmatch(
        r"model call failed after \d+\.\d s \(max_tokens=65536\): model endpoint returned non-text content; "
        r"a thinking model may have spent the reply budget on its reasoning, raise REEF_PROPOSER_MAX_TOKENS",
        proposal.notes["failure"],
    )
    assert down.calls == 2  # the plan call and the entries call, both failed; nothing to review
    assert "propose: served model call failed after" in caplog.text
    # Any other error gets no budget hint, and the budget in the reason is the one the call was given.
    monkeypatch.setenv("REEF_PROPOSER_MAX_TOKENS", "32768")
    unreachable = Model(failure=ModelBindingError("model endpoint unreachable: connection refused"))
    failure = failure_of(evolution.propose(NODES, (), unreachable, requests=(REQUEST,)))
    assert failure.endswith("(max_tokens=32768): model endpoint unreachable: connection refused")
    assert "REEF_PROPOSER_MAX_TOKENS" not in failure
    # A reply with nothing to apply says so, with the design when the model wrote one; an empty reply too.
    proposal = evolution.propose(NODES, (), canned(designed()), requests=(REQUEST,))
    assert proposal.mutations == () and proposal.notes == {"design": DESIGN, "failure": NO_ENTRY}
    assert failure_of(evolution.propose(NODES, (), canned(""), requests=(REQUEST,))) == "the reply is empty"
    # The failure path: still None, nothing recorded.
    assert evolution.propose(NODES, SAMPLES, Model(failure=no_text)) is None


def test_propose_lists_every_entry_id_and_updates_a_rules_entry_by_it(evolution) -> None:
    model = Model(designed(rules("brevity", "Be brief; one sentence.")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [(m.op, m.id, m.options) for m in proposal.mutations] == [
        ("update", "brevity", {"name": "rules", "config": {"text": "Be brief; one sentence."}})
    ]
    listing = model.prompt.split("Current harness entries", 1)[1].split("You may write", 1)[0]
    assert json.loads(listing.split(":", 1)[1]) == [
        {"id": "answer-style", "kind": "skill", "body": NODES[0][1]["text"]},
        {"id": "brevity", "kind": "rules", "body": "Be brief."},
    ]
    # An id another kind holds would be refused at admission: a rules entry takes one from its text and a named
    # kind is dropped, as before.
    reused = designed(rules("answer-style", "One sentence."), skill("brevity"), skill("plan-first", "# plan\n"))
    proposal = evolution.propose(NODES, (), Model(reused, json.dumps(REVIEW)), requests=(REQUEST,), entries=ENTRIES)
    assert [(m.op, m.id) for m in proposal.mutations] == [
        ("create", "rules-" + hashlib.sha256(b"One sentence.").hexdigest()[:8]),
        ("create", "plan-first"),
    ]
    # A rules text the tree already holds under its text's id updates that entry rather than failing admission.
    held_id = "rules-" + hashlib.sha256(b"Be brief.").hexdigest()[:8]
    held = (*ENTRIES, {"id": held_id, "name": "rules", "config": {"text": "Be brief."}})
    same = Model(designed(rules("answer-style", "Be brief.")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), same, requests=(REQUEST,), entries=held)
    assert [(m.op, m.id) for m in proposal.mutations] == [("update", held_id)]


def test_propose_brings_an_env_check_to_its_variable_before_the_shape_check(evolution) -> None:
    requires = [
        # A shell test naming one variable: the check becomes that name.
        {"name": "TWILIO_AUTH_TOKEN", "kind": "env", "check": 'test -n "$TWILIO_AUTH_TOKEN"'},
        {"name": "twilio-sid", "kind": "env", "check": "[ -n ${TWILIO_SID} ]"},
        # A check naming no variable is dropped when the name is one: the bare name says what to set.
        {"name": "SMS_FROM", "kind": "env", "check": "echo checked"},
        {"name": "SMS_FROM_NUMBER", "kind": "env", "check": "SMS_FROM_NUMBER"},
        {"name": "github-cli", "kind": "service", "check": "gh auth status"},
    ]
    request = {**REQUEST, "requires": [{"name": "SMS_TO", "kind": "env"}]}
    model = Model(designed(skill("sms"), requires=requires), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(request,), entries=ENTRIES)
    assert request["requires"] == [
        {"name": "SMS_TO", "kind": "env"},
        {"name": "TWILIO_AUTH_TOKEN", "kind": "env", "check": "TWILIO_AUTH_TOKEN"},
        {"name": "twilio-sid", "kind": "env", "check": "TWILIO_SID"},
        {"name": "SMS_FROM", "kind": "env"},
        {"name": "SMS_FROM_NUMBER", "kind": "env", "check": "SMS_FROM_NUMBER"},
        {"name": "github-cli", "kind": "service", "check": "gh auth status"},
    ]
    assert "refused_requires" not in proposal.notes
    # The review sees the requires items beside the entries.
    assert '"check": "gh auth status"' in model.prompts[2] and '"name": "SMS_TO"' in model.prompts[2]


def test_propose_keeps_the_prompt_of_a_requires_item_for_setup(evolution) -> None:
    """A prompt is the sentence setup shows when it asks for the item: kept on the request mapping stripped and
    cut at 200 characters, dropped when empty; the env normalization runs on the same item first."""
    check = "osascript -e 'tell application \"Messages\" to get name'"
    requires = [
        {"name": "REEF_AWAY_PHONE", "kind": "env", "prompt": "  The phone number to text, with the country code "},
        {"name": "messages-automation", "kind": "permission", "check": check, "prompt": "Allow " + "x" * 250},
        {"name": "github-cli", "kind": "service", "check": "gh auth status", "prompt": "   "},
        {"name": "SMS_FROM", "kind": "env", "check": 'test -n "$SMS_FROM"', "prompt": "The number to text from"},
    ]
    request = dict(REQUEST)
    model = Model(designed(skill("sms"), requires=requires), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(request,), entries=ENTRIES)
    assert request["requires"] == [
        {"name": "REEF_AWAY_PHONE", "kind": "env", "prompt": "The phone number to text, with the country code"},
        {"name": "messages-automation", "kind": "permission", "check": check, "prompt": ("Allow " + "x" * 250)[:200]},
        {"name": "github-cli", "kind": "service", "check": "gh auth status"},
        {"name": "SMS_FROM", "kind": "env", "check": "SMS_FROM", "prompt": "The number to text from"},
    ]
    assert "refused_requires" not in proposal.notes and "undeclared_env" not in proposal.notes
    # The review sees the prompts beside the items.
    assert '"prompt": "The number to text from"' in model.prompts[2]
    # A refused item is recorded as written, its prompt included.
    refused = [{"name": "phone", "kind": "sms", "prompt": "The number to text"}]
    model = Model(designed(skill("sms"), requires=refused), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(dict(REQUEST),), entries=ENTRIES)
    assert proposal.notes["refused_requires"] == [
        {"item": refused[0], "reason": "requires[0].kind must be one of ('permission', 'env', 'service')"}
    ]


def test_the_request_prompts_requires_examples_are_items_the_step_keeps_with_their_prompts(evolution) -> None:
    """The worked example of each kind in the prompt is JSON the shape check admits, and a reply that copies it
    lands on the request mapping as written, prompt included."""
    model = Model(designed(skill("run-tests")), json.dumps(REVIEW))
    evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    lines = [line for line in model.prompt.splitlines() if line.startswith(("- env,", "- permission,", "- service,"))]
    examples = [json.loads(line[line.index("{") :]) for line in lines]
    assert [item["kind"] for item in examples] == ["env", "permission", "service"]
    assert all(item["prompt"] for item in examples) and "check" not in examples[0]
    request = dict(REQUEST)
    model = Model(designed(skill("run-tests"), requires=examples), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(request,), entries=ENTRIES)
    assert request["requires"] == examples and "refused_requires" not in proposal.notes


def test_propose_records_the_requires_items_it_could_not_honor_with_the_reason(evolution) -> None:
    requires = [
        {"name": "phone", "kind": "sms"},
        "SLACK_WEBHOOK",
        {"name": "two-vars", "kind": "env", "check": 'test -n "$A" && test -n "$B"'},
        {"name": "bad name", "kind": "permission"},
        {"name": "TWILIO_SID", "kind": "env"},
    ]
    request = dict(REQUEST)
    model = Model(designed(skill("sms"), requires=requires), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(request,), entries=ENTRIES)
    assert request["requires"] == [{"name": "TWILIO_SID", "kind": "env"}]
    assert proposal.notes["refused_requires"] == [
        {
            "item": {"name": "phone", "kind": "sms"},
            "reason": "requires[0].kind must be one of ('permission', 'env', 'service')",
        },
        {"item": "SLACK_WEBHOOK", "reason": "requires[0] must be an object with a name and a kind"},
        {
            "item": {"name": "two-vars", "kind": "env", "check": 'test -n "$A" && test -n "$B"'},
            "reason": "requires[0].check must be a variable name matching ^[A-Za-z_][A-Za-z0-9_]*$ for kind env",
        },
        {
            "item": {"name": "bad name", "kind": "permission"},
            "reason": "requires[0].name must be a non-empty string matching ^[A-Za-z0-9][A-Za-z0-9._-]*$",
        },
    ]
    # A read only request mapping keeps the kept items out of it; the notes still record the refusals.
    frozen = types.MappingProxyType(dict(REQUEST))
    model = Model(designed(skill("sms"), requires=requires), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(frozen,), entries=ENTRIES)
    assert "requires" not in frozen and len(proposal.notes["refused_requires"]) == 4


def test_propose_lists_the_variables_an_extension_reads_that_no_requires_item_names(evolution) -> None:
    code = (
        "export default function (pi) {\n"
        "  const token = process.env.TWILIO_AUTH_TOKEN;\n"
        "  const sid = process.env[\"TWILIO_SID\"], from = process.env['SMS_FROM'];\n"
        "  const known = [process.env.HOME, process.env.PATH, process.env.REEF_URL, process.env.PI_OFFLINE];\n"
        "  return process.env.TWILIO_AUTH_TOKEN && process.env.SMS_TO;\n"
        "}\n"
    )
    request = {**REQUEST, "requires": [{"name": "SMS_TO", "kind": "env"}]}
    added = [{"name": "twilio-sid", "kind": "env", "check": "TWILIO_SID"}]
    model = Model(designed(extension("sms", code), requires=added), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(request,), entries=ENTRIES)
    # Named by an item's name or check, set by pi or the shell, or reef's own: not listed. Reading order, once.
    assert proposal.notes["undeclared_env"] == ["TWILIO_AUTH_TOKEN", "SMS_FROM"]
    # A skill reads nothing, and an extension whose every read is named leaves the key out.
    model = Model(designed(skill("sms")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(dict(REQUEST),), entries=ENTRIES)
    assert "undeclared_env" not in proposal.notes
    covered = designed(extension("sms", "process.env.SMS_TO"), requires=[{"name": "SMS_TO", "kind": "env"}])
    model = Model(covered, json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(dict(REQUEST),), entries=ENTRIES)
    assert "undeclared_env" not in proposal.notes


# -- evaluate: exact last-line grading ------------------------------------


def episode(trajectory: tuple[dict, ...]) -> EpisodeResult:
    return EpisodeResult(exit_code=0, stdout="", stderr="", trajectory=trajectory, residue=())


def pi_message(text: str) -> dict:
    return {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def test_evaluate_grades_exact_final_lines(evolution) -> None:
    prefix, answer = next(iter(evolution.ANSWERS.items()))
    task = f"{prefix} the task"
    assert evolution.evaluate(task, episode((pi_message(f"Working...\n\n{answer}"),))) == 1.0
    assert evolution.evaluate(task, episode(({"role": "assistant", "content": answer},))) == 1.0


def test_evaluate_grades_non_exact_as_zero(evolution) -> None:
    prefix, answer = next(iter(evolution.ANSWERS.items()))
    task = f"{prefix} the task"
    assert evolution.evaluate(task, episode((pi_message(f"{answer}!"),))) == 0.0
    assert evolution.evaluate(task, episode((pi_message(f"The answer is {answer}"),))) == 0.0
    assert evolution.evaluate(task, episode(())) == 0.0
    assert evolution.evaluate("no such prefix", episode((pi_message(answer),))) == 0.0


HEALTH_TASK = (
    "[health] Run the shell command `echo reef-ok` with your shell tool and reply with its exact output as a "
    "plain word alone on the last line."
)


def test_the_builtin_grades_the_health_task_and_the_tutorial_keeps_its_own_answers(monkeypatch) -> None:
    """The built-in scorer knows the profile's health task alone; the tutorial's shim keeps the three arithmetic
    answers its deployments still run, reading the trajectory through the built-in reader."""
    builtin = importlib.import_module("reef.recipe.reefine.evolution")
    tutorial = _method(monkeypatch, "evolution")
    assert builtin.ANSWERS == {"[health]": "reef-ok"}
    assert set(tutorial.ANSWERS) == {"[sieve]", "[fib]", "[csv]"}
    assert builtin.evaluate(HEALTH_TASK, episode((pi_message("$ echo reef-ok\nreef-ok"),))) == 1.0
    assert builtin.evaluate(HEALTH_TASK, episode((pi_message("reef-ok is the output."),))) == 0.0
    assert tutorial.evaluate(HEALTH_TASK, episode((pi_message("reef-ok"),))) == 0.0
    assert tutorial.evaluate("[sieve] count the primes", episode((pi_message("Sieving...\n\n9592"),))) == 1.0
    assert builtin.evaluate("[sieve] count the primes", episode((pi_message("9592"),))) == 0.0
    assert builtin.final_assistant_text([pi_message("a"), {"role": "user", "content": "b"}]) == "a"


# -- serve.yaml boots the recipe ------------------------------------------


@pytest.mark.parametrize("filename", ["serve.yaml", "serve-native.yaml", "deployment.yaml"])
@pytest.mark.parametrize("selector", ["role", "worker"])
def test_materializer_preserves_executor_profiles_and_recipe_selection(monkeypatch, tmp_path, filename, selector):
    materializer = _method(monkeypatch, "materialize_recipe")
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR.parents[1]))
    config = yaml.safe_load((EXAMPLE_DIR / "configs" / filename).read_text())
    config["executors"] = {"cpu-pool": {"backend": "mp", "workers": 2, "resources": {"cpus_per_worker": 2}}}
    config["execution"] = {"evolution": "cpu-pool"}
    if selector == "worker":
        config["recipe"]["config"]["evolution"]["worker_executor"] = "cpu-pool"
        config["execution"]["evolution"] = "uni"  # The explicit worker profile must win.
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    settings = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert settings["execution"] == config["execution"]
    assert settings["executors"] == config["executors"]
    assert "reef" not in settings and "services" not in settings
    assert json.loads((tmp_path / "work/tasks.json").read_text()) == config["recipe"]["config"]["evolution"]["tasks"]
    # Boot the real recipe; a retained selector without its profile would fail here.
    from reef.inference.http import InferenceProxyRuntime

    recipe = CordisRecipe.from_environment(
        {},
        config=recipe_config_from_mapping(settings),
        runtime=InferenceProxyRuntime(model_path="test", base_url="http://unused", api_key="dummy"),
    )
    assert recipe.worker_executor.backend == "mp"
    assert recipe.episode_workers == 2
    assert recipe.worker_executor.workers == 2
    assert recipe.worker_executor.resources.cpus_per_worker == 2


def test_materializer_accepts_config_without_execution_sections(monkeypatch, tmp_path):
    materializer = _method(monkeypatch, "materialize_recipe")
    config = yaml.safe_load((EXAMPLE_DIR / "configs/serve.yaml").read_text())
    config.pop("execution")
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    result = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert set(result) == {"schema-version", "recipe", "inference"}


def test_example_yaml_boots_the_recipe_through_from_environment(tmp_path, monkeypatch) -> None:
    """The run.sh contract, hermetic: interpolate serve.yaml through reef's
    config loader, materialize the recipe sections as a named config, and
    boot the recipe (seed validation included) with a fake binary."""
    evolution = _method(monkeypatch, "evolution")  # the tutorial's grader: its tasks are the three arithmetic ones
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    # serve.yaml names the real binary; this run has no pi on PATH.
    recipe_sections["evolution"] = {**recipe_sections["evolution"], "binary": str(tmp_path / "fake-pi")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))

    # The deployment names the upstream once, on the reef section; the
    # service builds the recipe's runtime from it.
    service = service_config_from_mapping(config)
    assert (service.upstream_url, service.upstream_api_key, service.upstream_model) == (
        "http://127.0.0.1:8000",
        "dummy",
        "qwen3-8b",
    )
    from reef.service.assembly import _upstream_runtime

    settings = load_recipe_config(materialized)
    built = CordisRecipe.from_environment({}, config=settings, runtime=_upstream_runtime(service))
    assert built.adapter == "pi"
    assert built.binary == str(tmp_path / "fake-pi")
    assert len(built.tasks) == 3
    assert all(any(task.startswith(prefix) for prefix in evolution.ANSWERS) for task in built.tasks)
    assert (built.batch_size, built.training_mode) == (1, "auto")

    # The seed carries no provider node and the binding comes from the runtime.
    assert [entry["id"] for entry in built.seed] == ["answer-style"]
    assert "upstream" not in yaml.safe_dump(list(built.seed))
    binding = built.model_binding()
    assert (binding.base_url, binding.model, binding.api_key, binding.api) == (
        "http://127.0.0.1:8000",
        "qwen3-8b",
        "dummy",
        "openai",
    )
    assert list(built.model_bindings()) == ["served"]
    assert isinstance(built.build("demo", SQLiteRecordStore()), Trainer)  # loads the seed; no episodes


# -- the native variant: skills, tools, and hooks --------------------------

NATIVE_NODES = (
    (
        "native_tool",
        {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {"type": "object"},
            "code": "def run(args, workdir):\n    return ''\n",
        },
    ),
    (
        "native_hook",
        {"name": "loop_guard", "event": "post_execute", "code": "def listen(payload, next):\n    return next()\n"},
    ),
    *NODES,
)

TOOL = {
    "description": "Run python.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    "code": "def run(args, workdir):\n    return 'ok'\n",
}
HOOK = {"event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"}


def native_proposal(kind: str, entry_id: str, **config) -> str:
    return json.dumps({"id": entry_id, "name": kind, "config": {"name": entry_id, **config}})


def native_message(text: str) -> dict:
    return {
        "type": "assistant/message",
        "seq": 4,
        "time": 0,
        "data": {"content": text, "tool_calls": [], "finish": "stop"},
    }


def test_native_propose_routes_skills_tools_and_hooks(native_evolution) -> None:
    tool = native_evolution.propose(
        NATIVE_NODES, SAMPLES, canned(native_proposal("native_tool", "run_python", **TOOL))
    )
    assert (tool.op, tool.id) == ("create", "run_python")
    assert tool.options == {"name": "native_tool", "config": {"name": "run_python", **TOOL}}
    hook = native_evolution.propose(
        NATIVE_NODES, SAMPLES, canned(native_proposal("native_hook", "answer_last", **HOOK))
    )
    assert (hook.op, hook.id, hook.options["name"]) == ("create", "answer_last", "native_hook")
    # A name held by a node of the same kind updates it: the shipped seed tools are addressable by name.
    same = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(native_proposal("native_tool", "read_file", **TOOL)))
    assert (same.op, same.id) == ("update", "read_file")
    skill = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(proposal("answer-style")))
    assert (skill.op, skill.id, skill.options["name"]) == ("update", "answer-style", "skill")
    # qwen2.5:7b was measured swapping the kind and the id; the config name settles it.
    swapped = json.dumps({"id": "native_tool", "name": "median", "config": {"name": "median", **TOOL}})
    mutation = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(swapped))
    assert (mutation.op, mutation.id, mutation.options["name"]) == ("create", "median", "native_tool")


def test_native_propose_refuses_malformed_shapes(native_evolution) -> None:
    for reply in (
        proposal("answer-style", name="rules"),
        native_proposal("native_hook", "h", event="on_exit", code="x = 1"),
        native_proposal("native_hook", "h", event="pre_step", code="  "),
        native_proposal("native_tool", "t", description="d", parameters="not a schema", code="x = 1"),
        native_proposal("native_tool", "t", description="", parameters={}, code="x = 1"),
        json.dumps({"id": "t", "name": "native_tool", "config": {"name": "other", **TOOL}}),
        "no json here",
    ):
        assert native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply)) is None


def test_native_propose_accepts_every_event_the_shape_offers(native_evolution) -> None:
    for event in ("pre_step", "pre_execute", "request_error", "post_execute"):
        reply = native_proposal(
            "native_hook", "guard", event=event, code="def listen(payload, next):\n    return next()\n"
        )
        mutation = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply))
        assert isinstance(mutation, Mutation) and mutation.options["config"]["event"] == event


MAIN_GRAPH = {
    "name": "main",
    "start": "think",
    "max_steps": 12,
    "stages": {
        "think": {"kind": "model"},
        "act": {"kind": "tools"},
        "check": {"kind": "verify", "check": "last_line_integer"},
        "done": {"kind": "end", "reason": "completed"},
    },
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "check"},
        {"from": "act", "when": "done", "to": "think"},
        {"from": "check", "when": "pass", "to": "done"},
        {"from": "check", "when": "fail", "to": "think"},
    ],
}


def test_native_propose_routes_the_main_graph_through_a_proposed_agent(native_evolution) -> None:
    """An agent alone can never win: only a subagent stage runs one, and its text comes back as a user message
    the grader never reads. The proposal carries the graph edit: ask the agent, then a model stage answers."""
    from reef.harness.tree.nodes import NODE_KINDS

    reply = native_proposal("native_agent", "helper", prompt="Solve the task alone.", tools=["read_file"])
    nodes = (*NATIVE_NODES, ("native_graph", MAIN_GRAPH))
    proposal = native_evolution.propose(nodes, SAMPLES, canned(reply))
    assert isinstance(proposal, list) and [m.op for m in proposal] == ["create", "update"]
    agent, route = proposal
    assert agent.id == "helper" and agent.options["config"]["prompt"] == "Solve the task alone."
    graph = route.options["config"]
    assert route.id == "main"
    assert graph["stages"]["ask-helper"] == {"kind": "subagent", "agent": "helper"}
    assert graph["stages"]["answer-helper"] == {"kind": "model"}
    assert {"from": "think", "when": "text", "to": "ask-helper"} in graph["edges"]
    assert [(e["when"], e["to"]) for e in graph["edges"] if e["from"] == "ask-helper"] == [
        ("completed", "answer-helper"),
        ("gave_up", "answer-helper"),
        ("budget", "answer-helper"),
        ("ask", "answer-helper"),
    ]
    assert sorted((e["when"], e["to"]) for e in graph["edges"] if e["from"] == "answer-helper") == [
        ("text", "check"),
        ("tool_calls", "act"),
    ]
    NODE_KINDS["native_graph"](None, graph)
    # The agent stands alone without a main graph to route through, when the graph already runs it under any
    # stage name, and when the stage names it would add are taken.
    assert isinstance(native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply)), Mutation)
    routed = (*NATIVE_NODES, ("native_graph", graph))
    assert isinstance(native_evolution.propose(routed, SAMPLES, canned(reply)), Mutation)
    consult = {
        **MAIN_GRAPH,
        "stages": {**MAIN_GRAPH["stages"], "consult": {"kind": "subagent", "agent": "helper"}},
        "edges": [
            {**e, "to": "consult"} if e["from"] == "think" and e["when"] == "text" else e for e in MAIN_GRAPH["edges"]
        ]
        + [{"from": "consult", "when": o, "to": "check"} for o in ("completed", "gave_up", "budget", "ask")],
    }
    NODE_KINDS["native_graph"](None, consult)
    assert isinstance(
        native_evolution.propose((*NATIVE_NODES, ("native_graph", consult)), SAMPLES, canned(reply)), Mutation
    )
    taken = {**MAIN_GRAPH, "stages": {**MAIN_GRAPH["stages"], "ask-helper": {"kind": "message", "text": "hi"}}}
    assert isinstance(
        native_evolution.propose((*NATIVE_NODES, ("native_graph", taken)), SAMPLES, canned(reply)), Mutation
    )


def test_native_evaluate_reads_the_last_assistant_message_with_content(native_evolution) -> None:
    task = "[sieve] count the primes below 100000"
    call = {
        "type": "assistant/message",
        "seq": 2,
        "time": 0,
        "data": {"content": "", "tool_calls": [{}], "finish": "x"},
    }
    assert native_evolution.evaluate(task, episode((call, native_message("Sieving...\n\n9592")))) == 1.0
    # A tool-call message carries no text; the last text answer is the one graded.
    assert native_evolution.evaluate(task, episode((native_message("9592"), call))) == 1.0
    assert native_evolution.evaluate(task, episode((native_message("The count is 9592"),))) == 0.0
    assert native_evolution.evaluate(task, episode(())) == 0.0


def test_native_example_yaml_boots_the_recipe_with_the_shipped_seed(native_evolution, tmp_path, monkeypatch) -> None:
    """The run.sh native contract, hermetic: the native serve file materializes
    like serve.yaml and boots with the loop's own tools and hook seeded by reference."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve-native.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))
    from reef.service.assembly import _upstream_runtime

    service = service_config_from_mapping(config)
    built = CordisRecipe.from_environment(
        {}, config=load_recipe_config(materialized), runtime=_upstream_runtime(service)
    )
    assert built.adapter == "native" and built.binary is None
    # The same three tasks as the pi variant, so the two runs are comparable.
    assert built.tasks == tuple(load_config(EXAMPLE_DIR / "configs" / "serve.yaml")["evolution"]["tasks"])
    assert built.training_mode == "auto"
    assert [entry["id"] for entry in built.seed] == [
        "read_file",
        "write_file",
        "run_bash",
        "execute",
        "loop_guard",
        "main",
        "answer-style",
    ]
    assert "upstream" not in yaml.safe_dump(list(built.seed))
    assert isinstance(built.build("demo", SQLiteRecordStore()), Trainer)  # loads the seed; no episodes


@pytest.mark.parametrize("model_id", ["provider/model-a", "provider/model-b"])
def test_deployment_yaml_names_directories_that_exist_and_boots_its_named_recipe(
    monkeypatch, tmp_path, model_id
) -> None:
    """The README deployment: ``reef.recipe: deployment`` is read back from the
    directory the service's own env names, and the harness package is on the
    PYTHONPATH the same env sets; a stale directory name here fails at boot, so
    the file's own paths are checked against the checkout."""
    import os

    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    repo_root = EXAMPLE_DIR.parents[1]
    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", model_id)
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(repo_root))
    path = EXAMPLE_DIR / "configs" / "deployment.yaml"
    config = load_config(path)
    recipe_dir = EXAMPLE_DIR / "configs"
    assert (recipe_dir / "deployment.yaml").resolve() == path.resolve()
    method_reference = config["evolution"]["propose"]
    module, _, function = method_reference.partition(":")
    assert callable(getattr(importlib.import_module(module), function))
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith("tutorials/evolve-your-harness/")
    assert config["run_dir"].startswith("tutorials/evolve-your-harness/")
    service = service_config_from_mapping(config)
    monkeypatch.delenv("REEF_UPSTREAM_MODEL")  # Recipe construction uses the resolved runtime, not the environment.
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(recipe_dir)},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, CordisRecipe) and built.adapter == "pi"
    assert service.upstream_model == model_id
    assert built.model_binding().model == model_id
    assert built.build_surface("demo").harness.served_model == model_id
    # The person asks while it keeps learning from failures, so the tutorial proposer must take requests.
    assert built.training_mode == "hybrid"
    records = SQLiteRecordStore()
    trainer = replace(built, binary=str(tmp_path / "fake-pi")).build("demo", records)
    assert trainer.training_mode == "hybrid"
    trainer.close()
    records.close()


def test_deployment_yaml_sets_the_review_default_for_evolved_extensions(evolution, monkeypatch) -> None:
    """The deployment sets the harness requests defaults: the seed carries the ask
    command and the API skill, the notice offers the update, and a win that
    touches a code_extension waits for a promote. The recipe folds the two
    booleans into its seed, so they are read back as the entries they append.
    The two demo files run in auto, where an ask is refused, so they set none
    of the three. The ``evolution`` fixture puts the tutorial's harness package
    on the path the deployment's dotted references resolve through."""
    import os

    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "provider/model-a")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(EXAMPLE_DIR.parents[1]))
    config = load_config(EXAMPLE_DIR / "configs" / "deployment.yaml")
    assert config["data"]["training_mode"] == "hybrid"
    assert config["evolution"]["requests"] is True and config["evolution"]["version_check"] is True
    assert config["evolution"]["review_kinds"] == ["code_extension"]
    for name in ("serve", "serve-native"):
        demo = load_config(EXAMPLE_DIR / "configs" / f"{name}.yaml")
        assert demo["data"]["training_mode"] == "auto"
        assert not {"requests", "version_check", "review_kinds"} & set(demo["evolution"])
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(EXAMPLE_DIR / "configs")},
        default_runtime=_upstream_runtime(service_config_from_mapping(config)),
    )
    assert isinstance(built, CordisRecipe)
    assert built.review_kinds == ("code_extension",)
    assert [entry["id"] for entry in built.seed] == [
        "answer-style",
        "reef-version-check",
        "reef-requests",
        "reef-pi-extension-api",
    ]
    assert [(entry["name"], entry["config"]["name"]) for entry in built.seed[1:]] == [
        ("code_extension", "reef-version-check"),
        ("code_extension", "reef-requests"),
        ("skill", "reef-pi-extension-api"),
    ]


def test_native_example_recipe_renders_its_seed_as_the_base_files(native_evolution, tmp_path, monkeypatch) -> None:
    """The seed a deployment ships is what a fresh scenario serves, rendered once by the recipe."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve-native.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))
    from reef.service.assembly import _upstream_runtime

    service = service_config_from_mapping(config)
    built = CordisRecipe.from_environment(
        {}, config=load_recipe_config(materialized), runtime=_upstream_runtime(service)
    )
    files = built.base_artifact_files()
    assert files is not None
    assert {"native/tools/read_file.py", "native/graphs/main.json", "native/skills/answer-style/SKILL.md"} <= set(
        files
    )
    assert "upstream" not in files["native/models.json"]  # the seed carries no provider
    info = built.build_surface("demo").harness
    assert info is not None
    assert [entry["id"] for entry in info.seed_entries][:3] == ["read_file", "write_file", "run_bash"]
    assert info.served_model == "qwen3-8b"
    assert (
        CordisRecipe.from_environment(
            {},
            config={**load_recipe_config(materialized), "evolution": {**recipe_sections["evolution"], "seed": []}},
            runtime=_upstream_runtime(service),
        ).base_artifact_files()
        is None
    )


# -- the replay page --------------------------------------------------------


def _session_events(session: str, release: str, stages, tool: str | None = None) -> list[dict]:
    events = [
        {
            "type": "session",
            "seq": 0,
            "time": 1000,
            "data": {"session": session, "release_id": release, "model": "m", "tools": ["read_file"], "agent": "root"},
        },
        {"type": "turn/start", "seq": 1, "time": 1001, "data": {"turn": 1, "prompt": "go"}},
    ]
    seq = 2
    for index, stage in enumerate(stages):
        events.append(
            {
                "type": "stage/enter",
                "seq": seq,
                "time": 1002 + index,
                "data": {"step": index, "stage": stage, "kind": "model"},
            }
        )
        seq += 1
        if tool and index == 0:
            events.append(
                {
                    "type": "tool/call",
                    "seq": seq,
                    "time": 1002 + index,
                    "data": {"step": index, "name": tool, "call_id": "c", "arguments": "{}"},
                }
            )
            seq += 1
        events.append(
            {
                "type": "stage/exit",
                "seq": seq,
                "time": 1003 + index,
                "data": {"step": index, "stage": stage, "outcome": "text", "to": "done"},
            }
        )
        seq += 1
    events.append({"type": "turn/end", "seq": seq, "time": 1100, "data": {"turn": 1, "reason": {"kind": "completed"}}})
    return events


def test_replay_collects_a_run_and_renders_one_self_contained_page(tmp_path: Path, monkeypatch) -> None:
    replay = _method(monkeypatch, "replay")
    work = tmp_path / "work"
    native = work / "tree" / "native"
    (native / "sessions" / "s1").mkdir(parents=True)
    (native / "sessions" / "s2").mkdir(parents=True)
    (work / "agent-record").mkdir()
    graph = {
        "name": "main",
        "start": "think",
        "stages": {"think": {"kind": "model"}, "done": {"kind": "end"}},
        "edges": [{"from": "think", "when": "text", "to": "done"}],
    }
    seed = [
        {"id": "read_file", "name": "native_tool", "config": {"name": "read_file"}},
        {"id": "main", "name": "native_graph", "config": graph},
    ]
    rule = {"id": "answer-format", "name": "rules", "config": {"text": "one line"}}
    (native / "tree.json").write_text(json.dumps([*seed, rule]))
    commit = {
        "recorded_at": 1050.0,
        "artifact_ref": {"release_id": "r2", "parent_release_id": "r1"},
        "algorithm_state": {"entries": [*seed, rule], "steps": 1},
        "metrics": {
            "steps": 1,
            "published": True,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "candidate_score": 2.0,
            "current_score": 0.0,
            "mutations": [
                {"op": "create", "id": "answer-format", "options": {"name": "rules", "config": {"text": "one line"}}}
            ],
            "proposal": {"id": "p1", "session": "s1", "release_id": "r1"},
            "selection": {"reason": "candidate won 2 task pairings and lost 0"},
        },
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(commit) + "\n")
    for name, release, tool in (("s1", "r1", "harness_propose"), ("s2", "r2", None)):
        events = _session_events(name, release, ["think"], tool)
        (native / "sessions" / name / "session.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (native / "sessions" / "serve.jsonl").write_text(
        json.dumps(
            {
                "type": "harness/mount",
                "seq": 0,
                "time": 999,
                "data": {"release_id": "r1", "source": "boot", "entries": 2},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "harness/mount",
                "seq": 1,
                "time": 1051,
                "data": {"release_id": "r2", "parent_release_id": "r1", "source": "release", "entries": 3},
            }
        )
        + "\n"
    )

    data = replay.collect(work)
    assert [(r["step"], r["kind"], r["release_id"]) for r in data["releases"]] == [
        (0, "seed", "r1"),
        (1, "published", "r2"),
    ]
    seed_release, published = data["releases"]
    # The seed is the published state with the step's creates undone; the diff names what the step added.
    assert [e["id"] for e in seed_release["entries"]] == ["read_file", "main"]
    assert published["diff"] == {"added": [{"id": "answer-format", "kind": "rules"}], "updated": [], "removed": []}
    assert published["proposal"] == {"id": "p1", "session": "s1", "release_id": "r1"}
    assert published["result"]["wins"] == 2 and published["result"]["reason"].startswith("candidate won")
    assert [(s["session"], s["release_id"], len(s["events"])) for s in data["sessions"]] == [
        ("s1", "r1", 6),
        ("s2", "r2", 5),
    ]
    assert [e["type"] for e in data["process"]] == ["harness/mount", "harness/mount"]

    page = replay.render(
        {
            **data,
            "sessions": [
                {
                    **data["sessions"][0],
                    "events": [
                        *data["sessions"][0]["events"],
                        {
                            "type": "user/message",
                            "seq": 9,
                            "time": 1200,
                            "data": {"content": "<!--<script>x</script>"},
                        },
                    ],
                }
            ],
        }
    )
    assert page.startswith("<title>Harness Evolution Replay</title>")
    assert '<script id="data" type="application/json">' in page and "harness_propose" in page
    # No "<" survives inside the data element: neither a closing tag nor the comment opener that would keep the
    # element from closing and swallow the page's own script.
    payload = page.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in payload and "\\u003c!--\\u003cscript" in payload
    assert page.count("</script>") == 2
    assert "http://" not in page.split("</style>")[0] and "cdn" not in page
    out = tmp_path / "replay.html"
    assert replay.main([str(work), str(out)]) == 0 and out.read_text(encoding="utf-8") == replay.render(
        replay.collect(work)
    )

    # A first step that updated an entry: its previous options are not on the record, so the seed keeps the
    # published entry and the step's diff names it from the mutation.
    updated = {
        **commit,
        "metrics": {
            **commit["metrics"],
            "mutations": [{"op": "update", "id": "main", "options": {"name": "native_graph", "config": graph}}],
        },
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(updated) + "\n")
    first_update = replay.collect(work)
    assert first_update["releases"][1]["diff"] == {
        "added": [],
        "updated": [{"id": "main", "kind": "native_graph"}],
        "removed": [],
    }
    assert [e["id"] for e in first_update["releases"][0]["entries"]] == ["read_file", "main", "answer-format"]

    # A rollback row is not a step: it is shown as its own kind, with no verdict.
    rollback = {
        "recorded_at": 1060.0,
        "operation": "rollback",
        "step": 2,
        "artifact_ref": {"release_id": "r1", "parent_release_id": "r2"},
        "metrics": None,
        "algorithm_state": None,
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(commit) + "\n" + json.dumps(rollback) + "\n")
    with_rollback = replay.collect(work)
    assert [(r["kind"], r["step"], r["release_id"]) for r in with_rollback["releases"]] == [
        ("seed", 0, "r1"),
        ("published", 1, "r2"),
        ("rollback", 2, "r1"),
    ]
    assert with_rollback["releases"][2]["result"] is None
    assert with_rollback["releases"][2]["entries"] == with_rollback["releases"][1]["entries"]

    # An empty work directory still renders a page.
    empty = replay.collect(tmp_path / "nothing")
    assert empty == {"releases": [], "sessions": [], "process": [], "seed_entries": []}
    assert '<script id="data" type="application/json">' in replay.render(empty)


# -- propose: a request whose review finds the answer short is answered again ------

SHORT = {"result": "partial", "delivers": True, "covered": ["the tests run first"], "uncovered": ["no retry"]}
SUBSTITUTE = {
    "result": "partial",
    "delivers": False,
    "covered": [],
    "uncovered": ["a rule tells the model about the session instead of opening it"],
}


def test_a_short_review_sends_the_request_back_with_its_findings_and_keeps_the_complete_answer(evolution) -> None:
    model = Model(
        designed(skill("run-tests")),
        json.dumps(SHORT),
        designed(skill("run-tests", "# improved\n\nretry once"), design="Second design."),
        json.dumps(REVIEW),
    )
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [(m.op, m.id) for m in proposal.mutations] == [("create", "run-tests")]
    assert proposal.mutations[0].options["config"]["text"].endswith("retry once")
    assert proposal.notes == {"design": "Second design.", "review": REVIEW, "attempts": 2}
    # plan, answer, review, answer again, review: a complete review ends the loop.
    assert model.calls == 5
    retry_prompt = model.prompts[3]
    assert retry_prompt.startswith(model.prompts[1])
    assert "An earlier answer to this request was reviewed and fell short." in retry_prompt
    assert f"Its design was:\n{DESIGN}\n" in retry_prompt and "- no retry\n" in retry_prompt
    assert "It did not deliver the behavior at all" not in retry_prompt
    # The prompts say what delivering means: no substitute in place of the behavior.
    assert "a rule, a note or a workaround that only imitates the behavior is not an answer" in model.prompts[1]
    assert '"delivers": true or false' in model.prompts[2]


def test_answers_that_never_deliver_skip_the_step_after_the_last_attempt_saying_why(evolution) -> None:
    answer = designed(rules("resume", "Resume the last session."))
    model = Model(answer, json.dumps(SUBSTITUTE), answer, json.dumps(SUBSTITUTE), answer, json.dumps(SUBSTITUTE))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.mutations == ()
    assert proposal.notes["failure"] == (
        "the change does not deliver the request: a rule tells the model about the session instead of opening it"
    )
    assert proposal.notes["attempts"] == 3
    assert model.calls == 1 + 2 * 3
    assert "It did not deliver the behavior at all" in model.prompts[-2]
    # A failed call after a substitute reports the substitute, which says more than the failed call.
    model = Model(answer, json.dumps(SUBSTITUTE), ModelBindingError("endpoint down"))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.mutations == () and proposal.notes["failure"].startswith("the change does not deliver")


def test_the_kept_answer_is_the_delivering_one_with_the_fewest_uncovered_points(evolution) -> None:
    two_gaps = {**SHORT, "uncovered": ["no retry", "no log"]}
    model = Model(
        designed(skill("first")),
        json.dumps(two_gaps),
        designed(rules("second", "A substitute.")),
        json.dumps(SUBSTITUTE),
        designed(skill("third")),
        json.dumps(SHORT),
    )
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [m.id for m in proposal.mutations] == ["third"]
    assert proposal.notes["review"] == SHORT and proposal.notes["attempts"] == 3
    # A failed call after a delivering answer keeps that answer rather than skipping the step.
    model = Model(designed(skill("first")), json.dumps(two_gaps), ModelBindingError("endpoint down"))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [m.id for m in proposal.mutations] == ["first"] and proposal.notes["attempts"] == 2
