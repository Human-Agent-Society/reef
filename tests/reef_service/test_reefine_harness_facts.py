"""The text proposer off pi: each harness's own facts reach the request, its plan and its review.

On pi the proposer writes extensions against the API reference; on every
other adapter it writes rules, skills and commands, plus a config entry
where the harness's config can enforce a behavior. These tests pin what the
prompts say per adapter (the rules file, how a command is typed, the tools,
how a mode is built), which config keys an entry may set, and the reasons a
step records when a reply gives nothing to apply.
"""

from __future__ import annotations

import json

import pytest
from reef_service.test_harness_example import NODES, PLAN_MARKER, REQUEST, Model, failure_of, request_reply

from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.recipe.reefine import evolution
from reef.recipe.reefine.harness_facts import FACTS

CHAT = {"id": "chat", "name": "agent_command", "config": {"name": "chat", "text": "# chat\n\nChat mode is on."}}
RULES = {"id": "chat-rules", "name": "rules", "config": {"text": "While chat mode is on, only search the web."}}


@pytest.mark.parametrize(
    ("adapter", "rules_file", "typed"),
    [
        ("claude", "CLAUDE.md", "/<id>"),
        ("codex", "AGENTS.md", "$<id>"),
        ("opencode", "AGENTS.md", "/<id>"),
        ("hermes", "SOUL.md", "/<id>"),
        ("dsh", "AGENTS.md", "/<id>"),
    ],
)
def test_the_request_prompt_names_the_harness_its_rules_file_its_command_and_its_tools(
    adapter: str, rules_file: str, typed: str
) -> None:
    model = Model(request_reply(RULES))
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter=adapter)
    prompt = model.prompt
    facts = FACTS[adapter]
    assert f"This harness is {facts.title}" in prompt and facts.command in prompt and facts.tools in prompt
    assert f"markdown appended to {rules_file}" in prompt and f"as {typed}" in prompt
    assert "- code_extension:" not in prompt and "pi.registerCommand" not in prompt
    # Only an extension branches on the platform; the command still names what the user must set up.
    assert "process.platform" not in prompt and f"reef-{adapter} setup" in prompt
    # The plan call hears the harness's own tools, so web search is not a tool the harness lacks.
    (plan,) = [text for text in model.prompts if PLAN_MARKER in text]
    assert f"Its own tools: {facts.tools}." in plan


def test_a_config_entry_is_offered_only_where_it_enforces_a_behavior_and_keeps_to_its_keys() -> None:
    assert evolution.request_kinds("opencode") == ("skill", "rules", "agent_command", "config")
    assert evolution.request_kinds("hermes") == ("skill", "rules", "agent_command")
    assert evolution.request_kinds("pi") == ("skill", "rules", "agent_command", "code_extension")
    agent = {"chat": {"mode": "primary", "prompt": "Chat.", "permission": {"*": "deny", "websearch": "allow"}}}
    entries = (
        {"id": "chat-agent", "name": "config", "config": {"target": "primary", "data": {"agent": agent}}},
        # A provider redirects the served model and a second target is not the harness config: both dropped.
        {"id": "hijack", "name": "config", "config": {"target": "primary", "data": {"provider": {"x": {}}}}},
        {"id": "elsewhere", "name": "config", "config": {"target": "env", "data": {"agent": agent}}},
        CHAT,
    )
    model = Model(request_reply(*entries))
    mutations = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode").mutations
    assert [(m.op, m.id, m.options["name"]) for m in mutations] == [
        ("create", "chat-agent", "config"),
        ("create", "chat", "agent_command"),
    ]
    assert mutations[0].options["config"] == {"target": "primary", "data": {"agent": agent}}
    assert "only the top level keys agent" in model.prompt and "opencode.json" in model.prompt
    # Where the harness has no config keys on the list, a config entry is not an entry at all.
    model = Model(request_reply(entries[0]))
    assert failure_of(evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="hermes")) == (
        "the reply holds no usable entry"
    )
    assert "- config:" not in model.prompt


def test_off_pi_a_step_the_harness_cannot_perform_is_named_undone_not_answered_with_an_extension() -> None:
    plan = json.dumps([{"step": "call the phone company", "needs_tool": True}])
    model = Model(request_reply(RULES), plan=plan)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="hermes")
    assert "- call the phone company" in model.prompt
    assert "No kind you may write adds a tool on this harness" in model.prompt
    assert "write a code_extension in this same reply" not in model.prompt
    model = Model(request_reply(RULES), plan=plan)
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert "write a code_extension in this same reply" in model.prompt


def test_the_review_judges_commands_by_the_harness_surface() -> None:
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(request_reply(CHAT, RULES), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="codex")
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    assert FACTS["codex"].command in text and FACTS["codex"].mode in text
    assert "pi.registerCommand" not in text
    model = Model(request_reply(CHAT, RULES), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    assert "pi.registerCommand" in text


def test_an_entry_dropped_for_its_id_is_named_in_the_notes_and_in_the_retry() -> None:
    """A command whose id is not its name is dropped, as ever; the step says so and the retry tells the model."""
    misnamed = {"id": "chat-mode", "name": "agent_command", "config": {"name": "chat", "text": "# chat\n"}}
    partial = json.dumps({"result": "partial", "delivers": True, "covered": [], "uncovered": ["no /chat command"]})
    complete = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(request_reply(misnamed, RULES), partial, request_reply(CHAT, RULES), complete)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="hermes")
    assert [m.id for m in proposal.mutations] == ["chat", "chat-rules"]
    dropped = "agent_command 'chat-mode' was dropped: its id must equal its config name 'chat'"
    retry = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert retry and f"- {dropped}" in retry[0]


class _FilteredBinding(ModelBinding):
    """A served model whose provider filtered every reply: the text is a refusal, the response says why."""

    def chat(self, messages, *, timeout_s=None, **params) -> str:
        return "I'm sorry, but I cannot assist with that request."

    def last_response(self) -> dict[str, object] | None:
        return {"choices": [{"finish_reason": "content_filter", "message": {"content": "I'm sorry."}}]}


def test_a_reply_the_provider_filtered_says_so_instead_of_naming_no_entry() -> None:
    models = ModelBindings(served=_FilteredBinding(base_url="http://127.0.0.1:1", model="m"))
    assert failure_of(evolution.propose(NODES, (), models, requests=(REQUEST,), adapter="opencode")) == (
        "the provider refused the reply (content_filter)"
    )


@pytest.mark.parametrize("adapter", ["claude", "codex", "hermes", "dsh"])
def test_a_prompt_level_mode_is_named_guidance_and_a_hard_restriction_stays_uncovered(adapter: str) -> None:
    """On a harness whose command cannot take a tool away, the proposer says the mode is followed by the model while
    every tool stays offered, and the review lists a hard restriction as uncovered rather than calling it delivered."""
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(request_reply(CHAT, RULES), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter=adapter)
    assert "every tool stays in its list" in model.prompt
    assert "never claim the other tools are unavailable" in model.prompt
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    assert "the review lists that point as uncovered" in text
    assert "it is the behavior itself, not a substitute" not in model.prompt


def test_opencode_enforces_a_mode_with_an_agent_and_leaves_it_through_agents() -> None:
    model = Model(request_reply(RULES))
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode")
    assert "enforces the mode" in model.prompt and "every tool stays in its list" not in model.prompt
    assert (
        "/agents, choosing build" in model.prompt and "Put the restriction in the chat agent's prompt" in model.prompt
    )
    assert "Quote a frontmatter value that holds ': '" in model.prompt
