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
def test_a_prompt_level_mode_keeps_its_state_in_the_conversation_and_a_hard_restriction_is_a_limit(
    adapter: str,
) -> None:
    """On a harness whose command cannot take a tool away, the proposer says the mode is followed by the model while
    every tool stays offered, keeps the mode's state in the conversation (a marker file shared every session and
    made the rules call a tool on every turn), and the review lists a hard restriction under limits."""
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(request_reply(CHAT, RULES), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter=adapter)
    assert "every tool stays in its list" in model.prompt
    assert "never claim the other tools are unavailable" in model.prompt
    assert "never in a file or a marker a tool writes or reads" in model.prompt
    assert "the rules make no tool call on any turn" in model.prompt
    # A skill the person types still loads its text on these harnesses: the mode declines it, and each harness says
    # how the loaded text shows in the message.
    assert "the model declines a skill that the person's message loads, other than the mode's own command" in (
        model.prompt
    )
    marker = {"claude": "Base directory for this skill:", "codex": "<skill> block", "hermes": "[IMPORTANT: The user"}
    assert marker.get(adapter, '<skill_content name="<name>">') in model.prompt
    assert "tries the tool the mode allows before it refuses a question that tool can answer" in model.prompt
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    assert "the review lists that point under limits" in text and 'goes in a "limits" list' in text
    assert "it is the behavior itself, not a substitute" not in model.prompt


def test_claude_permissions_from_a_request_may_only_pre_approve_web_tools() -> None:
    """Every session of a release runs under its permissions: an answer that allows a shell, a wildcard or sets
    another permissions key is written again, and the one that allows WebSearch is kept."""

    def permissions(value: dict) -> dict:
        return {
            "id": "chat-permissions",
            "name": "config",
            "config": {"target": "primary", "data": {"permissions": value}},
        }

    complete = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    widened = permissions({"allow": ["Bash(*)", "mcp__server"], "defaultMode": "bypassPermissions"})
    model = Model(request_reply(CHAT, widened), request_reply(CHAT, permissions({"allow": ["WebSearch"]})), complete)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="claude")
    kept = {m.id: m.options for m in proposal.mutations}
    assert kept["chat-permissions"]["config"]["data"] == {"permissions": {"allow": ["WebSearch"]}}
    (dropped,) = proposal.notes["dropped_attempts"]
    assert "a request may not set permissions.defaultMode" in dropped
    assert "not 'Bash(*)'" in dropped and "not 'mcp__server'" in dropped
    assert "WebFetch(domain:<host>)" in model.prompt


def test_dsh_names_its_setup_as_the_way_to_give_the_web_search_key() -> None:
    model = Model(request_reply(RULES))
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="dsh")
    assert "its How to use names reef-dsh setup as the way to give the key" in model.prompt


def test_the_tool_lists_are_what_a_session_offers_and_claude_replaces_every_arguments() -> None:
    """The tools a recorded reef-claude session (Claude Code 2.1.257) and reef-hermes session (v2026.8.31) offered,
    all of them, so a mode's wording can name what it keeps and what it declines."""
    claude = [name.split(" (")[0] for name in FACTS["claude"].tools.replace(" and ", ", ").split(", ")]
    hermes = [name.split(" (")[0] for name in FACTS["hermes"].tools.replace(" and ", ", ").split(", ")]
    assert len(claude) == 25 and {"Bash", "Skill", "TaskOutput", "WebSearch"} <= set(claude)
    assert len(hermes) == 21 and {"session_search", "text_to_speech", "vision_analyze", "web_search"} <= set(hermes)
    assert "replaces every $ARGUMENTS in the file" in FACTS["claude"].command


def test_opencode_enters_a_mode_by_selecting_its_agent_and_leaves_it_through_agents() -> None:
    """A command's agent: runs only that command's turn, and a new session's first message sets the session's agent:
    so the mode is entered with /agents or with its command as a new session's first message, and left with
    /agents (Tab reaches plan first); a leave command with agent: build does not leave, so none is written."""
    model = Model(request_reply(RULES))
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode")
    assert "enforces the mode" in model.prompt and "every tool stays in its list" not in model.prompt
    assert "agent: <name> to run that command's own turn with that agent" in model.prompt
    assert "as the first message of a new session, which starts the session in that agent" in model.prompt
    assert "/agents, choosing build" in model.prompt and "Tab from the mode's agent reaches plan first" in model.prompt
    assert "A command with agent: build does not leave the mode, so write no leave command" in model.prompt
    assert "The restriction's wording lives only in that agent's own prompt, never in rules" in model.prompt
    assert "switch the session to it" not in model.prompt and "second agent_command that leaves" not in model.prompt
    assert "Tab cycles the primary agents, build first" not in model.prompt
    assert "Quote a frontmatter value that holds ': '" in model.prompt


def _agent(permission: dict | None) -> dict:
    agent = {"mode": "primary", "prompt": "Only search the web."}
    if permission is not None:
        agent["permission"] = permission
    return {"id": "chat-agent", "name": "config", "config": {"target": "primary", "data": {"agent": {"chat": agent}}}}


def test_an_opencode_agent_without_a_permission_map_is_written_again() -> None:
    """An agent a request's config entry defines with no permission map is offered every tool, so the mode it builds
    restricts nothing: the answer goes back to the model with that reason, and the answer with a map is kept."""
    command = {
        "id": "chat",
        "name": "agent_command",
        "config": {"name": "chat", "text": "---\nagent: chat\n---\nChat."},
    }
    complete = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    mapped = {"*": "deny", "websearch": "allow"}
    model = Model(request_reply(_agent(None), command), request_reply(_agent(mapped), command), complete)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode")
    assert [m.id for m in proposal.mutations] == ["chat-agent", "chat"]
    assert proposal.mutations[0].options["config"]["data"]["agent"]["chat"]["permission"] == mapped
    reason = (
        "agent 'chat' has no permission map, so it is offered every tool: add agent.chat.permission, for example "
        '{"*": "deny", "websearch": "allow"}'
    )
    assert proposal.notes["dropped_attempts"] == [f"answer 1: {reason}"]
    (retry,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert reason in retry
    # An agent the tree already gives a map keeps it: the entry may change the prompt alone.
    tree = [
        {
            "id": "chat-agent",
            "name": "config",
            "config": {
                "target": "primary",
                "data": {"agent": {"chat": {"mode": "primary", "prompt": "Chat.", "permission": mapped}}},
            },
        }
    ]
    model = Model(request_reply(_agent(None)), complete)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode", entries=tree)
    assert [m.id for m in proposal.mutations] == ["chat-agent"] and "dropped_attempts" not in proposal.notes


def test_a_point_the_harness_notes_put_out_of_reach_is_a_limit_that_starts_no_retry() -> None:
    """A review whose only open points are limits ends the loop at the first answer and keeps them apart from the
    uncovered gaps; an uncovered gap still sends the request back."""
    limited = json.dumps(
        {"result": "partial", "delivers": True, "covered": ["chat"], "uncovered": [], "limits": ["no tool lockout"]}
    )
    model = Model(request_reply(CHAT, RULES), limited)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="hermes")
    assert model.answered == 2 and "attempts" not in proposal.notes
    assert proposal.notes["review"]["limits"] == ["no tool lockout"] and proposal.notes["review"]["uncovered"] == []
    gap = json.dumps(
        {"result": "partial", "delivers": True, "covered": [], "uncovered": ["no off"], "limits": ["no tool lockout"]}
    )
    model = Model(request_reply(CHAT, RULES), gap)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="hermes")
    assert proposal.notes["attempts"] == 3
    (retry, *_) = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert "- no off" in retry and "no tool lockout" not in retry


def test_terminus_gets_its_own_facts_and_no_setup_command() -> None:
    """terminus has no session and no install: the prompt describes its surface, names no reef-terminus setup, and the
    review judges commands by the harness rubric, not pi's."""
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(request_reply(CHAT, RULES), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="terminus")
    facts = FACTS["terminus"]
    assert f"This harness is {facts.title}" in model.prompt and facts.tools in model.prompt
    assert "reef-terminus setup" not in model.prompt and "this harness has no setup command" in model.prompt
    # Its runs happen in a task's Linux container: the prompt says so instead of the person's platforms.
    assert facts.machine in model.prompt and "macOS, Linux or Windows under WSL 2" not in model.prompt
    assert evolution.request_kinds("terminus") == ("skill", "rules", "agent_command")
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    assert "This harness is Terminus 2" in text and "pi.registerCommand" not in text
