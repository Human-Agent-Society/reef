"""The text proposer off pi: each harness's own facts reach the request, its plan and its review.

On pi the proposer writes extensions against the API reference; on every
other adapter it writes rules, skills and commands, plus the primary config
file where the harness's config can enforce a behavior. These tests pin what
the prompts say per adapter (the rules file, how a command is typed, the
tools, how a mode is built), which config keys a file may set, and the
reasons a step records when a reply gives nothing to apply.
"""

from __future__ import annotations

import json
import re

import pytest
from tests.recipes.reefine.reefine_support import PLAN_MARKER, TREE, Model, answer, designed, extension, failure_of

from recipes.beta.reefine import proposer
from recipes.beta.reefine.harness_facts import harness_facts
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import HarnessTree, TreeKind

CHAT = {"commands/chat.md": "# chat\n\nChat mode is on."}
RULES = {"rules.md": "While chat mode is on, only search the web.\n"}
COMPLETE = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})


def review_prompt(model: Model) -> str:
    (text,) = [prompt for prompt in model.prompts if "now you review the change" in prompt]
    return text


@pytest.mark.parametrize(
    ("adapter", "rules_file", "typed"),
    [
        ("claude", "CLAUDE.md", "/<name>"),
        ("codex", "AGENTS.md", "$<name>"),
        ("opencode", "AGENTS.md", "/<name>"),
        ("hermes", "SOUL.md", "/<name>"),
        ("dsh", "AGENTS.md", "/<name>"),
    ],
)
def test_the_request_prompt_names_the_harness_its_rules_file_its_command_and_its_tools(
    adapter: str, rules_file: str, typed: str
) -> None:
    model = Model(designed(RULES))
    answer(model, adapter)
    prompt = model.prompt
    facts = harness_facts(adapter)
    assert f"This harness is {facts.title}" in prompt and facts.command in prompt and facts.tools in prompt
    assert f"the markdown every session reads ({rules_file})" in prompt and f"as {typed}" in prompt
    assert "- extensions/<name>.ts:" not in prompt and "pi.registerCommand" not in prompt
    # Only an extension branches on the platform; the prompt still names what the user must set up.
    assert "process.platform" not in prompt and "name anything platform specific the user must set up" in prompt
    # The plan call hears the harness's own tools, so web search is not a tool the harness lacks.
    (plan,) = [text for text in model.prompts if PLAN_MARKER in text]
    assert f"Its own tools: {facts.tools}." in plan


def test_a_config_file_is_offered_only_where_it_enforces_a_behavior_and_keeps_to_its_keys() -> None:
    skill_rules_commands = (TreeKind.SKILL, TreeKind.RULES, TreeKind.AGENT_COMMAND)
    assert proposer.request_kinds("opencode") == (*skill_rules_commands, TreeKind.CONFIG)
    assert proposer.request_kinds("hermes") == skill_rules_commands
    assert proposer.request_kinds("pi") == (*skill_rules_commands, TreeKind.CODE_EXTENSION)
    agent = {"chat": {"mode": "primary", "prompt": "Chat.", "permission": {"*": "deny", "websearch": "allow"}}}
    # A second config target is not the harness config: dropped, while the primary config and the command land.
    files = {"config/primary.json": json.dumps({"agent": agent}), "config/env.json": json.dumps({"agent": agent})}
    model = Model(designed(files, CHAT), COMPLETE)
    proposal = answer(model, "opencode")
    assert proposal.patch.changes == {"config/primary.json": files["config/primary.json"], **CHAT}
    assert proposal.notes["dropped"] == [
        "config/env.json was dropped: a request may change config/primary.json only, and never remove it"
    ]
    assert "change only the top level keys agent" in model.prompt and "opencode.json" in model.prompt
    # A provider redirects the served model: a config file that changes it is dropped whole.
    model = Model(designed({"config/primary.json": json.dumps({"provider": {"x": {}}})}, CHAT), COMPLETE)
    assert answer(model, "opencode").notes["dropped"] == [
        "config/primary.json was dropped: it changes provider; a request may change agent"
    ]
    # Where the harness has no config keys on the list, a config file is not a file it takes at all.
    model = Model(designed({"config/primary.json": files["config/primary.json"]}))
    assert failure_of(answer(model, "hermes")) == "the reply holds no usable file"
    assert "- config/primary.json:" not in model.prompt


def test_off_pi_a_step_the_harness_cannot_perform_is_named_undone_not_answered_with_an_extension() -> None:
    plan = json.dumps([{"step": "call the phone company", "needs_tool": True}])
    model = Model(designed(RULES), plan=plan)
    answer(model, "hermes")
    assert "- call the phone company" in model.prompt
    assert "No file you may write adds a tool on this harness" in model.prompt
    assert "write an extension in this same reply" not in model.prompt
    model = Model(designed(RULES), plan=plan)
    answer(model)
    assert "write an extension in this same reply" in model.prompt


def test_the_review_judges_commands_by_the_harness_surface() -> None:
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model, "codex")
    text = review_prompt(model)
    assert harness_facts("codex").command in text and harness_facts("codex").mode in text
    assert "pi.registerCommand" not in text
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model)
    assert "pi.registerCommand" in review_prompt(model)


def test_a_file_dropped_for_its_kind_is_named_in_the_notes_and_in_the_retry() -> None:
    """An extension off pi is dropped, as ever; the step says so and the retry tells the model."""
    loader = extension("chat", "export default function (pi) {}\n")
    partial = json.dumps({"result": "partial", "delivers": True, "covered": [], "uncovered": ["no /chat command"]})
    model = Model(designed(loader, RULES), partial, designed(CHAT, RULES), COMPLETE)
    proposal = answer(model, "hermes")
    assert proposal.patch.changes == {**CHAT, **RULES}
    dropped = "extensions/chat.ts was dropped: this harness takes skill, rules, agent_command"
    retry = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert retry and f"- {dropped}" in retry[0]


class FilteredBinding(ModelBinding):
    """A served model whose provider filtered every reply: the text is a refusal, the response says why."""

    def chat(self, messages, *, timeout_s=None, **params) -> str:
        return "I'm sorry, but I cannot assist with that request."

    def last_response(self) -> dict[str, object] | None:
        return {"choices": [{"finish_reason": "content_filter", "message": {"content": "I'm sorry."}}]}


def test_a_reply_the_provider_filtered_says_so_instead_of_naming_no_file() -> None:
    models = ModelBindings(served=FilteredBinding(base_url="http://127.0.0.1:1", model="m"))
    assert failure_of(answer(models, "opencode")) == "the provider refused the reply (content_filter)"


@pytest.mark.parametrize("adapter", ["claude", "codex", "hermes", "dsh"])
def test_a_prompt_level_mode_keeps_its_state_in_the_conversation_and_a_hard_restriction_is_a_limit(
    adapter: str,
) -> None:
    """On a harness whose command cannot take a tool away, the proposer says the mode is followed by the model while
    every tool stays offered, keeps the mode's state in the conversation (a marker file shared every session and
    made the rules call a tool on every turn), and the review lists a hard restriction under limits."""
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model, adapter)
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
    text = review_prompt(model)
    assert "the review lists that point under limits" in text and 'goes in a "limits" list' in text
    assert "it is the behavior itself, not a substitute" not in model.prompt


def test_claude_permissions_from_a_request_may_only_pre_approve_web_tools() -> None:
    """Every session of a release runs under its permissions: an answer that allows a shell, a wildcard or sets
    another permissions key is written again, and the one that allows WebSearch is kept."""

    def permissions(value: dict) -> dict[str, str]:
        return {"config/primary.json": json.dumps({"permissions": value})}

    widened = permissions({"allow": ["Bash(*)", "mcp__server"], "defaultMode": "bypassPermissions"})
    model = Model(designed(CHAT, widened), designed(CHAT, permissions({"allow": ["WebSearch"]})), COMPLETE)
    proposal = answer(model, "claude")
    kept = json.loads(proposal.patch.changes["config/primary.json"])
    assert kept == {"permissions": {"allow": ["WebSearch"]}}
    (dropped,) = proposal.notes["dropped_attempts"]
    assert "a request may not set permissions.defaultMode" in dropped
    assert "not 'Bash(*)'" in dropped and "not 'mcp__server'" in dropped
    assert "WebFetch(domain:<host>)" in model.prompt


def test_dsh_tells_the_person_to_set_the_web_search_key_before_starting() -> None:
    model = Model(designed(RULES))
    answer(model, "dsh")
    assert "its How to use tells the person to set that variable before starting dsh" in model.prompt


def test_the_tool_lists_are_what_a_session_offers_and_claude_replaces_every_arguments() -> None:
    """The tools a recorded Claude Code session (Claude Code 2.1.257) and hermes session (v2026.8.31) offered,
    all of them, so a mode's wording can name what it keeps and what it declines."""
    claude = [name.split(" (")[0] for name in harness_facts("claude").tools.replace(" and ", ", ").split(", ")]
    hermes = [name.split(" (")[0] for name in harness_facts("hermes").tools.replace(" and ", ", ").split(", ")]
    assert len(claude) == 25 and {"Bash", "Skill", "TaskOutput", "WebSearch"} <= set(claude)
    assert len(hermes) == 21 and {"session_search", "text_to_speech", "vision_analyze", "web_search"} <= set(hermes)
    assert "replaces every $ARGUMENTS in the file" in harness_facts("claude").command


def test_the_dsh_tools_are_the_26_a_pinned_dsh_session_offers() -> None:
    """dsh 0.1.2-alpha.5's standard preset offers these 26 tools (read from a recorded dsh session); a version
    bump fails here, so someone checks the list again."""
    offered = [
        "ask_user_question",
        "bash",
        "create_goal",
        "edit",
        "exit_plan_mode",
        "get_goal",
        "glob",
        "grep",
        "interrupt_agent",
        "job_kill",
        "job_list",
        "job_output",
        "list_agents",
        "ralph",
        "read",
        "read_image",
        "send_message",
        "skill",
        "subagent",
        "subagent_fork",
        "todo_write",
        "update_goal",
        "web_fetch",
        "web_search",
        "workflow",
        "write",
    ]
    install = get_adapter("dsh").install
    assert install is not None and install.version == "0.1.2-alpha.5"
    listed = set(re.findall(r"[a-z_]+", harness_facts("dsh").tools))
    assert len(offered) == 26 and set(offered) <= listed


def test_opencode_enters_a_mode_by_selecting_its_agent_and_leaves_it_through_agents_and_a_leave_turn() -> None:
    """A command's agent: runs only that command's turn, and a new session's first message sets the session's agent:
    so the mode is entered with /agents or with its command as a new session's first message. It is left with
    /agents (Tab reaches plan first) together with a leave command, since opencode tells the model nothing when the
    agent changes; what the person starts (a typed skill, @file, !command) still runs, and is a limit."""
    model = Model(designed(RULES))
    answer(model, "opencode")
    assert "enforces the mode for the model" in model.prompt and "every tool stays in its list" not in model.prompt
    assert "agent: <name> to run that command's own turn with that agent" in model.prompt
    assert "as the first message of a new session, which starts the session in that agent" in model.prompt
    assert "/agents, choosing build" in model.prompt and "Tab from the mode's agent reaches plan first" in model.prompt
    assert "together with a leave command, a command file with agent: build" in model.prompt
    assert "opencode tells the model nothing when the agent changes" in model.prompt
    # The leave command alone runs one build turn; only /agents switches the session back.
    assert "The leave command alone runs one build turn and leaves the chat agent selected" in model.prompt
    # The mid session note stays in the command's reply: in the agent's prompt it would end every later reply.
    assert "belongs in that command's own reply alone, never in the agent's prompt or in rules" in model.prompt
    assert "reads the turns another agent answered in the history as that agent's, not its own" in model.prompt
    assert "a !command the person runs still run in any agent" in model.prompt
    assert "the review lists those person paths under limits" in model.prompt
    assert "the entering command states no restriction" in model.prompt
    assert "The restriction's wording lives only in that agent's own prompt, never in rules" in model.prompt
    assert "Quote a frontmatter value that holds ': '" in model.prompt


def agent_config(permission: dict | None) -> dict[str, str]:
    agent: dict[str, object] = {"mode": "primary", "prompt": "Only search the web."}
    if permission is not None:
        agent["permission"] = permission
    return {"config/primary.json": json.dumps({"agent": {"chat": agent}})}


def test_an_opencode_agent_without_a_permission_map_is_written_again() -> None:
    """An agent a request's config file defines with no permission map is offered every tool, so the mode it builds
    restricts nothing: the answer goes back to the model with that reason, and the answer with a map is kept."""
    command = {"commands/chat.md": "---\nagent: chat\n---\nChat."}
    mapped = {"*": "deny", "websearch": "allow"}
    model = Model(designed(agent_config(None), command), designed(agent_config(mapped), command), COMPLETE)
    proposal = answer(model, "opencode")
    assert proposal.patch.changes == {**agent_config(mapped), **command}
    reason = (
        "agent 'chat' has no permission map, so it is offered every tool: add agent.chat.permission, for example "
        '{"*": "deny", "websearch": "allow"}'
    )
    assert proposal.notes["dropped_attempts"] == [f"answer 1: {reason}"]
    (retry,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert reason in retry
    # An agent the tree already gives a map keeps it: the file may change the prompt alone.
    held = json.dumps({"agent": {"chat": {"mode": "primary", "prompt": "Chat.", "permission": mapped}}})
    tree = HarnessTree({**TREE.files, "config/primary.json": held})
    model = Model(designed(agent_config(None)), COMPLETE)
    proposal = answer(model, "opencode", tree=tree)
    assert proposal.patch.changes == agent_config(None) and "dropped_attempts" not in proposal.notes


def test_a_point_the_harness_notes_put_out_of_reach_is_a_limit_that_starts_no_retry() -> None:
    """A review whose only open points are limits ends the loop at the first answer and keeps them apart from the
    uncovered gaps; an uncovered gap still sends the request back."""
    limited = json.dumps(
        {"result": "partial", "delivers": True, "covered": ["chat"], "uncovered": [], "limits": ["no tool lockout"]}
    )
    model = Model(designed(CHAT, RULES), limited)
    proposal = answer(model, "hermes")
    assert model.answered == 2 and "attempts" not in proposal.notes
    assert proposal.notes["review"]["limits"] == ["no tool lockout"] and proposal.notes["review"]["uncovered"] == []
    gap = json.dumps(
        {"result": "partial", "delivers": True, "covered": [], "uncovered": ["no off"], "limits": ["no tool lockout"]}
    )
    model = Model(designed(CHAT, RULES), gap)
    proposal = answer(model, "hermes")
    assert proposal.notes["attempts"] == 3
    (retry, *_) = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert "- no off" in retry and "no tool lockout" not in retry


def test_terminus_gets_its_own_facts_and_runs_in_its_task_container() -> None:
    """terminus has no session and no install: the prompt describes its surface and where it runs, and the review
    judges commands by the harness rubric, not pi's."""
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model, "terminus")
    facts = harness_facts("terminus")
    assert f"This harness is {facts.title}" in model.prompt and facts.tools in model.prompt
    # Its runs happen in a task's Linux container: the prompt says so instead of the person's platforms.
    assert facts.machine in model.prompt and "macOS, Linux or Windows under WSL 2" not in model.prompt
    assert proposer.request_kinds("terminus") == (TreeKind.SKILL, TreeKind.RULES, TreeKind.AGENT_COMMAND)
    text = review_prompt(model)
    assert "This harness is Terminus 2" in text and "pi.registerCommand" not in text


def test_the_review_object_has_a_limits_key_where_the_harness_notes_send_points_there() -> None:
    """The harness review commands send a point no answer can deliver to a limits list in the object below, so the
    object carries that key; pi's review has no harness notes and no limits key."""
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model, "dsh")
    assert '"uncovered": ["<one point per item>"], "limits": ["<one point per item>"]}' in review_prompt(model)
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model)
    assert '"limits"' not in review_prompt(model)


@pytest.mark.parametrize("adapter", ["claude", "codex", "opencode", "hermes", "dsh", "terminus"])
def test_off_pi_the_prompts_speak_of_files_not_of_extensions_and_process_env(adapter: str) -> None:
    """Only a pi extension reads process.env: off pi the request prompt and the review say how a file gets a value,
    and name no extension, no process.env and no pi variable."""
    model = Model(designed(CHAT, RULES), COMPLETE)
    answer(model, adapter)
    for text in (model.prompt, review_prompt(model)):
        assert "process.env" not in text and "PI_OFFLINE" not in text and "the extension" not in text
    assert "reaches the harness's environment at run time" in model.prompt
    assert "which the harness finds in its environment at run time" in model.prompt
    assert "When these files cannot deliver" in model.prompt
    assert "no file of the change asks for it itself" in model.prompt
    assert "a variable a file relies on that no requires item names" in review_prompt(model)


def test_a_prompt_level_mode_names_its_off_command_and_no_route_around_it() -> None:
    """A reply that declines names the mode's off command and never suggests a shell line the person runs, such as
    Claude Code's ! prefix."""
    model = Model(designed(RULES))
    answer(model, "claude")
    assert "names the mode's off command as the way out and never suggests a route around the mode" in model.prompt
    assert "Claude Code runs a line that starts with ! in the person's shell" in model.prompt


def test_terminus_says_a_run_has_no_reply_and_codex_says_web_search_is_on_in_every_session() -> None:
    """A terminus run's visible result is its files, reward and trajectory; codex's web search config is on in
    every session of the release, which How to use and the review limits say."""
    model = Model(designed(RULES))
    answer(model, "terminus")
    assert "A run has no reply a person reads" in model.prompt and "the verifier's reward" in model.prompt
    model = Model(designed(RULES))
    answer(model, "codex")
    assert "turns the hosted search on in every session of the release, not only while a mode is on" in model.prompt


def test_the_step_names_the_earlier_answer_it_kept() -> None:
    """When a later answer covers less than an earlier one, the earlier one stands and the notes say which it was;
    when the last answer is kept the notes stay as they were."""
    first = json.dumps({"result": "partial", "delivers": True, "covered": [], "uncovered": ["no off"]})
    worse = json.dumps({"result": "partial", "delivers": True, "covered": [], "uncovered": ["no off", "no header"]})
    model = Model(designed(CHAT, RULES), first, designed(CHAT, RULES), worse, designed(CHAT), worse)
    proposal = answer(model, "hermes")
    assert proposal.notes["attempts"] == 3 and proposal.notes["kept_attempt"] == 1
    assert proposal.notes["review"]["uncovered"] == ["no off"]
