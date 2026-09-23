"""What the text proposer is told about each harness it writes entries for, beside pi.

pi's surface is its extension API, which the proposer reads from the tree's
own reference skill. Every other adapter takes rules, skills, commands and,
where its config can enforce a behavior, a few config keys; these facts say
how a person types a command there, what the command file holds, which tools
the harness has of its own (web search among them) and how a mode is built,
so a request is answered with that harness's own means, and its review
judges the answer by them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessFacts:
    """One harness's surface as a request prompt and its review describe it.

    ``command`` says how a person invokes an agent_command and what its text
    holds; ``tools`` names the harness's own tools; ``mode`` says how a mode
    is turned on, shown and turned off there. ``config_keys`` are the top
    level keys a request's config entry may set in the primary config file,
    and ``config_example`` shows one."""

    title: str
    command: str
    tools: str
    mode: str
    config_keys: tuple[str, ...] = ()
    config_example: str = ""


_CONVERSATION_MODE = (
    "{title} keeps no mode state of its own and a command cannot take a tool away, so a mode here is guidance the "
    "model follows while every tool stays in its list: the command that turns it on says so in its reply and "
    "names how to leave it, the rules say how the agent behaves while it is on and that every reply shows it is "
    "on, and the same command with the word off turns it off. The design, the How to use paragraph and the "
    "command's reply say the model follows the mode and never claim the other tools are unavailable. A request for "
    "a hard restriction (no other tool or skill may run at all) is only partly met this way: the review lists that "
    "point as uncovered."
)

FACTS = {
    "claude": HarnessFacts(
        title="Claude Code",
        command=(
            "A person types an agent_command as /<id> in the session. Its text is a Claude Code command file: "
            "YAML frontmatter with description (and allowed-tools, a list such as [WebSearch], which pre-approves "
            "those tools for the command's own turn), then the prompt, where $ARGUMENTS stands for what the person "
            "typed after /<id>."
        ),
        tools=(
            "Bash, Read, Edit, Write, NotebookEdit, Agent, Skill, Workflow, AskUserQuestion, WebFetch (reads one URL) "
            "and WebSearch (searches the web; no key needed)"
        ),
        mode=_CONVERSATION_MODE.format(title="Claude Code"),
        config_keys=("permissions",),
        config_example=(
            '{"target": "primary", "data": {"permissions": {"allow": ["WebSearch"]}}} lets WebSearch run without '
            "asking the person each time"
        ),
    ),
    "codex": HarnessFacts(
        title="Codex CLI",
        command=(
            "Codex has no custom slash commands: a typed /<word> that is not one of its own commands is refused "
            "before the model sees it. An agent_command renders as a skill, which a person types as $<id> in the "
            "session, so every place that tells the person how to use it says $<id>, never /<id>. Its text is a "
            "SKILL.md: YAML frontmatter with name and description, then the instructions; what the person typed "
            "after $<id> stays in their message."
        ),
        tools=(
            "exec_command (a shell, sandboxed to the workspace without network), write_stdin, request_user_input, "
            "view_image, multi_agent_v1, the goal tools and web_search (the hosted web search, off unless the config "
            "sets web_search)"
        ),
        mode=_CONVERSATION_MODE.format(title="Codex").replace("the same command", "the same skill"),
        config_keys=("web_search",),
        config_example=(
            '{"target": "primary", "data": {"web_search": "live"}} gives sessions the web_search tool (evaluation '
            "episodes keep it off). Web search comes only from this entry: it takes effect in a new reef-codex session "
            "after reef-codex update, and the person never edits a config file, since every install writes it again"
        ),
    ),
    "opencode": HarnessFacts(
        title="opencode",
        command=(
            "A person types an agent_command as /<id>. Its text is an opencode command file: YAML frontmatter with "
            "description, and agent: <name> to run the command with that agent and switch the session to it, then "
            "the prompt, where $ARGUMENTS stands for what the person typed after /<id>. Quote a frontmatter value "
            "that holds ': ' (description: \"Chat: web search only\")."
        ),
        tools=(
            "bash, read, edit, write, glob, grep, task, todowrite, skill, question, webfetch (reads one URL) and "
            "websearch (searches the web; no key needed)"
        ),
        mode=(
            "A mode is an agent. A config entry defines it in opencode.json with a permission map, which limits the "
            "tools the model gets and so enforces the mode, and a prompt, which replaces opencode's own system "
            "prompt while the agent runs; an agent_command with agent: <name> in its frontmatter switches the "
            "session to it. Write both in the same reply. Put the restriction in the chat agent's prompt, not in rules "
            "or the command text, so it ends when the session leaves the agent. The person leaves the mode with "
            "/agents, choosing build (Tab cycles the primary agents, build first); say so in the command's reply and "
            "in How to use."
        ),
        config_keys=("agent",),
        config_example=(
            '{"target": "primary", "data": {"agent": {"chat": {"mode": "primary", "description": "Chat, web search '
            'only", "prompt": "<how the agent behaves>", "permission": {"*": "deny", "websearch": "allow"}}}}} is '
            "an agent whose only tool is websearch"
        ),
    ),
    "hermes": HarnessFacts(
        title="Hermes Agent",
        command=(
            "A person types an agent_command as /<id> in the session. Its text is a SKILL.md: YAML frontmatter with "
            "name and description, then the instructions; what the person typed after /<id> reaches the model at "
            "the end of the message, after the line 'The user has provided the following instruction alongside "
            "the skill invocation:'."
        ),
        tools=(
            "terminal, process, execute_code, read_file, write_file, patch, search_files, skill_view, skills_list, "
            "skill_manage, delegate_task, browser_exec, memory, todo, web_extract (reads one page) and web_search "
            "(searches the web; no key needed)"
        ),
        mode=_CONVERSATION_MODE.format(title="Hermes"),
    ),
    "dsh": HarnessFacts(
        title="DeepSeek Harness (dsh)",
        command=(
            "A person uses dsh through its web UI (reef-dsh web) and types an agent_command as /<id>. Its text is a "
            "SKILL.md: YAML frontmatter with name and description, then the instructions, which dsh adds after the "
            "person's message; that message holds what they typed after /<id>."
        ),
        tools=(
            "bash (it writes only inside the working directory and the temp directories), read, write, edit, glob, "
            "grep, skill, subagent, subagent_fork, workflow, ask_user_question, todo_write, web_fetch (reads one URL) "
            "and web_search, which needs DEEPSEEK_API_KEY on the person's machine: a "
            'change that relies on web_search declares {"name": "DEEPSEEK_API_KEY", "kind": "env", "prompt": '
            '"<one sentence>"} as a requires item'
        ),
        mode=_CONVERSATION_MODE.format(title="dsh"),
    ),
}


def harness_facts(adapter: str) -> HarnessFacts | None:
    """The facts for ``adapter``; ``None`` for pi, whose surface is its extension API, and for an adapter the
    proposer knows nothing about beyond its entry kinds."""
    return FACTS.get(adapter)


__all__ = ["FACTS", "HarnessFacts", "harness_facts"]
