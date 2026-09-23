---
name: reef-claude-harness-api
description: What a harness change for Claude Code can be made of (commands, skills, rules, settings, hooks) and how Claude Code reads each file
---

# The Claude Code harness surface

Reef renders the harness tree into the directory `CLAUDE_CONFIG_DIR` points at, and Claude Code reads
these files from it. Nothing else in the tree changes Claude Code's behavior; in particular a
code_extension renders to `plugins/<name>/index.js`, a file Claude Code does not load without a plugin
marketplace entry, so a change for Claude Code never uses that kind.

| entry kind      | file                          | read when                                        |
|-----------------|-------------------------------|--------------------------------------------------|
| `rules`         | `CLAUDE.md`                   | every session, as standing instructions           |
| `skill`         | `skills/<name>/SKILL.md`      | when the user types `/<name>` or the model picks it |
| `agent_command` | `commands/<name>.md`          | when the user types `/<name>`                     |
| `config`        | `settings.json` (merged)      | at session start: permissions, hooks, env         |

## Commands (`agent_command`) and skills (`skill`)

Both are a markdown prompt with YAML frontmatter, and both give the user a `/<name>` slash command that
shows in the `/` menu with its description. The difference: a skill's description is also in the model's
context, so the model can invoke it on its own; a command file is the older single-file form. Fields
Claude Code reads (all optional; a field it does not know is ignored without an error):

```
---
description: One line, shown in the / menu; for a skill, what tells the model when to use it
argument-hint: <what to type after the command>
allowed-tools: Bash(git status *) Read
disable-model-invocation: true    # only the user runs it (a command with side effects)
user-invocable: false             # only the model runs it (background knowledge)
model: sonnet
hooks:                            # registered when the skill is invoked, for the rest of the session
  PreToolUse:
    - matcher: Edit|Write
      hooks:
        - type: command
          command: sh -c '...'
---
The prompt. $ARGUMENTS is everything the user typed after the command; $0 and $1 are its first and
second word; ${CLAUDE_SESSION_ID} is the session id.
A line that starts with ! followed by a backticked shell command runs before the prompt is sent, and
its output (stderr included) replaces the line: !`git status --short`
```

`allowed-tools` grants the listed tool rules for the turn that invokes the command, without a permission
prompt; the grant ends with the user's next message. A `!` line runs through the Bash tool with a two
minute timeout and goes through the permission check like any Bash call, with two limits of its own:
the check refuses a command that expands a shell variable (`$CLAUDE_CONFIG_DIR`, `$HOME`) and refuses
any write outside the session's working directory, whatever `allowed-tools` says. So a `!` line reads
(`git status --short`, `date`), and state a command must keep is written by a hook (below). Claude Code
substitutes `${CLAUDE_SESSION_ID}` and `${CLAUDE_PROJECT_DIR}` in the text before the check, so those
two are usable in a `!` line. The prompt text stays in context for the rest of the session; permissions
do not.

A command or a skill is the only way to add a slash command: rules and hooks register none.

## Rules (`rules`)

Markdown appended to `CLAUDE.md`, read at the start of every session. A rule steers the model; it
enforces nothing.

## Settings (`config`)

`{"target": "primary", "data": {...}}` merges the object into `settings.json`: objects merge key by key
and lists join, so several config entries can each add a rule or a hook without removing another's.

`permissions`: `allow`, `deny` and `ask` are lists of tool rules. A rule is a tool name (`WebFetch`,
`Edit`) or a tool name with a pattern (`Bash(git *)`, `Read(./secrets/**)`, `Skill(deploy *)`). `deny`
wins over `allow`. A deny rule is the hard way to keep a tool from running for the whole session; a rule
in `CLAUDE.md` is the soft way.

`hooks`: an object keyed by event, each a list of matcher groups:

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "sh -c '...'", "timeout": 10}]}
    ],
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "..."}]}
    ]
  }
}
```

Events a harness uses: `SessionStart` (matcher: `startup`, `resume`, `clear`, `compact`),
`UserPromptSubmit` (no matcher), `UserPromptExpansion` (matcher: the slash command's name; fires when
the user types a command, before it expands), `PreToolUse` and `PostToolUse` (matcher: the tool name),
`Stop`, `PreCompact`, `Notification`, `SessionEnd`. A matcher is an exact name, a `|` list of names, or,
when it holds any other character, a JavaScript regular expression tested unanchored (`^Edit$` for a
whole-string match, `^(?!WebSearch$)` for every tool but one); omit it to match every event of that kind.
Tool names: `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `Agent`, `Skill`,
`TodoWrite`, `AskUserQuestion`, and `mcp__<server>__<tool>` for MCP tools.

Each hook command runs through `sh` with the session's environment and receives the event as JSON on
stdin: `session_id`, `hook_event_name`, `cwd`, and for a tool event `tool_name` and `tool_input`, for
`UserPromptSubmit` the `prompt`, for `UserPromptExpansion` the `command_name`. Hook commands are not
permission checked: they may read and write anywhere the user can. Exit codes: 0 lets the
event proceed (stdout is context for the model on `SessionStart` and `UserPromptSubmit`, discarded
elsewhere); 2 blocks a `PreToolUse` tool call, a `UserPromptSubmit` prompt or a `UserPromptExpansion`
command, with stderr as the reason the model or the user sees; any other code is a non-blocking error.
Instead of exiting 2 a hook may print one JSON object and exit 0: `{"systemMessage": "..."}` shows
text to the user on any event; `{"decision": "block", "reason": "..."}` blocks on `UserPromptSubmit`
and `UserPromptExpansion`; `{"hookSpecificOutput": {"hookEventName": "PreToolUse",
"permissionDecision": "deny", "permissionDecisionReason": "..."}}` blocks a tool call. A hook cannot
ask the user anything, and a hook that reaches its timeout (600 s by default, 30 s on
`UserPromptSubmit`) is dropped without blocking. In an interactive session Claude Code runs settings
hooks only once the user has trusted the working folder (the dialog Claude Code shows at first start).

`env`: variables set for the session, as strings.

## State a mode needs

Claude Code keeps no mode state for a harness, and a command cannot write one (see the `!` limits above).
Hooks can: a `UserPromptExpansion` hook fires when the user types a slash command, with the command's name
as its matcher, before the command expands, and it runs without the permission check. A mode is then:

```json
{
  "hooks": {
    "UserPromptExpansion": [
      {"matcher": "^chat$", "hooks": [{"type": "command", "command": "touch \"$CLAUDE_CONFIG_DIR/.chat-mode\""}]},
      {"matcher": "^chat-off$", "hooks": [{"type": "command", "command": "rm -f \"$CLAUDE_CONFIG_DIR/.chat-mode\""}]},
      {"matcher": "^(?!chat$|chat-off$)", "hooks": [{"type": "command",
        "command": "if [ -f \"$CLAUDE_CONFIG_DIR/.chat-mode\" ]; then echo 'chat mode: only /chat-off is allowed' >&2; exit 2; fi"}]}
    ],
    "PreToolUse": [
      {"matcher": "^(?!WebSearch$)", "hooks": [{"type": "command",
        "command": "if [ -f \"$CLAUDE_CONFIG_DIR/.chat-mode\" ]; then echo 'chat mode: only WebSearch is allowed' >&2; exit 2; fi"}]}
    ]
  }
}
```

beside two agent_command entries, `chat` and `chat-off`, whose prompts tell the model the mode is on or
off so its replies match; the hooks are what enforce it. `CLAUDE_CONFIG_DIR` is the session's own copy of
the harness, removed when the session ends, so a marker there never outlives the session. Hooks that
match the same event run in parallel, so the blocking hook must exempt the commands that write the state.
A skill's frontmatter `hooks` are another way: they register when the user invokes the skill and stay for
the session, so a mode that never turns off needs no marker.

## What is not available

No new tools: a harness cannot give Claude Code a tool it does not have; the closest thing is a hook that
runs a shell command at an event, or a command the user runs. No widgets, status lines or footers
beyond `systemMessage`. No code that runs inside Claude Code's process.

## Reef's own entries

`reef-requests` is the `/reefine` command, `reef-version-check` the `SessionStart` hook that says when a
newer release is served or a release waits for review, and this skill is `reef-claude-harness-api`.
They are reef's and stay as they are.
