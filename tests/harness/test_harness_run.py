"""Guarantees of reef.harness.tasks.run, hermetic: a fake harness binary is
driven through the real descriptor, render, and run path (CI installs
neither pi nor opencode)."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.tasks.run import TaskRunError, remove_run_root, run_task
from reef.harness.tasks.trajectory import (
    ClaudeSessionReader,
    CodexSessionReader,
    DeepseekSessionReader,
    HermesSessionReader,
    TrajectoryError,
    final_assistant_text,
    read_opencode_session,
    read_pi_session,
    reader_for,
)
from reef.harness.tree import HarnessTree, render_tree

PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["--mode", "json"], args
prompt = args[args.index("-p") + 1]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = Path(args[args.index("-e") + 1]) / "rules.md"
events = [
    {
        "type": "session",
        "root": str(agent_dir.parent),
        "offline": os.environ.get("PI_OFFLINE"),
        "docker_host": os.environ.get("DOCKER_HOST"),
    },
    {"type": "agent_end", "prompt": prompt, "rules": rules_path.read_text() if rules_path.exists() else ""},
]
(session_dir / "session.jsonl").write_text("".join(json.dumps(event) + "\\n" for event in events))
print(json.dumps({"type": "agent_start"}))
print(json.dumps({"type": "agent_end"}))
sys.exit(3 if prompt == "fail" else 0)
"""

#: The two tables of opencode 1.18.18's session store the reader reads, as the pinned binary creates them.
OPENCODE_TABLES = """\
CREATE TABLE `message` (
  `id` text PRIMARY KEY,
  `session_id` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL,
  CONSTRAINT `fk_message_session_id_session_id_fk` FOREIGN KEY (`session_id`) REFERENCES `session`(`id`) ON DELETE CASCADE
);
CREATE TABLE `part` (
  `id` text PRIMARY KEY,
  `message_id` text NOT NULL,
  `session_id` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL,
  CONSTRAINT `fk_part_message_id_message_id_fk` FOREIGN KEY (`message_id`) REFERENCES `message`(`id`) ON DELETE CASCADE
);
"""

OPENCODE_FAKE = """\
#!/usr/bin/env python3
import json, os, sqlite3, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:3] == ["run", "--format", "json"] and "--auto" in args and args[-2] == "--", args
config_dir = Path(os.environ["OPENCODE_CONFIG_DIR"])
config = json.loads((config_dir / "opencode.json").read_text())
assert config["autoupdate"] is False and config["share"] == "disabled", config
# The session as opencode 1.18.18 stores it: a message row per turn, its pieces as part rows, and the JSON data
# columns without the ids the other columns hold. A shell call and its answer take two assistant turns.
data_dir = Path(os.environ["XDG_DATA_HOME"]) / "opencode"
data_dir.mkdir(parents=True)
database = sqlite3.connect(data_dir / "opencode.db")
database.execute("PRAGMA journal_mode=WAL")
database.executescript(OPENCODE_TABLES)
shell = {"status": "completed", "input": {"command": "echo reef-ok"}, "output": "reef-ok\\n"}
turns = [
    ("msg_01", {"role": "user", "agent": "build"}, [{"type": "text", "text": args[-1]}]),
    (
        "msg_02",
        {"parentID": "msg_01", "role": "assistant", "agent": "build", "finish": "tool-calls"},
        [{"type": "step-start"}, {"type": "tool", "tool": "bash", "callID": "c1", "state": shell},
         {"type": "step-finish", "reason": "tool-calls"}],
    ),
    (
        "msg_03",
        {"parentID": "msg_01", "role": "assistant", "agent": "build", "finish": "stop"},
        [{"type": "step-start"}, {"type": "text", "text": "reef-ok"}, {"type": "step-finish", "reason": "stop"}],
    ),
]
for turn, (message_id, message, parts) in enumerate(turns):
    database.execute("INSERT INTO message VALUES (?, 'ses_01', ?, ?, ?)", (message_id, turn, turn, json.dumps(message)))
    for index, part in enumerate(parts):
        row = (f"prt_{turn}{index}", message_id, "ses_01", turn, turn, json.dumps(part))
        database.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", row)
database.commit()
database.close()
# Boot mutations of the rendered config dir, whitelisted by the descriptor, and the npm cache of the boot install,
# which npm keeps in npm_config_cache when it is set and in ~/.npm otherwise.
(config_dir / ".gitignore").write_text("node_modules\\n")
(config_dir / "node_modules").mkdir()
(config_dir / "node_modules" / "dep.js").write_text("module.exports = {}\\n")
npm_cache = Path(os.environ.get("npm_config_cache", Path.home() / ".npm"))
(npm_cache / "_cacache").mkdir(parents=True)
(npm_cache / "_cacache" / "index").write_text("entry\\n")
# A file nothing declared: the residue scan must report it.
(config_dir.parent / "stray.txt").write_text("leak\\n")
print(json.dumps({"type": "text", "sessionID": "ses_01", "part": {"messageID": "msg_03", "type": "text", "text": "reef-ok"}}))
""".replace(
    "OPENCODE_TABLES", repr(OPENCODE_TABLES)
)

CODEX_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:11] == [
    "exec", "--json", "--strict-config", "--config", 'approval_policy="never"', "--config", 'web_search="disabled"',
    "--sandbox", "workspace-write", "--skip-git-repo-check", "--",
], args
prompt = args[11]
codex_home = Path(os.environ["CODEX_HOME"])
assert Path(os.environ["HOME"]) == codex_home.parent
config = (codex_home / "config.toml").read_text()
# The argv pins approvals off; config.toml leaves them to Codex's default for a person's reef-codex session.
assert "approval_policy" not in config and 'web_search = "disabled"' in config
run = codex_home / "sessions" / "2026" / "09" / "02" / "run.jsonl"
run.parent.mkdir(parents=True)
events = [
    {"type": "session_meta", "payload": {"cwd": str(codex_home.parent / "workspace")}},
    {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "done"}},
]
run.write_text("".join(json.dumps(event) + "\\n" for event in events))
(codex_home / "installation_id").write_text("test-install")
(codex_home / ".tmp").mkdir()
(codex_home / ".tmp" / "probe").write_text("ephemeral")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}))
sys.exit(3 if prompt == "fail" else 0)
"""

CLAUDE_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == "-p" and args[2:4] == ["--output-format", "json"], args
session = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "workspace" / "s1.jsonl"
session.parent.mkdir(parents=True)
# The shell commands an agent runs inherit this env.
switches = {key: os.environ.get(key) for key in ("DISABLE_AUTOUPDATER", "DISABLE_UPDATES")}
session.write_text(json.dumps({"type": "user", "env": switches}) + "\\n")
print(json.dumps({"type": "result", "result": "done"}))
"""


def fake_binary(tmp_path: Path, script: str) -> str:
    binary = tmp_path / "fake-harness"
    binary.write_text(script)
    binary.chmod(0o755)
    return str(binary)


def pi_files() -> dict[str, str]:
    return render_tree(HarnessTree({"rules.md": "Answer briefly."}), get_adapter("pi"))


def test_pi_run_collects_exit_stdout_and_trajectory(tmp_path: Path) -> None:
    result = run_task(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == ["agent_start", "agent_end"]
    assert [event["type"] for event in result.trajectory] == ["session", "agent_end"]
    assert result.residue == ()  # session storage is declared, nothing else appeared


def test_run_relocates_the_composition_and_pins_offline(tmp_path: Path) -> None:
    result = run_task(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    session, agent_end = result.trajectory
    assert session["offline"] == "1"  # PI_OFFLINE rode the relocation env
    assert agent_end["rules"] == "Answer briefly.\n"  # the rendered tree is what the binary read
    assert agent_end["prompt"] == "list files"


def test_run_root_is_removed_after_the_run(tmp_path: Path) -> None:
    result = run_task(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    root = Path(result.trajectory[0]["root"])
    assert not root.exists()


def test_an_adapter_without_host_env_keeps_no_service_variable_and_roots_in_the_temp_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """An adapter that declares no host_env and no bind-mounted root stays hermetic, on macOS too."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    result = run_task(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    session = result.trajectory[0]
    assert session["docker_host"] is None
    assert Path(session["root"]).parent == Path(tempfile.gettempdir())


def test_run_cleanup_repairs_permissions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "credential").write_text("secret")
    locked.chmod(0)
    try:
        remove_run_root(root)
    finally:
        if locked.exists():
            locked.chmod(stat.S_IRWXU)
    assert not root.exists()


def test_run_cleanup_failure_is_not_suppressed(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()

    def fail(*_args, **_kwargs):
        raise OSError("locked")

    monkeypatch.setattr("reef.harness.tasks.run.shutil.rmtree", fail)
    with pytest.raises(TaskRunError, match=r"cannot remove run root.*locked"):
        remove_run_root(root)


def test_nonzero_exit_code_is_reported_not_raised(tmp_path: Path) -> None:
    result = run_task(get_adapter("pi"), pi_files(), "fail", binary=fake_binary(tmp_path, PI_FAKE))
    assert result.exit_code == 3


def test_opencode_run_whitelists_boot_mutations_and_reports_residue(tmp_path: Path) -> None:
    files = render_tree(HarnessTree({"rules.md": "Answer briefly."}), get_adapter("opencode"))
    result = run_task(get_adapter("opencode"), files, "list files", binary=fake_binary(tmp_path, OPENCODE_FAKE))
    assert result.exit_code == 0
    assert [event["role"] for event in result.trajectory] == ["user", "assistant", "assistant"]
    assert result.trajectory[0]["content"] == [{"id": "prt_00", "type": "text", "text": "list files"}]
    assert result.residue == ("stray.txt",)  # boot mutations and the npm cache tolerated, the stray file is a finding


def test_opencode_run_grades_the_final_answer_of_its_session(tmp_path: Path) -> None:
    """The reader joins each message to its parts, so the grader finds the text of the last assistant turn."""
    files = render_tree(HarnessTree({}), get_adapter("opencode"))
    result = run_task(get_adapter("opencode"), files, "Run it.", binary=fake_binary(tmp_path, OPENCODE_FAKE))
    assert final_assistant_text(result.trajectory) == "reef-ok"


def test_codex_run_collects_nested_run_and_whitelists_boot_state(tmp_path: Path) -> None:
    files = render_tree(HarnessTree({"skills/fix/SKILL.md": "# Fix"}), get_adapter("codex"))
    result = run_task(get_adapter("codex"), files, "list files", binary=fake_binary(tmp_path, CODEX_FAKE))
    assert result.exit_code == 0
    assert json.loads(result.stdout)["item"]["text"] == "done"
    assert [event["type"] for event in result.trajectory] == ["session_meta", "event_msg"]
    assert result.residue == ()


HERMES_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
snapshot = {"session_id": "s1", "model": "m", "messages": [{"role": "user", "content": sys.argv[-1]}]}
(home / "sessions").mkdir()
(home / "sessions" / "session_s1.json").write_text(json.dumps(snapshot))
# Against an OpenRouter endpoint, hermes records the key it finds in its environment in the credential pool.
(home / "auth.lock").write_text("")
pool = {"openai-api": [{"source": "env:OPENAI_API_KEY", "secret_fingerprint": "sha256:0123456789abcdef"}]}
(home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": pool}))
(home / "logs").mkdir()
(home / "logs" / "agent.log").write_text("started\\n")
if not (home / "SOUL.md").exists():
    (home / "SOUL.md").write_text("You are Hermes Agent.\\n")  # the default rules file, for a tree with no rules
# A skill_view of a tree skill counts the load in the usage file beside the skills, under a lock file.
(home / "skills" / ".usage.json.lock").write_text("")
(home / "skills" / ".usage.json").write_text(json.dumps({"notes": {"view_count": 1, "use_count": 1}}))
print("done")
"""


def test_hermes_run_whitelists_what_hermes_writes_at_boot_and_on_a_skill_load(tmp_path: Path) -> None:
    files = render_tree(HarnessTree({"skills/notes/SKILL.md": "Keep notes."}), get_adapter("hermes"))
    result = run_task(get_adapter("hermes"), files, "list files", binary=fake_binary(tmp_path, HERMES_FAKE))
    assert result.exit_code == 0
    assert [event["type"] for event in result.trajectory] == ["session", "message"]
    # The credential pool, its lock, the default rules file, the log and the skill usage counts with their lock.
    assert result.residue == ()


def test_claude_run_runs_with_the_updater_and_its_update_commands_off(tmp_path: Path) -> None:
    """The run env sets both updater switches, and the agent's shell commands inherit them, so a claude update
    or claude install run there keeps the pinned version."""
    files = render_tree(HarnessTree({"rules.md": "Answer briefly."}), get_adapter("claude"))
    result = run_task(get_adapter("claude"), files, "list files", binary=fake_binary(tmp_path, CLAUDE_FAKE))
    assert result.exit_code == 0
    assert [event["env"] for event in result.trajectory] == [{"DISABLE_AUTOUPDATER": "1", "DISABLE_UPDATES": "1"}]
    assert result.residue == ()


def test_missing_binary_raises_run_error(tmp_path: Path) -> None:
    with pytest.raises(TaskRunError, match="not found"):
        run_task(get_adapter("pi"), pi_files(), "x", binary=str(tmp_path / "no-such-binary"))


def test_pi_reader_tolerates_exactly_one_torn_tail(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text('{"type": "agent_start"}\n{"type": "agent_end"\n')  # crash mid-write
    assert [event["type"] for event in read_pi_session(tmp_path)] == ["agent_start"]
    session.write_text('{"type": "agent_start"\n{"type": "agent_end"}\n')  # torn in the middle: corruption
    with pytest.raises(TrajectoryError, match="corrupt event at line 1"):
        read_pi_session(tmp_path)


def test_pi_reader_keeps_unicode_line_separators_inside_an_event(tmp_path: Path) -> None:
    events = [{"type": "message", "text": "one\u2028two\x85three"}, {"type": "agent_end"}]
    (tmp_path / "session.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events))
    assert list(read_pi_session(tmp_path)) == events


def test_pi_reader_parses_a_captured_real_session() -> None:
    # Real bytes from a live pi 0.84.2 run, refreshed by the harness-smoke
    # workflow's artifact (see the fixture directory's README.md).
    events = read_pi_session(Path(__file__).parent / "data" / "pi_session_real")
    assert len(events) == 5
    assert events[0]["type"] == "session" and events[0]["version"] == 3
    final = events[-1]["message"]
    assert final["role"] == "assistant"
    assert final["content"] == [{"type": "text", "text": "READY"}]


def test_claude_reader_reads_nested_sessions_in_path_order(tmp_path: Path) -> None:
    # Claude Code lays sessions out as projects/<cwd-slug>/<session-id>.jsonl.
    first = tmp_path / "projects" / "proj-a" / "01.jsonl"
    second = tmp_path / "projects" / "proj-a" / "02.jsonl"
    first.parent.mkdir(parents=True)
    first.write_text('{"type": "user"}\n')
    second.write_text('{"type": "assistant"}\n')
    assert [event["type"] for event in ClaudeSessionReader()(tmp_path)] == ["user", "assistant"]


def test_claude_reader_tolerates_exactly_one_torn_tail(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text('{"type": "user"}\n{"type": "assistant"\n')  # crash mid-write
    assert [event["type"] for event in ClaudeSessionReader()(tmp_path)] == ["user"]
    session.write_text('{"type": "user"\n{"type": "assistant"}\n')  # torn in the middle: corruption
    with pytest.raises(TrajectoryError, match="corrupt event at line 1"):
        ClaudeSessionReader()(tmp_path)


def test_claude_format_resolves_through_reader_for() -> None:
    assert reader_for("claude-session-jsonl").format == "claude-session-jsonl"


def test_codex_reader_reads_nested_sessions_and_tolerates_one_torn_tail(tmp_path: Path) -> None:
    first = tmp_path / "2026" / "09" / "01" / "run-01.jsonl"
    second = tmp_path / "2026" / "09" / "02" / "run-02.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text('{"type": "session_meta"}\n')
    second.write_text('{"type": "event_msg"}\n{"type": "turn_context"\n')
    assert [event["type"] for event in CodexSessionReader()(tmp_path)] == ["session_meta", "event_msg"]
    assert reader_for("codex-session-jsonl").format == "codex-session-jsonl"


def write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_claude_codex_and_pi_sessions_convert_to_atif_with_harbors_converters(tmp_path: Path) -> None:
    session = "11111111-1111-1111-1111-111111111111"
    write_jsonl(
        tmp_path / "claude" / "-app" / f"{session}.jsonl",
        [
            {"type": "user", "sessionId": session, "uuid": "u1", "message": {"role": "user", "content": "Say hi"}},
            {
                "type": "assistant",
                "sessionId": session,
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {"id": "m1", "role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            },
        ],
    )
    write_jsonl(
        tmp_path / "codex" / "2026" / "11" / "09" / "rollout-s1.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "s1", "cwd": "/app"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hi"}]},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            },
        ],
    )
    pi_session = Path(__file__).parent / "data" / "pi_session_real"
    for format_, path, final in [
        ("claude-session-jsonl", tmp_path / "claude", "hi"),
        ("codex-session-jsonl", tmp_path / "codex", "hi"),
        ("pi-session-jsonl", pi_session, "READY"),
    ]:
        (document,) = reader_for(format_).to_atif(path)
        assert document["schema_version"].startswith("ATIF-")
        assert document["steps"][-1]["source"] == "agent"
        assert document["steps"][-1]["message"] == final
        assert reader_for(format_).to_atif(tmp_path / "missing") == ()


def test_a_format_without_a_converter_gives_no_atif(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "session.jsonl", [{"type": "message"}])
    assert reader_for("json:loads").to_atif(tmp_path) == ()


def atif_steps(document: dict[str, object]) -> list[tuple[object, ...]]:
    """Each step's source, message, tool names and observation contents."""
    return [
        (
            step["source"],
            step["message"],
            [call["function_name"] for call in step.get("tool_calls") or []],
            [result["content"] for result in (step.get("observation") or {}).get("results", [])],
        )
        for step in document["steps"]
    ]


def test_real_dsh_and_opencode_sessions_convert_to_atif_with_their_tool_call() -> None:
    data = Path(__file__).parent / "data"
    task = "Run `echo reef-tool-ok` in the shell, then reply with exactly READY."
    (dsh,) = reader_for("deepseek-session-jsonl").to_atif(data / "dsh_session_real")
    steps = atif_steps(dsh)
    assert steps[0][0] == "system" and steps[1] == ("user", task, [], [])
    assert steps[-2][2:] == (["bash"], ["reef-tool-ok\n"]) and steps[-1] == ("agent", "READY", [], [])
    (opencode,) = reader_for("opencode-session-sqlite").to_atif(data / "opencode_session_real")
    steps = atif_steps(opencode)
    assert steps[0][0] == "user" and task in steps[0][1]
    assert steps[1][2:] == (["bash"], ["reef-tool-ok\n"]) and steps[2] == ("agent", "READY", [], [])


def native_event(type_: str, **data: object) -> dict[str, object]:
    return {"type": type_, "seq": 0, "time": 0, "data": data}


def test_a_native_session_converts_to_one_document_for_each_segment_and_one_for_each_agent(tmp_path: Path) -> None:
    (real,) = reader_for("native-jsonl").to_atif(Path(__file__).parent / "data" / "native_session_real")
    assert [step[0] for step in atif_steps(real)] == ["system", "user", "agent"]
    assert real["steps"][-1]["message"] == "READY" and real["agent"]["model_name"] == "smoke-model"
    call = {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": '{"command": "ls"}'}}
    write_jsonl(
        tmp_path / "session.jsonl",
        [
            native_event("session", task="List the files."),
            native_event("request/header", model="m", system="Be brief.", tools=[]),
            native_event("assistant/message", step=1, content="", tool_calls=[call]),
            native_event("tool/result", step=1, call_id="c1", name="shell", content="a.txt"),
            native_event("context/compacted", fired=True, summary="Listed a.txt."),
            native_event("assistant/message", step=2, content="a.txt", tool_calls=[]),
        ],
    )
    write_jsonl(tmp_path / "agents" / "001-helper.jsonl", [native_event("session", task="Help.")])
    write_jsonl(tmp_path / "agents" / "002-checker.jsonl", [native_event("context/compacted", fired=False)])
    helper, before, after = reader_for("native-jsonl").to_atif(tmp_path)
    assert atif_steps(helper) == [("user", "Help.", [], [])]
    assert atif_steps(before) == [
        ("system", "Be brief.", [], []),
        ("user", "List the files.", [], []),
        ("agent", "", ["shell"], ["a.txt"]),
    ]
    assert atif_steps(after) == [
        ("system", "Be brief.", [], []),
        ("user", "Summary of the earlier steps:\nListed a.txt.", [], []),
        ("agent", "a.txt", [], []),
    ]


def test_a_hermes_snapshot_converts_to_atif_with_harbors_converter(tmp_path: Path) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "echo ok"}'}}
    snapshot = {
        "session_id": "s1",
        "messages": [
            {"role": "user", "content": "Run echo"},
            {"role": "assistant", "content": "", "tool_calls": [call], "finish_reason": "tool_calls"},
            {"role": "tool", "tool_name": "terminal", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "READY", "finish_reason": "stop"},
        ],
    }
    (tmp_path / "session_s1.json").write_text(json.dumps(snapshot, indent=2))
    (document,) = reader_for("hermes-session-json").to_atif(tmp_path)
    steps = atif_steps(document)
    assert steps[0] == ("user", "Run echo", [], []) and steps[-1] == ("agent", "READY", [], [])
    assert steps[1][2:] == (["terminal"], ["ok"])


def test_a_python_programs_logged_calls_convert_to_one_document_for_each_conversation(tmp_path: Path) -> None:
    def call(messages: list[dict[str, str]], answer: str) -> dict[str, object]:
        return {
            "model": "m",
            "messages": messages,
            "response": {"choices": [{"message": {"role": "assistant", "content": answer}}]},
        }

    first = call([{"role": "user", "content": "hi"}], "hello")
    judge = call([{"role": "system", "content": "Judge."}, {"role": "user", "content": "Score it."}], "5")
    second = call(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ],
        "READY",
    )
    write_jsonl(tmp_path / "calls.jsonl", [first, {"event": "note"}, judge, second])
    talk, judged = reader_for("python-jsonl").to_atif(tmp_path)
    assert [step[:2] for step in atif_steps(talk)] == [
        ("user", "hi"),
        ("agent", "hello"),
        ("user", "bye"),
        ("agent", "READY"),
    ]
    assert [step[:2] for step in atif_steps(judged)] == [("system", "Judge."), ("user", "Score it."), ("agent", "5")]


@pytest.mark.parametrize("answer", ["reef-ok", "wrong", None])
def test_codex_run_reads_only_assistant_output_text(tmp_path: Path, answer: str | None) -> None:
    events = [
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "reef-ok"}]},
        },
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "reef-ok"}},
        {
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "reef-ok"}]},
        },
    ]
    if answer is not None:
        for text in ("Earlier reply", answer):
            events.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Output:"}, {"type": "output_text", "text": text}],
                    },
                }
            )
    events.append({"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": answer}})
    session = tmp_path / "run.jsonl"
    session.write_text("".join(json.dumps(event) + "\n" for event in events))
    trajectory = CodexSessionReader()(tmp_path)
    expected_reply = None if answer is None else f"Output:\n{answer}"
    assert final_assistant_text(trajectory) == expected_reply
    assert trajectory == tuple(events)


def test_deepseek_reader_reads_nested_sessions_and_tolerates_one_torn_tail(tmp_path: Path) -> None:
    # dsh lays sessions out as sessions/<cwd-slug>/session-<id>/session.jsonl.
    first = tmp_path / "slug" / "session-01" / "session.jsonl"
    second = tmp_path / "slug" / "session-02" / "session.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text('{"type": "session", "version": 0}\n{"type": "turn/start", "seq": 0}\n')
    second.write_text('{"type": "session", "version": 0}\n{"type": "turn/end", "seq": 0\n')  # crash mid-write
    assert [event["type"] for event in DeepseekSessionReader()(tmp_path)] == ["session", "turn/start", "session"]
    second.write_text('{"type": "session"\n{"type": "turn/end", "seq": 0}\n')  # torn in the middle: corruption
    with pytest.raises(TrajectoryError, match=r"deepseek session .* corrupt event at line 1"):
        DeepseekSessionReader()(tmp_path)
    assert reader_for("deepseek-session-jsonl").format == "deepseek-session-jsonl"


def test_deepseek_reader_lifts_each_events_data_so_the_final_assistant_text_is_found(tmp_path: Path) -> None:
    """dsh wraps every event in {type, seq, time, data}; the scorers read a message at the top, as pi writes it."""
    log = tmp_path / "slug" / "session-01" / "session.jsonl"
    log.parent.mkdir(parents=True)
    message = {
        "role": "assistant",
        "content": [{"type": "reasoning", "text": "run it"}, {"type": "text", "text": "The output:\n\nreef-ok"}],
    }
    log.write_text(
        '{"type": "session", "version": 0}\n'
        + json.dumps({"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}})
        + "\n"
        + json.dumps({"type": "assistant/message", "seq": 1, "time": 2, "data": {"turn": 1, "message": message}})
        + "\n"
    )
    events = DeepseekSessionReader()(tmp_path)
    assert events[1] == {"type": "turn/start", "seq": 0, "time": 1, "turn": 1}
    assert events[2]["message"] == message and "data" not in events[2]
    assert final_assistant_text(events) == "The output:\n\nreef-ok"


def test_hermes_reader_emits_a_session_event_then_one_event_per_message(tmp_path: Path) -> None:
    snapshot = {
        "session_id": "s1",
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "finish_reason": "tool_calls"},
            {"role": "tool", "tool_name": "terminal", "tool_call_id": "c1", "content": "ok"},
        ],
    }
    (tmp_path / "session_s1.json").write_text(json.dumps(snapshot))
    events = HermesSessionReader()(tmp_path)
    assert [event["type"] for event in events] == ["session", "message", "message", "message"]
    assert events[0] == {"type": "session", "session_id": "s1", "model": "m"}
    assert events[2]["tool_calls"] == [{"id": "c1"}] and events[3]["tool_name"] == "terminal"
    (tmp_path / "session_s2.json").write_text("{not json")
    with pytest.raises(TrajectoryError, match="not valid JSON"):
        HermesSessionReader()(tmp_path)
    (tmp_path / "session_s2.json").write_text(json.dumps({"session_id": "s2"}))
    with pytest.raises(TrajectoryError, match="messages list"):
        HermesSessionReader()(tmp_path)
    assert reader_for("hermes-session-json").format == "hermes-session-json"


def test_missing_trajectory_reads_as_empty(tmp_path: Path) -> None:
    assert read_pi_session(tmp_path / "never-written") == ()


def test_unknown_trajectory_format_is_rejected() -> None:
    with pytest.raises(TrajectoryError, match="unknown trajectory format"):
        reader_for("acme-log")


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    # Coding agents spawn tool children; a timeout must not orphan them.
    hang = f"""\
#!/usr/bin/env python3
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open({str(tmp_path)!r} + "/child.pid", "w").write(str(child.pid))
time.sleep(60)
"""
    with pytest.raises(TaskRunError, match="timed out"):
        run_task(get_adapter("pi"), pi_files(), "x", binary=fake_binary(tmp_path, hang), timeout=1.5)
    child_pid = int((tmp_path / "child.pid").read_text())
    time.sleep(0.2)  # give the group kill a beat to land
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_render_path_may_not_escape_the_run_root(tmp_path: Path) -> None:
    files = {**pi_files(), "../escape.txt": "x"}
    with pytest.raises(TaskRunError, match="escapes the run root"):
        run_task(get_adapter("pi"), files, "x", binary=fake_binary(tmp_path, PI_FAKE))
    assert not (tmp_path.parent / "escape.txt").exists()


def test_colliding_render_paths_raise_run_error(tmp_path: Path) -> None:
    files = {"a": "file", "a/b": "needs a to be a directory"}
    with pytest.raises(TaskRunError, match="cannot write render path"):
        run_task(get_adapter("pi"), files, "x", binary=fake_binary(tmp_path, PI_FAKE))


def test_opencode_reader_rejects_a_non_object_document(tmp_path: Path) -> None:
    assert read_opencode_session(tmp_path) == ()  # no session database yet
    (tmp_path / "opencode.db").write_text("not a database")
    with pytest.raises(TrajectoryError, match="cannot be read"):
        read_opencode_session(tmp_path)
    (tmp_path / "opencode.db").unlink()
    database = sqlite3.connect(tmp_path / "opencode.db")
    database.executescript(OPENCODE_TABLES)
    database.execute("INSERT INTO message VALUES ('msg_01', 'ses_01', 0, 0, '{\"role\": \"user\"}')")
    database.execute("INSERT INTO part VALUES ('prt_01', 'msg_01', 'ses_01', 0, 0, '[1, 2, 3]')")
    database.commit()
    database.close()
    with pytest.raises(TrajectoryError, match=r"part prt_01 .* is not an object"):
        read_opencode_session(tmp_path)
