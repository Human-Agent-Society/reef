"""Guarantees of reef.harness.episode, hermetic: a fake harness binary is
driven through the real descriptor, render, and episode path (CI installs
neither pi nor opencode)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episode import EpisodeError, run_episode
from reef.harness.render import render_composition
from reef.harness.trajectory import (
    TrajectoryError,
    read_claude_session,
    read_opencode_storage,
    read_pi_session,
    reader_for,
)

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
rules_path = agent_dir / "AGENTS.md"
events = [
    {"type": "session", "root": str(agent_dir.parent), "offline": os.environ.get("PI_OFFLINE")},
    {"type": "agent_end", "prompt": prompt, "rules": rules_path.read_text() if rules_path.exists() else ""},
]
(session_dir / "session.jsonl").write_text("".join(json.dumps(event) + "\\n" for event in events))
print(json.dumps({"type": "agent_start"}))
print(json.dumps({"type": "agent_end"}))
sys.exit(3 if prompt == "fail" else 0)
"""

OPENCODE_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:3] == ["run", "--format", "json"] and "--auto" in args, args
config_dir = Path(os.environ["OPENCODE_CONFIG_DIR"])
config = json.loads((config_dir / "opencode.json").read_text())
assert config["autoupdate"] is False and config["share"] == "disabled", config
storage = Path(os.environ["XDG_DATA_HOME"]) / "opencode" / "storage" / "session" / "message" / "s1"
storage.mkdir(parents=True)
(storage / "msg_001.json").write_text(json.dumps({"role": "user", "text": args[-1]}))
(storage / "msg_002.json").write_text(json.dumps({"role": "assistant", "text": "done"}))
# Boot mutations of the rendered config dir, whitelisted by the quirks.
(config_dir / ".gitignore").write_text("node_modules\\n")
(config_dir / "node_modules").mkdir()
(config_dir / "node_modules" / "dep.js").write_text("module.exports = {}\\n")
# A file nothing declared: the residue scan must report it.
(config_dir.parent / "stray.txt").write_text("leak\\n")
print(json.dumps({"type": "done"}))
"""


def fake_binary(tmp_path: Path, script: str) -> str:
    binary = tmp_path / "fake-harness"
    binary.write_text(script)
    binary.chmod(0o755)
    return str(binary)


def pi_files() -> dict[str, str]:
    return render_composition([("rules", {"text": "Answer briefly."})], get_adapter("pi"))


def test_pi_episode_collects_exit_stdout_and_trajectory(tmp_path: Path) -> None:
    result = run_episode(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == ["agent_start", "agent_end"]
    assert [event["type"] for event in result.trajectory] == ["session", "agent_end"]
    assert result.residue == ()  # session storage is declared, nothing else appeared


def test_episode_relocates_the_composition_and_pins_offline(tmp_path: Path) -> None:
    result = run_episode(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    session, agent_end = result.trajectory
    assert session["offline"] == "1"  # PI_OFFLINE rode the relocation env
    assert agent_end["rules"] == "Answer briefly.\n"  # the rendered tree is what the binary read
    assert agent_end["prompt"] == "list files"


def test_episode_root_is_removed_after_the_run(tmp_path: Path) -> None:
    result = run_episode(get_adapter("pi"), pi_files(), "list files", binary=fake_binary(tmp_path, PI_FAKE))
    root = Path(result.trajectory[0]["root"])
    assert not root.exists()


def test_nonzero_exit_code_is_reported_not_raised(tmp_path: Path) -> None:
    result = run_episode(get_adapter("pi"), pi_files(), "fail", binary=fake_binary(tmp_path, PI_FAKE))
    assert result.exit_code == 3


def test_opencode_episode_whitelists_boot_mutations_and_reports_residue(tmp_path: Path) -> None:
    files = render_composition([("rules", {"text": "Answer briefly."})], get_adapter("opencode"))
    result = run_episode(get_adapter("opencode"), files, "list files", binary=fake_binary(tmp_path, OPENCODE_FAKE))
    assert result.exit_code == 0
    assert [event["role"] for event in result.trajectory] == ["user", "assistant"]
    assert result.residue == ("stray.txt",)  # boot mutations tolerated, the stray file is a finding


def test_missing_binary_raises_episode_error(tmp_path: Path) -> None:
    with pytest.raises(EpisodeError, match="not found"):
        run_episode(get_adapter("pi"), pi_files(), "x", binary=str(tmp_path / "no-such-binary"))


def test_pi_reader_tolerates_exactly_one_torn_tail(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text('{"type": "agent_start"}\n{"type": "agent_end"\n')  # crash mid-write
    assert [event["type"] for event in read_pi_session(tmp_path)] == ["agent_start"]
    session.write_text('{"type": "agent_start"\n{"type": "agent_end"}\n')  # torn in the middle: corruption
    with pytest.raises(TrajectoryError, match="corrupt event at line 1"):
        read_pi_session(tmp_path)


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
    assert [event["type"] for event in read_claude_session(tmp_path)] == ["user", "assistant"]


def test_claude_reader_tolerates_exactly_one_torn_tail(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session.write_text('{"type": "user"}\n{"type": "assistant"\n')  # crash mid-write
    assert [event["type"] for event in read_claude_session(tmp_path)] == ["user"]
    session.write_text('{"type": "user"\n{"type": "assistant"}\n')  # torn in the middle: corruption
    with pytest.raises(TrajectoryError, match="corrupt event at line 1"):
        read_claude_session(tmp_path)


def test_claude_format_resolves_through_reader_for() -> None:
    assert reader_for("claude-session-jsonl").format == "claude-session-jsonl"


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
    with pytest.raises(EpisodeError, match="timed out"):
        run_episode(get_adapter("pi"), pi_files(), "x", binary=fake_binary(tmp_path, hang), timeout=1.5)
    child_pid = int((tmp_path / "child.pid").read_text())
    time.sleep(0.2)  # give the group kill a beat to land
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_render_path_may_not_escape_the_episode_root(tmp_path: Path) -> None:
    files = {**pi_files(), "../escape.txt": "x"}
    with pytest.raises(EpisodeError, match="escapes the episode root"):
        run_episode(get_adapter("pi"), files, "x", binary=fake_binary(tmp_path, PI_FAKE))
    assert not (tmp_path.parent / "escape.txt").exists()


def test_colliding_render_paths_raise_episode_error(tmp_path: Path) -> None:
    files = {"a": "file", "a/b": "needs a to be a directory"}
    with pytest.raises(EpisodeError, match="cannot write render path"):
        run_episode(get_adapter("pi"), files, "x", binary=fake_binary(tmp_path, PI_FAKE))


def test_opencode_reader_rejects_a_non_object_document(tmp_path: Path) -> None:
    (tmp_path / "part.json").write_text("[1, 2, 3]")
    with pytest.raises(TrajectoryError, match="not an event object"):
        read_opencode_storage(tmp_path)
