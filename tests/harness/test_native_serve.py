"""The serve form: one resident process holds the tree live, follows Reef's head between steps, and lets the model inspect and try.

A fake Reef on localhost plays the model (a scripted reply per request, the
receipt and release headers on every answer), the release catalog, and one
source tree per release, so every path of ``reef-native serve`` runs
hermetically."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from tests.support.harness_render import native_hook, native_json, native_tool

from reef.harness.adapters import get_adapter
from reef.harness.capture import CaptureProxy, ReleaseObserver
from reef.harness.runners.native import serve
from reef.harness.runners.native.release_client import HeadWatch, ReleaseClient, ReleaseClientError
from reef.harness.runners.native.serve import HARNESS_RELEASE_FILE, ServeError, Server
from reef.harness.tree import HarnessTree, render_tree, validate_tree
from reef.harness.tree.checks import NATIVE_RESERVED_TOOL_NAMES

SCENARIO = "serve-demo"
NATIVE = get_adapter("native")
RULES = {"rules.md": "Be brief."}


def _tool(name: str, body: str = "    return 'ok'") -> dict[str, str]:
    """The file of a tool that takes no arguments and runs ``body``."""
    return native_tool(
        name,
        f"def run(args, workdir):\n{body}\n",
        description=f"the {name} tool",
        parameters={"type": "object", "properties": {}},
    )


def _graph(stages: dict[str, Any], edges: list[dict[str, str]], start: str = "think") -> dict[str, str]:
    return native_json("graphs", {"name": "main", "start": start, "max_steps": 8, "stages": stages, "edges": edges})


SEED_STAGES: dict[str, Any] = {
    "think": {"kind": "model"},
    "act": {"kind": "tools"},
    "done": {"kind": "end", "reason": "completed"},
}
SEED_EDGES = [
    {"from": "think", "when": "tool_calls", "to": "act"},
    {"from": "think", "when": "text", "to": "done"},
    {"from": "act", "when": "done", "to": "think"},
]


def _reply(content: str | None = None, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class _FakeReef(ThreadingHTTPServer):
    """The model, the release catalog, and one source tree per release."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.releases: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.head = ""
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.fail_releases = False
        self.delay_s = 0.0
        self.hang_manifest_s = 0.0  # the next manifest read sleeps this long once, as behind a step's lock
        self.polls = 0
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def release(
        self, release_id: str, files: Mapping[str, str], parent: str | None = None, **evaluation_metrics: Any
    ) -> None:
        self.releases[release_id] = {
            "release_id": release_id,
            "artifact_id": release_id,
            "parent_release_id": parent,
            "content_id": f"content-{release_id}",
            "operation": "training",
            "files": dict(files),
            "metrics": {"wins": 1, **evaluation_metrics},
        }
        self.order.append(release_id)
        self.head = release_id

    def close(self) -> None:
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: Any, **headers: str) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        reef: _FakeReef = self.server  # type: ignore[assignment]
        split = urlsplit(self.path)
        if split.path == "/reef/harness/releases":
            reef.polls += 1
            if reef.fail_releases:
                self._json(500, {"error": {"message": "catalog down"}})
                return
            rows = [{k: v for k, v in reef.releases[rid].items() if k != "files"} for rid in reef.order]
            for row in rows:
                # Every release here was published to serve, so each one was activated when it was made.
                row["active"] = row["release_id"] == reef.head
                row["activated"] = True
            self._json(200, {"scenario": SCENARIO, "releases": rows})
        elif split.path == "/reef/harness":
            if reef.hang_manifest_s:
                hang, reef.hang_manifest_s = reef.hang_manifest_s, 0.0
                time.sleep(hang)
            query = parse_qs(split.query)
            release_id = query.get("release_id", [reef.head])[0]
            row = reef.releases.get(release_id)
            if row is None:
                self._json(404, {"error": {"message": f"unknown release {release_id}"}})
                return
            manifest = {k: v for k, v in row.items() if k not in ("metrics", "operation")}
            if query.get("form", ["rendered"])[0] != "source":
                # As Reef serves it: the release rendered for the adapter; the serve process asks for the source.
                manifest["files"] = render_tree(HarnessTree(row["files"]), NATIVE)
            manifest["evaluation"] = row["metrics"]
            self._json(200, manifest, x_reef_release_id=release_id)
        else:
            self._json(404, {"error": {"message": "no route"}})

    def do_POST(self) -> None:
        reef: _FakeReef = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.startswith("/v1/chat/completions"):
            reef.requests.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            if reef.delay_s:
                time.sleep(reef.delay_s)
            message = reef.replies.pop(0) if reef.replies else _reply("done")
            self._json(
                200,
                {"choices": [{"message": message}]},
                x_reef_agent_record_id=f"r-{len(reef.requests)}",
                x_reef_release_id=reef.head,
            )
        else:
            self._json(404, {"error": {"message": "no route"}})

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def reef() -> Iterator[_FakeReef]:
    with contextlib.closing(_FakeReef()) as server:
        yield server


def _tree(tmp_path: Path, reef: _FakeReef, release_id: str) -> Path:
    """The pulled tree: the model binding at the fake reef and the release file naming ``release_id``.

    No source tree is stored yet, so the process fetches the release's source tree at boot.
    """
    dest = tmp_path / "reef-harness"
    (dest / "native").mkdir(parents=True)
    (dest / "native" / "models.json").write_text(
        json.dumps({"api": "openai", "base_url": reef.base_url, "api_key": "tok", "model": "fake"}) + "\n",
        encoding="utf-8",
    )
    (dest / HARNESS_RELEASE_FILE).write_text(json.dumps({"release_id": release_id}) + "\n", encoding="utf-8")
    return dest


@contextlib.contextmanager
def _running(dest: Path, **options: Any) -> Iterator[Server]:
    server = Server(dest, scenario=SCENARIO, poll_interval_s=options.pop("poll_interval_s", 60.0), **options)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _typed(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [event["data"] for event in events if event["type"] == type_]


def _wait(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition did not hold in time")
        time.sleep(0.02)


class _Lines(serve.EventSink):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def send(self, line: str) -> None:
        self.lines.append(line)


def _turn(
    server: Server, prompt: str, session: str | None = None, workdir: Path | None = None
) -> tuple[dict, list[dict]]:
    """One turn over the socket: the result and the events that streamed on the way."""
    sink = _Lines()
    payload = {"turn": {"prompt": prompt, "session": session, "workdir": str(workdir or server.dest)}}
    reply = serve.request(server.socket_path, payload, sink)
    assert reply["type"] == "turn/result", reply
    return reply["data"], [json.loads(line) for line in sink.lines]


def _session_file(server: Server, session: str) -> Path:
    return server.sessions_dir / session / "session.jsonl"


def mounted_tools(server: Server) -> list[str]:
    """The tool files of every mount directory left on disk: only the current mount stays."""
    return sorted(path.name for path in server.mounts_dir.glob("*/native/tools/*.py"))


# -- sessions and turns ------------------------------------------------------------------------------------


def test_two_turns_in_one_session_keep_the_messages_and_number_the_turns(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", {**RULES, **_tool("shout", "    return 'LOUD'")})
    reef.replies = [_reply("hi"), _reply(tool_calls=[_call("shout", {}, "c1")]), _reply("still here")]
    with _running(_tree(tmp_path, reef, "r1")) as server:
        first, streamed = _turn(server, "say hi")
        assert first == {"exit": 0, "session": first["session"], "turn": 1, "text": "hi", "receipts": ["r-1"]}
        assert len(first["session"]) == 12
        assert [event["type"] for event in streamed][:3] == ["session", "turn/start", "stage/enter"]
        second, _ = _turn(server, "and again", session=first["session"])
        # Each turn returns the receipts of its own model calls, so its caller reports against them.
        assert second == {
            "exit": 0,
            "session": first["session"],
            "turn": 2,
            "text": "still here",
            "receipts": ["r-2", "r-3"],
        }
        assert server.status()["sessions"] == 1

    assert [message["content"] for message in reef.requests[1]["body"]["messages"]] == [
        "Be brief.",
        "say hi",
        "hi",
        "and again",
    ]
    events = _events(_session_file(server, first["session"]))
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert len(_typed(events, "session")) == 1
    header = _typed(events, "session")[0]
    assert header["mode"] == "serve" and header["release_id"] == "r1"
    assert [start["turn"] for start in _typed(events, "turn/start")] == [1, 2]
    assert [start["prompt"] for start in _typed(events, "turn/start")] == ["say hi", "and again"]
    assert [end["turn"] for end in _typed(events, "turn/end")] == [1, 2]
    assert [step["turn"] for step in _typed(events, "step/start")] == [1, 2, 2]
    # One header for the whole session: nothing the model saw changed between the turns.
    assert len(_typed(events, "request/header")) == 1
    assert _typed(events, "tool/result")[0]["content"] == "LOUD"
    assert {request["headers"]["x-reef-tag-release"] for request in reef.requests} == {"r1"}
    assert {request["headers"]["x-reef-scenario"] for request in reef.requests} == {SCENARIO}
    assert list(server.sessions_dir.glob("*/session.jsonl")) == [_session_file(server, first["session"])]
    # stop took the mounts and the socket down
    assert not server.mounts_dir.exists() and not server.socket_path.exists()


def test_a_response_header_moves_the_head_and_the_mount_lands_between_two_steps(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", {**RULES, **_tool("whisper", "    return 'quiet'")})
    reef.replies = [
        _reply(tool_calls=[_call("shout", {}, "c1")]),
        _reply(tool_calls=[_call("shout", {}, "c2")]),
        _reply("ok"),
    ]
    dest = _tree(tmp_path, reef, "r1")
    r2 = {**RULES, **_tool("whisper", "    return 'quiet'"), **_tool("shout", "    return 'LOUD'")}
    with _running(dest) as server:
        # The head moves after the boot poll: the only way this process learns of r2 is the answer's header.
        _wait(lambda: reef.polls >= 1)
        reef.release("r2", r2, parent="r1")
        result, _ = _turn(server, "shout")
        assert result["exit"] == 0 and server.status()["release_id"] == "r2"
        assert sorted(server.host.tools) == ["shout", "whisper"]

    events = _events(_session_file(server, result["session"]))
    kinds = [(event["type"], event["data"].get("step", event["data"].get("release_id"))) for event in events]
    assert kinds.index(("harness/mount", "r2")) > kinds.index(("step/end", 1))
    assert kinds.index(("harness/mount", "r2")) < kinds.index(("step/start", 2))
    mount = _typed(events, "harness/mount")[0]
    assert mount == {"release_id": "r2", "parent_release_id": "r1", "source": "release", "files": 3}
    headers = _typed(events, "request/header")
    assert [[t["function"]["name"] for t in h["tools"]] for h in headers] == [["whisper"], ["shout", "whisper"]]
    results = _typed(events, "tool/result")
    assert results[0]["error"]["code"] == "UNKNOWN_TOOL" and results[1]["content"] == "LOUD"
    assert [request["headers"]["x-reef-tag-release"] for request in reef.requests] == ["r1", "r2", "r2"]
    # What a restart boots from: the stored source tree and the release file name the mounted release.
    assert json.loads((dest / HARNESS_RELEASE_FILE).read_text())["release_id"] == "r2"
    assert HarnessTree.read(dest / serve.SOURCE_DIR) == HarnessTree(r2)


def test_a_restart_boots_from_the_stored_source_tree_without_asking_reef(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", {**RULES, **_tool("shout")})
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest) as server:
        assert server.status()["release_id"] == "r1"
    # Reef no longer knows r1: the next start boots from source/ as the release file names it.
    reef.releases.pop("r1")
    reef.order.remove("r1")
    with _running(dest) as server:
        log = server.sessions_dir / serve.SERVE_LOG
        assert server.status()["release_id"] == "r1" and list(server.host.tools) == ["shout"]
        assert [(m["release_id"], m["source"]) for m in _typed(_events(log), "harness/mount")][-1] == ("r1", "boot")


def test_the_poll_mounts_a_new_head_at_the_turn_boundary_and_the_next_turn_runs_the_new_graph(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", _graph(SEED_STAGES, SEED_EDGES))
    reef.replies = [_reply("one"), _reply("two")]
    with _running(_tree(tmp_path, reef, "r1"), poll_interval_s=0.1) as server:
        first, _ = _turn(server, "first")
        stages = {**SEED_STAGES, "hello": {"kind": "message", "text": "Hello from r2."}}
        edges = [*SEED_EDGES, {"from": "hello", "when": "done", "to": "think"}]
        reef.release("r2", _graph(stages, edges, start="hello"), parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: any(m["release_id"] == "r2" for m in _typed(_events(log), "harness/mount")))
        assert server.status()["release_id"] == "r2" and server.status()["pending_mount"] is None
        _, streamed = _turn(server, "second", session=first["session"])

    # The mount landed while no turn was open, so it went to the process's own log, not a session.
    assert [m["source"] for m in _typed(_events(log), "harness/mount")] == ["boot", "release"]
    assert "harness/mount" not in {event["type"] for event in _events(_session_file(server, first["session"]))}
    assert [e["stage"] for e in _typed(streamed, "stage/enter")][:2] == ["hello", "think"]
    assert reef.requests[1]["body"]["messages"][-1]["content"] == "Hello from r2."


def test_a_failed_mount_keeps_the_previous_tree_serving(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", {**_tool("good"), **_tool("spare")})
    reef.replies = [_reply(tool_calls=[_call("good", {}, "c1")]), _reply("fine")]
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, poll_interval_s=0.1) as server:
        broken = native_tool("broken", "def helper(args, workdir):\n    return 1\n")
        # r2 changes good, drops spare and adds a file that cannot load: none of it may reach the host.
        reef.release("r2", {**_tool("good", "    return 'changed'"), **broken}, parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/mount-failed")))
        failed = _typed(_events(log), "harness/mount-failed")[0]
        assert failed["release_id"] == "r2" and failed["source"] == "release"
        assert "no top level statement of broken.py binds run" in failed["error"]
        assert server.status()["release_id"] == "r1" and sorted(server.host.tools) == ["good", "spare"]
        assert server.host.tools["good"].run({}, ".") == "ok"
        assert mounted_tools(server) == ["good.py", "spare.py"]
        result, streamed = _turn(server, "go")
        assert result["exit"] == 0 and _typed(streamed, "tool/result")[0]["content"] == "ok"
        # Announced once: the head that failed is not retried every poll while it stands.
        time.sleep(0.25)
        assert len(_typed(_events(log), "harness/mount-failed")) == 1

    assert json.loads((dest / HARNESS_RELEASE_FILE).read_text())["release_id"] == "r1"
    assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1"]
    assert reef.requests[0]["headers"]["x-reef-tag-release"] == "r1"


def test_follow_pinned_announces_the_head_and_the_mount_control_applies_it(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", _tool("one"))
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, follow="pinned", poll_interval_s=0.1) as server:
        reef.release("r2", {**_tool("one"), **_tool("two")}, parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "release/available")))
        assert _typed(_events(log), "release/available") == [{"release_id": "r2"}]
        assert server.status()["release_id"] == "r1" and server.status()["follow"] == "pinned"
        reply = serve.request(server.socket_path, {"control": "mount", "release_id": "r2"})
        assert reply == {"type": "control/result", "data": {"mounted": True, "release_id": "r2", "error": None}}
        assert server.status()["release_id"] == "r2" and sorted(server.host.tools) == ["one", "two"]
        missing = serve.request(server.socket_path, {"control": "mount", "release_id": "r9"})
        assert missing["data"]["mounted"] is False and "404" in missing["data"]["error"]

    assert json.loads((dest / HARNESS_RELEASE_FILE).read_text())["parent_release_id"] == "r1"
    assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1", "r2"]


def test_a_poll_failure_is_logged_with_backoff_and_the_process_keeps_serving(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", _tool("one"))
    reef.replies = [_reply("served")]
    reef.fail_releases = True
    with _running(_tree(tmp_path, reef, "r1"), poll_interval_s=0.1) as server:
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: len(_typed(_events(log), "release/poll-failed")) >= 2)
        failures = _typed(_events(log), "release/poll-failed")
        assert "500" in failures[0]["error"] and [f["retry_in_s"] for f in failures[:2]] == [0.1, 0.2]
        result, _ = _turn(server, "go")
        assert result["text"] == "served"
        reef.fail_releases = False
        reef.release("r2", {**_tool("one"), **_tool("two")}, parent="r1")
        _wait(lambda: server.status()["release_id"] == "r2")


def test_a_poll_that_times_out_retries_at_the_interval_and_a_reef_that_is_down_backs_off() -> None:
    """A catalog read that waits behind a running evolve step's lock times out: a busy Reef, polled again at
    the interval so the head lands as soon as the step ends. Only a Reef that does not answer at all backs off."""

    class _Client:
        def __init__(self, outcomes: list[Any]) -> None:
            self.outcomes = outcomes

        def poll(self) -> str | None:
            outcome = self.outcomes.pop(0) if self.outcomes else "r1"
            if isinstance(outcome, Exception):
                raise outcome
            return str(outcome)

    class _ReleaseListener(serve.ReleaseUpdateListener):
        def __init__(self) -> None:
            self.heads: list[tuple[str, str]] = []

        def new_head(self, release_id: str, source: str) -> None:
            self.heads.append((release_id, source))

    events: list[dict[str, Any]] = []

    class _Log(serve.EventWriter):
        def write(self, type_: str, data: Any) -> None:
            events.append({"type": type_, **data})

    busy = ReleaseClientError("GET /reef/harness/releases failed: TimeoutError: timed out", timed_out=True)
    down = ReleaseClientError("GET /reef/harness/releases failed: URLError: Connection refused")
    release_listener = _ReleaseListener()
    watch = HeadWatch(_Client([busy, busy, busy, down, down, "r2"]), release_listener, _Log(), 0.01, mounted="r1")  # type: ignore[arg-type]
    watch.start()
    try:
        _wait(lambda: release_listener.heads == [("r2", "poll")])
    finally:
        watch.stop()
    assert [event["type"] for event in events] == ["release/poll-failed"] * 5
    assert [event["retry_in_s"] for event in events] == [0.01, 0.01, 0.01, 0.01, 0.02]


def test_a_manifest_fetch_that_times_out_is_retried_by_the_next_poll(tmp_path: Path, reef: _FakeReef) -> None:
    """The head moved, the poll saw it, and the manifest read then waited behind the next step's lock: the mount
    fails on the timeout, and the next poll that names the head announces it again instead of leaving it."""
    reef.release("r1", _tool("one"))
    with _running(_tree(tmp_path, reef, "r1"), poll_interval_s=0.1, release_timeout_s=0.2) as server:
        log = server.sessions_dir / serve.SERVE_LOG
        reef.hang_manifest_s = 0.6
        reef.release("r2", {**_tool("one"), **_tool("two")}, parent="r1")
        # The release ID changes before installation and the mount log write finish.
        _wait(
            lambda: any(m["release_id"] == "r2" for m in _typed(_events(log), "harness/mount")),
            timeout_s=10.0,
        )
        assert server.status()["release_id"] == "r2"
        failed = _typed(_events(log), "harness/mount-failed")
        assert [f["release_id"] for f in failed] == ["r2"] and "timed out" in failed[0]["error"]
        assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1", "r2"]


def test_a_release_request_that_times_out_is_marked_busy_and_a_refused_one_is_not() -> None:
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        # A port that accepts the connection and never answers: the read times out, as behind a step's lock.
        client = ReleaseClient(f"http://127.0.0.1:{listener.getsockname()[1]}", None, "s", timeout_s=0.05)
        with pytest.raises(ReleaseClientError) as busy:
            client.releases()
        assert busy.value.timed_out is True and "timed out" in str(busy.value)
        listener.close()
        with pytest.raises(ReleaseClientError) as refused:
            client.releases()
        assert refused.value.timed_out is False and refused.value.status is None
    finally:
        with contextlib.suppress(OSError):
            listener.close()


def test_the_turn_timeout_ends_the_turn_before_the_next_step(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", _tool("slow"))
    reef.replies = [_reply(tool_calls=[_call("slow", {}, "c1")]), _reply("never")]
    reef.delay_s = 0.15
    with _running(_tree(tmp_path, reef, "r1"), turn_timeout_s=0.05) as server:
        result, streamed = _turn(server, "go")
    assert result["exit"] == 0 and result["text"] == ""
    assert _typed(streamed, "turn/end") == [{"turn": 1, "reason": {"kind": "turn-timeout", "seconds": 0.05}}]
    assert len(reef.requests) == 1 and len(_typed(streamed, "tool/result")) == 1


def test_a_malformed_request_answers_an_error_and_the_socket_path_falls_back_to_tmp(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", {})
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest) as server:
        assert serve.request(server.socket_path, {"nothing": 1}) == {
            "type": "error",
            "data": {"message": "request must carry turn or control"},
        }
        assert serve.request(server.socket_path, {"turn": {"prompt": 3}})["type"] == "error"
        assert serve.request(server.socket_path, {"control": "reboot"})["data"]["message"].startswith(
            "unknown control"
        )
        assert serve.request(server.socket_path, {"turn": {"prompt": "x", "session": "bad id"}})["type"] == "error"
        with pytest.raises(ServeError, match="already listens"):
            Server(dest, scenario=SCENARIO).start()
    with pytest.raises(ServeError, match="no serve process"):
        serve.request(server.socket_path, {"control": "status"})
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    (deep / "native").mkdir(parents=True)
    (deep / "native" / "models.json").write_text("{}")
    assert str(serve.socket_path(deep)).startswith("/tmp/reef-native-") and serve.socket_path(deep).suffix == ".sock"
    assert serve.socket_path(deep) == serve.socket_path(deep / "native") != serve.socket_path(dest)
    assert serve.socket_path(dest, Path("/x/own.sock")) == Path("/x/own.sock")
    assert len(str(serve.socket_path(dest)).encode()) <= serve.MAX_SOCKET_PATH_BYTES
    assert serve.tree_layout(dest / "native") == (dest, dest / "native")
    with pytest.raises(ServeError, match="scenario is required"):
        Server(dest, scenario="")
    # No stored source tree, and a release Reef does not know: nothing to boot.
    (dest / HARNESS_RELEASE_FILE).write_text(json.dumps({"release_id": "r9"}) + "\n", encoding="utf-8")
    shutil.rmtree(dest / serve.SOURCE_DIR)
    with pytest.raises(ServeError, match=r"no source tree under .* and Reef answered none"):
        Server(dest, scenario=SCENARIO).start()


# -- the subcommands, end to end ---------------------------------------------------------------------------


def _launcher(tmp_path: Path) -> Path:
    """reef-native as the console script would be: this interpreter running the subcommand dispatch."""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "reef-native"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(root)!r})\n"
        "from reef.harness.runners.native.__main__ import main\nsys.exit(main())\n"
    )
    path.chmod(0o755)
    return path


def test_the_serve_status_turn_and_mount_subcommands_run_end_to_end(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", {**RULES, **_tool("shout", "    return 'LOUD'")})
    reef.replies = [_reply("hello"), _reply(tool_calls=[_call("shout", {}, "c1")]), _reply("loud enough")]
    dest = _tree(tmp_path, reef, "r1")
    binary = _launcher(tmp_path)
    env = {**os.environ, "REEF_HARNESS_SCENARIO": SCENARIO}
    env.pop("REEF_NATIVE_ENFORCE", None)
    process = subprocess.Popen(
        [str(binary), "serve", "--tree", str(dest), "--poll-interval", "0.1"],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        socket_path = serve.socket_path(dest)  # under /tmp when the tree's own path is too long for a socket
        _wait(lambda: socket_path.exists() or process.poll() is not None, timeout_s=30.0)
        assert process.poll() is None, process.stderr.read()  # type: ignore[union-attr]

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run([str(binary), *args, "--tree", str(dest)], capture_output=True, text=True, env=env)

        status = run("status")
        assert status.returncode == 0, status.stderr
        assert json.loads(status.stdout) == {
            "release_id": "r1",
            "parent_release_id": None,
            "follow": "head",
            "files": 2,
            "pending_mount": None,
            "sessions": 0,
            "socket": str(socket_path),
            "self_tools": False,
        }
        turn = run("turn", "-p", "say hello", "--workdir", str(tmp_path))
        assert turn.returncode == 0, turn.stderr
        lines = [json.loads(line) for line in turn.stdout.splitlines()]
        assert lines[0]["type"] == "session" and lines[-1]["type"] == "turn/result"
        session = lines[-1]["data"]["session"]
        assert lines[-1]["data"] == {"exit": 0, "session": session, "turn": 1, "text": "hello", "receipts": ["r-1"]}
        quiet = run("turn", "-p", "shout", "--session", session, "--quiet")
        assert (quiet.returncode, quiet.stdout) == (0, "loud enough\n")
        assert len(reef.requests) == 3 and reef.requests[2]["body"]["messages"][1]["content"] == "say hello"

        reef.release("r2", {**RULES, **_tool("shout", "    return 'LOUD'"), **_tool("hum")}, parent="r1")
        # The wait reads the status over the socket, so it starts no process per poll.
        _wait(lambda: serve.request(socket_path, {"control": "status"})["data"]["release_id"] == "r2", timeout_s=10.0)
        mounted = run("mount", "r1")
        assert mounted.returncode == 0 and json.loads(mounted.stdout)["mounted"] is True
        # A manual move off the head sticks: the head was announced once and is not redone every poll.
        time.sleep(0.3)
        assert json.loads(run("status").stdout)["release_id"] == "r1"
        assert run("mount", "r9").returncode == 1
    finally:
        process.send_signal(signal.SIGTERM)
        stderr = process.communicate(timeout=30)[1]
    assert process.returncode == 0, stderr
    assert "release r1 on" in stderr
    assert not socket_path.exists() and not (dest / "native" / "mounts").exists()
    # The rollout form and its flags are untouched behind the dispatch.
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env)
    assert version.returncode == 0 and version.stdout.startswith("reef-native ")


# -- the self tools ----------------------------------------------------------------------------------------

WHISPER = _tool("whisper", "    return 'quiet'")


def test_a_trial_mount_tags_its_calls_and_is_unmounted_at_turn_end(tmp_path: Path, reef: _FakeReef) -> None:
    served = {**RULES, **_tool("shout", "    return 'LOUD'")}
    reef.release("r1", served)
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"patch": WHISPER}, "c1")]),
        _reply(tool_calls=[_call("whisper", {}, "c2")]),
        _reply("done"),
    ]
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, self_tools=True) as server:
        assert sorted(server.host.tools) == ["harness_inspect", "harness_try", "shout"]
        result, streamed = _turn(server, "try a whisper")
        assert result["exit"] == 0 and result["text"] == "done"
        assert sorted(server.host.tools) == ["harness_inspect", "harness_try", "shout"]
        assert server.live_tree() == HarnessTree(served)
        assert "whisper.py" not in mounted_tools(server)

    results = _typed(streamed, "tool/result")
    tried = json.loads(results[0]["content"])
    assert tried["mounted"] is True and len(tried["try_id"]) == 8
    assert results[0]["enforcement"] == {"mode": "none", "denied": []} and results[1]["content"] == "quiet"
    mount = _typed(streamed, "harness/mount")[0]
    assert mount["source"] == "try" and mount["try_id"] == tried["try_id"] and mount["release_id"] == "r1"
    assert mount["paths"] == ["native/tools/whisper.py"] and mount["files"] == 3
    headers = _typed(streamed, "request/header")
    assert [t["function"]["name"] for t in headers[1]["tools"]][-2:] == ["shout", "whisper"]
    types = [event["type"] for event in streamed]
    assert types.index("harness/unmount") > types.index("turn/end")
    assert _typed(streamed, "harness/unmount")[0]["try_id"] == tried["try_id"]
    assert [r["headers"].get("x-reef-tag-trial") for r in reef.requests] == [None, tried["try_id"], tried["try_id"]]
    assert {r["headers"]["x-reef-tag-release"] for r in reef.requests} == {"r1"}
    # A trial is not served: the release file and the stored source tree still name r1.
    assert json.loads((dest / HARNESS_RELEASE_FILE).read_text())["release_id"] == "r1"
    assert HarnessTree.read(dest / serve.SOURCE_DIR) == HarnessTree(served)


def test_a_turns_receipts_leave_out_the_calls_a_trial_made(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", {**RULES, **_tool("shout", "    return 'LOUD'")})
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"patch": WHISPER}, "c1")]),
        _reply(tool_calls=[_call("whisper", {}, "c2")]),
        _reply("done"),
    ]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, _ = _turn(server, "try a whisper")
    assert result["receipts"] == ["r-1"]


def test_a_trial_that_fails_validation_or_load_is_refused_and_the_served_tree_stays(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", _tool("shout", "    return 'LOUD'"))
    off_layout = {"native/tools/shout.md": "not a tool"}
    broken = native_tool("whisper", "def helper(args, workdir):\n    return 1\n")
    # Validation passes it (listen is bound, EVENT is a literal), and its top level raises when the mount imports it.
    raising = native_hook(
        "boom", "pre_step", "raise RuntimeError('boom')\ndef listen(payload, next):\n    return next()\n"
    )
    loop = native_json("graphs", {"name": "x"}) | {"native/loops/main.py": "def run_turn(ctx):\n    return\n"}
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"patch": off_layout}, "c1")]),
        _reply(tool_calls=[_call("harness_try", {"patch": broken}, "c2")]),
        _reply(tool_calls=[_call("harness_try", {"patch": raising}, "c6")]),
        _reply(tool_calls=[_call("harness_try", {"patch": loop}, "c3")]),
        _reply(tool_calls=[_call("harness_try", {"patch": {}}, "c4")]),
        _reply(tool_calls=[_call("harness_try", {"patch": "nope"}, "c5")]),
        _reply("done"),
    ]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "try things")
        assert result["exit"] == 0 and server.live_tree() == HarnessTree(_tool("shout", "    return 'LOUD'"))
    results = _typed(streamed, "tool/result")
    assert "native/tools/shout.md: not a path of the native harness tree" in json.loads(results[0]["content"])["error"]
    assert "binds no run(args, workdir) at its top level" in json.loads(results[1]["content"])["error"]
    assert "boom.py failed to import: RuntimeError: boom" in json.loads(results[2]["content"])["error"]
    # A native loop is reviewed code: a trial never mounts one.
    assert results[3]["error"]["code"] == "TOOL_FAILED" and "is reviewed code" in results[3]["content"]
    assert (
        results[4]["error"]["code"] == "TOOL_FAILED" and "patch must change at least one path" in results[4]["content"]
    )
    assert results[5]["error"]["code"] == "INVALID_ARGS"  # the declaration's schema holds before run sees the call
    # Validation refuses the first two before a mount; only the hook that raises on import reaches one and fails.
    failed = _typed(streamed, "harness/mount-failed")
    assert [f["source"] for f in failed] == ["try"]
    assert "harness/unmount" not in {event["type"] for event in streamed}
    assert [r["headers"].get("x-reef-tag-trial") for r in reef.requests] == [None] * 7


def test_harness_inspect_reads_the_tree_the_graphs_the_results_and_the_status(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r0", _tool("shout"))
    rejected = [{"step": 2, "patch": {"skills/x/SKILL.md": "# x"}, "reason": "lost"}]
    served = {**_tool("shout"), **_graph(SEED_STAGES, SEED_EDGES)}
    reef.release("r1", served, parent="r0", rejected=rejected)
    reef.replies = [
        _reply(tool_calls=[_call("harness_inspect", {"what": what}, f"c-{what}")])
        for what in ("tree", "graph", "results", "status", "nope")
    ]
    reef.replies.append(_reply("looked"))
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "look around")
        socket_path = server.socket_path
    assert result["text"] == "looked"
    tree, graph, results, status, nope = _typed(streamed, "tool/result")
    assert json.loads(tree["content"]) == {"release_id": "r1", "files": served}
    graphs = json.loads(graph["content"])
    assert graphs["main"]["start"] == "think" and list(graphs["graphs"]) == ["main"]
    seen = json.loads(results["content"])
    assert [row["release_id"] for row in seen["releases"]] == ["r1", "r0"]
    assert seen["releases"][0]["evaluation"] == {"wins": 1, "rejected": rejected} and seen["head"] == "r1"
    assert seen["rejected"] == rejected
    assert json.loads(status["content"]) == {
        "release_id": "r1",
        "parent_release_id": "r0",
        "follow": "head",
        "files": 2,
        "pending_mount": None,
        "sessions": 1,
        "socket": str(socket_path),
        "self_tools": True,
    }
    assert nope["error"]["code"] == "TOOL_FAILED" and "what must be one of" in nope["content"]


def test_a_tree_tool_with_a_self_tools_name_does_not_mount(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", _tool("shout"))
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, self_tools=True, poll_interval_s=0.1) as server:
        reef.release("r2", {**_tool("shout"), **_tool(NATIVE_RESERVED_TOOL_NAMES[0])}, parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/mount-failed")))
        failed = _typed(_events(log), "harness/mount-failed")[0]
        assert failed["release_id"] == "r2" and "'harness_inspect' is already installed" in failed["error"]
        assert server.status()["release_id"] == "r1" and "shout" in server.host.tools
    # A stored source tree that holds one does not boot either.
    (dest / serve.SOURCE_DIR / "native" / "tools" / "harness_try.py").write_text(
        _tool(NATIVE_RESERVED_TOOL_NAMES[1])["native/tools/harness_try.py"], encoding="utf-8"
    )
    with pytest.raises(ServeError, match="'harness_try' is already installed"):
        Server(dest, scenario=SCENARIO, self_tools=True).start()
    assert not (dest / "native" / "serve.sock").exists()


def test_validation_refuses_a_tree_tool_with_a_self_tools_name() -> None:
    for name in NATIVE_RESERVED_TOOL_NAMES:
        (problem,) = validate_tree(HarnessTree(_tool(name)), NATIVE).problems
        assert problem == f"native/tools/{name}.py: tool name {name!r} is reserved for a built-in tool"


# -- the pieces on their own -------------------------------------------------------------------------------


def test_the_capture_proxy_tags_calls_live_and_drains_each_capture_with_its_tags(reef: _FakeReef) -> None:
    import urllib.request

    reef.release("r1", {})
    seen: list[str] = []

    class Observer(ReleaseObserver):
        def observe(self, release_id: str) -> None:
            seen.append(release_id)

    proxy = CaptureProxy(reef.base_url, SCENARIO, "tok", tags={"release": "r1"}, observer=Observer())
    proxy.start()
    try:

        def call() -> None:
            request = urllib.request.Request(
                f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=5).read()

        call()
        proxy.tags["trial"] = "t1"
        call()
        proxy.tags.pop("trial")
        first = [(capture["receipt"], capture["tags"]) for capture in proxy.drain()]
        assert proxy.drain() == []
        call()
        second = [(capture["receipt"], capture["tags"]) for capture in proxy.drain()]
    finally:
        proxy.stop()
    assert seen == ["r1", "r1", "r1"]
    tags = [(r["headers"].get("x-reef-tag-release"), r["headers"].get("x-reef-tag-trial")) for r in reef.requests]
    assert tags == [("r1", None), ("r1", "t1"), ("r1", None)]
    assert {r["headers"]["authorization"] for r in reef.requests} == {"Bearer tok"}
    assert first == [("r-1", {"release": "r1"}), ("r-2", {"release": "r1", "trial": "t1"})]
    assert second == [("r-3", {"release": "r1"})]


def test_a_mount_that_orphans_the_pinned_graph_closes_the_turn_with_an_error(tmp_path: Path, reef: _FakeReef) -> None:
    helper = native_json("agents", {"name": "helper", "prompt": "You help.", "max_steps": 1})
    stages = {
        "think": {"kind": "model"},
        "act": {"kind": "tools"},
        "delegate": {"kind": "subagent", "agent": "helper"},
        "done": {"kind": "end", "reason": "completed"},
    }
    edges = [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "delegate"},
        {"from": "act", "when": "done", "to": "think"},
        *(
            {"from": "delegate", "when": outcome, "to": "done"}
            for outcome in ("completed", "gave_up", "budget", "ask")
        ),
    ]
    reef.release("r1", {**_tool("shout", "    return 'LOUD'"), **helper, **_graph(stages, edges)})
    dest = _tree(tmp_path, reef, "r1")
    # The first answer names r2, which drops the helper and the delegate stage; the graph this turn walks is r1's.
    reef.replies = [_reply(tool_calls=[_call("shout", {}, "c1")]), _reply("thinking done"), _reply("helper")]
    with _running(dest) as server:
        _wait(lambda: reef.polls >= 1)
        reef.release("r2", {**_tool("shout", "    return 'LOUD'"), **_graph(SEED_STAGES, SEED_EDGES)}, parent="r1")
        result, streamed = _turn(server, "go")
        assert result["exit"] == 1 and result["text"] == ""
        ended = _typed(streamed, "turn/end")[-1]
        assert ended["reason"]["kind"] == "error" and ended["reason"]["error"]["code"] == "TURN_ERROR"
        assert "helper" in ended["reason"]["error"]["message"]
        assert server.status()["release_id"] == "r2"
        # The process and the session both survive: the next turn runs the mounted graph.
        reef.replies = [_reply("second")]
        again, _ = _turn(server, "again", session=result["session"])
        assert again["exit"] == 0 and again["text"] == "second" and again["turn"] == 2


def test_a_failed_mount_keeps_the_current_host_without_importing_it_again(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "once.marker"
    monkeypatch.setenv("ONCE_MARKER", str(marker))
    # A hook imports at mount (a tool module imports only where a call runs), and this import is not idempotent:
    # a second import of the same module fails on the marker it left.
    once = native_hook(
        "once",
        "pre_step",
        "import os\nopen(os.environ['ONCE_MARKER'], 'x').close()\n\n\ndef listen(payload, next):\n    return next()\n",
    )
    reef.release("r1", {**once, **_tool("other")})
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, poll_interval_s=0.1) as server:
        assert [hook.name for hook in server.host.hooks["pre_step"]] == ["once"] and marker.exists()
        reef.release("r2", native_tool("broken", "def helper(args, workdir):\n    return 1\n"), parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/mount-failed")))
        # The new tree was built beside the current one, so the current host stands as it was.
        assert server.status()["release_id"] == "r1" and sorted(server.host.tools) == ["other"]
        assert [hook.name for hook in server.host.hooks["pre_step"]] == ["once"]
        assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1"]
        # A mount of a tree with the hook imports it afresh in its own directory.
        marker.unlink()
        reef.release("r3", {**once, **_tool("other")}, parent="r1")
        _wait(lambda: server.status()["release_id"] == "r3")
        assert [hook.name for hook in server.host.hooks["pre_step"]] == ["once"] and marker.exists()


def test_a_session_from_a_previous_process_is_not_silently_resumed(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", _graph(SEED_STAGES, SEED_EDGES))
    dest = _tree(tmp_path, reef, "r1")
    reef.replies = [_reply("first")]
    with _running(dest) as server:
        first, _ = _turn(server, "one", session="abc123")
        assert first["turn"] == 1
    reef.replies = [_reply("second")]
    with _running(dest) as server:
        payload = {"turn": {"prompt": "two", "session": "abc123", "workdir": str(dest)}}
        reply = serve.request(server.socket_path, payload, _Lines())
        assert reply["type"] == "error" and "predates this process" in reply["data"]["message"]
        fresh, _ = _turn(server, "two")
        assert fresh["turn"] == 1 and fresh["session"] != "abc123"
    events = _events(_session_file(server, "abc123"))
    assert [e["data"]["turn"] for e in events if e["type"] == "turn/start"] == [1]


def test_the_release_client_poll_reads_the_active_row() -> None:
    class _Catalog(serve.ReleaseClient):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__("http://127.0.0.1:9", "tok", SCENARIO)
            self._rows = rows

        def releases(self) -> list[dict[str, Any]]:
            return list(self._rows)

    assert _Catalog([]).poll() is None
    r1 = {"release_id": "r1", "artifact_id": "r1", "activated": True}
    r2 = {"release_id": "r2", "artifact_id": "r2", "activated": False}
    # A newer inactive release is not the active release.
    assert _Catalog([{**r1, "active": True}, {**r2, "active": False}]).poll() == "r1"
    assert _Catalog([{**r1, "active": False}, {**r2, "active": True, "activated": True}]).poll() == "r2"
    assert _Catalog([{**r1, "active": False}]).poll() is None


def test_the_self_tools_run_in_process_when_the_environment_names_bwrap(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "bwrap").write_text("#!/bin/sh\necho 'bwrap: must not run for a built-in tool' >&2\nexit 1\n")
    (fake / "bwrap").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("REEF_NATIVE_ENFORCE", "bwrap")
    reef.release("r1", {**_tool("shout"), **_graph(SEED_STAGES, SEED_EDGES)})
    reef.replies = [_reply(tool_calls=[_call("harness_inspect", {"what": "status"}, "c1")]), _reply("seen")]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "look")
    assert result["text"] == "seen"
    status = _typed(streamed, "tool/result")[0]
    assert status["is_error"] is False and json.loads(status["content"])["release_id"] == "r1"
    assert status["enforcement"] == {"mode": "none", "denied": []}
    assert _typed(streamed, "session")[0]["enforcement"] == "bwrap"
