"""The serve form: one resident process holds the tree live, follows Reef's head between steps, and lets the model inspect, try and propose.

A fake Reef on localhost plays the model (a scripted reply per request, the
receipt and release headers on every answer), the release catalog, one
manifest per release, the proposals route and the report route, so every
path of ``reef-native serve`` runs hermetically."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from reef.harness import harness_wrapper
from reef.harness.harness_wrapper import HARNESS_RELEASE_SIDECAR, CaptureProxy
from reef.harness.native import serve
from reef.harness.native.release_client import HeadWatch, ReleaseClient, ReleaseClientError
from reef.harness.native.selftools import RESERVED_NAMES
from reef.harness.native.serve import ServeError, Server, admit_mutations
from reef.train.cordis_backend.strategies import Mutation

SCENARIO = "serve-demo"


def _entry(id_: str, kind: str, **config: Any) -> dict[str, Any]:
    return {"id": id_, "name": kind, "config": config}


def _tool(name: str, body: str = "    return 'ok'", id_: str | None = None) -> dict[str, Any]:
    return _entry(
        id_ or name,
        "native_tool",
        name=name,
        description=f"the {name} tool",
        parameters={"type": "object", "properties": {}},
        code=f"def run(args, workdir):\n{body}\n",
    )


def _graph(stages: dict[str, Any], edges: list[dict[str, str]], start: str = "think") -> dict[str, Any]:
    return _entry("main", "native_graph", name="main", start=start, max_steps=8, stages=stages, edges=edges)


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
    """The model, the release catalog, one manifest per release, the proposals route and the report route."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.releases: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.head = ""
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []
        self.proposals: list[dict[str, Any]] = []
        self.proposals_route = True
        self.fail_releases = False
        self.delay_s = 0.0
        self.polls = 0
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def release(self, release_id: str, entries: list[dict], parent: str | None = None, **gate: Any) -> None:
        self.releases[release_id] = {
            "release_id": release_id,
            "parent_release_id": parent,
            "content_id": f"content-{release_id}",
            "operation": "training",
            "pending": False,
            "files": {"native/tree.json": json.dumps(entries, indent=2, sort_keys=True) + "\n"},
            "metrics": {"wins": 1, **gate},
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
                row["current"] = row["release_id"] == reef.head
            self._json(200, {"scenario": SCENARIO, "releases": rows})
        elif split.path == "/reef/harness":
            release_id = parse_qs(split.query).get("release_id", [reef.head])[0]
            row = reef.releases.get(release_id)
            if row is None:
                self._json(404, {"error": {"message": f"unknown release {release_id}"}})
                return
            manifest = {k: v for k, v in row.items() if k not in ("metrics", "operation", "pending")}
            manifest["gate"] = row["metrics"]
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
        elif self.path == "/reef/harness/proposals":
            if not reef.proposals_route:
                self._json(404, {"error": {"message": "no route"}})
                return
            reef.proposals.append({"headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            self._json(
                200,
                {"proposal_id": f"p{len(reef.proposals)}", "admitted": True, "reason": None, "release_id": reef.head},
            )
        elif self.path == "/reef/report":
            reef.reports.append(body)
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": {"message": "no route"}})

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def reef() -> Iterator[_FakeReef]:
    server = _FakeReef()
    try:
        yield server
    finally:
        server.close()


def _tree(tmp_path: Path, reef: _FakeReef, release_id: str) -> Path:
    """The pulled tree of one release: native/tree.json, the model binding at the fake reef, the sidecar."""
    dest = tmp_path / "reef-harness"
    (dest / "native").mkdir(parents=True)
    (dest / "native" / "tree.json").write_text(
        reef.releases[release_id]["files"]["native/tree.json"], encoding="utf-8"
    )
    (dest / "native" / "models.json").write_text(
        json.dumps({"api": "openai", "base_url": reef.base_url, "api_key": "tok", "model": "fake"}) + "\n",
        encoding="utf-8",
    )
    (dest / HARNESS_RELEASE_SIDECAR).write_text(json.dumps({"release_id": release_id}) + "\n", encoding="utf-8")
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


class _Lines:
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


# -- sessions and turns ------------------------------------------------------------------------------------


def test_two_turns_in_one_session_keep_the_messages_and_number_the_turns(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_entry("r", "rules", text="Be brief."), _tool("shout", "    return 'LOUD'")])
    reef.replies = [_reply("hi"), _reply(tool_calls=[_call("shout", {}, "c1")]), _reply("still here")]
    with _running(_tree(tmp_path, reef, "r1")) as server:
        first, streamed = _turn(server, "say hi")
        assert first == {"exit": 0, "session": first["session"], "turn": 1, "text": "hi"}
        assert len(first["session"]) == 12
        assert [event["type"] for event in streamed][:3] == ["session", "turn/start", "stage/enter"]
        second, _ = _turn(server, "and again", session=first["session"])
        assert second == {"exit": 0, "session": first["session"], "turn": 2, "text": "still here"}
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
    assert header["mode"] == "serve" and header["release_id"] == "r1" and header["tree"] == "tree.json"
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
    assert (
        not server.mount_dir.exists() and not server.socket_path.exists()
    )  # stop took the modules and the socket down


def test_a_response_header_moves_the_head_and_the_mount_lands_between_two_steps(
    tmp_path: Path, reef: _FakeReef
) -> None:
    rule = _entry("r", "rules", text="Be brief.")
    reef.release("r1", [rule, _tool("whisper", "    return 'quiet'")])
    reef.replies = [
        _reply(tool_calls=[_call("shout", {}, "c1")]),
        _reply(tool_calls=[_call("shout", {}, "c2")]),
        _reply("ok"),
    ]
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest) as server:
        # The head moves after the boot poll: the only way this process learns of r2 is the answer's header.
        _wait(lambda: reef.polls >= 1)
        reef.release(
            "r2", [rule, _tool("whisper", "    return 'quiet'"), _tool("shout", "    return 'LOUD'")], parent="r1"
        )
        result, _ = _turn(server, "shout")
        assert result["exit"] == 0 and server.status()["release_id"] == "r2"
        assert sorted(server.host.tools) == ["shout", "whisper"]

    events = _events(_session_file(server, result["session"]))
    kinds = [(event["type"], event["data"].get("step", event["data"].get("release_id"))) for event in events]
    assert kinds.index(("harness/mount", "r2")) > kinds.index(("step/end", 1))
    assert kinds.index(("harness/mount", "r2")) < kinds.index(("step/start", 2))
    mount = _typed(events, "harness/mount")[0]
    assert mount == {"release_id": "r2", "parent_release_id": "r1", "source": "release", "entries": 3}
    headers = _typed(events, "request/header")
    assert [[t["function"]["name"] for t in h["tools"]] for h in headers] == [["whisper"], ["shout", "whisper"]]
    results = _typed(events, "tool/result")
    assert results[0]["error"]["code"] == "UNKNOWN_TOOL" and results[1]["content"] == "LOUD"
    assert [request["headers"]["x-reef-tag-release"] for request in reef.requests] == ["r1", "r2", "r2"]
    # What a restart boots from: the tree file and the sidecar name the mounted release.
    assert json.loads((dest / HARNESS_RELEASE_SIDECAR).read_text())["release_id"] == "r2"
    assert [entry["id"] for entry in json.loads((dest / "native" / "tree.json").read_text())] == [
        "r",
        "whisper",
        "shout",
    ]


def test_the_poll_mounts_a_new_head_at_the_turn_boundary_and_the_next_turn_runs_the_new_graph(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", [_graph(SEED_STAGES, SEED_EDGES)])
    reef.replies = [_reply("one"), _reply("two")]
    with _running(_tree(tmp_path, reef, "r1"), poll_interval_s=0.1) as server:
        first, _ = _turn(server, "first")
        stages = {**SEED_STAGES, "hello": {"kind": "message", "text": "Hello from r2."}}
        edges = [*SEED_EDGES, {"from": "hello", "when": "done", "to": "think"}]
        reef.release("r2", [_graph(stages, edges, start="hello")], parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: any(m["release_id"] == "r2" for m in _typed(_events(log), "harness/mount")))
        assert server.status()["release_id"] == "r2" and server.status()["pending_mount"] is None
        _, streamed = _turn(server, "second", session=first["session"])

    # The mount landed while no turn was open, so it went to the process's own log, not a session.
    assert [m["source"] for m in _typed(_events(log), "harness/mount")] == ["boot", "release"]
    assert "harness/mount" not in {event["type"] for event in _events(_session_file(server, first["session"]))}
    assert [e["stage"] for e in _typed(streamed, "stage/enter")][:2] == ["hello", "think"]
    assert reef.requests[1]["body"]["messages"][-1]["content"] == "Hello from r2."


def test_a_failed_mount_rolls_back_and_the_previous_composition_keeps_serving(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("good"), _tool("spare")])
    reef.replies = [_reply(tool_calls=[_call("good", {}, "c1")]), _reply("fine")]
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, poll_interval_s=0.1) as server:
        broken = _tool("broken", "    return 1")
        broken["config"]["code"] = "raise RuntimeError('boom')\n\ndef run(args, workdir):\n    return 1\n"
        # r2 changes good, drops spare and adds an entry that cannot import: the whole of it must come undone.
        reef.release("r2", [_tool("good", "    return 'changed'"), broken], parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/mount-failed")))
        failed = _typed(_events(log), "harness/mount-failed")[0]
        assert failed["release_id"] == "r2" and failed["source"] == "release" and failed["entry"] == "broken"
        assert "boom" in failed["error"]
        assert server.status()["release_id"] == "r1" and sorted(server.host.tools) == ["good", "spare"]
        assert server.host.tools["good"].run({}, ".") == "ok"
        assert sorted(p.name for p in (server.mount_dir / "tools").glob("*.py")) == ["good.py", "spare.py"]
        result, streamed = _turn(server, "go")
        assert result["exit"] == 0 and _typed(streamed, "tool/result")[0]["content"] == "ok"
        # Announced once: the head that failed is not retried every poll while it stands.
        time.sleep(0.35)
        assert len(_typed(_events(log), "harness/mount-failed")) == 1

    assert json.loads((dest / HARNESS_RELEASE_SIDECAR).read_text()) == {"release_id": "r1"}
    assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1"]
    assert reef.requests[0]["headers"]["x-reef-tag-release"] == "r1"


def test_follow_pinned_announces_the_head_and_the_mount_control_applies_it(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("one")])
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, follow="pinned", poll_interval_s=0.1) as server:
        reef.release("r2", [_tool("one"), _tool("two")], parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "release/available")))
        assert _typed(_events(log), "release/available") == [{"release_id": "r2"}]
        assert server.status()["release_id"] == "r1" and server.status()["follow"] == "pinned"
        reply = serve.request(server.socket_path, {"control": "mount", "release_id": "r2"})
        assert reply == {"type": "control/result", "data": {"mounted": True, "release_id": "r2", "error": None}}
        assert server.status()["release_id"] == "r2" and sorted(server.host.tools) == ["one", "two"]
        missing = serve.request(server.socket_path, {"control": "mount", "release_id": "r9"})
        assert missing["data"]["mounted"] is False and "404" in missing["data"]["error"]

    assert json.loads((dest / HARNESS_RELEASE_SIDECAR).read_text())["parent_release_id"] == "r1"
    assert [m["release_id"] for m in _typed(_events(log), "harness/mount")] == ["r1", "r2"]


def test_a_poll_failure_is_logged_with_backoff_and_the_process_keeps_serving(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("one")])
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
        reef.release("r2", [_tool("one"), _tool("two")], parent="r1")
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

    class _Sink:
        def __init__(self) -> None:
            self.heads: list[tuple[str, str]] = []

        def new_head(self, release_id: str, source: str) -> None:
            self.heads.append((release_id, source))

    events: list[dict[str, Any]] = []

    class _Log:
        def write(self, type_: str, data: Any) -> None:
            events.append({"type": type_, **data})

    busy = ReleaseClientError("GET /reef/harness/releases failed: TimeoutError: timed out", timed_out=True)
    down = ReleaseClientError("GET /reef/harness/releases failed: URLError: Connection refused")
    sink = _Sink()
    watch = HeadWatch(_Client([busy, busy, busy, down, down, "r2"]), sink, _Log(), 0.01, mounted="r1")  # type: ignore[arg-type]
    watch.start()
    try:
        _wait(lambda: sink.heads == [("r2", "poll")])
    finally:
        watch.stop()
    assert [event["type"] for event in events] == ["release/poll-failed"] * 5
    assert [event["retry_in_s"] for event in events] == [0.01, 0.01, 0.01, 0.01, 0.02]


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
    reef.release("r1", [_tool("slow")])
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
    reef.release("r1", [])
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
    (dest / "native" / "tree.json").unlink()
    with pytest.raises(ServeError, match=r"tree\.json is missing"):
        Server(dest, scenario=SCENARIO).start()


# -- the subcommands, end to end ---------------------------------------------------------------------------


def _launcher(tmp_path: Path) -> Path:
    """reef-native as the console script would be: this interpreter running the subcommand dispatch."""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "reef-native"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(root)!r})\n"
        "from reef.harness.native.__main__ import main\nsys.exit(main())\n"
    )
    path.chmod(0o755)
    return path


def test_the_serve_status_turn_and_mount_subcommands_run_end_to_end(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_entry("r", "rules", text="Be brief."), _tool("shout", "    return 'LOUD'")])
    reef.replies = [_reply("hello"), _reply(tool_calls=[_call("shout", {}, "c1")]), _reply("loud enough")]
    dest = _tree(tmp_path, reef, "r1")
    binary = _launcher(tmp_path)
    env = {**os.environ, "REEF_HARNESS_SCENARIO": SCENARIO}
    env.pop("REEF_NATIVE_ENFORCE", None)
    process = subprocess.Popen(
        [str(binary), "serve", "--tree", str(dest), "--poll-interval", "0.2"],
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
            "entries": 2,
            "degraded": [],
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
        assert lines[-1]["data"] == {"exit": 0, "session": session, "turn": 1, "text": "hello"}
        quiet = run("turn", "-p", "shout", "--session", session, "--quiet")
        assert (quiet.returncode, quiet.stdout) == (0, "loud enough\n")
        assert len(reef.requests) == 3 and reef.requests[2]["body"]["messages"][1]["content"] == "say hello"

        reef.release(
            "r2",
            [_entry("r", "rules", text="Be brief."), _tool("shout", "    return 'LOUD'"), _tool("hum")],
            parent="r1",
        )
        _wait(lambda: json.loads(run("status").stdout)["release_id"] == "r2", timeout_s=10.0)
        mounted = run("mount", "r1")
        assert mounted.returncode == 0 and json.loads(mounted.stdout)["mounted"] is True
        # A manual move off the head sticks: the head was announced once and is not redone every poll.
        time.sleep(0.5)
        assert json.loads(run("status").stdout)["release_id"] == "r1"
        assert run("mount", "r9").returncode == 1
    finally:
        process.send_signal(signal.SIGTERM)
        stderr = process.communicate(timeout=30)[1]
    assert process.returncode == 0, stderr
    assert "release r1 on" in stderr
    assert not socket_path.exists() and not (dest / "native" / "mounts").exists()
    # The episode form and its flags are untouched behind the dispatch.
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env)
    assert version.returncode == 0 and version.stdout.startswith("reef-native ")


# -- the self tools ----------------------------------------------------------------------------------------


def _whisper_mutation() -> dict[str, Any]:
    tool = _tool("whisper", "    return 'quiet'")
    return {"op": "create", "id": "whisper", "options": {"name": tool["name"], "config": tool["config"]}}


def test_a_trial_mount_tags_its_calls_and_is_unmounted_at_turn_end(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    reef.release("r1", [_entry("r", "rules", text="Be brief."), _tool("shout", "    return 'LOUD'")])
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"mutations": [_whisper_mutation()]}, "c1")]),
        _reply(tool_calls=[_call("whisper", {}, "c2")]),
        _reply("done"),
    ]
    dest = _tree(tmp_path, reef, "r1")
    captures = tmp_path / "captures"
    monkeypatch.setenv("REEF_HARNESS_CAPTURES_DIR", str(captures))
    with _running(dest, self_tools=True) as server:
        assert sorted(server.host.tools) == ["harness_inspect", "harness_propose", "harness_try", "shout"]
        result, streamed = _turn(server, "try a whisper")
        assert result["exit"] == 0 and result["text"] == "done"
        assert sorted(server.host.tools) == ["harness_inspect", "harness_propose", "harness_try", "shout"]
        assert [entry["id"] for entry in server.live_entries()] == ["r", "shout"]
        assert not (server.mount_dir / "tools" / "whisper.py").exists()

    results = _typed(streamed, "tool/result")
    tried = json.loads(results[0]["content"])
    assert tried["mounted"] is True and len(tried["try_id"]) == 8
    assert results[0]["enforcement"] == {"mode": "none", "denied": []} and results[1]["content"] == "quiet"
    mount = _typed(streamed, "harness/mount")[0]
    assert mount["source"] == "try" and mount["try_id"] == tried["try_id"] and mount["release_id"] == "r1"
    assert mount["mutations"] == [_whisper_mutation()]
    headers = _typed(streamed, "request/header")
    assert [t["function"]["name"] for t in headers[1]["tools"]][-2:] == ["shout", "whisper"]
    types = [event["type"] for event in streamed]
    assert types.index("harness/unmount") > types.index("turn/end")
    assert _typed(streamed, "harness/unmount")[0]["try_id"] == tried["try_id"]
    assert [r["headers"].get("x-reef-tag-trial") for r in reef.requests] == [None, tried["try_id"], tried["try_id"]]
    assert {r["headers"]["x-reef-tag-release"] for r in reef.requests} == {"r1"}
    assert json.loads((dest / HARNESS_RELEASE_SIDECAR).read_text()) == {"release_id": "r1"}
    # report claims the turn's spool and sends the one receipt that was not a trial's.
    monkeypatch.setenv("REEF_HARNESS_COMPOSE", str(dest / "native"))
    harness_wrapper.report(SCENARIO, "native", 1.0, "tried it")
    assert [report["references"] for report in reef.reports] == [["r-1"]]
    assert reef.reports[0]["metadata"] == {"client_release": "r1"}


def test_a_trial_that_fails_admission_or_import_is_refused_and_rolled_back(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("shout", "    return 'LOUD'")])
    bad_kind = {"op": "update", "id": "shout", "options": {"name": "skill", "config": {"name": "shout", "text": "x"}}}
    broken = _whisper_mutation()
    broken["options"]["config"]["code"] = "raise RuntimeError('boom')\n\ndef run(args, workdir):\n    return 1\n"
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"mutations": [bad_kind]}, "c1")]),
        _reply(tool_calls=[_call("harness_try", {"mutations": [broken]}, "c2")]),
        _reply(tool_calls=[_call("harness_try", {"mutations": [{"op": "explode", "id": "x"}]}, "c3")]),
        _reply(tool_calls=[_call("harness_try", {"mutations": "nope"}, "c4")]),
        _reply("done"),
    ]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "try things")
        assert result["exit"] == 0 and [entry["id"] for entry in server.live_entries()] == ["shout"]
    results = _typed(streamed, "tool/result")
    assert "cannot change the entry's kind" in json.loads(results[0]["content"])["error"]
    second = json.loads(results[1]["content"])
    assert second["entry"] == "whisper" and "boom" in second["error"]
    assert (
        results[2]["error"]["code"] == "TOOL_FAILED" and "mutations[0]: mutation op must be" in results[2]["content"]
    )
    assert results[3]["error"]["code"] == "INVALID_ARGS"  # the declaration's schema holds before run sees the call
    failed = _typed(streamed, "harness/mount-failed")
    assert [f["source"] for f in failed] == ["try"] and failed[0]["entry"] == "whisper"
    assert "harness/unmount" not in {event["type"] for event in streamed}
    assert [r["headers"].get("x-reef-tag-trial") for r in reef.requests] == [None] * 5


def test_harness_inspect_reads_the_tree_the_graphs_the_verdicts_and_the_status(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r0", [_tool("shout")])
    rejected = [
        {"step": 2, "mutations": [{"op": "create", "id": "x", "options": {"name": "skill"}}], "reason": "lost"}
    ]
    reef.release("r1", [_tool("shout"), _graph(SEED_STAGES, SEED_EDGES)], parent="r0", rejected=rejected)
    reef.replies = [
        _reply(tool_calls=[_call("harness_inspect", {"what": what}, f"c-{what}")])
        for what in ("tree", "graph", "verdicts", "status", "nope")
    ]
    reef.replies.append(_reply("looked"))
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "look around")
        socket_path = server.socket_path
    assert result["text"] == "looked"
    tree, graph, verdicts, status, nope = _typed(streamed, "tool/result")
    assert json.loads(tree["content"]) == {
        "release_id": "r1",
        "entries": [
            {"id": "shout", "kind": "native_tool", "config": _tool("shout")["config"]},
            {"id": "main", "kind": "native_graph", "config": _graph(SEED_STAGES, SEED_EDGES)["config"]},
        ],
    }
    graphs = json.loads(graph["content"])
    assert graphs["main"]["start"] == "think" and list(graphs["graphs"]) == ["main"]
    seen = json.loads(verdicts["content"])
    assert [row["release_id"] for row in seen["releases"]] == ["r1", "r0"]
    assert seen["releases"][0]["gate"] == {"wins": 1, "rejected": rejected} and seen["head"] == "r1"
    assert seen["rejected"] == rejected
    assert json.loads(status["content"]) == {
        "release_id": "r1",
        "parent_release_id": None,
        "follow": "head",
        "entries": 2,
        "degraded": [],
        "pending_mount": None,
        "sessions": 1,
        "socket": str(socket_path),
        "self_tools": True,
    }
    assert nope["error"]["code"] == "TOOL_FAILED" and "what must be one of" in nope["content"]


def test_harness_propose_sends_the_proposal_and_a_reef_without_the_route_is_a_tool_error(
    tmp_path: Path, reef: _FakeReef
) -> None:
    reef.release("r1", [_tool("shout")])
    proposal = {"mutations": [_whisper_mutation()], "reason": "a quieter tool"}
    reef.replies = [
        _reply(tool_calls=[_call("harness_propose", proposal, "c1")]),
        _reply("proposed"),
        _reply(tool_calls=[_call("harness_propose", proposal, "c2")]),
        _reply("went on"),
    ]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        first, streamed = _turn(server, "propose it")
        reef.proposals_route = False
        second, later = _turn(server, "propose again", session=first["session"])
    assert first["text"] == "proposed" and second["text"] == "went on"
    (sent,) = reef.proposals
    assert sent["headers"]["x-reef-scenario"] == SCENARIO and sent["headers"]["authorization"] == "Bearer tok"
    assert sent["body"] == {**proposal, "session": first["session"], "release_id": "r1"}
    result = _typed(streamed, "tool/result")[0]
    assert json.loads(result["content"]) == {"proposal_id": "p1", "admitted": True, "reason": None, "release_id": "r1"}
    refused = _typed(later, "tool/result")[0]
    assert refused["error"]["code"] == "TOOL_FAILED" and "this reef has no proposals route" in refused["content"]
    assert second["exit"] == 0


def test_a_tree_entry_with_a_reserved_name_does_not_mount(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("shout")])
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, poll_interval_s=0.1) as server:
        reef.release("r2", [_tool("shout"), _tool(RESERVED_NAMES[0])], parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/mount-failed")))
        failed = _typed(_events(log), "harness/mount-failed")[0]
        assert failed["entry"] == "harness_inspect" and "reserved name" in failed["error"]
        assert server.status()["release_id"] == "r1" and list(server.host.tools) == ["shout"]
    (dest / "native" / "tree.json").write_text(json.dumps([_tool(RESERVED_NAMES[2])]), encoding="utf-8")
    with pytest.raises(ServeError, match="reserved name"):
        Server(dest, scenario=SCENARIO).start()
    assert not (dest / "native" / "serve.sock").exists()


# -- the pieces on their own -------------------------------------------------------------------------------


def test_admit_mutations_refuses_what_the_backend_refuses() -> None:
    from dataclasses import replace

    from reef.harness.adapters import get_adapter

    # The stage 2 descriptor renders no agent_command; until it lands, the same check runs on this copy.
    base = get_adapter("native")
    descriptor = replace(base, node_paths={k: v for k, v in base.node_paths.items() if k != "agent_command"})
    entries = [_tool("shout"), _graph(SEED_STAGES, SEED_EDGES)]
    created = {"name": "native_tool", "config": _tool("hum")["config"]}
    after, reason = admit_mutations(entries, [Mutation("create", "hum", created)], descriptor)
    assert reason is None and [entry["id"] for entry in after] == ["shout", "main", "hum"]
    assert [entry["id"] for entry in entries] == ["shout", "main"]  # the input is never touched
    cases = [
        (Mutation("create", "shout", created), "already exists"),
        (Mutation("update", "nope", created), "cannot resolve entry nope"),
        (Mutation("remove", "nope"), "cannot resolve entry nope"),
        (Mutation("update", "shout", {"name": "skill"}), "cannot change the entry's kind"),
        (Mutation("update", "shout", {"config": {"name": "shout", "description": "", "code": "x"}}), "rejected"),
        (
            Mutation("create", "cmd", {"name": "agent_command", "config": {"name": "cmd", "text": "t"}}),
            "does not render",
        ),
        (Mutation("create", "why", {"name": "no_such_kind", "config": {}}), "unknown node kind"),
        (
            Mutation(
                "update",
                "main",
                {
                    "config": {
                        **_graph({**SEED_STAGES, "act": {"kind": "tools", "allow": ["gone"]}}, SEED_EDGES)["config"]
                    }
                },
            ),
            "allows tools the tree lacks",
        ),
    ]
    for mutation, expected in cases:
        same, reason = admit_mutations(entries, [mutation], descriptor)
        assert reason is not None and expected in reason, (mutation, reason)
        assert same == entries


def test_the_capture_proxy_tags_calls_live_and_publishes_per_turn(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    monkeypatch.setenv("REEF_HARNESS_CAPTURES_DIR", str(tmp_path / "captures"))
    reef.release("r1", [])
    seen: list[str] = []

    class Observer:
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
        assert proxy.publish_turn() == 2
        assert proxy.publish_turn() == 0
        call()
        assert proxy.publish_turn() == 1
    finally:
        proxy.stop()
    assert seen == ["r1", "r1", "r1"]
    tags = [(r["headers"].get("x-reef-tag-release"), r["headers"].get("x-reef-tag-trial")) for r in reef.requests]
    assert tags == [("r1", None), ("r1", "t1"), ("r1", None)]
    assert {r["headers"]["authorization"] for r in reef.requests} == {"Bearer tok"}
    spools = sorted((tmp_path / "captures").glob("*.pending.json"))
    turns = [json.loads(spool.read_text())["turns"] for spool in spools]
    assert [[(t["receipt"], t["tags"]) for t in turn] for turn in turns] == [
        [("r-1", {"release": "r1"}), ("r-2", {"release": "r1", "trial": "t1"})],
        [("r-3", {"release": "r1"})],
    ]
    harness_wrapper.report(SCENARIO, "native", 0.5, "first turn")
    harness_wrapper.report(SCENARIO, "native", 0.5, "second turn")
    assert [report["references"] for report in reef.reports] == [["r-1"], ["r-3"]]


# -- what the refuter pass found -----------------------------------------------------------------------------


def test_a_mount_that_orphans_the_pinned_graph_closes_the_turn_with_an_error(tmp_path: Path, reef: _FakeReef) -> None:
    helper = _entry("helper", "native_agent", name="helper", prompt="You help.", max_steps=1)
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
    reef.release("r1", [_tool("shout", "    return 'LOUD'"), helper, _graph(stages, edges)])
    dest = _tree(tmp_path, reef, "r1")
    # The first answer names r2, which drops the helper and the delegate stage; the graph this turn walks is r1's.
    reef.replies = [_reply(tool_calls=[_call("shout", {}, "c1")]), _reply("thinking done"), _reply("helper")]
    with _running(dest) as server:
        _wait(lambda: reef.polls >= 1)
        reef.release("r2", [_tool("shout", "    return 'LOUD'"), _graph(SEED_STAGES, SEED_EDGES)], parent="r1")
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


def test_a_rollback_that_cannot_reimport_is_logged_and_named_by_status(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "once.marker"
    monkeypatch.setenv("ONCE_MARKER", str(marker))
    once = _tool("once", "    return 'once'")
    # An import that is not idempotent: the second import of the same module fails on the marker it left.
    once["config"][
        "code"
    ] = "import os\nopen(os.environ['ONCE_MARKER'], 'x').close()\n\n\ndef run(args, workdir):\n    return 'once'\n"
    reef.release("r1", [once, _tool("other")])
    dest = _tree(tmp_path, reef, "r1")
    with _running(dest, poll_interval_s=0.1) as server:
        assert sorted(server.host.tools) == ["once", "other"] and marker.exists()
        broken = _tool("broken")
        broken["config"]["code"] = "raise RuntimeError('boom')\n\n\ndef run(args, workdir):\n    return 1\n"
        reef.release("r2", [broken], parent="r1")
        log = server.sessions_dir / serve.SERVE_LOG
        _wait(lambda: bool(_typed(_events(log), "harness/rollback-failed")))
        failed = _typed(_events(log), "harness/rollback-failed")[0]
        assert failed["entry"] == "once" and "FileExistsError" in failed["error"] and failed["release_id"] == "r1"
        assert server.status()["degraded"] == ["once"] and sorted(server.host.tools) == ["other"]
        # A mount that lands clears the mark: the composition is again what the release says.
        marker.unlink()
        reef.release("r3", [once, _tool("other")], parent="r1")
        _wait(lambda: server.status()["release_id"] == "r3")
        assert server.status()["degraded"] == [] and sorted(server.host.tools) == ["once", "other"]


def test_a_session_from_a_previous_process_is_not_silently_resumed(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_graph(SEED_STAGES, SEED_EDGES)])
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


def test_the_release_client_poll_skips_rows_pending_review() -> None:
    class _Catalog(serve.ReleaseClient):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            super().__init__("http://127.0.0.1:9", "tok", SCENARIO)
            self._rows = rows

        def releases(self) -> list[dict[str, Any]]:
            return list(self._rows)

    assert _Catalog([]).poll() is None
    assert _Catalog([{"release_id": "r1"}, {"release_id": "r2", "pending": True}]).poll() == "r1"
    assert _Catalog([{"release_id": "r1"}, {"release_id": "r2", "pending": False}]).poll() == "r2"
    assert _Catalog([{"release_id": "r1", "pending": True}]).poll() is None


def test_a_tree_tool_cannot_take_a_self_tool_name_at_admission() -> None:
    from reef.harness.nodes import NODE_KINDS

    for name in RESERVED_NAMES:
        config = {**_tool(name)["config"]}
        with pytest.raises(ValueError, match="reserved for the host plane"):
            NODE_KINDS["native_tool"](None, config)


def test_the_self_tools_run_in_process_when_the_environment_names_bwrap(
    tmp_path: Path, reef: _FakeReef, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "bwrap").write_text("#!/bin/sh\necho 'bwrap: must not run for a host plane tool' >&2\nexit 1\n")
    (fake / "bwrap").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("REEF_NATIVE_ENFORCE", "bwrap")
    reef.release("r1", [_tool("shout"), _graph(SEED_STAGES, SEED_EDGES)])
    reef.replies = [_reply(tool_calls=[_call("harness_inspect", {"what": "status"}, "c1")]), _reply("seen")]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "look")
    assert result["text"] == "seen"
    status = _typed(streamed, "tool/result")[0]
    assert status["is_error"] is False and json.loads(status["content"])["release_id"] == "r1"
    assert status["enforcement"] == {"mode": "none", "denied": []}
    assert _typed(streamed, "session")[0]["enforcement"] == "bwrap"
