"""The LangGraph example runs a real tool and produces the verifier's input.

Install the script's pinned LangGraph and LangChain packages to run the agent
tests. The verifier tests need only Reef's development environment.
"""

import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.tasks.trajectory import reader_for
from reef.harness.tree import HarnessTree, validate_tree

EXAMPLE = Path(__file__).resolve().parents[2] / "tutorials" / "custom-agent"


@pytest.mark.parametrize("refuse_model", [False, True], ids=["tool-and-answer", "model-error"])
def test_langgraph_program_records_calls_and_propagates_model_errors(tmp_path: Path, refuse_model: bool) -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_openai")
    smoke = runpy.run_path(str(EXAMPLE / "smoke_provider.py"))

    class RefusingProvider(BaseHTTPRequestHandler):
        """An HTTP handler that returns 401 for each model POST.

        Values are BaseHTTPRequestHandler instances attached to the test server.
        The test constructs them to check that model rejection fails the graph.
        """

        def do_POST(self) -> None:
            payload = b'{"error":{"message":"model access refused"}}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    source = HarnessTree.read(EXAMPLE / "tree")
    assert validate_tree(source, get_adapter("python")).passed
    program = tmp_path / "program"
    shutil.copytree(EXAMPLE / "tree" / "program", program)
    handler = RefusingProvider if refuse_model else smoke["SmokeHandler"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    key = "test-only-model-key"
    (program / "model.json").write_text(
        json.dumps(
            {
                "api": "openai",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "model": "langgraph-smoke",
                "api_key": key,
            }
        ),
        encoding="utf-8",
    )
    sessions = tmp_path / "sessions"
    try:
        result = subprocess.run(
            [sys.executable, str(program / "main.py"), "Calculate 17 multiplied by 23."],
            cwd=tmp_path,
            env={**os.environ, "REEF_PROGRAM_SESSION_DIR": str(sessions)},
            text=True,
            capture_output=True,
            timeout=45,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    log = (sessions / "calls.jsonl").read_text(encoding="utf-8")
    assert key not in log
    events = [json.loads(line) for line in log.splitlines()]
    answer = tmp_path / "answer.txt"
    if refuse_model:
        assert result.returncode != 0 and not answer.exists()
        assert events == [{"type": "model/error", "status": 401}]
    else:
        assert result.returncode == 0, result.stderr
        assert answer.read_text(encoding="utf-8").strip() == "391"
        assert len(events) == 2
        assert events[1]["messages"][-1]["role"] == "tool"
        assert events[1]["messages"][-1]["content"] == "391"
        (trajectory,) = reader_for("python-jsonl").to_atif(sessions)
        trajectory_path = tmp_path / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
        grade = runpy.run_path(str(EXAMPLE / "harbor" / "tests" / "grade.py"))["grade"]
        assert grade(answer, trajectory_path) == 1.0


@pytest.mark.parametrize("answer", [None, "391", "wrong"])
def test_verifier_refuses_missing_or_wrong_answer_and_missing_tool_trace(tmp_path: Path, answer: str | None) -> None:
    grade = runpy.run_path(str(EXAMPLE / "harbor" / "tests" / "grade.py"))["grade"]
    answer_path = tmp_path / "answer.txt"
    if answer is not None:
        answer_path.write_text(answer, encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text('{"steps": []}', encoding="utf-8")
    assert grade(answer_path, trajectory_path) == 0.0
