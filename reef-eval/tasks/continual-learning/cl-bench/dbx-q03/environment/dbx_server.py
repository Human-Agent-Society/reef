"""Database-exploration judge: the query-metered sidecar for one question.

Runs in the judge container with the SQLite database and the correct
answer; the agent reaches it only over HTTP. Semantics are ported from
CL-Bench's ``database_exploration/task.py`` (Apache-2.0): read-only
queries (SELECT/WITH/EXPLAIN/PRAGMA plus ``.tables``/``.schema``),
results capped at 50 rows, a 10-second timeout, errors consume budget,
and the reward is ``1 - queries/15`` when the final answer is correct
and 0 otherwise. Exceeding the 15-query budget force-fails the question.

Endpoints: ``POST /query`` {"sql"} — ``POST /answer`` {"answer"},
terminal — ``GET /final``, the verifier's verdict — ``GET /health``.
"""

import json
import os
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
PORT = int(os.environ.get("PORT", "8082"))

MAX_QUERY_RESULT_ROWS = 50
SQL_QUERY_TIMEOUT_SECONDS = 10
MAX_QUERIES = 15

_lock = threading.Lock()
_state = {"queries": 0, "done": False, "reward": 0.0, "verdict": None}

QUESTION = json.loads((JUDGE_DIR / "question.json").read_text())
DB_PATH = str(JUDGE_DIR / "question.db")


# ---- upstream answer checking, ported verbatim in behavior ----


def evaluate_answer(submitted: str, question: dict) -> bool:
    if not submitted or not submitted.strip():
        return False
    submitted_clean = submitted.strip()
    expected = str(question["answer"]).strip()
    answer_type = question.get("answer_type", "text")
    tolerance = question.get("tolerance", 0)
    if answer_type == "integer":
        nums = re.findall(r"-?\d+", submitted_clean)
        try:
            return bool(nums) and int(nums[0]) == int(expected)
        except (ValueError, IndexError):
            return False
    if answer_type == "float":
        nums = re.findall(r"-?\d+\.?\d*", submitted_clean)
        try:
            if not nums:
                return False
            return abs(float(nums[0]) - float(expected)) <= max(
                tolerance, abs(float(expected)) * 0.01
            )
        except (ValueError, IndexError):
            return False
    return submitted_clean.lower() == expected.lower()


# ---- upstream query execution semantics ----


def execute_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if sql == ".tables":
        sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    elif sql.startswith(".schema"):
        parts = sql.split()
        if len(parts) > 1:
            sql = "SELECT sql FROM sqlite_master WHERE name = " + repr(parts[1])
        else:
            sql = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA"):
        return (
            f"ERROR: Only SELECT / WITH / EXPLAIN / PRAGMA statements "
            f"are allowed. Got: {first_word}"
        )
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    deadline = time.monotonic() + SQL_QUERY_TIMEOUT_SECONDS
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_QUERY_RESULT_ROWS + 1)
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error):
            return f"ERROR: query exceeded {SQL_QUERY_TIMEOUT_SECONDS}s timeout"
        return f"ERROR: {error}"
    except sqlite3.Error as error:
        return f"ERROR: {error}"
    finally:
        conn.close()

    truncated = len(rows) > MAX_QUERY_RESULT_ROWS
    rows = rows[:MAX_QUERY_RESULT_ROWS]

    def cell(v):
        if v is None:
            return "NULL"
        return re.sub(r"[\n\t]+", " ", str(v))

    table = [" | ".join(columns), "-+-".join("-" * len(c) for c in columns)]
    table += [" | ".join(cell(v) for v in row) for row in rows]
    text = "\n".join(table)
    if truncated:
        text += f"\n... (showing first {MAX_QUERY_RESULT_ROWS} rows)"
    return text


def _finalize(correct: bool, submitted: str) -> dict:
    queries = _state["queries"]
    reward = (1 - queries / MAX_QUERIES) if correct else 0.0
    _state["done"] = True
    _state["reward"] = round(reward, 6)
    _state["verdict"] = {
        "correct": correct,
        "your_answer": submitted,
        "correct_answer": str(QUESTION["answer"]),
        "exploratory_queries_used": queries,
        "reward": _state["reward"],
    }
    return _state["verdict"]


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._reply({"ok": True})
        if self.path == "/final":
            with _lock:
                if not _state["done"]:
                    _finalize(False, "")
                    _state["verdict"]["note"] = "no answer submitted"
                return self._reply({"reward": _state["reward"], **_state["verdict"]})
        return self._reply({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
        except (json.JSONDecodeError, ValueError):
            return self._reply({"error": "body must be JSON"}, 400)
        with _lock:
            if _state["done"]:
                return self._reply(
                    {"error": "question is finished", **(_state["verdict"] or {})},
                    409,
                )
            if self.path == "/query":
                _state["queries"] += 1  # errors consume budget too, as upstream
                if _state["queries"] > MAX_QUERIES:
                    verdict = _finalize(False, "")
                    verdict["note"] = "query budget exceeded — treated as incorrect"
                    return self._reply(verdict, 429)
                result = execute_sql(str(body.get("sql", "")))
                remaining = MAX_QUERIES - _state["queries"]
                return self._reply(
                    {
                        "result": result,
                        "queries_used": _state["queries"],
                        "remaining": remaining,
                    }
                )
            if self.path == "/answer":
                submitted = str(body.get("answer", ""))
                correct = evaluate_answer(submitted, QUESTION)
                return self._reply(_finalize(correct, submitted))
        return self._reply({"error": "unknown endpoint"}, 404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
