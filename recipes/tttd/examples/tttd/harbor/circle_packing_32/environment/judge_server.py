"""The judge: one HTTP server, one scoring implementation, a submission budget.

Generic — task authors never edit this file. It loads the scoring from the
files next to it:

- ``score.py``     — ``grade(path) -> {"reward": float, "reason": str}``,
  run on every submission (the session score the agent sees);
- ``final.py``     — optional, same signature, run once on the best
  submission when the verifier asks for the final result. This is where
  hidden tests live: the file ships only in the judge image, and querying
  it costs the whole session (see /final below).
- ``judge_config.json`` — ``{"max_submissions": int|null,
  "min_interval_sec": float}``. The submission budget is the task's
  anti-probing dial: generous for public metrics, tight where feedback
  would leak information.

Endpoints (body in, JSON out):

- ``POST /submit``  — body = the raw solution file. Scores it, appends to
  the submission log, returns ``{"n", "score", "reason", "best", "remaining"}``.
  Over budget or too soon → 429.
- ``GET /status``   — ``{"used", "remaining", "best"}``.
- ``GET /final``    — ``{"reward", "reason", "best_n", "submissions"}``: the
  final judgment (final.py on the best submission if present, else the
  best session score) plus the full submission log. **Finalizing is terminal**:
  the first call locks the session — the result is computed once and
  cached (safe for the verifier to retry), and every later /submit is
  refused. An agent that peeks early ends its own run.
- ``GET /health``   — liveness.

Environment: ``PORT`` (default 8082), ``JUDGE_DIR`` (where score.py etc.
live; defaults to this file's directory), ``DATA_DIR`` (payload storage,
default ``/judge/data``).
"""

import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/judge/data"))
PORT = int(os.environ.get("PORT", "8082"))


def _load(name: str):
    path = JUDGE_DIR / name
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade


def _config() -> dict:
    path = JUDGE_DIR / "judge_config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


class Judge:
    def __init__(self):
        self.score = _load("score.py")
        self.final = _load("final.py")
        config = _config()
        self.max_submissions = config.get("max_submissions")
        self.min_interval_sec = float(config.get("min_interval_sec", 0))
        max_concurrent = config.get("max_concurrent_submissions")
        if max_concurrent is not None and int(max_concurrent) < 1:
            raise ValueError("max_concurrent_submissions must be positive")
        self.score_slots = None if max_concurrent is None else threading.BoundedSemaphore(int(max_concurrent))
        self.t0 = time.monotonic()
        self.last_submission_at: float | None = None
        self.log: list[dict] = []
        self.finalized: dict | None = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def submit(self, body: bytes) -> tuple[int, dict]:
        # Reserve an immutable submission number under the lock, then run the
        # expensive scorer outside it. ThreadingHTTPServer can therefore
        # evaluate independent submissions concurrently without racing the
        # budget, ordering, or finalization state.
        with self.condition:
            if self.finalized is not None:
                return 429, {"error": "session finalized — no more submissions"}
            now = time.monotonic() - self.t0
            if self.max_submissions is not None and len(self.log) >= int(self.max_submissions):
                return 429, {
                    "error": "submission limit reached",
                    "used": len(self.log),
                }
            if (
                self.min_interval_sec > 0
                and self.last_submission_at is not None
                and now - self.last_submission_at < self.min_interval_sec
            ):
                wait = self.min_interval_sec - (now - self.last_submission_at)
                return 429, {"error": "too soon", "retry_after_sec": round(wait, 1)}

            n = len(self.log)
            self.last_submission_at = now
            entry = {"n": n, "t": round(now, 3), "score": None, "reason": "scoring"}
            self.log.append(entry)

        payload = DATA_DIR / f"submission_{n}"
        try:
            payload.write_bytes(body)
            if self.score_slots is None:
                result = self.score(payload)
            else:
                with self.score_slots:
                    result = self.score(payload)
            score, reason = float(result["reward"]), result.get("reason", "")
        except Exception as e:  # a broken submission scores 0, loudly
            score, reason = 0.0, f"scoring failed: {e!r}"

        with self.condition:
            entry["score"] = score
            entry["reason"] = reason
            best = max(item["score"] for item in self.log if item["score"] is not None)
            remaining = None if self.max_submissions is None else int(self.max_submissions) - len(self.log)
            self.condition.notify_all()
            return 200, {
                "n": n,
                "score": score,
                "reason": reason,
                "best": best,
                "remaining": remaining,
            }

    def status(self) -> dict:
        with self.condition:
            completed_scores = [entry["score"] for entry in self.log if entry["score"] is not None]
            return {
                "used": len(self.log),
                "remaining": (None if self.max_submissions is None else int(self.max_submissions) - len(self.log)),
                "best": max(completed_scores, default=None),
                "pending": len(self.log) - len(completed_scores),
            }

    def final_result(self) -> dict:
        with self.condition:
            while any(entry["score"] is None for entry in self.log):
                self.condition.wait()
            if self.finalized is not None:
                return self.finalized
            self.finalized = self._compute_final()
            return self.finalized

    def _compute_final(self) -> dict:
        if not self.log:
            return {
                "reward": 0.0,
                "reason": "no submissions",
                "best_n": None,
                "submissions": [],
            }
        best = max(self.log, key=lambda e: (e["score"], -e["n"]))
        if self.final is not None:
            payload = DATA_DIR / f"submission_{best['n']}"
            try:
                result = self.final(payload)
                reward, reason = float(result["reward"]), result.get("reason", "")
            except Exception as e:
                reward, reason = 0.0, f"final scoring failed: {e!r}"
        else:
            reward, reason = best["score"], best["reason"] or "best submission"
        return {
            "reward": reward,
            "reason": reason,
            "best_n": best["n"],
            "submissions": list(self.log),
        }


judge = Judge()


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"ok": True})
        elif self.path == "/status":
            self._reply(200, judge.status())
        elif self.path == "/final":
            self._reply(200, judge.final_result())
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/submit":
            self._reply(404, {"error": "unknown path"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        code, payload = judge.submit(self.rfile.read(length))
        self._reply(code, payload)

    def log_message(self, *args):
        pass  # keep container logs quiet


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
