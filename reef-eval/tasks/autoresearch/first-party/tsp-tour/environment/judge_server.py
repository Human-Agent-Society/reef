"""The judge: one scoring implementation, a submission budget, two ports.

Generic: task authors never edit this file. It loads the scoring from the
files next to it:

- ``score.py``: ``grade(path) -> {"reward": float, "reason": str}``,
  run on every submission (the session score the agent sees);
- ``final.py``: optional, same signature, run once on the best
  submission when the verifier asks for the final result. This is where
  hidden tests live: the file ships only in the judge image, and querying
  it costs the whole session (see /final below).
- ``judge_config.json``: ``{"max_submissions": int|null,
  "min_interval_sec": float}``. The submission budget is the task's
  anti-probing dial: generous for public metrics, tight where feedback
  would leak information.

Two ports, two capability levels:

Agent port (``PORT``, default 8082), the URL handed to the agent as
``$JUDGE_URL``:

- ``POST /submit``: body = the raw solution file. Scores it, appends to
  the submission log, returns ``{"n", "score", "reason", "best", "remaining"}``.
  Over budget or too soon returns 429.
- ``GET /status``: ``{"used", "remaining", "best"}``.
- ``GET /final``: 403. Finalization is a verifier-only capability; the
  agent cannot trigger or observe hidden evaluation.
- ``GET /health``: liveness.

Verifier port (``VERIFIER_PORT``, default ``PORT + 1``), reachable
only by the trusted verifier (via network topology in containers, via
filesystem-hidden token locally):

- ``GET /final``: ``{"reward", "reason", "best_n", "submissions"}``,
  the final judgment (final.py on the best submission if present, else the
  best session score) plus the full submission log. Finalizing is
  terminal: the first call locks the session, the result is computed
  once and cached (safe for the verifier to retry), and every later
  /submit is refused. Requires ``Authorization: Bearer <token>``.
- ``GET /token``: returns the verifier token (so the verifier can
  fetch it without filesystem access).
- ``GET /health``: liveness.

The token is generated at startup with ``secrets.token_hex(32)`` and
written to ``{DATA_DIR}/.verifier_token``, a file inside the judge's own
filesystem/container, never in the agent's environment. The local executor
reads it from there; the container verifier fetches it via ``GET /token``
on the verifier port (which the agent should not be able to reach; see
``allowed_hosts`` and the deployment docs).

Environment: ``PORT`` (default 8082), ``VERIFIER_PORT`` (default
``PORT + 1``), ``JUDGE_DIR`` (where score.py etc. live; defaults to this
file's directory), ``DATA_DIR`` (payload storage, default ``/judge/data``).
"""

import importlib.util
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JUDGE_DIR = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/judge/data"))
PORT = int(os.environ.get("PORT", "8082"))
VERIFIER_PORT = int(os.environ.get("VERIFIER_PORT", str(PORT + 1)))
VERIFIER_TOKEN = secrets.token_hex(32)


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
        self.t0 = time.monotonic()
        self.log: list[dict] = []
        self.finalized: dict | None = None
        self.lock = threading.Lock()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def submit(self, body: bytes) -> tuple[int, dict]:
        with self.lock:
            if self.finalized is not None:
                return 429, {"error": "session finalized, no more submissions"}
            now = time.monotonic() - self.t0
            if self.max_submissions is not None and len(self.log) >= int(
                self.max_submissions
            ):
                return 429, {
                    "error": "submission limit reached",
                    "used": len(self.log),
                }
            if self.log and now - self.log[-1]["t"] < self.min_interval_sec:
                wait = self.min_interval_sec - (now - self.log[-1]["t"])
                return 429, {"error": "too soon", "retry_after_sec": round(wait, 1)}

            n = len(self.log)
            payload = DATA_DIR / f"submission_{n}"
            payload.write_bytes(body)
            try:
                result = self.score(payload)
                score, reason = float(result["reward"]), result.get("reason", "")
            except Exception as e:  # a broken submission scores 0, loudly
                score, reason = 0.0, f"scoring failed: {e!r}"
            self.log.append(
                {"n": n, "t": round(now, 3), "score": score, "reason": reason}
            )
            best = max(entry["score"] for entry in self.log)
            remaining = (
                None
                if self.max_submissions is None
                else int(self.max_submissions) - len(self.log)
            )
            return 200, {
                "n": n,
                "score": score,
                "reason": reason,
                "best": best,
                "remaining": remaining,
            }

    def status(self) -> dict:
        with self.lock:
            return {
                "used": len(self.log),
                "remaining": (
                    None
                    if self.max_submissions is None
                    else int(self.max_submissions) - len(self.log)
                ),
                "best": max((e["score"] for e in self.log), default=None),
            }

    def final_result(self) -> dict:
        with self.lock:
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

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / ".verifier_token").write_text(VERIFIER_TOKEN)


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _is_verifier(self) -> bool:
        return getattr(self.server, "is_verifier", False)

    def _check_token(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {VERIFIER_TOKEN}"

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"ok": True})
        elif self._is_verifier and self.path == "/token":
            self._reply(200, {"token": VERIFIER_TOKEN})
        elif self.path == "/status":
            self._reply(200, judge.status())
        elif self.path == "/final":
            if not self._is_verifier:
                self._reply(403, {"error": "finalization is verifier-only"})
                return
            if not self._check_token():
                self._reply(403, {"error": "invalid verifier token"})
                return
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
    agent_server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    agent_server.is_verifier = False
    verifier_server = ThreadingHTTPServer(("0.0.0.0", VERIFIER_PORT), Handler)
    verifier_server.is_verifier = True
    threading.Thread(target=agent_server.serve_forever, daemon=True).start()
    verifier_server.serve_forever()
