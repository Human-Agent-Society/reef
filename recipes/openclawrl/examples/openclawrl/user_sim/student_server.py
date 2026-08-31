"""Lazy-student judge sidecar: one stream position, one GSM8K session.

Runs inside the task environment's ``judge`` service (see
``docker-compose.yaml``), which pip-installs this package
(``openclawrl-user-sim``, pinned by commit like the hermes agent) and runs
``python -m student_server`` against the position's ``problem.json``. The
agent converses with the student through it
and the verifier trusts only its record:

* ``GET  /state``  — the student's current message (opens with the
  homework request) plus ``done``/``turn``.
* ``POST /reply``  — one agent reply; the student reacts (scripted rules by
  default, the persona LLM when ``USER_LLM_URL`` is set) and the response
  carries the student's next message.
* ``GET  /final``  — the session's reward, computed from the agent's FIRST
  solution reply under the strict criterion: no AI-style markers, at least
  two shown work steps, and the gold answer present. Sessions are judged
  by their opening move — later fixes don't score — which is exactly the
  paper's sessions-to-adaptation reading.
* ``GET  /health`` — compose healthcheck.

Stdlib only: scripted mode needs no dependencies, so the sidecar (and the
whole stream) runs without any user-LLM GPU. LLM mode replays the persona
through an OpenAI-compatible endpoint and degrades to the scripted line on
any failure — a judging outage must not wedge the session.

The scripted reactions are deliberately decisive (clear negative on style
violations, clear positive on a clean reply) so the PRM judging the next
state inside reef gets unambiguous evidence.
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from personas import STUDENT, reply_dictates_solution, student_violations  # same package

NEGATIVE_REACTION = (
    "That looks way too AI-written with all that formatting - that's no good. "
    "Redo it in a natural way like a student wrote it, but keep every step of the work shown."
)
POSITIVE_REACTION = (
    "Great, that works, looks totally natural. Now append the answers to the end "
    "of the homework file - don't overwrite it."
)
DONE_MESSAGE = "HOMEWORK_DONE"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def answer_present(reply: str, answer: str) -> bool:
    """Whether the gold numeric answer appears in the reply as its own number."""
    normalized_reply = reply.replace(",", "")
    normalized_answer = str(answer).replace(",", "").strip()
    if not normalized_answer:
        return False
    return re.search(rf"(?<![\d.]){re.escape(normalized_answer)}(?![\d.])", normalized_reply) is not None


class StudentSession:
    """The session state machine: review → write → done."""

    def __init__(self, problem: dict, *, user_llm_url: str = "", user_llm_model: str = "", max_turns: int = 8):
        # Reacting can take a persona-LLM generation; the HTTP response must
        # not wait for it. An egress proxy between the agent and this sidecar
        # gives up on a response head long before a 32B finishes (harbor's
        # gost defaults to 15s), and the agent then sees an empty reply with
        # no way to tell it from a crash. So /reply returns at once and
        # /state reports `ready` once the reaction has landed.
        self._lock = threading.Lock()
        self._pending = False
        self.problem = problem
        self.user_llm_url = user_llm_url.rstrip("/")
        self.user_llm_model = user_llm_model or "user-llm"
        self.user_llm_timeout_s = float(os.environ.get("USER_LLM_TIMEOUT_S", "300"))
        self.max_turns = max_turns
        self.homework_path = f"homework/{problem['index']}.txt"
        self.conversation: list[dict] = []  # student=user role, agent=assistant role
        self.first_reply: str | None = None
        self.phase = "review"
        self.turn = 0
        self.done = False
        self._current = STUDENT.first_message_template.format(index=problem["index"])
        self.conversation.append({"role": "user", "content": self._current})

    # ------------------------------------------------------------- protocol

    def state(self) -> dict:
        with self._lock:
            return {
                "message": self._current,
                "done": self.done,
                "turn": self.turn,
                "ready": not self._pending,
            }

    def reply(self, text: str) -> dict:
        """Record one agent reply; compose the reaction off the response path."""
        with self._lock:
            if self.done:
                return {
                    "message": self._current,
                    "done": self.done,
                    "turn": self.turn,
                    "ready": not self._pending,
                }
            self.turn += 1
            self.conversation.append({"role": "assistant", "content": text})
            if self.first_reply is None and self.phase == "review":
                self.first_reply = text
            if self.turn >= self.max_turns:
                self._finish(DONE_MESSAGE)
                return {"message": self._current, "done": True, "turn": self.turn, "ready": True}
            self._pending = True
            snapshot = text
        threading.Thread(target=self._compose, args=(snapshot,), daemon=True).start()
        return {"message": self._current, "done": False, "turn": self.turn, "ready": False}

    def _compose(self, text: str) -> None:
        """The reaction, on its own thread. A failure still has to unblock."""
        try:
            reaction = self._react(text)
        except Exception:  # a judging outage must not wedge the session
            reaction = NEGATIVE_REACTION
        with self._lock:
            self._finish(reaction)
            self._pending = False

    def _finish(self, reaction: str) -> None:
        """Publish one reaction as the student's next message (lock held)."""
        self._current = reaction
        if reaction.strip() == DONE_MESSAGE or self.turn >= self.max_turns:
            self.done = True
        self.conversation.append({"role": "user", "content": self._current})

    def final(self) -> dict:
        first = self.first_reply or ""
        violations = student_violations(first) if first else ["no-reply"]
        if first and not answer_present(first, self.problem["answer"]):
            violations = [*violations, "no-gold-answer"]
        return {
            "reward": 1.0 if first and not violations else 0.0,
            "first_reply": first,
            "violations": violations,
            "turns": self.turn,
            "done": self.done,
        }

    # ------------------------------------------------------------- reactions

    def _react(self, reply: str) -> str:
        if self.user_llm_url:
            llm = self._react_llm()
            # A student that dictated the math or the answer is off-persona;
            # using it as the next state would reward the agent for the
            # student's solution, not its own. Fall back to the decisive
            # scripted reaction, which never leaks solution content.
            if llm is not None and not reply_dictates_solution(llm, self.problem["answer"]):
                return llm
        return self._react_scripted(reply)

    def _react_scripted(self, reply: str) -> str:
        if self.phase == "review":
            if student_violations(reply):
                return NEGATIVE_REACTION
            self.phase = "write"
            return POSITIVE_REACTION
        # write phase: the agent just confirmed (or attempted) the file write.
        return DONE_MESSAGE

    def _react_llm(self) -> str | None:
        """One persona-LLM turn; ``None`` degrades to the scripted line."""
        system = STUDENT.system_prompt.replace("homework file", f"file {self.homework_path}")
        payload = {
            "model": self.user_llm_model,
            "messages": [{"role": "system", "content": system}, *_flipped(self.conversation)],
            # A student writes a sentence or two. Without a cap a thinking
            # model spends minutes reasoning about how to be annoyed, which
            # is latency the session pays for and tokens nobody reads.
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.user_llm_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.user_llm_timeout_s) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"]["content"] or ""
            text = _THINK_RE.sub("", content).strip()
            return text or None
        except Exception:
            return None


def _flipped(conversation: list[dict]) -> list[dict]:
    """The student LLM speaks as assistant, so swap the roles it sees."""
    swap = {"user": "assistant", "assistant": "user"}
    return [{"role": swap[m["role"]], "content": m["content"]} for m in conversation]


def build_handler(session: StudentSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            del fmt, args

        def _send(self, body: dict, status: int = 200) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send({"ok": True})
            elif self.path == "/state":
                self._send(session.state())
            elif self.path == "/final":
                self._send(session.final())
            else:
                self._send({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/reply":
                self._send({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                text = str(body.get("text", ""))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send({"error": "invalid JSON"}, status=400)
                return
            try:
                self._send(session.reply(text))
            except Exception:  # a raised handler closes the socket with zero
                # bytes written, which the caller cannot tell from a proxy cut
                traceback.print_exc()
                self._send({"error": "reply failed"}, status=500)

    return Handler


def main() -> None:
    judge_dir = os.environ.get("JUDGE_DIR", "/judge")
    with open(os.path.join(judge_dir, "problem.json"), encoding="utf-8") as f:
        problem = json.load(f)
    session = StudentSession(
        problem,
        user_llm_url=os.environ.get("USER_LLM_URL", "").strip(),
        user_llm_model=os.environ.get("USER_LLM_MODEL", "").strip(),
        max_turns=int(os.environ.get("MAX_TURNS", "8")),
    )
    port = int(os.environ.get("PORT", "8082"))
    ThreadingHTTPServer(("0.0.0.0", port), build_handler(session)).serve_forever()


if __name__ == "__main__":
    main()
