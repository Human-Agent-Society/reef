"""Score one generated candidate through the external judge service.

The harness never runs a candidate itself and never knows what the task is:
it hands the extracted program to a ``Scorer`` and turns the judge's answer
into a :class:`VerificationResult`. ``JudgeScorer`` speaks one wire
protocol — multipart submit, poll by submission id — which the deployment
points at whichever judge service scores the task.

Speaking that protocol directly keeps a benchmark's generation and
cloud-runner dependencies out of the Reef actor environment; the privileged
judge stays external and authoritative.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .state import VerificationResult

FENCE_LANGUAGES = {
    "cpp": ("cpp", r"c\+\+", "cxx"),
    "python": ("python", "py"),
}
SOURCE_FILES = {"cpp": ("solution.cpp", "text/x-c++src"), "python": ("solution.py", "text/x-python")}


class JudgeUnavailableError(RuntimeError):
    """Raised when the external judge service is unavailable."""


@dataclass(frozen=True)
class JudgeResult:
    valid: bool
    score: float | None
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class Scorer(Protocol):
    """Score one extracted candidate program."""

    def __call__(self, code: str) -> VerificationResult: ...


def extract_solution_code(text: str, language: str = "cpp") -> str | None:
    """Return the fenced program inside ``<solution>``, or the last fence."""
    solution_match = re.search(r"<solution>\s*([\s\S]*?)\s*</solution>", text)
    if solution_match:
        return extract_solution_code(solution_match.group(1), language)
    alternatives = FENCE_LANGUAGES.get(language.lower(), (re.escape(language.lower()),))
    pattern = r"```(?:" + "|".join(alternatives) + r")\s*([\s\S]*?)\s*```"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip() or None
    return None


class JudgeScorer:
    """Submit one candidate to the judge and wait for its verdict."""

    def __init__(
        self,
        judge_url: str,
        *,
        problem_id: str = "0",
        language: str = "cpp",
        timeout_s: float = 340.0,
        poll_interval_s: float = 2.0,
        base_dir: str | Path | None = None,
    ) -> None:
        if not judge_url.strip():
            raise ValueError("the judge URL must be non-empty")
        if timeout_s <= 0 or poll_interval_s < 0:
            raise ValueError("invalid judge timeout or poll interval")
        self.judge_url = judge_url.rstrip("/")
        self.problem_id = str(problem_id)
        self.language = language
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else None

    def __call__(self, code: str) -> VerificationResult:
        try:
            result = self.evaluate(code)
        except JudgeUnavailableError as exc:
            return VerificationResult(
                reward=0.0,
                raw_score=None,
                valid=False,
                status="environment_error",
                message=str(exc),
                artifacts={"code": code},
            )
        artifacts = {"code": code, **(result.artifacts or {})}
        if not result.valid:
            return VerificationResult(
                reward=0.0,
                raw_score=None,
                valid=False,
                status="invalid",
                message=result.message,
                artifacts=artifacts,
            )
        score = 0.0 if result.score is None else float(result.score)
        return VerificationResult(
            reward=score,
            raw_score=score,
            valid=True,
            status="valid",
            message=result.message or f"judge score: {score:.2f}",
            artifacts=artifacts,
        )

    def evaluate(self, code: str) -> JudgeResult:
        """Submit one candidate and poll the judge API for its result."""
        if not code.strip():
            raise ValueError("a judge submission must be non-empty")

        deadline = time.monotonic() + self.timeout_s
        submission_id = _submit(
            self.judge_url,
            problem_id=self.problem_id,
            language=self.language,
            code=code,
            timeout_s=min(30.0, self.timeout_s),
        )
        artifacts: dict[str, Any] = {
            "problem_id": self.problem_id,
            "submission_id": submission_id,
        }
        if self.base_dir is not None:
            artifacts["judge_base_dir"] = str(self.base_dir)

        while time.monotonic() < deadline:
            remaining_s = deadline - time.monotonic()
            try:
                result = _get_json(
                    f"{self.judge_url}/result/{urllib.parse.quote(submission_id, safe='')}",
                    timeout_s=max(0.001, min(10.0, remaining_s)),
                )
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise JudgeUnavailableError(_http_error_message("poll", exc)) from exc
                result = None
            except (UnicodeError, ValueError) as exc:
                raise JudgeUnavailableError(f"the judge returned an invalid result payload: {exc}") from exc
            except (OSError, TimeoutError, urllib.error.URLError):
                # Match the reference runner: a transient polling failure
                # stays pending until the evaluation deadline.
                result = None

            if result is not None:
                status = str(result.get("status") or "")
                if status == "done":
                    try:
                        score = float(result.get("score", 0.0))
                    except (TypeError, ValueError) as exc:
                        raise JudgeUnavailableError("the judge returned a non-numeric score") from exc
                    artifacts.update(
                        judge_status=status,
                        score_unbounded=result.get("scoreUnbounded"),
                    )
                    return JudgeResult(
                        valid=True,
                        score=score,
                        message=str(result.get("message") or "accepted"),
                        artifacts=artifacts,
                    )
                if status == "error":
                    artifacts.update(judge_status=status, judge_logs=result.get("logs") or result.get("stderr") or "")
                    message = result.get("message") or result.get("error") or "judge evaluation failed"
                    return JudgeResult(False, None, str(message), artifacts)

            delay_s = min(self.poll_interval_s, max(0.0, deadline - time.monotonic()))
            if delay_s:
                time.sleep(delay_s)

        artifacts["judge_status"] = "timeout"
        return JudgeResult(False, None, f"evaluation timed out after {self.timeout_s:g}s", artifacts)


def _submit(judge_url: str, *, problem_id: str, language: str, code: str, timeout_s: float) -> str:
    boundary = f"reef-guidance-{uuid4().hex}"
    body = b"".join(
        (
            _multipart_field(boundary, "pid", problem_id),
            _multipart_field(boundary, "lang", language),
            _multipart_file(boundary, "code", language, code),
            f"--{boundary}--\r\n".encode(),
        )
    )
    request = urllib.request.Request(
        f"{judge_url}/submit",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = _decode_json(response.read())
    except urllib.error.HTTPError as exc:
        raise JudgeUnavailableError(_http_error_message("submission", exc)) from exc
    except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
        raise JudgeUnavailableError(f"judge submission failed: {type(exc).__name__}: {exc}") from exc
    submission_id = payload.get("sid")
    if isinstance(submission_id, int) and not isinstance(submission_id, bool):
        submission_id = str(submission_id)
    if not isinstance(submission_id, str) or not submission_id.strip():
        raise JudgeUnavailableError("the judge submission response has no sid")
    return submission_id


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return _decode_json(response.read())


def _decode_json(raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict):
        raise ValueError("the judge returned a non-object JSON response")
    return payload


def _multipart_field(boundary: str, name: str, value: str) -> bytes:
    return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()


def _multipart_file(boundary: str, name: str, language: str, value: str) -> bytes:
    filename, content_type = SOURCE_FILES.get(language.lower(), (f"solution.{language.lower()}", "text/plain"))
    header = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    return header + value.encode() + b"\r\n"


def _http_error_message(operation: str, error: urllib.error.HTTPError) -> str:
    body = error.read().decode(errors="replace")[:1_000]
    return f"judge {operation} failed with HTTP {error.code}: {body}"


__all__ = [
    "JudgeResult",
    "JudgeScorer",
    "JudgeUnavailableError",
    "Scorer",
    "extract_solution_code",
]
