"""Operator-owned checks for the tutorial's application tasks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import final_assistant_text
from reef.train.evaluation.reefine import RequestVerifier, VerificationResult


class BugfixVerifier(RequestVerifier):
    def verify(self, result: EpisodeResult, workspace: Path) -> VerificationResult:
        source = workspace / "adder.py"
        if not source.is_file():
            return VerificationResult(False, "adder.py missing")
        # Keep arithmetic assertions outside the agent's application workspace.
        with TemporaryDirectory(prefix="reef-bugfix-check-") as temporary:
            root = Path(temporary)
            (root / "adder.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "check.py").write_text(
                "from adder import sum_to\n"
                "for n in (0, 1, 4, 5, 100):\n"
                "    if sum_to(n) != n*(n+1)//2: raise SystemExit(1)\n",
                encoding="utf-8",
            )
            try:
                tested = subprocess.run([sys.executable, "check.py"], cwd=root, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                return VerificationResult(False, "trusted arithmetic checks timed out")
        if tested.returncode != 0:
            return VerificationResult(False, "trusted arithmetic checks failed")
        calls = tool_operations(result)
        edits = [
            index
            for index, (name, arguments, output, error) in enumerate(calls)
            if name in {"edit", "write"} and Path(str(arguments.get("path", ""))).name == "adder.py"
        ]
        first_edit = edits[0] if edits else None
        last_edit = edits[-1] if edits else None
        reproduced = (
            any(
                name == "bash"
                and "test" in str(arguments).lower()
                and any(marker in output for marker in ("FAILED", "AssertionError", "1 failed"))
                for name, arguments, output, error in calls[:first_edit]
            )
            if first_edit is not None
            else False
        )
        passing_tests = (
            [
                index + last_edit + 1
                for index, (name, arguments, output, error) in enumerate(calls[last_edit + 1 :])
                if name == "bash" and "test" in str(arguments).lower() and "passed" in output and not error
            ]
            if last_edit is not None
            else []
        )
        tested_after = bool(passing_tests)
        review_calls = calls[passing_tests[-1] + 1 :] if passing_tests else []
        reviewed = any(
            name not in {"read", "write", "edit", "bash", "grep", "find", "ls"}
            and "review" in name.lower()
            and bool(output.strip())
            and not error
            and "REQUEST_CHANGES" not in output
            and not re.search(
                r'"?(?:exitCode|exit_code)"?\s*[:=]\s*[1-9]|No API key|Error:|timed out|no output|inconclusive|cancelled',
                output,
                re.IGNORECASE,
            )
            for name, arguments, output, error in review_calls
        )
        parent_session = None
        session_id = None
        requested_reviews = set()
        completed_reviews = set()
        for event in result.trajectory:
            if event.get("type") == "session":
                session_id = event.get("id")
                if parent_session is None:
                    parent_session = session_id
            message = event.get("message")
            if not isinstance(message, Mapping) or not isinstance(session_id, str) or session_id == parent_session:
                continue
            content = message.get("content") or []
            text = " ".join(str(block.get("text", "")) for block in content if isinstance(block, Mapping))
            if message.get("role") == "user" and "review" in text.lower():
                requested_reviews.add(session_id)
            elif message.get("role") == "assistant" and message.get("stopReason") == "stop" and text.strip():
                completed_reviews.add(session_id)
        reviewed = reviewed and bool(requested_reviews & completed_reviews)
        missing = [
            name
            for name, passed in (
                ("failing test before edit", reproduced),
                ("passing test after edit", tested_after),
                ("successful review tool and separate completed review session", reviewed),
            )
            if not passed
        ]
        return VerificationResult(
            not missing, "missing " + ", ".join(missing) if missing else "trusted tests and bug-fix workflow passed"
        )


class ResearchVerifier(RequestVerifier):
    def verify(self, result: EpisodeResult, workspace: Path) -> VerificationResult:
        calls = tool_operations(result)
        answer = final_assistant_text(result.trajectory) or ""
        citations = re.findall(
            r"(?:https://arxiv\.org/(?:abs|pdf)/|arxiv[:\s]+)(\d{4}\.\d{4,5})(?:v\d+)?",
            answer,
            re.IGNORECASE,
        )
        for paper in workspace.rglob("*.pdf"):
            extracted = paper.with_suffix(".txt")
            if not paper.read_bytes().startswith(b"%PDF") or not extracted.is_file():
                continue
            try:
                converted = subprocess.run(["pdftotext", str(paper), "-"], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return VerificationResult(False, f"trusted PDF extraction failed: {exc}")
            if converted.returncode != 0:
                continue
            # Check the retained text against the PDF outside the candidate's tool implementation.
            source_text = " ".join(converted.stdout.split())
            saved_text = " ".join(extracted.read_text(encoding="utf-8").split())
            if (
                len(saved_text) < 1000
                or saved_text[:1000] not in source_text
                or not any(identifier in source_text for identifier in citations)
            ):
                continue
            read_source = False
            for operation in calls:
                if operation[3]:
                    continue
                observed_text = " ".join(operation[2].split())
                # A read may select a relevant section rather than the paper's first page.
                if any(
                    observed_text[offset : offset + 1000] in source_text
                    for offset in range(0, len(observed_text) - 999, 500)
                ):
                    read_source = True
                    break
            requested_source = any(
                not error and any(identifier in str(arguments) for identifier in citations)
                for name, arguments, output, error in calls
            )
            if read_source and requested_source:
                return VerificationResult(True, "valid PDF, matching extracted and observed text, and citation passed")
        return VerificationResult(
            False,
            "research requires a valid PDF, matching extracted text, observed reading and an arXiv citation",
        )


def tool_operations(result: EpisodeResult) -> list[tuple[str, Mapping[str, object], str, bool]]:
    pending: dict[str, tuple[str, Mapping[str, object]]] = {}
    operations = []
    for event in result.trajectory:
        message = event.get("message") or event
        if not isinstance(message, Mapping):
            continue
        blocks = message.get("content") or []
        if not isinstance(blocks, list):
            continue
        if message.get("role") == "assistant":
            for block in blocks:
                if isinstance(block, Mapping) and block.get("type") == "toolCall":
                    arguments = block.get("arguments")
                    pending[str(block.get("id"))] = (
                        str(block.get("name")),
                        arguments if isinstance(arguments, Mapping) else {},
                    )
        elif message.get("role") == "toolResult":
            name, arguments = pending.pop(str(message.get("toolCallId")), (str(message.get("toolName")), {}))
            output = "\n".join(str(block.get("text", "")) for block in blocks if isinstance(block, Mapping))
            details = message.get("details")
            if isinstance(details, Mapping):
                output += "\n" + json.dumps(details)
            operations.append((name, arguments, output, message.get("isError") is True))
    return operations
