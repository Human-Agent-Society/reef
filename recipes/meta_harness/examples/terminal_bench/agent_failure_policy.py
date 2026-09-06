"""Keep raw terminal failures separate from upstream's benchmark zero convention.

Upstream Meta-Harness 44b9942 parse_job_results counts an errored trial without
a verifier reward as zero. We admit only the narrowly evidenced case where
the agent exits its interactive shell and subsequent tmux polling fails.
Raw validity and verifier reward remain unchanged; benchmark_score is separate.
The completed-terminal policy also admits a fully recorded, billed terminal
loss without claiming the agent caused it. Causal attribution is diagnostic,
not a requirement in upstream's scoring function. Transport failures, missing
usage and other execution errors still stop admission. Policies must match
between comparison arms.
"""

import copy
import hashlib
import json
import math
import re
import shlex
from collections import Counter
from datetime import datetime
from pathlib import Path

from reef.harness.terminus.runner import atif_steps
from reef.harness.terminus.trial import trial_outcome

PROTOCOL = "upstream-agent-terminal-exit-zero-v1"
ERREXIT_PROTOCOL = "upstream-agent-terminal-exit-zero-v2"
PIPESTATUS_PROTOCOL = "upstream-agent-terminal-exit-zero-v3"
PANE_PROTOCOL = "upstream-agent-terminal-exit-zero-v4"
COMPLETED_PROTOCOL = "upstream-completed-terminal-error-zero-v5"
SUPPORTED = (PROTOCOL, ERREXIT_PROTOCOL, PIPESTATUS_PROTOCOL, PANE_PROTOCOL, COMPLETED_PROTOCOL)
OUTCOME_FIELDS = (
    "valid",
    "reward",
    "phase",
    "exception_type",
    "error",
    "agent_execution_started",
    "timing",
    "cost_usd",
    "residue",
)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _money(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _direct_exit(command):
    if not isinstance(command, str) or "\n" in command.rstrip("\n") or "<<" in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    if any(
        t in ("(", ")", "|") or (t == "&" and not (i and tokens[i - 1].endswith((">", "<"))))
        for i, t in enumerate(tokens)
    ):
        return False  # A subshell, pipeline or background exit need not close the terminal.
    last = max((i for i, t in enumerate(tokens) if t in (";", "&&", "||")), default=-1)
    tail = tokens[last + 1 :]
    return tail == ["exit"] or (
        len(tail) == 2
        and tail[0] == "exit"
        and re.fullmatch(r"(?:[0-9]{1,3}|\$\?|\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\})", tail[1])
        is not None
    )


def _pipeline_exit(command):
    """Recognize only a foreground pipeline followed by a top-level exit."""
    if not isinstance(command, str) or "\n" in command.rstrip("\n") or "<<" in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    forbidden = {
        "(",
        ")",
        "&&",
        "||",
        "if",
        "then",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "{",
        "}",
        "function",
    }
    if any(
        token in forbidden or (token == "&" and not (i and tokens[i - 1].endswith((">", "<"))))
        for i, token in enumerate(tokens)
    ):
        return False
    last = max((i for i, token in enumerate(tokens) if token == ";"), default=-1)
    return last > 0 and "|" in tokens[:last] and tokens[last + 1 :] == ["exit", "${PIPESTATUS[0]}"]


def _pane_size_check(evidence):
    """A final, visibly failed numeric assertion in the parent interactive shell.

    Only recognize a restricted straight-line shell grammar. The bracketed
    paste markers delimit the last submitted command, including terminal
    wrapping. A later prompt/output, ambiguous syntax, or passing predicate
    is insufficient. No shell command from the evidence is executed.
    """
    from .terminal_evidence import MAX_BYTES
    from .terminal_evidence import PROTOCOL as PANE

    if not isinstance(evidence, dict) or evidence.get("protocol") != PANE:
        return None
    text = evidence.get("text")
    if (
        not isinstance(text, str)
        or len(text.encode()) > MAX_BYTES
        or evidence.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest()
        or type(evidence.get("truncated")) is not bool
    ):
        return None
    marker = "\x1b[?2004h"
    start = text.rfind(marker)
    if start < 0 or "\x1b[?2004l" not in text[start:]:
        return None
    submitted, output = text[start + len(marker) :].split("\x1b[?2004l", 1)
    prompt = re.fullmatch(r"(/[A-Za-z0-9_./-]*) [$#] ([\s\S]+)\n", submitted)
    if not prompt or "\x1b" in output:
        return None
    command = prompt[2].replace("\n", "")
    # These prelude clauses cannot disable errexit in the parent shell.
    path = r"[A-Za-z0-9_./-]+"
    prelude = rf"(?:(?:cd {path}|test -s {path}|[./]{path}(?: {path})*); )*"
    pattern = (
        rf"set -e; {prelude}(?P<var>[A-Za-z_][A-Za-z0-9_]*)=\$\(stat -c%s {path}\); "
        r'echo "(?P<label>[A-Za-z_][A-Za-z0-9_]*)=\$(?P=var)'
        r'(?: [A-Za-z_][A-Za-z0-9_]*=\$\(awk "BEGIN\{print \$(?P=var)(?:/[1-9][0-9]*)+\}"\))?"; '
        r'test "\$(?P=var)" -lt (?P<limit>[1-9][0-9]{0,17})'
    )
    match = re.fullmatch(pattern, command)
    if not match:
        return None
    # The echoed integer must be the final output, with no subsequent prompt
    # or diagnostic that could point to a different cause of terminal loss.
    observed = re.search(
        rf"(?:^|\n){re.escape(match['label'])}=(?P<value>[0-9]{{1,18}})(?: [A-Za-z_][A-Za-z0-9_]*=[0-9.]+)?\n?\Z",
        output,
    )
    if not observed or re.search(r"(?m)^/[^\n]* [$#] ", output):
        return None
    value, limit = int(observed["value"]), int(match["limit"])
    if value < limit:
        return None
    return {
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "observed_integer": value,
        "required_less_than": limit,
        "terminal_evidence_sha256": _sha(evidence),
    }


def evidence(row, policy=PROTOCOL):
    if (
        row.get("valid") is not False
        or row.get("reward") is not None
        or row.get("phase") != "execution_failure"
        or row.get("exception_type") != "RuntimeError"
        or not row.get("agent_execution_started")
        or not _money(row.get("cost_usd"))
        or row.get("residue")
        or (row.get("timing") or {}).get("verifier") is not None
    ):
        return None
    error = row.get("error") or ""
    if "tmux send-keys" not in error or "no server running on /tmp/tmux-" not in error:
        return None
    steps = [s for s in row.get("trajectory", []) if isinstance(s, dict) and s.get("source") == "agent"]
    if policy == COMPLETED_PROTOCOL:
        # This is a measurement convention, not an agent-causation classifier.
        # Require a complete trial and agent phase with a retained trajectory.
        # Missing-result/transport/cleanup failures must not become free zeros.
        if not steps or row.get("transport_diagnostics") or row.get("e2b_diagnostics"):
            return None
        execution = (row.get("timing") or {}).get("agent_execution") or {}
        stamps = [
            row.get("started_at"),
            execution.get("started_at"),
            execution.get("finished_at"),
            row.get("finished_at"),
        ]
        try:
            times = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in stamps]
            if any(value.tzinfo is None for value in times) or times != sorted(times):
                return None
        except (AttributeError, TypeError, ValueError):
            return None
        terminal = row.get("terminal_evidence")
        if terminal is not None:
            from .terminal_evidence import MAX_BYTES
            from .terminal_evidence import PROTOCOL as TERMINAL

            text = terminal.get("text") if isinstance(terminal, dict) else None
            if (
                not isinstance(text, str)
                or len(text.encode()) > MAX_BYTES
                or terminal.get("protocol") != TERMINAL
                or type(terminal.get("truncated")) is not bool
                or terminal.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest()
            ):
                return None
        causal = evidence(row, PANE_PROTOCOL)
        return {
            "protocol": policy,
            "upstream_commit": "44b9942127847f7421db70d8c7e48407f09a3c70",
            "basis": "completed_billed_terminal_loss_uses_upstream_zero_convention",
            "attribution": "agent_exit_evidence" if causal else "unresolved",
            "causal_evidence": causal,
            "verifier_ran": False,
            "raw_outcome_sha256": _sha({k: row.get(k) for k in OUTCOME_FIELDS}),
            "completion_sha256": _sha(stamps),
            "agent_steps_sha256": _sha(steps),
            "terminal_evidence_sha256": _sha(terminal),
        }
    recent = steps[-4:]
    outputs = "\n".join(
        str(item.get("content", ""))
        for step in recent
        for item in (step.get("observation") or {}).get("results", [])
        if isinstance(item, dict)
    )
    for step in recent:
        for call in step.get("tool_calls") or []:
            command = (call.get("arguments") or {}).get("keystrokes")
            if (
                call.get("function_name") == "bash_command"
                and _direct_exit(command)
                and re.search(r"(?:^|\n)[^\n]{0,120}[$#] " + re.escape(command.rstrip("\n")) + r"(?:\n|$)", outputs)
            ):
                return {
                    "protocol": policy,
                    "upstream_commit": "44b9942127847f7421db70d8c7e48407f09a3c70",
                    "basis": "agent_exit_observed_then_terminal_missing",
                    "verifier_ran": False,
                    "raw_outcome_sha256": _sha({k: row.get(k) for k in OUTCOME_FIELDS}),
                    "agent_steps_sha256": _sha(steps),
                    "exit_step_id": step.get("step_id"),
                    "exit_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                }
    if policy in (ERREXIT_PROTOCOL, PIPESTATUS_PROTOCOL, PANE_PROTOCOL):
        for step in recent:
            for call in step.get("tool_calls") or []:
                command = (call.get("arguments") or {}).get("keystrokes")
                if not isinstance(command, str) or call.get("function_name") != "bash_command":
                    continue
                first = command.splitlines()[0].strip() if command.splitlines() else ""
                if first not in ("set -e", "set -o errexit") or re.search(
                    r"(?m)^\s*set (?:\+e|\+o errexit)\s*$", command
                ):
                    continue
                observed = re.search(r"(?:^|\n)[^\n]{0,120}[$#] " + re.escape(first) + r"\r?\n", outputs)
                if observed and "asciinema: recording finished" in outputs[observed.end() :]:
                    return {
                        "protocol": policy,
                        "upstream_commit": "44b9942127847f7421db70d8c7e48407f09a3c70",
                        "basis": "agent_errexit_observed_then_recording_finished_and_terminal_missing",
                        "verifier_ran": False,
                        "raw_outcome_sha256": _sha({k: row.get(k) for k in OUTCOME_FIELDS}),
                        "agent_steps_sha256": _sha(steps),
                        "errexit_step_id": step.get("step_id"),
                        "errexit_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    }
    if policy in (PIPESTATUS_PROTOCOL, PANE_PROTOCOL):
        for step in recent:
            observed = "\n".join(
                str(item.get("content", ""))
                for item in (step.get("observation") or {}).get("results", [])
                if isinstance(item, dict)
            )
            for call in step.get("tool_calls") or []:
                command = (call.get("arguments") or {}).get("keystrokes")
                if (
                    call.get("function_name") == "bash_command"
                    and _pipeline_exit(command)
                    and re.search(r"(?:^|\n)exit\r?\nasciinema: recording finished(?:\r?\n|$)", observed)
                ):
                    return {
                        "protocol": policy,
                        "upstream_commit": "44b9942127847f7421db70d8c7e48407f09a3c70",
                        "basis": "agent_pipeline_status_exit_then_recording_finished_and_terminal_missing",
                        "verifier_ran": False,
                        "raw_outcome_sha256": _sha({k: row.get(k) for k in OUTCOME_FIELDS}),
                        "agent_steps_sha256": _sha(steps),
                        "exit_step_id": step.get("step_id"),
                        "exit_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    }
    if policy == PANE_PROTOCOL and (check := _pane_size_check(row.get("terminal_evidence"))):
        return {
            "protocol": policy,
            "upstream_commit": "44b9942127847f7421db70d8c7e48407f09a3c70",
            "basis": "agent_parent_shell_errexit_failed_numeric_check_then_terminal_missing",
            "verifier_ran": False,
            "raw_outcome_sha256": _sha({k: row.get(k) for k in OUTCOME_FIELDS}),
            "agent_steps_sha256": _sha(steps),
            **check,
        }
    return None


def annotate(row, policy=None):
    if policy is None:
        return copy.deepcopy(row)
    if policy not in SUPPORTED:
        raise ValueError("unknown agent failure scoring policy")
    projected = copy.deepcopy(row)
    if proof := evidence(row, policy):
        projected.update(benchmark_score=0.0, benchmark_failure=proof)
    return projected


def eligible(row, policy=None):
    """Compact bridge admission; the raw trial is rechecked by job_health."""
    proof_policy = (row.get("benchmark_failure") or {}).get("protocol")
    allowed = (
        policy in SUPPORTED and proof_policy in SUPPORTED and SUPPORTED.index(proof_policy) <= SUPPORTED.index(policy)
    )
    return bool(
        row.get("valid")
        or (
            policy in SUPPORTED
            and allowed
            and row.get("benchmark_score") == 0.0
            and not isinstance(row.get("benchmark_score"), bool)
        )
    )


def score(row, policy=None):
    if row.get("valid"):
        return row.get("reward")
    if eligible(row, policy):
        if evidence(row, row["benchmark_failure"]["protocol"]) != row["benchmark_failure"]:
            raise ValueError("agent failure score differs from its retained terminal evidence")
        return 0.0
    return None


def read_trial(path, policy=None):
    path = Path(path)
    raw = json.loads(path.read_text())
    row = trial_outcome(raw)
    if policy in SUPPORTED and not row["valid"]:
        row["trajectory"] = atif_steps(path.parent / "agent")
        if policy in (PANE_PROTOCOL, COMPLETED_PROTOCOL):
            from .terminal_evidence import read_terminal_evidence

            if terminal := read_terminal_evidence(path.parent / "agent/terminus_2.pane"):
                row["terminal_evidence"] = terminal
        row = annotate(row, policy)
    return row


def job_health(job, expected, policy=None):
    rows = []
    for path in sorted(Path(job).glob("*/result.json")):
        row = read_trial(path, policy)
        row["task"] = json.loads(path.read_text()).get("task_name")
        rows.append(row)
    return {
        "valid": Counter(r["task"] for r in rows) == Counter(expected) and all(eligible(r, policy) for r in rows),
        "scheduled": len(expected),
        "recorded": len(rows),
        "invalid": sum(not eligible(r, policy) for r in rows),
        "verified": sum(bool(r["valid"]) for r in rows),
        "agent_failures_scored_zero": sum(
            not r["valid"]
            and eligible(r, policy)
            and (r.get("benchmark_failure") or {}).get("attribution") != "unresolved"
            for r in rows
        ),
        "unattributed_terminal_failures_scored_zero": sum(
            not r["valid"]
            and eligible(r, policy)
            and (r.get("benchmark_failure") or {}).get("attribution") == "unresolved"
            for r in rows
        ),
        "trials": rows,
    }
