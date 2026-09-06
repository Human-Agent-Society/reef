"""Retain terminal output omitted when Harbor fails partway through a batch.

The shared host executor collects this from the completed episode archive,
before its temporary root is removed. It is staged evidence, not a score or
an external algorithm archive. Reef makes it durable with the episode commit.
"""

import hashlib
import json
from pathlib import Path

PROTOCOL = "harbor-terminal-pane-tail-v1"
MAX_BYTES = 1024 * 1024


def read_terminal_evidence(path, *, secrets=()):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("terminal evidence must be a regular collected file")
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - MAX_BYTES))
        text = handle.read(MAX_BYTES).decode("utf-8", errors="replace")
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[REDACTED]")
    return {
        "protocol": PROTOCOL,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "truncated": size > MAX_BYTES,
    }


def preserve_terminal_evidence(root, writable_paths, env):
    """Enrich the private summary; never edit Harbor's raw result or pane."""
    root = Path(root)
    sessions, trials = root / "terminus/sessions", root / "terminus/trials"
    if sessions.resolve() not in {Path(path).resolve() for path in writable_paths}:
        return
    summaries, results = list(sessions.glob("*.json")), list(trials.rglob("result.json"))
    if len(summaries) != 1 or len(results) != 1:
        return
    summary = json.loads(summaries[0].read_text())
    outcome = summary.get("outcome") or {}
    if outcome.get("valid") is not False:
        return
    evidence = read_terminal_evidence(
        results[0].parent / "agent/terminus_2.pane",
        secrets=[env.get(key) for key in ("OPENAI_API_KEY", "E2B_API_KEY", "ANTHROPIC_API_KEY")],
    )
    if evidence is None:
        return
    if "terminal_evidence" in outcome and outcome["terminal_evidence"] != evidence:
        raise ValueError("runner summary conflicts with collected terminal evidence")
    summary["outcome"] = {**outcome, "terminal_evidence": evidence}
    summaries[0].write_text(json.dumps(summary))
