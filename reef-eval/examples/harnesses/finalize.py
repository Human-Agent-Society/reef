"""Best-effort final-artifact submission, the ``_finalize`` step.

Under the judge protocol the verifier's only input is the judge's
submission log: the final reward is the best submission, and zero
submissions score 0.0 no matter what the agent left in the workspace.
Harnesses whose feedback hook submits structurally (OpenEvolve's evaluator)
are covered by construction; harnesses that are only told to submit
(Codex via the instruction, CORAL workers via AGENTS.md) are not. This
script closes that gap: when the judge saw zero submissions, spend one on
the run's final artifact. When the run already submitted, best-of semantics
cover it and there is nothing to do, so no budget is spent and no score the
agent never confirmed enters the trace.

Never fails: any error (no ``JUDGE_URL``, no artifact, budget exhausted,
judge unreachable) just skips the submission, so the trial is unaffected.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def finalize(artifact: Path, judge_url: str | None = None) -> dict | None:
    """Submit ``artifact`` iff the judge has seen zero submissions so far."""
    url = (judge_url or os.environ.get("JUDGE_URL") or "").rstrip("/")
    artifact = Path(artifact)
    if not url or not artifact.is_file():
        return None
    with urllib.request.urlopen(f"{url}/status", timeout=10) as response:
        status = json.loads(response.read())
    if status.get("used"):
        return None
    request = urllib.request.Request(
        f"{url}/submit",
        data=artifact.read_bytes(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


if __name__ == "__main__":
    try:
        result = finalize(Path(sys.argv[1]))
        if result is not None:
            print(json.dumps(result))
    except Exception as error:
        print(f"final-artifact submission skipped: {error!r}", file=sys.stderr)
