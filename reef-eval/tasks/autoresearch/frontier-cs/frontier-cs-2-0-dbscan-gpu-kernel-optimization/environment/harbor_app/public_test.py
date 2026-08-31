"""Public self-test == the graded evaluation, run by the judge.

The data generator is judge-only and is NOT present in this (agent) image, so you
cannot read how the workloads are produced and hardcode a data-specific shortcut.
Instead, this packages your current `/app/dbscanlib`, submits it to the judge --
which runs the EXACT graded workloads on a GPU and grades your clustering against
the exact reference (ARI) -- and prints your per-workload result + score. It is
byte-for-byte the same evaluation used for your final grade.

(Submission is asynchronous; this waits for the result. You can also submit
directly with `bash /app/make_submission.sh && bash /app/submit.sh` and poll with
`bash /app/submissions.sh`.)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SUBMISSIONS_LOG = Path("/logs/agent/submissions.jsonl")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _latest_uuid_from_log() -> str | None:
    if not SUBMISSIONS_LOG.exists():
        return None
    lines = [l for l in SUBMISSIONS_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    for l in reversed(lines):
        try:
            u = json.loads(l).get("submission_uuid")
            if u:
                return u
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    # 1) package the current package into /app/solution.patch
    subprocess.run(["bash", "/app/make_submission.sh"], check=True)
    # 2) submit to the judge (async); it runs the exact graded eval on GPU
    print("Submitting to the judge (runs the exact graded workloads + returns your score)...",
          flush=True)
    r = subprocess.run(["python3", "/app/submit.py", *sys.argv[1:]],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    # 3) find the submission uuid and wait for the result
    m = _UUID.findall(r.stdout + r.stderr)
    uuid = m[-1] if m else _latest_uuid_from_log()
    if not uuid:
        print("Submitted. Poll for the result with: bash /app/submissions.sh")
        return 0
    return subprocess.call(["python3", "/app/wait_submission.py", uuid])


if __name__ == "__main__":
    sys.exit(main())
