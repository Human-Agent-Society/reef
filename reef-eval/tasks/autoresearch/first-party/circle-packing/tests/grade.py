"""The verifier's whole job under the judge protocol: ask the judge for the
final result and write it down.

Generic: task authors never edit this file. It runs inside the task
environment after the agent's budget ends. The judge exposes finalization
on a separate verifier port (``$JUDGE_URL`` with the port incremented by
one) behind a token; this script fetches the token from the verifier port
and calls ``GET {verifier_url}/final`` (final.py on the best submission if
the task has one, else the best session score), then writes:

- ``/logs/verifier/reward.json``: ``{"reward": <float>}``
- ``/logs/verifier/reason.txt``: the human-readable reason
- ``/logs/verifier/submissions.jsonl``: one ``{"t", "score"}`` line per
  submission; reef-eval ingests these as the trusted score-over-time curve.

If ``$VERIFIER_JUDGE_URL`` is set it is used directly; otherwise the
verifier URL is derived from ``$JUDGE_URL`` by incrementing the port.
Falls back to the agent port without a token for old judges that don't
have the two-port separation.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")


def _verifier_url(judge_url: str) -> str:
    vurl = os.environ.get("VERIFIER_JUDGE_URL")
    if vurl:
        return vurl.rstrip("/")
    parsed = urllib.parse.urlparse(judge_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8082
    return f"http://{host}:{port + 1}"


def _fetch_token(verifier_url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{verifier_url}/token", timeout=10) as resp:
            return json.loads(resp.read()).get("token")
    except Exception:
        return None


def finalize(judge_url: str) -> dict:
    verifier_url = _verifier_url(judge_url)

    token = _fetch_token(verifier_url)
    if token:
        try:
            req = urllib.request.Request(
                f"{verifier_url}/final",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {
                "reward": 0.0,
                "reason": f"judge unreachable: {e!r}",
                "submissions": [],
            }

    # Fall back to the agent port (old judge without two-port separation)
    try:
        with urllib.request.urlopen(f"{judge_url}/final", timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"reward": 0.0, "reason": f"judge unreachable: {e!r}", "submissions": []}


if __name__ == "__main__":
    result = finalize(os.environ.get("JUDGE_URL", "http://judge:8082"))
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps({"reward": result["reward"]}))
    (VERIFIER_DIR / "reason.txt").write_text(result.get("reason") or "ok")
    with (VERIFIER_DIR / "submissions.jsonl").open("w") as f:
        for entry in result.get("submissions", []):
            f.write(json.dumps({"t": entry["t"], "score": entry["score"]}) + "\n")
    print(json.dumps({"reward": result["reward"]}), result.get("reason", ""))
