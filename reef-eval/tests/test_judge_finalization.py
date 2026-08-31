"""Adversarial and normal-path tests for verifier-only finalization.

The judge exposes two ports: an agent port (``POST /submit``, ``GET /status``,
``GET /health``) and a verifier port (``GET /final`` with token, ``GET /token``,
``GET /health``).  The agent port's ``/final`` returns 403 without executing
``final.py``: the agent cannot trigger or observe hidden evaluation.

These tests start the real judge server as a subprocess on two ports and
exercise the HTTP boundary directly.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

from reef_eval.executors import _start_judge as start_judge

TEMPLATE = Path(__file__).parent.parent / "tasks" / "_template"
JUDGE_SERVER = TEMPLATE / "environment" / "judge_server.py"
SCORE_PY = TEMPLATE / "environment" / "score.py"


def _start_judge(tmp_path, judge_dir=None, with_final=False):
    """Start the real judge on two ports.

    Returns ``(agent_url, verifier_url, token, proc, marker_path)``.
    If *with_final* is true, a final.py with an observable side effect
    (writing *marker_path*) is installed.
    """
    if judge_dir is None:
        judge_dir = tmp_path / "judge"
        judge_dir.mkdir()
        (judge_dir / "score.py").write_text(SCORE_PY.read_text())
        (judge_dir / "judge_config.json").write_text(json.dumps({}))

    marker = tmp_path / "final_ran"
    if with_final:
        (judge_dir / "final.py").write_text(
            "import json\n"
            "from pathlib import Path\n"
            "def grade(artifact):\n"
            f"    Path({str(marker)!r}).write_text('ran')\n"
            "    x = float(json.loads(Path(artifact).read_text())['x'])\n"
            "    return {'reward': x, 'reason': 'ok'}\n"
        )

    # Reuse reef-eval's launcher rather than a second copy of it: it retries a
    # lost port race, which is what made this file flaky.
    (judge_dir / "judge_server.py").write_text(JUDGE_SERVER.read_text())
    data_dir = tmp_path / "data"
    proc, agent_url, verifier_url = start_judge(
        judge_dir, data_dir, tmp_path / "judge.log"
    )
    token = (data_dir / ".verifier_token").read_text()
    return agent_url, verifier_url, token, proc, marker


def _submit(url, payload):
    req = urllib.request.Request(f"{url}/submit", data=json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url, path, token=None):
    req = urllib.request.Request(f"{url}{path}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# Adversarial tests: the agent path cannot finalize
# ---------------------------------------------------------------------------


def test_agent_port_final_is_refused(tmp_path):
    """GET /final on the agent port returns 403."""
    agent, verifier, token, proc, _ = _start_judge(tmp_path)
    try:
        _submit(agent, {"x": 0.5})
        code, body = _get(agent, "/final")
        assert code == 403
        assert "verifier-only" in body["error"]
    finally:
        proc.kill()
        proc.wait()


def test_agent_port_final_does_not_execute_final_py(tmp_path):
    """Calling /final on the agent port must NOT trigger final.py."""
    agent, verifier, token, proc, marker = _start_judge(tmp_path, with_final=True)
    try:
        _submit(agent, {"x": 0.9})
        code, body = _get(agent, "/final")
        assert code == 403
        assert not marker.exists(), "final.py was executed via the agent port"
    finally:
        proc.kill()
        proc.wait()


def test_agent_port_no_token_endpoint(tmp_path):
    """The agent port does not expose GET /token."""
    agent, _, _, proc, _ = _start_judge(tmp_path)
    try:
        code, _ = _get(agent, "/token")
        assert code == 404
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Normal-path tests: the trusted verifier can finalize
# ---------------------------------------------------------------------------


def test_verifier_finalizes_with_token(tmp_path):
    """GET /final on the verifier port with the correct token succeeds."""
    agent, verifier, token, proc, _ = _start_judge(tmp_path)
    try:
        _submit(agent, {"x": 0.7})
        code, body = _get(verifier, "/final", token)
        assert code == 200
        assert body["reward"] == 0.7
        assert body["best_n"] == 0
    finally:
        proc.kill()
        proc.wait()


def test_verifier_final_without_token_is_refused(tmp_path):
    """The verifier port requires the token."""
    agent, verifier, _, proc, _ = _start_judge(tmp_path)
    try:
        _submit(agent, {"x": 0.5})
        code, body = _get(verifier, "/final")
        assert code == 403
        assert "token" in body["error"]
    finally:
        proc.kill()
        proc.wait()


def test_verifier_final_with_wrong_token_is_refused(tmp_path):
    """A wrong token is rejected."""
    agent, verifier, _, proc, _ = _start_judge(tmp_path)
    try:
        _submit(agent, {"x": 0.5})
        code, _ = _get(verifier, "/final", "wrong-token")
        assert code == 403
    finally:
        proc.kill()
        proc.wait()


def test_verifier_retry_returns_cached_result(tmp_path):
    """Verifier retries are idempotent: same result, no re-execution."""
    agent, verifier, token, proc, marker = _start_judge(tmp_path, with_final=True)
    try:
        _submit(agent, {"x": 0.8})
        code1, body1 = _get(verifier, "/final", token)
        assert code1 == 200
        assert marker.exists(), "final.py should have run on first call"

        # second call: marker already exists, final.py should not run again
        marker.write_text("already")  # tamper to detect re-execution
        code2, body2 = _get(verifier, "/final", token)
        assert code2 == 200
        assert body1 == body2
        assert marker.read_text() == "already", "final.py ran again on retry"
    finally:
        proc.kill()
        proc.wait()


def test_submissions_rejected_after_finalization(tmp_path):
    """After the verifier finalizes, further submissions are refused."""
    agent, verifier, token, proc, _ = _start_judge(tmp_path)
    try:
        _submit(agent, {"x": 0.5})
        _get(verifier, "/final", token)
        code, body = _submit(agent, {"x": 0.9})
        assert code == 429
        assert "finalized" in body["error"]
    finally:
        proc.kill()
        proc.wait()


def test_verifier_token_endpoint(tmp_path):
    """The verifier port exposes GET /token for fetching the token."""
    agent, verifier, token, proc, _ = _start_judge(tmp_path)
    try:
        code, body = _get(verifier, "/token")
        assert code == 200
        assert body["token"] == token
    finally:
        proc.kill()
        proc.wait()
