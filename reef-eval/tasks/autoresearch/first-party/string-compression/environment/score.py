"""The scoring rule, run by the judge on every submission. Teaching point: grading code the agent SHIPS, safely.

The artifact contains a decompressor program plus a payload. The grader runs
the decompressor in a subprocess and requires a byte-exact round trip;
reward = corpus bytes / (decompressor source bytes + payload string bytes).

Walls:
- the reference corpus is loaded into memory and every on-disk copy the
  verifier container holds is DELETED before the decompressor runs, so
  "decompress by reading the reference file" finds nothing;
- the subprocess gets a hard timeout and an empty temp cwd.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CORPUS = (Path(__file__).parent / "corpus.txt").read_text()
TIMEOUT_SEC = 15
MAX_DECOMPRESSOR_BYTES = 100_000


def _delete_reference_copies() -> None:
    """Remove every corpus copy the verifier container might hold (no-op on
    dev machines, where these absolute paths don't exist)."""
    import contextlib

    for path in (
        Path("/judge/corpus.txt"),
        Path("/tests/corpus.txt"),
        Path("/app/corpus.txt"),
    ):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution.json artifact"}
    try:
        solution = json.loads(Path(artifact).read_text())
        decompressor, payload = solution["decompressor"], solution["payload"]
        if not isinstance(decompressor, str) or not isinstance(payload, str):
            raise TypeError("decompressor and payload must be strings")
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}
    if len(decompressor.encode()) > MAX_DECOMPRESSOR_BYTES:
        return {"reward": 0.0, "reason": "decompressor too large"}

    _delete_reference_copies()

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "decompress.py"
        program.write_text(f"PAYLOAD = {payload!r}\n" + decompressor)
        try:
            run = subprocess.run(
                [sys.executable, str(program)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {"reward": 0.0, "reason": f"decompressor exceeded {TIMEOUT_SEC}s"}
    if run.returncode != 0:
        return {"reward": 0.0, "reason": f"decompressor failed: {run.stderr[:200]}"}
    if run.stdout != CORPUS:
        return {"reward": 0.0, "reason": "round trip is not byte-exact"}

    compressed = len(decompressor.encode()) + len(payload.encode())
    return {
        "reward": len(CORPUS.encode()) / compressed,
        "compressed_bytes": compressed,
    }
