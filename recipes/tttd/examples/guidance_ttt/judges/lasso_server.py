"""Expose the official 17-case Lasso evaluator through the judge protocol."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

if __package__:
    from .protocol import Judge, serve
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import Judge, serve


class LassoJudge(Judge):
    def __init__(self, image: str) -> None:
        self.image = image
        subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)

    def __call__(self, pid: str, language: str, code: str) -> dict:
        if pid != "lasso_path" or language != "python":
            return {"valid": False, "score": 0, "message": "expected lasso_path and a Python wrapper"}
        with tempfile.TemporaryDirectory(prefix="reef-lasso-") as directory:
            root = Path(directory)
            (root / "candidate").mkdir()
            (root / "output").mkdir()
            (root / "candidate/solution.py").write_text(code)
            name = f"reef-lasso-{uuid4().hex}"
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                "4",
                "--memory",
                "16g",
                "--pids-limit",
                "256",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-v",
                f'{root / "candidate"}:/candidate:ro',
                "-v",
                f'{root / "output"}:/output',
                self.image,
            ]
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
                return {"valid": False, "score": 0, "message": "candidate exceeded the 600-second limit"}
            if process.returncode in (125, 126, 127):
                raise RuntimeError("Lasso evaluator container could not start")
            result = root / "output/result.json"
            if process.returncode != 0 or not result.is_file():
                raise RuntimeError(f"Lasso evaluator did not return metrics: {process.stderr[-2000:]}")
            metrics = json.loads(result.read_text())
            score = float(metrics["combined_score"])
            valid = math.isfinite(score) and score > 0 and float(metrics.get("validity", 1)) == 1
            valid = valid and not metrics.get("error")
            return {
                "valid": bool(valid),
                "score": score if valid else 0,
                "score_unbounded": score if valid else 0,
                "message": str(metrics.get("error") or ("accepted" if valid else "correctness check failed")),
                "artifacts": {
                    "official_metrics": metrics,
                    "simpletes_commit": "47d3413da1d85dc24341219d47452d2601e56a57",
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="reef-guidance-lasso")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    serve(LassoJudge(args.image), host=args.host, port=args.port, max_workers=1)


if __name__ == "__main__":
    main()
