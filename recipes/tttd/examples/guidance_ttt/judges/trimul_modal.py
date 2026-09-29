"""Deploy the pinned TriMul evaluator on an H100 with Modal."""

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DISCOVER = Path(os.environ.get("GUIDANCE_DISCOVER_ROOT", HERE.parent / "work/trimul/discover")).resolve()
EVALUATOR = DISCOVER / "examples/gpu_mode/lib/bioml/trimul"
app = modal.App("reef-guidance-trimul")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.7.1", "triton==3.3.1", "pyyaml==6.0.2", "numpy==2.2.6"
)
if modal.is_local():
    if not (EVALUATOR / "task.yml").is_file():
        raise RuntimeError("Run prepare.py trimul before deploying the evaluator")
    image = image.add_local_dir(EVALUATOR, remote_path="/opt/trimul", copy=True).add_local_file(
        HERE / "trimul_runner.py", remote_path="/opt/trimul_runner.py", copy=True
    )


@app.function(image=image, gpu="H100", timeout=1200, max_containers=1)
def evaluate(solution: str) -> dict:
    import sys

    sys.path.insert(0, "/opt")
    from trimul_runner import run_official_trimul_evaluation

    return run_official_trimul_evaluation(solution, evaluator_dir="/opt/trimul", timeout_s=1100)
