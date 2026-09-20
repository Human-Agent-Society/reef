"""Figure 3 of arXiv:2601.19897 through reef-eval: one model learns Tool Use, then Science Q&A, by SDFT.

The stream is two Harbor tasks, ``harbor/tooluse`` then ``harbor/science``,
run under ``harness:HarborAgent``. Each stage is a fresh Reef stack started
from the previous stage's exported weights, the way the reference
implementation chains its single-task runs; the stack's recipe is the
``sdft`` recipe (``serve.yaml``) and its learning-rate schedule spans
exactly the stage's steps. The stack takes four GPUs (the actor with the
rollout engines colocated) and one host port, and the stream names it, so
a second stream on other GPUs and another port (``CUDA_VISIBLE_DEVICES``,
``REEF_PORT``) runs beside it: a smoke beside the full run.

For each stage:

    start   — ``docker compose up`` with the stage's starting weights and
              its step count
    run     — ``lab.run`` builds the task's images, starts the judge, and
              runs the stage in the task container; the judge scores the
              served model on both skills every few steps
    record  — the verifier's final reward (this task's accuracy after the
              last step, the other skill's beside it) and every judge score
              land in the Lab store tagged ``stream``, ``arm``, ``position``
    stop    — ``docker compose down``; the stage's HF export seeds the next

Rows already recorded are skipped, so a crashed stream resumes; a new
``--stream`` name starts over.

    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client \\
        --with-editable . run.py
    SDFT_STEPS=2 ... run.py --stream smoke      # two steps per stage
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
HARBOR = HERE / "harbor"
STAGES = ("tooluse", "science")
#: The reference's training splits: the stage's step count follows from them.
TRAINING_PROMPTS = {"tooluse": 4046, "science": 2674}
#: The rows' ``arm`` tag in the Lab store.
ARM = "sdft"
#: The stack's GPUs (four comma-separated ids) and the host port its Reef publishes.
DEFAULT_GPUS = "0,1,2,3"
DEFAULT_PORT = 28902
#: serve.yaml's lr-warmup-iters; Megatron requires the decay span to be longer.
LR_WARMUP_STEPS = 10
AGENT = {"name": "harness:HarborAgent", "model_name": "reef"}
#: The stack's container name prefix, as docker-compose.yaml fixes it; the stream's name follows.
CONTAINER = "reef-sdft-skills"
#: Container paths: the base model under the models mount, a stage's export under the run mount.
BASE_MODEL = "/root/models/Qwen2.5-7B-Instruct"
EXPERIMENT_MOUNT = "/var/lib/experiment"

RUN_DIR = Path(os.environ.get("RUN_DIR", HERE / "work")).resolve()


def stage_steps(task: str) -> int:
    """Optimizer steps the stage runs: the runner's schedule over the split, capped like the runner."""
    epochs = int(os.environ.get("SDFT_EPOCHS", "2"))
    prompts_per_step = int(os.environ.get("SDFT_PROMPTS_PER_STEP", "32"))
    cap = int(os.environ.get("SDFT_STEPS", "0"))
    steps = epochs * TRAINING_PROMPTS[task] // prompts_per_step
    return min(steps, cap) if cap else steps


def latest_export(stage_dir: Path) -> Path:
    """The stage's last HF export (the bridge names them by training step)."""
    exports = [path for path in (stage_dir / "checkpoints" / "hf").glob("*") if path.name.isdigit()]
    if not exports:
        raise FileNotFoundError(f"no HF export under {stage_dir}/checkpoints/hf; the stage did not train")
    return max(exports, key=lambda path: int(path.name))


def stack_environment(stream: str) -> dict[str, str]:
    """What the stack and the tasks read from the environment: the stack's name, its GPUs and its host port."""
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS).split(",")
    if len(gpus) != 4:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must name four GPUs, got {gpus}")
    return {
        "STACK": stream,
        "REEF_HOST_PORT": os.environ.get("REEF_PORT", str(DEFAULT_PORT)),
        **{f"REEF_GPU_{index}": gpu.strip() for index, gpu in enumerate(gpus)},
    }


def compose(stream: str, *arguments: str, environment: dict[str, str]) -> None:
    subprocess.run(
        ["docker", "compose", "-p", f"skills-{stream}", "-f", str(HERE / "docker-compose.yaml"), *arguments],
        check=True,
        env={**os.environ, **stack_environment(stream), **environment},
    )


def start_stack(stream: str, model_path: str, steps: int, stage_dir: Path) -> None:
    """Start the Reef stack on ``model_path`` with a schedule over ``steps``; state goes to ``stage_dir``.

    A smoke run's few steps still get a schedule longer than the warmup, as
    Megatron insists; its learning rate then never leaves the warmup ramp.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "SDFT_MODEL_PATH": model_path,
        "SDFT_LR_DECAY_ITERS": str(max(steps, LR_WARMUP_STEPS + 1)),
        "RUN_DIR": str(stage_dir),
        "EXPERIMENT_DIR": str(RUN_DIR),
    }
    print(f"==> the {stream} stack on {model_path}, {steps} steps, state in {stage_dir}", flush=True)
    compose(stream, "up", "-d", "--wait", environment=environment)


def stop_stack(stream: str, stage_dir: Path) -> None:
    environment = {"RUN_DIR": str(stage_dir), "EXPERIMENT_DIR": str(RUN_DIR)}
    compose(stream, "down", "--timeout", "120", environment=environment)
    # A stack mid-step can outlive compose's grace; the next stage needs its GPUs.
    subprocess.run(["docker", "rm", "-f", f"{CONTAINER}-{stream}"], check=False, capture_output=True)


async def run_stream(stream: str, seed: int) -> None:
    lab = Lab(RUN_DIR / "lab")
    # The task containers read the host port from this process's environment.
    os.environ.update(stack_environment(stream))
    model_path = BASE_MODEL
    for position, task in enumerate(STAGES):
        stage_dir = RUN_DIR / stream / ARM / task
        tags = {"stream": stream, "arm": ARM, "position": position, "seed": seed}
        # The scenario names the stage; the task's compose file hands it to both containers.
        os.environ["REEF_SCENARIO"] = f"{stream}-{ARM}-{task}"
        os.environ["SDFT_SEED"] = str(seed)
        key = f"{stream}@{position:03d}:{ARM}:{task}:seed{seed}"
        row = lab.store.get(key)
        if row is None:
            start_stack(stream, model_path, stage_steps(task), stage_dir)
            try:
                row = await lab.run(str(HARBOR / task), AGENT, tags=tags, key=key)
            finally:
                stop_stack(stream, stage_dir)
        else:
            print(f"[{stream} {position}] {task}: recorded already, reusing its export", flush=True)
        if row.tags.get("error") or "reward" not in row.rewards:
            raise RuntimeError(
                f"stage {position} ({task}) of {stream} failed: {row.tags.get('error') or 'no reward'}; "
                f"its row is recorded under {lab.root}, so rerun with a new --stream name"
            )
        print(f"[{stream} {position}] {task}: {json.dumps(row.rewards)}", flush=True)
        export = latest_export(stage_dir)
        model_path = f"{EXPERIMENT_MOUNT}/{export.relative_to(RUN_DIR)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stream", default="skill-stream", help="the stream's name in the Lab store (a new name reruns)"
    )
    parser.add_argument("--seed", type=int, default=42, help="the training order's seed")
    arguments = parser.parse_args()
    for variable in ("REEF_TOKEN", "REEF_IMAGE", "MODEL_DIR", "REEF_ROOT"):
        if not os.environ.get(variable):
            sys.exit(f"run.py: {variable} is not set; run.sh sets it")
    asyncio.run(run_stream(arguments.stream, arguments.seed))


if __name__ == "__main__":
    main()
