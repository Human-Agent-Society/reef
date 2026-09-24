"""Run one explicitly budgeted author SDPO/GRPO experiment from its pinned checkout.

Use the author's training environment. This launcher changes paths, GPU count,
seed and an optional smoke-step ceiling; it records those changes before running.
It never submits a sweep or silently resumes an old output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REFERENCE_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
DATASETS = {"chemistry": "datasets/sciknoweval/chemistry", "tooluse": "datasets/tooluse"}


def build_command(args: argparse.Namespace) -> list[str]:
    reference = args.reference.resolve()
    output = args.output.resolve()
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "sdpo" if args.method == "sdpo" else "baseline_grpo",
    ]
    overrides = {
        "vars.dir": str(reference),
        "vars.task": DATASETS[args.dataset],
        "vars.log_dir": str(output),
        "vars.ckpt_dir": str(output / "checkpoints"),
        "custom_reward_function.path": str(reference / "verl/utils/reward_score/feedback/__init__.py"),
        "actor_rollout_ref.model.path": str(args.model_path.resolve()),
        "actor_rollout_ref.actor.fsdp_config.seed": args.seed,
        "data.seed": args.seed,
        "data.train_batch_size": 32,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.minibatch,
        "actor_rollout_ref.actor.optim.lr": args.learning_rate,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 10,
        "actor_rollout_ref.rollout.val_kwargs.n": 16,
        "algorithm.rollout_correction.rollout_is": "token",
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.logger": "[console]",
        "trainer.project_name": "reef-sdpo-reproduction",
        "trainer.group_name": f"{args.dataset}-{args.method}",
        "trainer.experiment_name": f"{args.method}-seed-{args.seed}",
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.total_epochs": 30,
        "trainer.resume_mode": "disable",
    }
    if args.method == "sdpo":
        overrides.update(
            {
                "actor_rollout_ref.actor.self_distillation.distillation_topk": 100,
                "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success": "True",
                "actor_rollout_ref.actor.self_distillation.alpha": 0.5,
                "actor_rollout_ref.actor.self_distillation.include_environment_feedback": "False",
            }
        )
    if args.steps:
        overrides["trainer.total_training_steps"] = args.steps
    return [*command, *(f"{key}={value}" for key, value in overrides.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True, help="Hugging Face snapshot SHA used at model-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("sdpo", "grpo"), default="sdpo")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="chemistry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--minibatch", type=int, choices=(8, 32), default=32)
    parser.add_argument("--learning-rate", type=float, choices=(1e-5, 1e-6), default=1e-5)
    parser.add_argument(
        "--steps", type=int, default=2, help="0 selects the full 30-epoch budget; otherwise a labeled smoke run"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.gpus < 1 or 256 % args.gpus:
        parser.error("steps must be nonnegative; GPU count must divide 256")
    if args.method == "sdpo" and (args.minibatch != 32 or args.learning_rate != 1e-5):
        parser.error("Section 3 SDPO uses minibatch 32 and LR 1e-5")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f"reference must be {REFERENCE_COMMIT}, got {revision}")
    subprocess.run(["git", "-C", str(args.reference), "diff", "--exit-code", "HEAD"], check=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be empty; exact EMA resume is not assumed")
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.reference / DATASETS[args.dataset]
    hashes = {
        split: hashlib.sha256((source_dir / f"{split}.json").read_bytes()).hexdigest() for split in ("train", "test")
    }
    command = build_command(args)
    manifest = {
        "reference_commit": revision,
        "model_path": str(args.model_path.resolve()),
        "model_revision": args.model_revision,
        "method": args.method,
        "dataset": args.dataset,
        "data_sha256": hashes,
        "seed": args.seed,
        "gpus": args.gpus,
        "kind": "smoke" if args.steps else "paper_budget",
        "steps": args.steps,
        "command": command,
        "dry_run": args.dry_run,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not (args.model_path / "config.json").is_file():
        raise ValueError("model-path must contain a downloaded, revision-pinned Hugging Face model")
    environment = {
        **os.environ,
        "TASK": DATASETS[args.dataset],
        "EXPERIMENT": f"{args.method}-seed-{args.seed}",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
    }
    subprocess.run(
        [sys.executable, "data/preprocess.py", "--data_source", DATASETS[args.dataset]],
        cwd=args.reference,
        env=environment,
        check=True,
    )
    # Record the resolved Hydra configuration before any model training.
    resolved = subprocess.check_output([*command, "--cfg", "job", "--resolve"], cwd=args.reference, env=environment)
    (args.output / "resolved.yaml").write_bytes(resolved)
    with (args.output / "train.log").open("w") as stream:
        subprocess.run(
            command, cwd=args.reference, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


if __name__ == "__main__":
    main()
