"""CLI for the Qwen3-8B summary-only Guidance-TTT deployment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from examples.guidance_ttt import task_ids

from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=task_ids(), default="polyomino_packing")
    parser.add_argument("--executor", choices=("gpt_oss_120b", "openrouter_glm_5_2"), required=True)
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--executor-temperature", type=float, default=None)
    parser.add_argument("--executor-max-tokens", type=int, default=None)
    parser.add_argument(
        "--gpt-oss-reasoning-effort",
        choices=("low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--guidance-max-tokens", type=int, default=8_192)
    parser.add_argument("--guidance-temperature", type=float, default=1.0)
    parser.add_argument("--guidance-top-p", type=float, default=1.0)
    parser.add_argument("--guidance-top-k", type=int, default=-1)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=12_288)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--training-horizon-steps",
        type=int,
        default=None,
        help="fixed optimizer-scheduler horizon; defaults to steps and must stay unchanged on resume",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--verifier-timeout-s",
        type=int,
        default=None,
        help="task-level timeout; defaults to 340s (Polyomino), 120s (VLIW), or 1160s (TriMul)",
    )
    parser.add_argument("--verifier-concurrency", type=int, default=None)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        default=None,
        help="JSON object with task-specific verifier settings; credentials should be referenced by env var",
    )
    parser.add_argument("--frontiercs-base-dir", type=Path, default=None)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8081")
    parser.add_argument("--seed-library-path", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--frozen-policy-control",
        action="store_true",
        help="run the matched search-only control without reporting rollouts for LoRA updates",
    )
    parser.add_argument(
        "--require-nonconstant-step",
        action="store_true",
        help="fail qualification if no candidate verifies or every reward group is constant",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(os.environ.get("REEF_GUIDANCE_TTT_STATE_ROOT", "/workspace/state")),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    verifier_config = None
    if args.verifier_config is not None:
        verifier_config = json.loads(args.verifier_config.read_text())
        if not isinstance(verifier_config, dict):
            raise ValueError("--verifier-config must contain a JSON object")
    result = run_experiment(
        gpu_count=args.gpu_count,
        state_root=args.state_root,
        executor=args.executor,
        tensor_parallel_size=args.tensor_parallel_size,
        executor_base_url=args.executor_base_url,
        executor_temperature=args.executor_temperature,
        executor_max_tokens=args.executor_max_tokens,
        gpt_oss_reasoning_effort=args.gpt_oss_reasoning_effort,
        model_path=args.model_path,
        groups=args.groups,
        rollouts=args.rollouts,
        guidance_max_tokens=args.guidance_max_tokens,
        guidance_temperature=args.guidance_temperature,
        guidance_top_p=args.guidance_top_p,
        guidance_top_k=args.guidance_top_k,
        sampling_seed=args.sampling_seed,
        sequence_length=args.sequence_length,
        lora_rank=args.lora_rank,
        steps=args.steps,
        training_horizon_steps=args.training_horizon_steps,
        max_workers=args.max_workers,
        verifier_timeout_s=args.verifier_timeout_s,
        verifier_concurrency=args.verifier_concurrency,
        verifier_config=verifier_config,
        frontiercs_base_dir=args.frontiercs_base_dir,
        judge_url=args.judge_url,
        seed_library_path=args.seed_library_path,
        run_name=args.run_name,
        resume=args.resume,
        require_nonconstant_step=args.require_nonconstant_step,
        task_id=args.task,
        adaptation_enabled=not args.frozen_policy_control,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
