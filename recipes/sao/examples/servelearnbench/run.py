"""Run the fixed Retail L1/W1 pilot using official tasks and scoring."""

import argparse
import json
import os
import time
from pathlib import Path

from reef_client import ReefClient
from servelearnbench import llm, protocol
from servelearnbench.methods.baselines import Oracle
from servelearnbench.scenarios import get_bundle
from servelearnbench.scoring import category

from harness.adapter import (
    ReefLLM,
    episode_end_reason,
    episode_runtime_load_id,
    report_episode,
    run_episode,
    training_releases,
    wait_for_training,
)

BENCHMARK_REVISION = "57aa919b5ae71d60c38a7ed559262447406f1e8b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("blind", "oracle", "sao"), required=True)
    parser.add_argument("--scenario", default="retail-l1-w1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="Run two serving batches and two test tasks; label results as smoke"
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=protocol.MAX_STEPS, help="Maximum model calls per task")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SAO_BATCH", "8")))
    parser.add_argument("--training-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_steps <= 0 or args.training_timeout <= 0 or args.batch_size <= 0:
        parser.error("token, turn, and timeout limits must be positive")
    if "SAO_BATCH" in os.environ and args.batch_size != int(os.environ["SAO_BATCH"]):
        parser.error("--batch-size must match the trainer SAO_BATCH")
    client = ReefClient(
        os.environ["REEF_SERVICE_URL"],
        token=os.environ.get("REEF_TOKEN"),
        timeout_s=60,
    )
    if training_releases(client, args.scenario):
        raise RuntimeError("The scenario already trained; start an independent stack and run directory")
    cfg = llm.ModelConfig(model="reef", reasoning_effort=None, max_tokens=args.max_tokens)
    inference_client = ReefClient(client.service_url, token=client.token, timeout_s=1800)
    model = ReefLLM(inference_client, args.scenario, cfg, args.seed)
    bundle = get_bundle("retail_l1", fresh=True)
    protocol.stamp_serving(bundle["serving"])
    serving = [task for task in bundle["serving"] if task.z["window"] == 1]
    tests = bundle["test_set"](1)
    protocol.stamp_test(tests, bundle["serving"], 1)
    if (len(serving), len(tests)) != (140, 64):
        raise RuntimeError("Retail L1/W1 differs from the pinned 140/64 split")
    if args.smoke:
        serving, tests = serving[: 2 * args.batch_size], tests[:2]
    trainable_count = len(serving) // args.batch_size * args.batch_size
    if args.method == "sao" and not trainable_count:
        parser.error("The serving split must contain at least one complete training batch")
    oracle = Oracle()
    oracle.setup(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_releases = 0
    expected_version: str | None = None
    test_rewards: list[float] = []
    pending_tasks: list[str] = []
    ending_counts: dict[str, dict[str, int]] = {"serving": {}, "test": {}}
    model_name = os.environ.get("REEF_MODEL_NAME") or Path(os.environ.get("MODEL_DIR", "reef")).name
    config = {
        "type": "config",
        "benchmark_revision": BENCHMARK_REVISION,
        "scenario": "retail_l1",
        "window": 1,
        "method": args.method,
        "seed": args.seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "smoke": args.smoke,
        "max_tokens": args.max_tokens,
        "global_batch_size": args.batch_size,
        "micro_batch_size": 1,
        "critic_only_batches": int(os.environ.get("SAO_WARMUP", "10")),
        "context_length": 131072,
        "tail_policy": "score_without_training",
        "untrained_tail_tasks": len(serving) - trainable_count,
        "max_steps": args.max_steps,
        "max_output_per_task": protocol.MAX_OUTPUT_PER_TASK,
        "tool_budget": protocol.ENGINE_TOOL_BUDGET,
        "serving_count": len(serving),
        "test_count": len(tests),
        "update_batch_size": 1,
        "reef_scenario": args.scenario,
        "model": model_name,
    }
    # Exclusive creation prevents mixing a fresh model with previous results.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"config": config}) + "\n")
        output.flush()
        for phase, tasks in (("serving", serving), ("test", tests)):
            for task_index, task in enumerate(tasks):
                model.begin_episode()
                releases_before = training_releases(client, args.scenario)
                if len(releases_before) != expected_releases:
                    raise RuntimeError("Unexpected model update outside the serving-task boundary")
                extra = oracle.context(task, phase, 1) if args.method == "oracle" else ""
                started = time.monotonic()
                # Infrastructure errors abort the run. They are never reward=0.
                reward, trace, usage, judge_usage = run_episode(
                    model, bundle, task, max_steps=args.max_steps, extra_system=extra
                )
                if training_releases(client, args.scenario) != releases_before:
                    raise RuntimeError("Model changed during an episode")
                reason = episode_end_reason(trace)
                ending_counts[phase][reason] = ending_counts[phase].get(reason, 0) + 1
                version = episode_runtime_load_id(model) if args.method == "sao" else None
                if args.method == "sao" and expected_version is None:
                    expected_version = version
                if expected_version is not None and version != expected_version:
                    raise RuntimeError("Episode did not use the latest committed training weights")
                row = {
                    "type": "episode",
                    "scenario": "retail_l1",
                    "method": args.method,
                    "model": model_name,
                    "instruction": task.instruction,
                    "phase": phase,
                    "window": 1,
                    "task_id": task.task_id,
                    "timestamp": (
                        task.z["stream_pos"] if phase == "serving" else protocol.window_end(bundle["serving"], 1)
                    ),
                    "category": category("retail", task.z.get("slice")),
                    "reward": reward.reward,
                    "tokens": usage,
                    "judge_tokens": judge_usage,
                    "answer_detail": reward.answer_detail,
                    "trace": trace,
                    "receipts": list(model.receipts),
                    "end_reason": reason,
                    "training_releases_before": expected_releases,
                    "runtime_load_id": version,
                    "duration_seconds": time.monotonic() - started,
                }
                # Save scored episodes before reporting so a failed training
                # step remains inspectable, rather than silently disappearing.
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if phase == "serving" and args.method == "sao":
                    if task_index >= trainable_count:
                        output.write(json.dumps({"type": "untrained_tail", "task_id": task.task_id}) + "\n")
                    else:
                        report = report_episode(model, args.scenario, task.task_id, reward.reward, reason)
                        pending_tasks.append(task.task_id)
                        output.write(json.dumps({"type": "report", "task_id": task.task_id, "report": report}) + "\n")
                        if len(pending_tasks) == args.batch_size:
                            expected_releases += 1
                            release = wait_for_training(
                                client, args.scenario, expected_releases, args.training_timeout
                            )
                            expected_version = release["runtime_load_id"]
                            output.write(
                                json.dumps({"type": "training", "task_ids": list(pending_tasks), "release": release})
                                + "\n"
                            )
                            pending_tasks.clear()
                    output.flush()
                if phase == "test":
                    test_rewards.append(reward.reward)
                print(f"{phase} {task.task_id}: reward={reward.reward:g}, end={reason}", flush=True)
        summary = {
            "type": "summary",
            "test_score": 100 * sum(test_rewards) / len(test_rewards),
            "training_releases": expected_releases,
            "smoke": args.smoke,
            "ending_counts": ending_counts,
            "trained_tasks": trainable_count if args.method == "sao" else 0,
            "untrained_tail_tasks": len(serving) - trainable_count if args.method == "sao" else 0,
        }
        output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
