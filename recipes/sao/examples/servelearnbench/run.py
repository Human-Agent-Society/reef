"""Run the Retail L1 stream through Reef: every window in order, as the benchmark's runner does.

For each window: the serving tasks run in order and, for SAO, every full
batch trains the policy before the next batch starts; then the window's
held-out tests run against the frozen weights and report nothing. The
weights carry into the next window. The summary's ``test_score`` is the
mean reward over every test task run, times 100: the benchmark's setting
score for the scenario when every window runs.
"""

import argparse
import json
import os
import time
import urllib.error
from pathlib import Path
from urllib.parse import quote

from reef_client import ReefClient, ReefClientError
from servelearnbench import llm, protocol
from servelearnbench.methods.baselines import Oracle
from servelearnbench.scenarios import get_bundle, n_windows
from servelearnbench.scoring import category

from harness.agent import ReefLLM, run_episode
from harness.report import episode_end_reason, episode_runtime_load_id, report_episode

BENCHMARK_REVISION = "57aa919b5ae71d60c38a7ed559262447406f1e8b"


def training_releases(client: ReefClient, scenario: str) -> list[dict]:
    """Return committed training releases oldest first; an uncreated scenario has none."""
    try:
        response = client.get(f"/reef/scenarios/{quote(scenario, safe='')}/releases")
    except ReefClientError as error:
        if error.status == 404:
            return []
        raise
    return [
        release
        for release in reversed(response["releases"])
        if release["operation"] == "training" and not release.get("pending", False)
    ]


def wait_for_training(client: ReefClient, scenario: str, expected: int, timeout_seconds: float) -> dict:
    """Wait for exactly expected commits, including serving-weight publication."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            releases = training_releases(client, scenario)
            status = client.get("/reef/status")
        except TimeoutError:
            continue  # Registry reads can wait behind an in-flight update.
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                continue
            raise
        if status["error"]:
            raise RuntimeError(f"Training failed: {status['error']}")
        if len(releases) > expected:
            raise RuntimeError("Unexpected training release; use a dedicated scenario for this run")
        if len(releases) == expected:
            current = status["scenarios"][scenario]
            version = current["current_runtime_load_id"]
            # Saved-artifact release rows do not carry runtime_load_id.
            # Read the published version only after admission reopens.
            if current["inference_admission"]["open"] and isinstance(version, str) and version:
                return {**releases[-1], "runtime_load_id": version}
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"Training did not commit release {expected} within {timeout_seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("blind", "oracle", "sao"), required=True)
    parser.add_argument("--scenario", default="retail-l1", help="The Reef scenario that records and trains")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--windows", default="", help="Comma-separated window indices to run, in order; default every window"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first window only, with two serving batches and two test tasks; label results as smoke",
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
    windows = [int(value) for value in args.windows.split(",") if value] or list(range(n_windows(bundle)))
    if any(window < 0 or window >= n_windows(bundle) for window in windows):
        parser.error(f"--windows must name windows 0 to {n_windows(bundle) - 1}")
    if args.smoke:
        windows = windows[:1]
    # The stream per window: its serving tasks in order, then its tests stamped
    # with the window's end position, as the benchmark's runner does.
    plan: dict[int, tuple[list, list, int]] = {}
    for window in windows:
        serving = [task for task in bundle["serving"] if task.z["window"] == window]
        tests = bundle["test_set"](window)
        test_timestamp = protocol.stamp_test(tests, bundle["serving"], window)
        if args.smoke:
            serving, tests = serving[: 2 * args.batch_size], tests[:2]
        plan[window] = (serving, tests, test_timestamp)
    serving_total = sum(len(serving) for serving, _, _ in plan.values())
    test_total = sum(len(tests) for _, tests, _ in plan.values())
    trained_total = sum(len(serving) // args.batch_size * args.batch_size for serving, _, _ in plan.values())
    if args.method == "sao" and not trained_total:
        parser.error("The serving stream must contain at least one complete training batch")
    oracle = Oracle()
    oracle.setup(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_releases = 0
    expected_version: str | None = None
    test_rewards: list[float] = []
    window_scores: dict[str, float] = {}
    pending_tasks: list[str] = []
    ending_counts: dict[str, dict[str, int]] = {"serving": {}, "test": {}}
    model_name = os.environ.get("REEF_MODEL_NAME") or Path(os.environ.get("MODEL_DIR", "reef")).name
    config = {
        "type": "config",
        "benchmark_revision": BENCHMARK_REVISION,
        "scenario": "retail_l1",
        "windows": windows,
        "method": args.method,
        "seed": args.seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "smoke": args.smoke,
        "max_tokens": args.max_tokens,
        "global_batch_size": args.batch_size,
        "micro_batch_size": 1,
        "critic_only_batches": int(os.environ.get("SAO_WARMUP", "10")),
        "context_length": 40960,
        "tail_policy": "score_without_training",
        "untrained_tail_tasks": serving_total - trained_total,
        "max_steps": args.max_steps,
        "max_output_per_task": protocol.MAX_OUTPUT_PER_TASK,
        "tool_budget": protocol.ENGINE_TOOL_BUDGET,
        "serving_count": serving_total,
        "test_count": test_total,
        "update_batch_size": 1,
        "reef_scenario": args.scenario,
        "model": model_name,
    }
    # Exclusive creation prevents mixing a fresh model with previous results.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"config": config}) + "\n")
        output.flush()
        for window in windows:
            serving, tests, test_timestamp = plan[window]
            trainable_count = len(serving) // args.batch_size * args.batch_size
            window_rewards: list[float] = []
            for phase, tasks in (("serving", serving), ("test", tests)):
                for task_index, task in enumerate(tasks):
                    model.begin_episode()
                    releases_before = training_releases(client, args.scenario)
                    if len(releases_before) != expected_releases:
                        raise RuntimeError("Unexpected model update outside the serving-task boundary")
                    extra = oracle.context(task, phase, window) if args.method == "oracle" else ""
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
                        "window": window,
                        "task_id": task.task_id,
                        "timestamp": task.z["stream_pos"] if phase == "serving" else test_timestamp,
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
                            output.write(
                                json.dumps({"type": "report", "task_id": task.task_id, "report": report}) + "\n"
                            )
                            if len(pending_tasks) == args.batch_size:
                                expected_releases += 1
                                release = wait_for_training(
                                    client, args.scenario, expected_releases, args.training_timeout
                                )
                                expected_version = release["runtime_load_id"]
                                output.write(
                                    json.dumps(
                                        {"type": "training", "task_ids": list(pending_tasks), "release": release}
                                    )
                                    + "\n"
                                )
                                pending_tasks.clear()
                        output.flush()
                    if phase == "test":
                        test_rewards.append(reward.reward)
                        window_rewards.append(reward.reward)
                    print(f"W{window} {phase} {task.task_id}: reward={reward.reward:g}, end={reason}", flush=True)
            window_scores[str(window)] = 100 * sum(window_rewards) / len(window_rewards)
            print(
                f"W{window} test: {len(window_rewards)} tasks, mean reward {window_scores[str(window)]:.1f}",
                flush=True,
            )
        summary = {
            "type": "summary",
            "test_score": 100 * sum(test_rewards) / len(test_rewards),
            "window_scores": window_scores,
            "windows": windows,
            "training_releases": expected_releases,
            "smoke": args.smoke,
            "ending_counts": ending_counts,
            "trained_tasks": trained_total if args.method == "sao" else 0,
            "untrained_tail_tasks": serving_total - trained_total if args.method == "sao" else 0,
        }
        output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
