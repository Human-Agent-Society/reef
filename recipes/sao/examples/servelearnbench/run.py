"""Run the Retail L1 stream through Reef: every window in order, as the benchmark's runner does.

For each window, serving tasks run in order. Each SAO batch, including the
window tail, trains before the next batch starts. Held-out tests then run
against the frozen weights and report nothing. The weights carry into
the next window. The summary's ``test_score`` is the
combined mean reward over every test task run, times 100. Official setting
scores are reported separately for hidden and fully specified test tasks.
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
from harness.campaign import ResultLog, category_scores
from harness.config import training_config
from harness.report import episode_end_reason, episode_runtime_load_id, report_episode

BENCHMARK_REVISION = "57aa919b5ae71d60c38a7ed559262447406f1e8b"


def training_releases(client: ReefClient, scenario: str) -> list[dict]:
    """Return selected training checkpoints oldest first, excluding job lifecycle events."""
    try:
        response = client.get(f"/reef/scenarios/{quote(scenario, safe='')}/releases")
    except ReefClientError as error:
        if error.status == 404:
            return []
        raise
    return [
        release
        for release in reversed(response["releases"])
        if release["operation"] == "training"
        and not release.get("pending", False)
        and (release.get("metrics") or {}).get("training_job", {}).get("event") == "checkpoint"
        and (release.get("metrics") or {}).get("selected") is True
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
            if (
                current["finished_jobs"] == expected
                and current["training_job"] is None
                and current["inference_admission"]["open"]
                and isinstance(version, str)
                and version
            ):
                return {**releases[-1], "runtime_load_id": version}
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"Training did not commit release {expected} within {timeout_seconds}s")


def recover_campaign(log: ResultLog, client: ReefClient, model: ReefLLM, config: dict, timeout: float) -> None:
    """Resume committed batches; replay identical reports for a fully collected pending batch."""
    if not log.rows or log.rows[0] != {"config": config}:
        raise RuntimeError(
            "Resume configuration differs from the saved run; keep the original model, paths, and settings"
        )
    boundary = 1
    pending: dict | None = None
    recorded_releases = []
    for index, row in enumerate(log.rows[1:], 1):
        if row.get("type") == "batch_ready":
            if pending is not None:
                raise RuntimeError("Result log contains overlapping training batches")
            pending = row
        elif row.get("type") == "training":
            if pending is None or row["task_ids"] != pending["task_ids"]:
                raise RuntimeError("Training result does not match its collected batch")
            recorded_releases.append(row["release"]["release_id"])
            pending = None
            boundary = index + 1
        elif (row.get("type") == "episode" and not row["trainable"]) or row.get("type") in (
            "untrained_tail",
            "summary",
        ):
            boundary = index + 1
    releases = training_releases(client, model.scenario)
    if [row["release_id"] for row in releases[: len(recorded_releases)]] != recorded_releases:
        raise RuntimeError(
            "Saved results and restored Reef training history disagree; use the matching checkpoint and storage"
        )
    if len(releases) < len(recorded_releases) or len(releases) > len(recorded_releases) + int(pending is not None):
        raise RuntimeError("Restored training progress does not match this campaign")
    if pending is None:
        # No report is sent until batch_ready is durable. Partial collection can be repeated safely.
        log.discard_after(boundary)
        return
    tasks = pending["task_ids"]
    episodes = {row["task_id"]: row for row in log.rows if row.get("type") == "episode"}
    batch = [episodes[task_id] for task_id in tasks]
    if len(releases) == len(recorded_releases):
        for row in batch:
            model.receipts = list(row["receipts"])
            report_episode(
                model, model.scenario, row["task_id"], row["reward"], row["end_reason"], batch_task_ids=tasks
            )
    release = wait_for_training(client, model.scenario, len(recorded_releases) + 1, timeout)
    log.append({"type": "training", "task_ids": tasks, "batch_size": len(tasks), "release": release})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("blind", "oracle", "sao"), required=True)
    parser.add_argument("--scenario", default="retail-l1", help="The Reef scenario that records and trains")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", default="", help="Unique, increasing window indices; default every window")
    parser.add_argument(
        "--smoke", action="store_true", help="First selected window: up to two serving batches and two tests"
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=protocol.MAX_STEPS)
    parser.add_argument("--batch-size", type=int, help="Must match SAO_BATCH when that variable is set")
    parser.add_argument("--training-timeout", type=float, default=1800)
    parser.add_argument(
        "--resume", action="store_true", help="Continue with the original checkpoints, Reef storage, and results"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Print and validate the plan without connecting to Reef"
    )
    parser.add_argument("--stop-after-batches", type=int, help="Stop cleanly after this many total committed batches")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_steps <= 0 or args.training_timeout <= 0:
        parser.error("Token, turn, and timeout limits must be positive")
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        parser.error("--stop-after-batches must be positive")
    if args.batch_size is not None:
        if "SAO_BATCH" in os.environ and args.batch_size != int(os.environ["SAO_BATCH"]):
            parser.error("--batch-size must match the trainer SAO_BATCH")
        os.environ["SAO_BATCH"] = str(args.batch_size)
    try:
        settings = training_config()
        windows = [int(value) for value in args.windows.split(",")] if args.windows else []
    except ValueError as error:
        parser.error(str(error))
    args.batch_size = settings["SAO_BATCH"]
    if args.max_tokens >= settings["SAO_CONTEXT"]:
        parser.error("--max-tokens must leave room for the prompt within SAO_CONTEXT")
    bundle = get_bundle("retail_l1", fresh=True)
    protocol.stamp_serving(bundle["serving"])
    windows = windows or list(range(n_windows(bundle)))
    if any(window < 0 or window >= n_windows(bundle) for window in windows):
        parser.error(f"--windows must name windows 0 to {n_windows(bundle) - 1}")
    if windows != sorted(set(windows)):
        parser.error("--windows must be unique and in increasing chronological order")
    if args.smoke:
        windows = windows[:1]
    plan: dict[int, tuple[list, list, int]] = {}
    window_plan = []
    for window in windows:
        serving = [task for task in bundle["serving"] if task.z["window"] == window]
        tests = bundle["test_set"](window)
        test_timestamp = protocol.stamp_test(tests, bundle["serving"], window)
        if args.smoke:
            serving, tests = serving[: max(2, 2 * args.batch_size - 1)], tests[:2]
        plan[window] = (serving, tests, test_timestamp)
        batches = (len(serving) + args.batch_size - 1) // args.batch_size if args.method == "sao" else 0
        tail = len(serving) % args.batch_size if args.method == "sao" else 0
        window_plan.append(
            {
                "window": window,
                "serving": len(serving),
                "test": len(tests),
                "training_batches": batches,
                "partial_batch_tasks": tail,
            }
        )
    serving_total = sum(len(serving) for serving, _, _ in plan.values())
    test_total = sum(len(tests) for _, tests, _ in plan.values())
    batch_total = sum(row["training_batches"] for row in window_plan)
    trained_total = serving_total if args.method == "sao" else 0
    warmup = settings["SAO_WARMUP"] if args.method == "sao" else 0
    training_plan = {
        "windows": window_plan,
        "training_batches": batch_total,
        "critic_only_batches": min(warmup, batch_total),
        "actor_updates": max(0, batch_total - warmup),
    }
    print(json.dumps({"training_plan": training_plan}), flush=True)
    if args.method == "sao" and batch_total <= warmup:
        parser.error(
            f"Plan has {batch_total} training batches and {warmup} warmup batches: no actor update. Reduce SAO_WARMUP or SAO_BATCH, or select more serving tasks"
        )
    if args.check_only:
        return
    client = ReefClient(os.environ["REEF_SERVICE_URL"], token=os.environ.get("REEF_TOKEN"), timeout_s=60)
    if not args.resume and training_releases(client, args.scenario):
        raise RuntimeError("The scenario already trained; use --resume with its original directories")
    cfg = llm.ModelConfig(model="reef", reasoning_effort=None, max_tokens=args.max_tokens)
    model = ReefLLM(ReefClient(client.service_url, token=client.token, timeout_s=1800), args.scenario, cfg, args.seed)
    oracle = Oracle()
    oracle.setup(bundle)
    model_name = os.environ.get("REEF_MODEL_NAME") or Path(os.environ.get("MODEL_DIR", "reef")).name
    config = {
        "type": "config",
        "campaign_format": 3,
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
        "micro_batch_size": settings["SAO_MICRO_BATCH"],
        "critic_only_batches": settings["SAO_WARMUP"],
        "context_length": settings["SAO_CONTEXT"],
        "tail_policy": "train_partial_batch",
        "untrained_tail_tasks": 0,
        "partial_batch_tasks": sum(row["partial_batch_tasks"] for row in window_plan),
        "max_steps": args.max_steps,
        "max_output_per_task": protocol.MAX_OUTPUT_PER_TASK,
        "tool_budget": protocol.ENGINE_TOOL_BUDGET,
        "serving_count": serving_total,
        "test_count": test_total,
        "update_batch_size": args.batch_size,
        "reef_scenario": args.scenario,
        "model": model_name,
        "model_path": os.environ.get("MODEL_DIR"),
        "checkpoint_dir": os.environ.get("CHECKPOINT_DIR"),
        "training_plan": training_plan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = ResultLog(args.output, resume=args.resume)
    try:
        if args.resume:
            recover_campaign(log, client, model, config, args.training_timeout)
        else:
            log.append({"config": config})
        episodes = [row for row in log.rows if row.get("type") == "episode"]
        completed = {(row["phase"], row["task_id"]) for row in episodes}
        if len(completed) != len(episodes):
            raise RuntimeError("Saved results contain duplicate task results")
        expected_releases = sum(row.get("type") == "training" for row in log.rows)
        expected_version = None
        if expected_releases:
            release = wait_for_training(client, args.scenario, expected_releases, args.training_timeout)
            expected_version = release["runtime_load_id"]
        if any(row.get("type") == "summary" for row in log.rows):
            print(json.dumps(log.rows[-1]), flush=True)
            return
        if args.stop_after_batches is not None and expected_releases >= args.stop_after_batches:
            print(
                f"Already completed {expected_releases} batches; remove --stop-after-batches to continue", flush=True
            )
            return
        for window in windows:
            serving, tests, test_timestamp = plan[window]
            pending: list[dict] = []
            for phase, tasks in (("serving", serving), ("test", tests)):
                for task_index, task in enumerate(tasks):
                    if (phase, task.task_id) in completed:
                        continue
                    model.begin_episode()
                    releases_before = training_releases(client, args.scenario)
                    if len(releases_before) != expected_releases:
                        raise RuntimeError("Unexpected model update outside the serving-task boundary")
                    extra = oracle.context(task, phase, window) if args.method == "oracle" else ""
                    started = time.monotonic()
                    reward, trace, usage, judge_usage = run_episode(
                        model, bundle, task, max_steps=args.max_steps, extra_system=extra
                    )
                    if training_releases(client, args.scenario) != releases_before:
                        raise RuntimeError("Model changed during an episode")
                    reason = episode_end_reason(trace)
                    version = episode_runtime_load_id(model) if args.method == "sao" else None
                    if args.method == "sao" and expected_version is None:
                        expected_version = version
                    if expected_version is not None and version != expected_version:
                        raise RuntimeError("Episode did not use the latest committed training weights")
                    trainable = phase == "serving" and args.method == "sao"
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
                        "trainable": trainable,
                    }
                    log.append(row)
                    episodes.append(row)
                    if trainable:
                        pending.append(row)
                        if len(pending) == args.batch_size or task_index == len(tasks) - 1:
                            task_ids = [episode["task_id"] for episode in pending]
                            # Persist the entire batch before any report can cause an update.
                            log.append({"type": "batch_ready", "task_ids": task_ids})
                            for episode in pending:
                                model.receipts = list(episode["receipts"])
                                report_episode(
                                    model,
                                    args.scenario,
                                    episode["task_id"],
                                    episode["reward"],
                                    episode["end_reason"],
                                    batch_task_ids=task_ids,
                                )
                            expected_releases += 1
                            release = wait_for_training(
                                client, args.scenario, expected_releases, args.training_timeout
                            )
                            expected_version = release["runtime_load_id"]
                            log.append(
                                {
                                    "type": "training",
                                    "task_ids": task_ids,
                                    "batch_size": len(task_ids),
                                    "release": release,
                                }
                            )
                            pending.clear()
                            if args.stop_after_batches is not None and expected_releases >= args.stop_after_batches:
                                print(
                                    f"Stopped after {expected_releases} committed batches; continue with --resume",
                                    flush=True,
                                )
                                return
                    print(f"W{window} {phase} {task.task_id}: reward={reward.reward:g}, end={reason}", flush=True)
        test_rewards = [row["reward"] for row in episodes if row["phase"] == "test"]
        window_scores = {}
        window_categories = {}
        for window in windows:
            selected = [row for row in episodes if row["window"] == window]
            rewards = [row["reward"] for row in selected if row["phase"] == "test"]
            window_scores[str(window)] = 100 * sum(rewards) / len(rewards)
            window_categories[str(window)] = category_scores(selected)
        ending_counts: dict[str, dict[str, int]] = {"serving": {}, "test": {}}
        for row in episodes:
            counts = ending_counts[row["phase"]]
            counts[row["end_reason"]] = counts.get(row["end_reason"], 0) + 1
        summary = {
            "type": "summary",
            "test_score": 100 * sum(test_rewards) / len(test_rewards),
            "test_score_definition": "combined_test_mean",
            "window_scores": window_scores,
            "category_scores": category_scores(episodes),
            "window_category_scores": window_categories,
            "windows": windows,
            "training_releases": expected_releases,
            "smoke": args.smoke,
            "ending_counts": ending_counts,
            "trained_tasks": trained_total,
            "untrained_tail_tasks": 0,
            "partial_batch_tasks": config["partial_batch_tasks"],
        }
        log.append(summary)
        print(json.dumps(summary), flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
