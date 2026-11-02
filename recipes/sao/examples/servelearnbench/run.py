"""Run the Retail L1 stream through Reef: every window in order, as the benchmark's runner does.

For each window, serving tasks run in order. Each SAO batch, including the
window tail, trains before the next batch starts: the driver reports the
batch's episodes, puts their samples into the scenario's manual training
buffer, and queues one training instruction for them. Held-out tests then run
against the frozen weights and report nothing. The weights carry into the
next window. The summary's ``test_score`` is the combined mean reward over
every test task run, times 100. Official setting scores are reported
separately for hidden and fully specified test tasks.
"""

import argparse
import json
import os
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.parse import quote

from reef_client import ReefClient
from servelearnbench import llm, protocol
from servelearnbench.methods.baselines import Oracle
from servelearnbench.scenarios import get_bundle, n_windows
from servelearnbench.scoring import category

from harness.agent import ReefLLM, run_episode
from harness.campaign import ResultLog, category_scores
from harness.config import training_config
from harness.report import episode_artifact_id, episode_end_reason, report_episode, report_id

BENCHMARK_REVISION = "57aa919b5ae71d60c38a7ed559262447406f1e8b"
#: The pause between two reads of the training state.
POLL_SECONDS = 1.0


def scenario_path(scenario: str, suffix: str) -> str:
    return f"/reef/scenarios/{quote(scenario, safe='')}{suffix}"


def scenario_status(client: ReefClient, scenario: str) -> dict:
    """Return the scenario's block of ``GET /reef/status``; a recorded training error fails the campaign."""
    status = client.get("/reef/status")
    if status["error"]:
        raise RuntimeError(f"Training failed: {status['error']}")
    if scenario not in status["scenarios"]:
        raise RuntimeError(f"Reef does not hold the scenario {scenario!r}")
    return status["scenarios"][scenario]


def training_jobs(client: ReefClient, scenario: str) -> list[dict]:
    """Return the scenario's training jobs, oldest first."""
    return client.get(scenario_path(scenario, "/jobs"))["jobs"]


def instruction_jobs(jobs: list[dict]) -> dict[str, dict]:
    """Return the jobs that training instructions made, by instruction ID, oldest first."""
    return {job["instruction"]["id"]: job for job in jobs if job["instruction"] is not None}


def ensure_scenario(client: ReefClient, scenario: str, *, trains: bool) -> dict:
    """Create the scenario when it does not exist; return its status block. A training run needs manual mode."""
    client.post("/reef/scenarios", scenario, {"name": scenario})
    block = scenario_status(client, scenario)
    if trains and block["training_mode"] != "manual":
        raise RuntimeError("The campaign queues each batch's training itself; set recipe.config.training_mode: manual")
    return block


def training_instruction_id(scenario: str, task_ids: list[str]) -> str:
    """The instruction ID of one declared batch; a resumed campaign queues the same instruction again."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"servelearnbench:{scenario}:train:{','.join(task_ids)}").hex


def training_instruction(instruction_id: str, text: str, epochs: int, served_artifact_id: str) -> dict:
    """The ``POST /reef/train`` body that trains the manual buffer in ``epochs`` passes, one optimizer step each."""
    return {
        "agent_record_id": instruction_id,
        "text": text,
        "session": "servelearnbench",
        "release_id": served_artifact_id,
        "parameters": {"scheduling": {"epochs": epochs, "shuffle": False, "remainder": "partial"}},
    }


def wait_for_samples(client: ReefClient, scenario: str, report_ids: list[str], timeout_seconds: float) -> list[dict]:
    """Wait until the sample store holds the sample of every report; return them in report order.

    Reef makes one sample per report, with the ID ``<report id>:0``. A recorded
    training error fails the wait.
    """
    wanted = {f"{value}:0": index for index, value in enumerate(report_ids)}
    deadline = time.monotonic() + timeout_seconds
    while True:
        scenario_status(client, scenario)
        found: dict[str, dict] = {}
        after = 0
        while True:
            page = client.get(scenario_path(scenario, f"/samples?after={after}&limit=1000"))
            found.update({row["sample_id"]: row for row in page["samples"] if row["sample_id"] in wanted})
            if page["next"] is None:
                break
            after = page["next"]
        if len(found) == len(wanted):
            return sorted(found.values(), key=lambda row: wanted[row["sample_id"]])
        if time.monotonic() >= deadline:
            missing = sorted(set(wanted) - set(found))
            raise TimeoutError(
                f"Reef made no sample for {len(missing)} reports within {timeout_seconds}s: {missing[:3]}"
            )
        time.sleep(POLL_SECONDS)


def wait_for_training(
    client: ReefClient,
    scenario: str,
    instruction_id: str,
    expected_jobs: int,
    served_before: str,
    timeout_seconds: float,
) -> dict:
    """Wait until the instruction's job finished and the scenario serves its checkpoint.

    The scenario then counts ``expected_jobs`` finished jobs, runs no job,
    admits inference again, and serves an artifact other than ``served_before``
    that is also the trainer's training start. Return the job and that artifact.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            block = scenario_status(client, scenario)
            job = instruction_jobs(training_jobs(client, scenario)).get(instruction_id)
        except TimeoutError:
            continue  # Status reads can wait behind an in-flight commit.
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                continue
            raise
        if job is not None and job["state"] == "failed":
            raise RuntimeError(f"Training job {job['id']} failed: {job['error']}")
        if block["finished_jobs"] > expected_jobs:
            raise RuntimeError("Unexpected training job; use a dedicated scenario for this run")
        served = block["served_artifact_id"]
        if (
            job is not None
            and job["state"] == "finished"
            and block["finished_jobs"] == expected_jobs
            and block["training_job"] is None
            and block["inference_admission"]["open"]
            and served != served_before
            and all(row["active"] == row["training_start"] == served for row in block["artifacts"].values())
        ):
            return {"job_id": job["id"], "artifact_id": served, "scenario_step": block["scenario_step"]}
        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"Training job {expected_jobs} did not serve its checkpoint within {timeout_seconds}s")


def train_batch(
    client: ReefClient,
    model: ReefLLM,
    scenario: str,
    batch: list[dict],
    *,
    instruction_id: str,
    text: str,
    epochs: int,
    expected_jobs: int,
    served_artifact_id: str,
    timeout_seconds: float,
) -> dict:
    """Report the batch's episodes, queue one training job for them, and wait for its checkpoint to serve.

    Every step repeats safely after an interruption: a report and the
    instruction keep their IDs, and a batch whose job exists is not queued
    again.
    """
    for episode in batch:
        model.receipts = list(episode["receipts"])
        report_episode(model, scenario, episode["task_id"], episode["reward"], episode["end_reason"])
    report_ids = [report_id(scenario, episode["task_id"]) for episode in batch]
    samples = wait_for_samples(client, scenario, report_ids, timeout_seconds)
    if instruction_id not in instruction_jobs(training_jobs(client, scenario)):
        trained = [row["sample_id"] for row in samples if row["train_counts"]]
        if trained:
            raise RuntimeError(f"Samples of this batch trained outside the campaign: {trained[:3]}")
        sample_ids = [row["sample_id"] for row in samples]
        client.post(scenario_path(scenario, "/buffer/insert"), scenario, {"sample_ids": sample_ids})
        client.post("/reef/train", scenario, training_instruction(instruction_id, text, epochs, served_artifact_id))
    return wait_for_training(client, scenario, instruction_id, expected_jobs, served_artifact_id, timeout_seconds)


def recover_campaign(log: ResultLog, client: ReefClient, model: ReefLLM, config: dict, timeout_seconds: float) -> None:
    """Resume committed batches; finish a fully collected pending batch with its original reports."""
    if not log.rows or log.rows[0] != {"config": config}:
        raise RuntimeError(
            "Resume configuration differs from the saved run; keep the original model, paths, and settings"
        )
    boundary = 1
    pending: dict | None = None
    recorded_instructions = []
    for index, row in enumerate(log.rows[1:], 1):
        if row.get("type") == "batch_ready":
            if pending is not None:
                raise RuntimeError("Result log contains overlapping training batches")
            pending = row
        elif row.get("type") == "training":
            if pending is None or row["task_ids"] != pending["task_ids"]:
                raise RuntimeError("Training result does not match its collected batch")
            recorded_instructions.append(row["instruction_id"])
            pending = None
            boundary = index + 1
        elif (row.get("type") == "episode" and not row["trainable"]) or row.get("type") == "summary":
            boundary = index + 1
    jobs = list(instruction_jobs(training_jobs(client, model.scenario)))
    if jobs[: len(recorded_instructions)] != recorded_instructions:
        raise RuntimeError(
            "Saved results and restored Reef training history disagree; use the matching checkpoint and storage"
        )
    if len(jobs) > len(recorded_instructions) + int(pending is not None):
        raise RuntimeError("Restored training progress does not match this campaign")
    if pending is None:
        # No report is sent until batch_ready is durable. Partial collection can be repeated safely.
        log.discard_after(boundary)
        return
    episodes = {row["task_id"]: row for row in log.rows if row.get("type") == "episode"}
    result = train_batch(
        client,
        model,
        model.scenario,
        [episodes[task_id] for task_id in pending["task_ids"]],
        instruction_id=pending["instruction_id"],
        text=pending["text"],
        epochs=pending["epochs"],
        expected_jobs=len(recorded_instructions) + 1,
        served_artifact_id=pending["served_artifact_id"],
        timeout_seconds=timeout_seconds,
    )
    log.append(training_row(pending, result))


def training_row(declared: dict, result: dict) -> dict:
    """The result row of a declared batch: its tasks, instruction and passes, and the job and served artifact."""
    return {
        "type": "training",
        "task_ids": declared["task_ids"],
        "batch_size": len(declared["task_ids"]),
        "instruction_id": declared["instruction_id"],
        "epochs": declared["epochs"],
        **result,
    }


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
        "training_jobs": batch_total,
        "training_batches": batch_total + warmup,
        "critic_only_batches": warmup,
        "actor_updates": batch_total,
    }
    if args.method == "sao" and window_plan:
        # Warmup reuses the first job's data; it does not consume a normal actor batch.
        window_plan[0]["training_batches"] += warmup
    print(json.dumps({"training_plan": training_plan}), flush=True)
    if args.method == "sao" and batch_total == 0:
        parser.error("Plan has no serving tasks: no actor update")
    if args.check_only:
        return
    client = ReefClient(os.environ["REEF_SERVICE_URL"], token=os.environ.get("REEF_TOKEN"), timeout_s=60)
    scenario = ensure_scenario(client, args.scenario, trains=args.method == "sao")
    if not args.resume and scenario["finished_jobs"]:
        raise RuntimeError("The scenario already trained; use --resume with its original directories")
    cfg = llm.ModelConfig(model="reef", reasoning_effort=None, max_tokens=args.max_tokens)
    model = ReefLLM(ReefClient(client.service_url, token=client.token, timeout_s=1800), args.scenario, cfg, args.seed)
    oracle = Oracle()
    oracle.setup(bundle)
    model_name = os.environ.get("REEF_MODEL_NAME") or Path(os.environ.get("MODEL_DIR", "reef")).name
    config = {
        "type": "config",
        "campaign_format": 5,
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
        "warmup_policy": "reuse_first_batch",
        "training_mode": "manual",
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
        training_rows = [row for row in log.rows if row.get("type") == "training"]
        expected_jobs = len(training_rows)
        # The artifact that every episode must be served by: the last training's checkpoint, or the start.
        expected_artifact = training_rows[-1]["artifact_id"] if training_rows else scenario["served_artifact_id"]
        if scenario_status(client, args.scenario)["served_artifact_id"] != expected_artifact:
            raise RuntimeError("Reef serves another artifact than the campaign's last training produced")
        if any(row.get("type") == "summary" for row in log.rows):
            print(json.dumps(log.rows[-1]), flush=True)
            return
        if args.stop_after_batches is not None and expected_jobs >= args.stop_after_batches:
            print(f"Already completed {expected_jobs} batches; remove --stop-after-batches to continue", flush=True)
            return
        for window in windows:
            serving, tests, test_timestamp = plan[window]
            pending: list[dict] = []
            for phase, tasks in (("serving", serving), ("test", tests)):
                for task_index, task in enumerate(tasks):
                    if (phase, task.task_id) in completed:
                        continue
                    model.begin_episode()
                    before = scenario_status(client, args.scenario)
                    if before["finished_jobs"] != expected_jobs or before["training_job"] is not None:
                        raise RuntimeError("Unexpected model update outside the serving-task boundary")
                    extra = oracle.context(task, phase, window) if args.method == "oracle" else ""
                    started = time.monotonic()
                    reward, trace, usage, judge_usage = run_episode(
                        model, bundle, task, max_steps=args.max_steps, extra_system=extra
                    )
                    reason = episode_end_reason(trace)
                    artifact_id = episode_artifact_id(model)
                    if artifact_id != expected_artifact:
                        raise RuntimeError("Episode was not served by the latest committed training weights")
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
                        "training_jobs_before": expected_jobs,
                        "artifact_id": artifact_id,
                        "duration_seconds": time.monotonic() - started,
                        "trainable": trainable,
                    }
                    log.append(row)
                    episodes.append(row)
                    if trainable:
                        pending.append(row)
                        if len(pending) == args.batch_size or task_index == len(tasks) - 1:
                            task_ids = [episode["task_id"] for episode in pending]
                            # The first job of the campaign repeats its batch for the critic warmup passes.
                            epochs = warmup + 1 if expected_jobs == 0 else 1
                            declared = {
                                "type": "batch_ready",
                                "window": window,
                                "task_ids": task_ids,
                                "instruction_id": training_instruction_id(args.scenario, task_ids),
                                "text": f"Retail L1 window {window}: SAO on {len(task_ids)} serving tasks, "
                                f"{epochs} passes",
                                "epochs": epochs,
                                "served_artifact_id": expected_artifact,
                            }
                            # Persist the entire batch before any report can cause an update.
                            log.append(declared)
                            result = train_batch(
                                client,
                                model,
                                args.scenario,
                                pending,
                                instruction_id=declared["instruction_id"],
                                text=declared["text"],
                                epochs=epochs,
                                expected_jobs=expected_jobs + 1,
                                served_artifact_id=expected_artifact,
                                timeout_seconds=args.training_timeout,
                            )
                            expected_jobs += 1
                            expected_artifact = result["artifact_id"]
                            log.append(training_row(declared, result))
                            pending.clear()
                            if args.stop_after_batches is not None and expected_jobs >= args.stop_after_batches:
                                print(
                                    f"Stopped after {expected_jobs} committed batches; continue with --resume",
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
            "training_jobs": expected_jobs,
            "critic_only_batches": warmup,
            "actor_updates": expected_jobs,
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
