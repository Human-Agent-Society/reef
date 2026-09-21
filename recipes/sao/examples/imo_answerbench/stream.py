"""Stream single rollouts of many problems through Reef, N in flight, the paper's way.

SAO (arXiv:2607.07508, §4.1) trains with a batch of 128 rollouts and a group
size of 1: every rollout in a step comes from a different prompt, and a rollout
joins the next step the moment it is scored, without waiting for siblings. The
Harbor loop in ``run.py`` is the smallest possible instance of that (six
rollouts of one problem, one optimizer step each) and is meant as a smoke
test. This driver is the paper's shape at a budget one node can afford: it
keeps ``IN_FLIGHT`` requests open at all times, each on a problem drawn from a
training pool, grades each completion with the same strict rule the Harbor
verifier uses, and reports the score against that rollout's receipt. Reef
trains whenever ``recipe.config.batch-size`` reports have accumulated.

Problems come from a JSONL file with ``problem_idx``, ``problem`` and ``gold``
columns (``export_problems.py`` writes it from the Hugging Face dump). The
held-out indices are never served, so a later evaluation on them measures
generalization rather than memorization.

Environment:
  SAO_PROBLEMS      path to the problems JSONL (required)
  SAO_HOLDOUT       comma-separated problem_idx values to exclude (default: none)
  SAO_POOL          optional comma-separated problem_idx values to train on
  SAO_IN_FLIGHT     concurrent rollouts (default 8; equal to the recipe batch size)
  SAO_GROUP         GRPO control: rollouts per prompt (default 1 = SAO). With
                    G > 1 the driver runs IN_FLIGHT // G groups at once; each
                    group waits for all G of its rollouts and then posts their
                    reports together, the barrier GRPO needs and SAO lacks.
  SAO_BUDGET        total scored rollouts before the driver stops (default 320)
  SAO_MAX_TOKENS    generation window per rollout (default 61440)
  SAO_RECORDS_PATH  where one JSON line per scored rollout is appended
  SAO_SEED          sampling seed for the problem order (default 0)
  SAO_BATCH         rollouts per optimizer step (the recipe's batch-size); the
                    driver waits for BUDGET // SAO_BATCH training releases
                    before exiting, so the stack is not torn down mid-step
  SAO_TRAIN_DRAIN_TIMEOUT_S  how long to wait for that drain (default 14400)
  SAO_PROGRESS_FILE  optional file holding the number of optimizer steps done
                    (written by whoever can see the trainer's checkpoints); with
                    it the driver keeps at most SAO_AHEAD batches (default 2)
                    generated beyond the trained ones, so engines do not run
                    far ahead of a slower trainer and pile up stale rollouts
  SAO_STALL_S       seconds without trainer progress before the pacer releases
                    one extra batch (default 2700). Must exceed the step time,
                    or the pacer releases batches the trainer will drop as stale
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from reef_client import ReefClient, ReefClientError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness.grader import answers_equal, extract_answer

SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = "sao-smoke"
RECIPE = "sao"

IN_FLIGHT = int(os.environ.get("SAO_IN_FLIGHT", "8"))
GROUP = int(os.environ.get("SAO_GROUP", "1"))
BUDGET = int(os.environ.get("SAO_BUDGET", "320"))
MAX_TOKENS = int(os.environ.get("SAO_MAX_TOKENS", "61440"))
RECORDS_PATH = Path(os.environ.get("SAO_RECORDS_PATH", "work/records/stream.jsonl"))
SEED = int(os.environ.get("SAO_SEED", "0"))
MAX_FAILURE_STREAK = 24
BATCH = int(os.environ.get("SAO_BATCH", "1"))
TRAIN_DRAIN_TIMEOUT_S = int(os.environ.get("SAO_TRAIN_DRAIN_TIMEOUT_S", "14400"))
PROGRESS_FILE = os.environ.get("SAO_PROGRESS_FILE")
AHEAD = int(os.environ.get("SAO_AHEAD", "3"))
FAILURE_PAUSE_S = 60
INSTRUCTION_SUFFIX = "\n\nPut your final answer within \\boxed{}."

_write_lock = threading.Lock()


def load_problems() -> list[dict]:
    path = os.environ.get("SAO_PROBLEMS")
    if not path:
        raise SystemExit("SAO_PROBLEMS must point at the problems JSONL")
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    holdout = {int(x) for x in os.environ.get("SAO_HOLDOUT", "").split(",") if x.strip()}
    pool_env = os.environ.get("SAO_POOL", "").strip()
    pool = {int(x) for x in pool_env.split(",") if x.strip()} if pool_env else None
    selected = [r for r in rows if r["problem_idx"] not in holdout and (pool is None or r["problem_idx"] in pool)]
    if not selected:
        raise SystemExit("the training pool is empty after applying SAO_HOLDOUT/SAO_POOL")
    return selected


def problem_order(problems: list[dict], budget: int) -> list[dict]:
    """Uniform without replacement within an epoch, reshuffled each epoch."""
    rng = random.Random(SEED)
    order: list[dict] = []
    while len(order) < budget:
        epoch = list(problems)
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:budget]


def releases() -> list[dict] | None:
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())["releases"]
    except Exception:
        return None


def serving_release() -> str | None:
    """The scenario's latest release id, so each record names the weights that served it."""
    rows = releases()
    return str(rows[-1]["release_id"]) if rows else None


STALL_S = int(os.environ.get("SAO_STALL_S", "2700"))
_started_at = time.time()
_credit = 0
_credit_lock = threading.Lock()


def trained_steps() -> tuple[int, float] | None:
    """(steps done, seconds since the progress file last changed); None when pacing is off."""
    if not PROGRESS_FILE:
        return None
    try:
        with open(PROGRESS_FILE) as handle:
            steps = int(handle.read().strip() or 0)
        return steps, time.time() - os.path.getmtime(PROGRESS_FILE)
    except (OSError, ValueError):
        return 0, time.time() - _started_at  # no file yet: pace from zero, count idleness from launch


def pace(started: int) -> None:
    """Hold new submissions while more than AHEAD batches are out beyond the trainer.

    Reef drops batches whose rollouts exceed max-staleness, and the driver
    cannot see that, so an idle trainer (no progress for STALL_S) releases one
    batch of credit; otherwise a burst of dropped batches would deadlock.
    """
    global _credit
    while True:
        progress = trained_steps()
        if progress is None:
            return
        steps, idle = progress
        with _credit_lock:
            if started < (steps + 1 + AHEAD) * BATCH + _credit:
                return
            if idle > STALL_S:
                _credit += BATCH
                print(f"pace: trainer idle {idle:.0f}s at step {steps}; releasing one more batch", flush=True)
                continue
        time.sleep(10)


def wait_for_training(expected: int) -> None:
    """Block until ``expected`` training releases exist, so the last step is not cut off."""
    deadline = time.time() + TRAIN_DRAIN_TIMEOUT_S
    trained = None
    while time.time() < deadline:
        rows = releases()
        trained = None if rows is None else sum(1 for r in rows if r.get("operation") == "training")
        if trained is not None and trained >= expected:
            print(f"trained: {trained}/{expected} steps committed", flush=True)
            return
        time.sleep(15)
    print(f"WARNING: only {trained}/{expected} steps trained within {TRAIN_DRAIN_TIMEOUT_S}s", flush=True)


def one_rollout(client: ReefClient, model: str, problem: dict, position: int, report: bool = True) -> dict:
    started = time.time()
    release = serving_release()
    response, receipt = client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": problem["problem"] + INSTRUCTION_SUFFIX}],
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    )
    completion = response["choices"][0]["message"]["content"]
    predicted = extract_answer(completion)
    score = 1.0 if answers_equal(str(problem["gold"]), predicted) else 0.0
    if report:
        client.report(SCENARIO, {"score": score, "references": [receipt]}, recipe=RECIPE)
    record = {
        "position": position,
        "problem_idx": problem["problem_idx"],
        "score": score,
        "predicted": predicted,
        "gold": str(problem["gold"]),
        "serving_release_id": release,
        "agent_record_id": receipt,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "finish_reason": response["choices"][0].get("finish_reason"),
        "seconds": round(time.time() - started, 1),
        "recorded_at": time.time(),
    }
    with _write_lock:
        RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RECORDS_PATH, "a") as out:
            out.write(json.dumps(record) + "\n")
    print(
        f"[{RECIPE} {position}] idx={problem['problem_idx']} score={score:.0f} "
        f"tokens={record['completion_tokens']} release={release} predicted={predicted!r}",
        flush=True,
    )
    return record


def main() -> None:
    problems = load_problems()
    order = problem_order(problems, BUDGET)
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=7200)
    model = os.environ.get("SAO_MODEL_NAME", "reef")  # the name run.py sends; Reef's SGLang serves it
    print(
        f"pool={len(problems)} problems, budget={BUDGET}, in_flight={IN_FLIGHT}, max_tokens={MAX_TOKENS}", flush=True
    )

    done = 0
    failures = 0
    streak = 0

    def settle(futures) -> list[dict]:
        """Count finished rollouts; a failure is retried later rather than spent from the budget."""
        nonlocal done, failures, streak
        retry = []
        for future in futures:
            try:
                future.result()
            except ReefClientError as error:
                failures += 1
                streak += 1
                retry.append(future.problem)
                print(f"rollout failed: {error}", flush=True)
            except Exception as error:  # a single failed rollout must not end the run
                failures += 1
                streak += 1
                retry.append(future.problem)
                print(f"rollout failed: {type(error).__name__}: {error}", flush=True)
            else:
                done += 1
                streak = 0
        if streak >= MAX_FAILURE_STREAK:
            raise SystemExit(f"{streak} rollouts failed in a row; the serving stack is down")
        if retry:
            time.sleep(FAILURE_PAUSE_S)  # give the engine time to come back before re-sending
        return retry

    def submit(pool, problem, position):
        future = pool.submit(one_rollout, client, model, problem, position)
        future.problem = problem
        return future

    if GROUP > 1:
        # GRPO control: GROUP rollouts of one prompt form a group; IN_FLIGHT // GROUP
        # groups run at once. A group's reports are posted together, in one
        # go, so Slime's group-relative baseline sees complete groups in order.
        report_lock = threading.Lock()

        def run_group(problem: dict, first_position: int) -> int:
            nonlocal done, failures
            pace(done)
            with ThreadPoolExecutor(max_workers=GROUP) as members:
                records: list[dict] = []
                attempts = 0
                while len(records) < GROUP:
                    need = GROUP - len(records)
                    if attempts >= MAX_FAILURE_STREAK:
                        raise SystemExit("a group could not be completed; the serving stack is down")
                    futures = [
                        members.submit(one_rollout, client, model, problem, first_position + len(records) + k, False)
                        for k in range(need)
                    ]
                    for future in wait(futures).done:
                        try:
                            records.append(future.result())
                        except Exception as error:
                            failures += 1
                            attempts += 1
                            print(f"rollout failed: {type(error).__name__}: {error}", flush=True)
                    if len(records) < GROUP:
                        time.sleep(FAILURE_PAUSE_S)
            with report_lock:
                for record in records:
                    client.report(
                        SCENARIO, {"score": record["score"], "references": [record["agent_record_id"]]}, recipe=RECIPE
                    )
            done += GROUP
            return GROUP

        groups_in_flight = max(1, IN_FLIGHT // GROUP)
        with ThreadPoolExecutor(max_workers=groups_in_flight) as pool:
            futures = [
                pool.submit(run_group, problem, step * GROUP) for step, problem in enumerate(order[: BUDGET // GROUP])
            ]
            for future in futures:
                future.result()
    else:
        with ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
            pending = set()
            queue = list(order)
            position = 0
            while done < BUDGET:
                while len(pending) < IN_FLIGHT and queue:
                    pace(done)  # completed rollouts, so the in-flight set stays full while the trainer catches up
                    pending.add(submit(pool, queue.pop(0), position))
                    position += 1
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                queue = settle(finished) + queue  # failed prompts go back to the front
    print(f"finished: {done} rollouts, {failures} failures", flush=True)
    wait_for_training(done // BATCH)


if __name__ == "__main__":
    main()
