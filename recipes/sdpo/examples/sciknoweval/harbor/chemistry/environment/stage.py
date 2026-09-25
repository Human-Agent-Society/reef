"""SDPO on SciKnowEval Chemistry, run inside the task's container: train the served model through Reef.

For each step, over the question order ``chemistry.question_schedule`` fixes
from the seed:

    sample  — ``GROUPS_PER_STEP`` questions, each answered
              ``ROLLOUTS_PER_GROUP`` times through Reef at temperature 1: the
              student's on-policy grid, recorded with its tokens and log-probs
    report  — every rollout against its receipt, carrying its coordinates in
              the grid and the reference's binary score
    learn   — the recipe holds the step until the grid is complete, builds it
              as one batch (each rollout's teacher rereads the question with a
              successful sibling's response) and publishes the weights; the
              loop blocks on that release, so the next grid is on-policy

Every ``EVAL_EVERY`` steps, and before the first, the stage evaluates avg@16 on
the 210 test questions and appends it to ``RESULT_PATH``: that series is the
learning curve. The evaluation samples are not recorded, so they never become
training data.

Section 3 of the paper disables environment feedback, so a report carries no
teacher context: the teacher's privileged information is a successful sibling's
response alone.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import chemistry

GROUPS_PER_STEP = int(os.environ.get("SDPO_GROUPS_PER_STEP", "32"))  # MUST equal serve.yaml's groups-per-step
ROLLOUTS_PER_GROUP = int(os.environ.get("SDPO_ROLLOUTS_PER_GROUP", "8"))  # MUST equal serve.yaml's rollouts-per-group
EPOCHS = int(os.environ.get("SDPO_EPOCHS", "30"))  # the reference's total_epochs ceiling
SEED = int(os.environ.get("SDPO_SEED", "42"))
EVAL_EVERY = int(os.environ.get("SDPO_EVAL_EVERY", "5"))  # the reference's test_freq
EVAL_SAMPLES = int(os.environ.get("SDPO_EVAL_SAMPLES", "16"))  # avg@16, the paper's metric
EVAL_CONCURRENCY = int(os.environ.get("SDPO_EVAL_CONCURRENCY", "32"))
#: A ceiling on the steps to run (0 runs the whole schedule).
STEPS = int(os.environ.get("SDPO_STEPS", "0"))
#: Stop after the first evaluation past this much pure training time (0 disables the budget).
TRAINING_HOURS = float(os.environ.get("SDPO_TRAINING_HOURS", "0"))
TRAIN_TIMEOUT_S = float(os.environ.get("SDPO_TRAIN_TIMEOUT_S", "7200"))
RESULT_PATH = Path(os.environ.get("SDPO_RESULT_PATH", "/workspace/chemistry.json"))


def write_record(record: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    train = chemistry.load_split("train")
    test = chemistry.load_split("test")
    schedule = chemistry.question_schedule(len(train), EPOCHS, GROUPS_PER_STEP, SEED)
    if STEPS:
        schedule = schedule[:STEPS]
    client = chemistry.make_client(timeout_s=TRAIN_TIMEOUT_S)
    print(
        f"[chemistry] {len(train)} training and {len(test)} test questions; {len(schedule)} steps of "
        f"{GROUPS_PER_STEP}x{ROLLOUTS_PER_GROUP} (seed {SEED}); service {chemistry.SERVICE_URL}, "
        f"scenario {chemistry.SCENARIO}",
        flush=True,
    )
    releases_before = chemistry.wait_for_training(0, TRAIN_TIMEOUT_S)

    record: dict = {
        "task": "sciknoweval/chemistry",
        "grid": {"groups_per_step": GROUPS_PER_STEP, "rollouts_per_group": ROLLOUTS_PER_GROUP},
        "seed": SEED,
        "planned_steps": len(schedule),
        "evaluations": [],
        "steps": [],
    }
    initial = chemistry.evaluate(client, test, n=EVAL_SAMPLES, concurrency=EVAL_CONCURRENCY)
    record["evaluations"].append({"step": 0, "training_seconds": 0.0, **initial})
    write_record(record)
    print(f"[chemistry step 0] {json.dumps(initial)}", flush=True)

    training_seconds = 0.0
    for step, question_indices in enumerate(schedule, start=1):
        started = time.time()
        solved = 0
        for group, question_index in enumerate(question_indices):
            row = train[question_index]
            rollouts = chemistry.sample_group(client, row, ROLLOUTS_PER_GROUP)
            for rollout, (text, receipt) in enumerate(rollouts):
                score = float(chemistry.is_correct(text, row["answer"]))
                solved += score
                client.report(
                    chemistry.SCENARIO,
                    {
                        "references": [receipt],
                        "score": score,
                        # Section 3 disables environment feedback; a successful
                        # sibling is the teacher's only privileged information.
                        "metadata": {"step": step, "group": group, "rollout": rollout, "teacher_context": ""},
                    },
                )
        releases = chemistry.wait_for_training(releases_before + step, TRAIN_TIMEOUT_S)
        elapsed = time.time() - started
        training_seconds += elapsed
        record["steps"].append(
            {
                "step": step,
                "training_releases": releases,
                "train_accuracy": solved / (GROUPS_PER_STEP * ROLLOUTS_PER_GROUP),
                "step_seconds": round(elapsed, 1),
                "training_seconds": round(training_seconds, 1),
            }
        )
        print(json.dumps(record["steps"][-1]), flush=True)
        if step == len(schedule) or (EVAL_EVERY and step % EVAL_EVERY == 0):
            scores = chemistry.evaluate(client, test, n=EVAL_SAMPLES, concurrency=EVAL_CONCURRENCY)
            record["evaluations"].append({"step": step, "training_seconds": round(training_seconds, 1), **scores})
            print(f"[chemistry step {step}] {json.dumps(scores)}", flush=True)
            write_record(record)
            # The paper reports the best score within a pure-training budget;
            # the evaluation that crosses it is the run's last.
            if TRAINING_HOURS and training_seconds >= TRAINING_HOURS * 3600:
                record["stopped_at_budget_boundary"] = True
                break
        write_record(record)
    write_record(record)


if __name__ == "__main__":
    main()
