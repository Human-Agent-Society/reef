"""One stage of the skill stream, run inside the task's container: train the served model on this task.

For each step of ``PROMPTS_PER_STEP`` training prompts, in the order
``skills.wave_schedule`` fixes from the seed:

    ask     — one chat completion per prompt through Reef at temperature 1.0:
              the student's on-policy sample, recorded with its tokens and
              log-probs
    report  — the dataset's demonstration as the report's ``teacher_context`` against
              that sample's receipt
    learn   — Reef's recipe batches the step's reports, runs one optimizer
              step and publishes the weights; the loop blocks on that
              release, so the next step samples from the updated policy

Every ``EVAL_EVERY`` steps, and before the first, the stage submits its step
number to the task's judge, which scores the served model on both skills'
test splits and keeps the score log reef-eval turns into the learning curve.
The final submission, after the last step, is the stage's result.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import skills

TASK = json.loads((Path(__file__).with_name("task.json")).read_text())["task"]
EPOCHS = int(os.environ.get("SKILLS_EPOCHS", "2"))
PROMPTS_PER_STEP = int(os.environ.get("SKILLS_PROMPTS_PER_STEP", "32"))  # MUST equal serve.yaml's batch size
SEED = int(os.environ.get("SKILLS_SEED", "42"))
MAX_TOKENS = int(os.environ.get("SKILLS_MAX_TOKENS", "2048"))
EVAL_EVERY = int(os.environ.get("SKILLS_EVAL_EVERY", "10"))
#: A ceiling on the steps to run (0 runs the whole schedule); a smoke run sets a few.
STEPS = int(os.environ.get("SKILLS_STEPS", "0"))
TRAIN_TIMEOUT_S = float(os.environ.get("SKILLS_TRAIN_TIMEOUT_S", "3600"))
JUDGE_URL = os.environ["JUDGE_URL"].rstrip("/")


def submit(step: int) -> dict:
    """Ask the judge to score the served model now; the judge records the scores."""
    body = json.dumps({"task": TASK, "step": step}).encode()
    request = urllib.request.Request(f"{JUDGE_URL}/submit", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def main() -> None:
    examples = skills.training_examples(TASK)
    schedule = skills.wave_schedule(len(examples), EPOCHS, PROMPTS_PER_STEP, SEED)
    if STEPS:
        schedule = schedule[:STEPS]
    client = skills.make_client()
    print(
        f"[{TASK}] {len(examples)} training prompts, {len(schedule)} steps of {PROMPTS_PER_STEP} "
        f"({EPOCHS} epochs, seed {SEED}); service {skills.SERVICE_URL}, scenario {skills.SCENARIO}",
        flush=True,
    )
    releases_before = skills.wait_for_training(0, TRAIN_TIMEOUT_S)
    print(f"[{TASK} step 0] judge: {submit(0)}", flush=True)
    for step, prompt_indices in enumerate(schedule, start=1):
        started = time.time()
        wave = [examples[index] for index in prompt_indices]
        outputs = skills.ask_all(
            client,
            [messages for messages, _ in wave],
            temperature=1.0,
            max_tokens=MAX_TOKENS,
            concurrency=PROMPTS_PER_STEP,
        )
        sampled = time.time()
        for (_, demonstration), (_, receipt, _) in zip(wave, outputs, strict=True):
            client.report(skills.SCENARIO, {"references": [receipt], "metadata": {"teacher_context": demonstration}})
        releases = skills.wait_for_training(releases_before + step, TRAIN_TIMEOUT_S)
        finished = time.time()
        print(
            json.dumps(
                {
                    "task": TASK,
                    "step": step,
                    "steps": len(schedule),
                    "training_releases": releases,
                    "mean_completion_tokens": sum(tokens for _, _, tokens in outputs) / len(outputs),
                    "truncated": sum(1 for _, _, tokens in outputs if tokens >= MAX_TOKENS),
                    "sample_s": round(sampled - started, 1),
                    "train_s": round(finished - sampled, 1),
                }
            ),
            flush=True,
        )
        if step == len(schedule) or (EVAL_EVERY and step % EVAL_EVERY == 0):
            print(f"[{TASK} step {step}] judge: {submit(step)}", flush=True)


if __name__ == "__main__":
    main()
