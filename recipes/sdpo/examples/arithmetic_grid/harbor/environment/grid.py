"""One SDPO run, inside the task's container: train the served model on an arithmetic grid.

For each step of the run:

    sample  — every question of the grid, answered ``ROLLOUTS_PER_GROUP``
              times through Reef at temperature 1: the student's on-policy
              samples, recorded with their tokens and log-probs
    report  — every rollout against its receipt, carrying its coordinates in
              the grid, its score, and the format feedback a wrong answer
              earns
    learn   — the recipe holds the step until every coordinate has arrived,
              builds the whole grid as one batch, runs one optimizer step and
              publishes the weights; the loop blocks on that release, so the
              next step samples from the updated policy

The questions are arithmetic with a formatting instruction, so the protocol is
visible without a dataset: a wrong answer earns the feedback the teacher reads,
and a right one becomes its siblings' demonstration. The run writes its record
to ``RESULT_PATH``, where the verifier reads the last grid's accuracy.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reef_client import ReefClient

SERVICE_URL = os.environ.get("REEF_SERVICE_URL", "http://host.docker.internal:28902").rstrip("/")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")
SCENARIO = os.environ.get("REEF_SCENARIO", "sdpo-arithmetic")
MODEL = "reef"  # the model name the requests carry; Reef's SGLang serves it
#: The sampling grid. Both MUST equal serve.yaml's groups-per-step and rollouts-per-group.
GROUPS_PER_STEP = int(os.environ.get("SDPO_GROUPS_PER_STEP", "2"))
ROLLOUTS_PER_GROUP = int(os.environ.get("SDPO_ROLLOUTS_PER_GROUP", "2"))
STEPS = int(os.environ.get("SDPO_STEPS", "2"))
SEED = int(os.environ.get("SDPO_SEED", "42"))
MAX_TOKENS = int(os.environ.get("SDPO_MAX_TOKENS", "128"))
TRAIN_TIMEOUT_S = float(os.environ.get("SDPO_TRAIN_TIMEOUT_S", "1800"))
RESULT_PATH = Path(os.environ.get("SDPO_RESULT_PATH", "/workspace/grid.json"))
#: What the teacher reads for a rollout whose question no sibling solved.
FORMAT_FEEDBACK = "The required answer format is a single integer, without explanation."
#: The service is gone or rejecting requests; waiting cannot help.
SERVICE_GONE = -1


def questions(step: int) -> list[tuple[str, str]]:
    """This step's questions and their answers, drawn from the seed so a run is reproducible."""
    generator = random.Random(f"{SEED}:{step}")
    drawn = []
    for _ in range(GROUPS_PER_STEP):
        left, right = generator.randint(2, 49), generator.randint(2, 49)
        drawn.append((f"What is {left} + {right}? Reply with only the integer.", str(left + right)))
    return drawn


def ask(client: ReefClient, question: str, answer: str) -> tuple[str, str, float]:
    """One attempt at one question: its receipt, its text, and whether it is correct."""
    response, receipt = client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": question}],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    text = response["choices"][0]["message"]["content"]
    return receipt, text, float(text.strip() == answer)


def sample_grid(client: ReefClient, step: int) -> list[dict[str, Any]]:
    """Every coordinate of one grid, sampled from the release the service serves now."""
    drawn = questions(step)
    coordinates = [(group, rollout) for group in range(GROUPS_PER_STEP) for rollout in range(ROLLOUTS_PER_GROUP)]

    def attempt(coordinate: tuple[int, int]) -> dict[str, Any]:
        group, rollout = coordinate
        question, answer = drawn[group]
        receipt, text, score = ask(client, question, answer)
        return {"group": group, "rollout": rollout, "receipt": receipt, "answer": text, "score": score}

    with ThreadPoolExecutor(max_workers=len(coordinates)) as pool:
        return list(pool.map(attempt, coordinates))


def report_grid(client: ReefClient, step: int, rollouts: list[dict[str, Any]]) -> None:
    """Report the grid: each rollout at its coordinates, with the feedback a failed attempt earns."""
    for sample in rollouts:
        client.report(
            SCENARIO,
            {
                "references": [sample["receipt"]],
                "score": sample["score"],
                "metadata": {
                    "step": step,
                    "group": sample["group"],
                    "rollout": sample["rollout"],
                    "teacher_context": "" if sample["score"] else FORMAT_FEEDBACK,
                },
            },
        )


def training_release_count() -> int | None:
    """Training releases committed so far; ``None`` while the service is busy, 0 before the scenario exists."""
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 0  # the scenario does not exist yet: the first request creates it
        return SERVICE_GONE  # answered and rejected: not our deployment
    except urllib.error.URLError as error:
        if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
            return SERVICE_GONE
        return None  # stalled behind a train step; try again
    except TimeoutError:
        return None
    return sum(1 for row in payload["releases"] if row.get("operation") == "training")


def wait_for_training(expected: int, timeout_s: float) -> int:
    """Block until the scenario has committed ``expected`` training releases; return the count seen."""
    deadline = time.time() + timeout_s
    while True:
        count = training_release_count()
        if count == SERVICE_GONE:
            raise RuntimeError(f"the Reef service at {SERVICE_URL} is gone or rejects scenario {SCENARIO}")
        if count is not None and count >= expected:
            return count
        if time.time() > deadline:
            raise TimeoutError(f"training release {expected} did not commit within {timeout_s:.0f}s (seen: {count})")
        time.sleep(2.0)


def main() -> None:
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=TRAIN_TIMEOUT_S)
    print(
        f"[sdpo] {STEPS} steps of {GROUPS_PER_STEP} questions x {ROLLOUTS_PER_GROUP} attempts, seed {SEED}; "
        f"service {SERVICE_URL}, scenario {SCENARIO}",
        flush=True,
    )
    releases_before = wait_for_training(0, TRAIN_TIMEOUT_S)
    records = []
    for step in range(STEPS):
        started = time.time()
        rollouts = sample_grid(client, step)
        sampled = time.time()
        report_grid(client, step, rollouts)
        releases = wait_for_training(releases_before + step + 1, TRAIN_TIMEOUT_S)
        accuracy = sum(sample["score"] for sample in rollouts) / len(rollouts)
        records.append(
            {
                "step": step,
                "accuracy": accuracy,
                "training_releases": releases,
                "sample_s": round(sampled - started, 1),
                "train_s": round(time.time() - sampled, 1),
                "rollouts": rollouts,
            }
        )
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(
            json.dumps(
                {
                    "grid": {"groups_per_step": GROUPS_PER_STEP, "rollouts_per_group": ROLLOUTS_PER_GROUP},
                    "seed": SEED,
                    "accuracy": accuracy,
                    "steps": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({key: value for key, value in records[-1].items() if key != "rollouts"}), flush=True)


if __name__ == "__main__":
    main()
