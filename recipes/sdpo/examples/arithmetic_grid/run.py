"""SDPO on a synthetic arithmetic grid: the sampling protocol in its smallest complete form.

Each step samples every question of the grid ``--rollouts-per-group`` times
from one policy release, reports each rollout with its coordinates and its
score, and waits for the training release before sampling the next step. The
questions are arithmetic with a formatting instruction, so a wrong answer
earns the environment feedback the teacher reads and a right one becomes its
siblings' demonstration.

The grid here is two questions by two attempts, far below the paper's 32 by 8.
This example shows the protocol and the training cycle; it is not a benchmark
result. The SciKnowEval reproduction is Reef issue #428.

Run it through ``run.sh``, which starts the service from ``serve.yaml`` first.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef_client import ReefClient

#: Each question is one group of the grid; every attempt answers it independently.
QUESTIONS = (
    ("What is 2 + 2? Reply with only the integer.", "4"),
    ("What is 3 + 5? Reply with only the integer.", "8"),
)
FORMAT_FEEDBACK = "The required answer format is a single integer, without explanation."


@dataclass(frozen=True)
class Rollout:
    """One attempt at one question: where it sits in the grid, what it answered, and what it scored."""

    group: int
    rollout: int
    receipt: str
    answer: str
    score: float


class GridCampaign:
    """Sample complete SDPO grids against a running Reef service and report each rollout.

    A step trains only once every coordinate of its grid has arrived, so the
    campaign reports the whole grid before waiting for that step's release.
    """

    def __init__(self, url: str, token: str, scenario: str, rollouts_per_group: int, timeout_s: float) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.scenario = scenario
        self.rollouts_per_group = rollouts_per_group
        self.timeout_s = timeout_s
        self.client = ReefClient(self.url, token=token, timeout_s=timeout_s)

    def training_releases(self) -> list[dict[str, Any]]:
        """The scenario's training releases, oldest first; empty before its first one."""
        request = urllib.request.Request(
            f"{self.url}/reef/scenarios/{self.scenario}/releases",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.load(response)["releases"]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            raise
        return [row for row in rows if row.get("operation") == "training"]

    def ask(self, coordinates: tuple[int, int]) -> Rollout:
        """One attempt at one question, recorded so its receipt can carry a report."""
        group, rollout = coordinates
        question, answer = QUESTIONS[group]
        response, receipt = self.client.inference_with_record(
            self.scenario,
            "/v1/chat/completions",
            {
                "model": "reef",
                "messages": [{"role": "user", "content": question}],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        text = response["choices"][0]["message"]["content"]
        return Rollout(group, rollout, receipt, text, float(text.strip() == answer))

    def sample_step(self, step: int) -> list[Rollout]:
        """Every coordinate of one grid, sampled from the release the service serves now."""
        grid = [(group, rollout) for group in range(len(QUESTIONS)) for rollout in range(self.rollouts_per_group)]
        with ThreadPoolExecutor(max_workers=len(grid)) as pool:
            return list(pool.map(self.ask, grid))

    def report_step(self, step: int, rollouts: list[Rollout]) -> None:
        """Report the grid, each rollout at its coordinates, with the feedback a failed attempt earns."""
        for sample in rollouts:
            self.client.report(
                self.scenario,
                {
                    "references": [sample.receipt],
                    "score": sample.score,
                    "metadata": {
                        "step": step,
                        "group": sample.group,
                        "rollout": sample.rollout,
                        "teacher_context": "" if sample.score else FORMAT_FEEDBACK,
                    },
                },
            )

    def wait_for_release(self, expected: int) -> list[dict[str, Any]]:
        """Block until the scenario has published ``expected`` training releases."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            releases = self.training_releases()
            if len(releases) >= expected:
                return releases
            if time.monotonic() >= deadline:
                raise TimeoutError(f"training release {expected} did not arrive within {self.timeout_s:.0f}s")
            time.sleep(2)

    def run(self, steps: int, output: Path) -> None:
        if self.training_releases():
            raise ValueError("this example needs a scenario that has not trained yet; use a fresh run directory")
        records = []
        for step in range(steps):
            started = time.monotonic()
            rollouts = self.sample_step(step)
            sampled = time.monotonic()
            self.report_step(step, rollouts)
            releases = self.wait_for_release(step + 1)
            records.append(
                {
                    "step": step,
                    "sampling_seconds": sampled - started,
                    "training_seconds": time.monotonic() - sampled,
                    "solved": sum(sample.score for sample in rollouts),
                    "rollouts": [vars(sample) for sample in rollouts],
                    "releases": releases,
                }
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps({"grid": [len(QUESTIONS), self.rollouts_per_group], "steps": records}, indent=2) + "\n"
            )
            print(json.dumps(records[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28902")
    parser.add_argument("--token", required=True, help="the service token, as serve.yaml reads it from REEF_TOKEN")
    parser.add_argument("--scenario", default="sdpo-arithmetic")
    parser.add_argument("--steps", type=int, default=2, help="sampling grids to train on, one optimizer step each")
    parser.add_argument(
        "--rollouts-per-group", type=int, default=2, help="attempts per question; MUST equal serve.yaml's value"
    )
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = GridCampaign(args.url, args.token, args.scenario, args.rollouts_per_group, args.timeout_s)
    campaign.run(args.steps, args.output)


if __name__ == "__main__":
    main()
