"""Exercise two complete SDPO updates against an already-running Reef service.

This synthetic arithmetic task is an infrastructure smoke test, not a paper
benchmark. It uses environmental formatting feedback and records each release.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reef_client import ReefClient


class SmokeRun:
    def __init__(self, url: str, token: str, scenario: str, output: Path) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.scenario = scenario
        self.output = output
        self.client = ReefClient(self.url, token=token, timeout_s=1200)

    def releases(self) -> list[dict[str, Any]]:
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

    def ask(self, item: tuple[int, int, str, str]) -> tuple[int, int, str, str, float]:
        group, rollout, question, answer = item
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
        return group, rollout, receipt, text, float(text.strip() == answer)

    def run(self) -> None:
        if self.releases():
            raise ValueError("use a fresh scenario for this two-update smoke test")
        rows = []
        questions = [
            ("What is 2 + 2? Reply with only the integer.", "4"),
            ("What is 3 + 5? Reply with only the integer.", "8"),
        ]
        for step in range(2):
            start = time.monotonic()
            grid = [
                (group, rollout, question, answer)
                for group, (question, answer) in enumerate(questions)
                for rollout in range(2)
            ]
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(self.ask, grid))
            sampled = time.monotonic()
            for group, rollout, receipt, _, score in outputs:
                self.client.report(
                    self.scenario,
                    {
                        "references": [receipt],
                        "score": score,
                        "metadata": {
                            "step": step,
                            "group": group,
                            "rollout": rollout,
                            "teacher_context": "The required answer format is a single integer, without explanation.",
                        },
                    },
                )
            deadline = time.monotonic() + 1200
            while True:
                releases = self.releases()
                if len(releases) >= step + 1:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"training release {step + 1} did not arrive")
                time.sleep(2)
            row = {
                "step": step,
                "sampling_seconds": sampled - start,
                "training_seconds": time.monotonic() - sampled,
                "releases": releases,
                "samples": outputs,
            }
            rows.append(row)
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps({"kind": "synthetic_smoke", "steps": rows}, indent=2) + "\n")
            print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28902")
    parser.add_argument("--token", default="reef-local-smoke")
    parser.add_argument("--scenario", default="sdpo-smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SmokeRun(args.url, args.token, args.scenario, args.output).run()


if __name__ == "__main__":
    main()
