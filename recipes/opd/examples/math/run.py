"""Run sequential OPD batches and held-out AIME'24 evaluation through Reef."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from reef_client import ReefClient, ReefClientError


@dataclass(frozen=True)
class Question:
    id: str
    prompt: str
    answer: str = ""


def load_questions(path: Path) -> list[Question]:
    with path.open() as handle:
        return [Question(**json.loads(line)) for line in handle if line.strip()]


def boxed_answer(text: str) -> int | None:
    """Extract the final boxed integer; no answer guessed from intermediate work."""
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    remaining = text[start + len(r"\boxed{") :]
    match = re.match(r"\s*(?:\\text\{)?\s*([0-9]{1,3})\s*\}?\s*\}", remaining)
    return int(match.group(1)) if match else None


class Campaign:
    """One scenario with synchronous version boundaries and separate evaluation records."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        stack = yaml.safe_load(args.config.read_text())
        expected = args.prompts_per_step * args.samples_per_prompt
        if args.steps and (
            stack["recipe"]["config"]["batch-size"] != expected
            or stack["training"]["config"]["global_batch_size"] != expected
        ):
            raise ValueError("Driver batch must match both recipe and trainer in the deployed configuration")
        if int(stack["inference"]["options"]["context-length"]) <= args.eval_tokens:
            raise ValueError("Inference context must fit evaluation tokens plus the prompt")
        self.client = ReefClient(args.url, token=os.environ.get("REEF_TOKEN"), timeout_s=args.timeout)
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "config.json").write_text(
            json.dumps(
                {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2
            )
            + "\n"
        )

    def releases(self) -> list[dict[str, Any]]:
        try:
            result = self.client.get(f"/reef/scenarios/{quote(self.args.scenario, safe='')}/releases")
        except ReefClientError as error:
            if error.status == 404:
                return []
            raise
        return result["releases"]

    def wait_for_release(self, before: int) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            rows = self.releases()
            count = sum(row.get("operation") == "training" for row in rows)
            if count == before + 1:
                return rows
            if count > before + 1:
                raise RuntimeError("More than one training update occurred for this batch")
            status = self.client.get("/reef/status")
            if status.get("error"):
                raise RuntimeError(f"Training failed: {status['error']}")
            time.sleep(2)
        raise TimeoutError(f"No publication after training step {before}")

    def sample(self, work: tuple[Question, int, bool]) -> dict[str, Any]:
        question, seed, evaluation = work
        prompt = question.prompt
        if evaluation:
            prompt += "\nPlease reason step by step, and put your final answer within \\boxed{}."
        response, headers = self.client.post(
            "/v1/chat/completions",
            self.args.scenario,
            {
                "model": self.args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": seed,
                "max_tokens": self.args.eval_tokens if evaluation else self.args.train_tokens,
            },
        )
        receipt = headers.get("x-reef-agent-record-id")
        if not receipt:
            raise RuntimeError("Inference did not return a Reef receipt")
        # Weight releases belong to the persisted inference receipt; the
        # x-reef-release-id response header describes file/harness releases.
        record = self.client.get(
            f"/reef/scenarios/{quote(self.args.scenario, safe='')}/records/{quote(receipt, safe='')}"
        )
        artifact = record.get("artifact_ref") or {}
        release_id = artifact.get("release_id")
        if record.get("agent_record_id") != receipt or not release_id:
            raise RuntimeError("Inference receipt has no identifiable weight release")
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        return {
            "question_id": question.id,
            "seed": seed,
            "receipt": receipt,
            "release_id": release_id,
            "evaluation": evaluation,
            "response": content,
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage", {}),
            "answer": question.answer if evaluation else None,
            "correct": boxed_answer(content) == int(question.answer) if evaluation else None,
        }

    def collect(self, work: list[tuple[Question, int, bool]], filename: str) -> list[dict[str, Any]]:
        rows = []
        with (
            ThreadPoolExecutor(max_workers=self.args.concurrency) as pool,
            (self.output / filename).open("x") as handle,
        ):
            for row in pool.map(self.sample, work):
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        versions = {row["release_id"] for row in rows}
        if len(versions) != 1 or None in versions:
            raise RuntimeError(f"Batch did not use exactly one identifiable weight release: {versions}")
        return rows

    def evaluate(self, questions: list[Question], step: int) -> None:
        work = [(q, self.args.seed + repeat, True) for q in questions for repeat in range(self.args.eval_repeats)]
        started = time.monotonic()
        rows = self.collect(work, f"eval-{step:04d}.jsonl")
        metric = {
            "step": step,
            "release_id": rows[0]["release_id"],
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "samples": len(rows),
            "questions": len(questions),
            "elapsed_s": time.monotonic() - started,
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
        }
        with (self.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)

    def run(self) -> None:
        if self.releases():
            raise RuntimeError("Use a fresh scenario; this driver does not silently resume a previous campaign")
        evaluation = load_questions(self.args.eval_data)
        self.evaluate(evaluation, 0)
        if self.args.steps == 0:
            return
        training = load_questions(self.args.train_data)
        if self.args.steps * self.args.prompts_per_step > len(training):
            raise ValueError("Not enough unique prompts for the requested schedule")
        for step in range(1, self.args.steps + 1):
            batch = training[(step - 1) * self.args.prompts_per_step : step * self.args.prompts_per_step]
            work = [
                (q, self.args.seed + step * self.args.samples_per_prompt + sample, False)
                for q in batch
                for sample in range(self.args.samples_per_prompt)
            ]
            rows = self.collect(work, f"train-{step:04d}.jsonl")
            before = sum(row.get("operation") == "training" for row in self.releases())
            for row in rows:
                self.client.report(
                    self.args.scenario, {"metadata": {"teacher_context": ""}}, references=[row["receipt"]]
                )
            releases = self.wait_for_release(before)
            (self.output / f"releases-{step:04d}.json").write_text(json.dumps(releases, indent=2) + "\n")
            if step % self.args.eval_every == 0 or step == self.args.steps:
                self.evaluate(evaluation, step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:28982")
    parser.add_argument("--scenario", default="opd-math")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=512)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--train-tokens", type=int, default=16384)
    parser.add_argument("--eval-tokens", type=int, default=64000)
    parser.add_argument("--eval-repeats", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if (
        args.steps < 0
        or min(
            args.prompts_per_step,
            args.samples_per_prompt,
            args.train_tokens,
            args.eval_tokens,
            args.eval_repeats,
            args.eval_every,
            args.concurrency,
            args.timeout,
        )
        <= 0
    ):
        parser.error("Steps must be nonnegative and batch sizes, budgets and intervals must be positive")
    Campaign(args).run()


if __name__ == "__main__":
    main()
