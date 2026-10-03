"""Run sequential OPD batches and held-out AIME'24 evaluation through Reef."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def write_json(path: Path, value: Any) -> None:
    """Publish complete metadata so an interrupted write cannot replace good state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_id(scenario: str, receipt: str) -> str:
    """Keep a report retry identical even when its first HTTP response was lost."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"reef:opd:{scenario}:{receipt}").hex


class Campaign:
    """One scenario with synchronous version boundaries and separate evaluation records."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        stack = yaml.safe_load(args.config.read_text())
        if args.steps and stack["recipe"]["config"]["batch-size"] != args.prompts_per_step * args.samples_per_prompt:
            raise ValueError("The driver's prompts times samples per step must equal the recipe's batch-size")
        if int(stack["inference"]["options"]["context-length"]) <= args.eval_tokens:
            raise ValueError("Inference context must fit evaluation tokens plus the prompt")
        self.client = ReefClient(args.url, token=os.environ.get("REEF_TOKEN"), timeout_s=args.timeout)
        self.output = args.output
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        settings.pop("resume", None)
        inputs = {name: file_digest(Path(settings[name])) for name in ("config", "train_data", "eval_data")}
        inputs["driver"] = file_digest(Path(__file__))
        if args.resume:
            previous = json.loads((self.output / "config.json").read_text())
            # Parallelism and network timeouts do not change the sampling protocol.
            for key in ("concurrency", "timeout"):
                previous.pop(key, None)
                settings.pop(key, None)
            if previous != settings or json.loads((self.output / "inputs.json").read_text()) != inputs:
                raise ValueError("Resume requires the same protocol, input files, configuration and driver")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_json(self.output / "config.json", settings)
            write_json(self.output / "inputs.json", inputs)

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

    def history(self) -> list[dict[str, Any]]:
        """Require an uninterrupted creation/training history, oldest first."""
        rows = list(reversed(self.releases()))
        if rows and (
            rows[0].get("operation") != "creation"
            or any(row.get("operation") != "training" for row in rows[1:])
            or any(row.get("pending") for row in rows)
            or [row.get("current", False) for row in rows] != [False] * (len(rows) - 1) + [True]
            or len({row["release_id"] for row in rows}) != len(rows)
        ):
            raise RuntimeError("Campaign requires a single current release and an unchanged training history")
        return rows

    def collect(
        self,
        work: list[tuple[Question, int, bool]],
        filename: str,
        *,
        release_id: str | None = None,
        allow_new: bool = True,
    ) -> list[dict[str, Any]]:
        expected = {(question.id, seed): evaluation for question, seed, evaluation in work}
        if len(expected) != len(work) or not work:
            raise ValueError("Each batch needs unique, nonempty question/seed pairs")
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        path = self.output / filename
        if path.exists():
            with path.open("r+b") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # A killed append may leave only the last record incomplete.
                        handle.truncate(offset)
                        break
                    row = json.loads(line)
                    key = (row["question_id"], row["seed"])
                    if key in rows or key not in expected or row.get("evaluation") is not expected[key]:
                        raise ValueError(f"Unexpected or duplicate sample in {path}: {key}")
                    if not row.get("receipt") or not row.get("release_id"):
                        raise ValueError(f"Sample in {path} lacks a receipt or release")
                    rows[key] = row
        versions = {row["release_id"] for row in rows.values()}
        if len(versions) > 1 or (release_id is not None and versions - {release_id}):
            raise RuntimeError(f"Saved batch does not match expected release {release_id}: {versions}")
        if release_id is None and versions:
            release_id = next(iter(versions))
        missing = [item for item in work if (item[0].id, item[1]) not in rows]
        if missing and not allow_new:
            raise RuntimeError(f"Cannot regenerate missing samples from a historical release: {filename}")
        failure: Exception | None = None
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool, path.open("ab") as handle:
            futures = [pool.submit(self.sample, item) for item in missing]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    # Drain other completed requests before surfacing this failure.
                    failure = failure or error
                    continue
                if release_id is None:
                    release_id = row["release_id"]
                if row["release_id"] != release_id:
                    raise RuntimeError("Batch crossed a weight release boundary")
                handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
                rows[(row["question_id"], row["seed"])] = row
        if failure is not None:
            raise failure
        # Reports retain dataset order even though completed requests are saved immediately.
        return [rows[(question.id, seed)] for question, seed, _ in work]

    def evaluate(
        self,
        questions: list[Question],
        step: int,
        *,
        release_id: str | None = None,
        allow_new: bool = True,
    ) -> None:
        work = [(q, self.args.seed + repeat, True) for q in questions for repeat in range(self.args.eval_repeats)]
        started = time.monotonic()
        rows = self.collect(work, f"eval-{step:04d}.jsonl", release_id=release_id, allow_new=allow_new)
        metric = {
            "step": step,
            "release_id": rows[0]["release_id"],
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "samples": len(rows),
            "questions": len(questions),
            "elapsed_s": time.monotonic() - started,
            "elapsed_scope": "completing_attempt",
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
        }
        metric_path = self.output / f"eval-{step:04d}.metrics.json"
        if metric_path.exists():
            saved = json.loads(metric_path.read_text())
            if any(
                saved[key] != metric[key]
                for key in ("step", "release_id", "accuracy", "samples", "questions", "truncated")
            ):
                raise ValueError(f"Saved metric does not match predictions at step {step}")
        else:
            write_json(metric_path, metric)
        metrics = [json.loads(path.read_text()) for path in sorted(self.output.glob("eval-*.metrics.json"))]
        temporary = self.output / "metrics.jsonl.tmp"
        with temporary.open("w") as handle:
            handle.writelines(json.dumps(value) + "\n" for value in metrics)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.output / "metrics.jsonl")
        print(json.dumps(metric), flush=True)

    def verify_update(self, step: int, rows: list[dict[str, Any]], history: list[dict[str, Any]]) -> None:
        result = self.client.get(
            f"/reef/scenarios/{quote(self.args.scenario, safe='')}/commits?after_step={step - 1}&limit=1"
        )
        commits = result["commits"]
        expected = {
            identifier
            for row in rows
            for identifier in (row["receipt"], report_id(self.args.scenario, row["receipt"]))
        }
        if len(commits) != 1 or (
            commits[0]["step"] != step
            or commits[0]["operation"] != "training"
            or commits[0].get("pending")
            or commits[0]["artifact_ref"]["release_id"] != history[step]["release_id"]
            or set(commits[0]["consumed_ids"]) != expected
        ):
            raise RuntimeError(f"Committed update {step} does not match this campaign's exact batch")
        write_json(self.output / f"commit-{step:04d}.json", commits[0])

    def run(self) -> None:
        with (self.output / ".driver.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another driver is already using this output directory") from error
            self.run_locked()

    def run_locked(self) -> None:
        history = self.history()
        if history and not self.args.resume:
            raise RuntimeError("Use a fresh scenario, or --resume with its original output directory")
        if len(history) > self.args.steps + 1:
            raise RuntimeError("Scenario has more updates than the declared campaign")
        if self.args.resume and not history and any(path.stat().st_size for path in self.output.glob("*-*.jsonl")):
            raise RuntimeError("Saved samples exist but their Reef scenario is missing")
        evaluation = load_questions(self.args.eval_data)
        training = load_questions(self.args.train_data) if self.args.steps else []
        if self.args.steps * self.args.prompts_per_step > len(training):
            raise ValueError("Not enough unique prompts for the requested schedule")
        self.evaluate(
            evaluation, 0, release_id=history[0]["release_id"] if history else None, allow_new=len(history) <= 1
        )
        for step in range(1, self.args.steps + 1):
            history = self.history()
            if len(history) < step:
                raise RuntimeError("Scenario lost a previously completed training release")
            batch = training[(step - 1) * self.args.prompts_per_step : step * self.args.prompts_per_step]
            work = [
                (q, self.args.seed + step * self.args.samples_per_prompt + sample, False)
                for q in batch
                for sample in range(self.args.samples_per_prompt)
            ]
            rows = self.collect(
                work,
                f"train-{step:04d}.jsonl",
                release_id=history[step - 1]["release_id"],
                allow_new=len(history) == step,
            )
            if len(history) == step:
                for row in rows:
                    self.client.report(
                        self.args.scenario,
                        {
                            "agent_record_id": report_id(self.args.scenario, row["receipt"]),
                            "metadata": {"teacher_context": ""},
                        },
                        references=[row["receipt"]],
                    )
                self.wait_for_release(step - 1)
                history = self.history()
                if len(history) != step + 1:
                    raise RuntimeError("Scenario advanced beyond this campaign's batch")
            self.verify_update(step, rows, history)
            release_path = self.output / f"releases-{step:04d}.json"
            if release_path.exists():
                saved = list(reversed(json.loads(release_path.read_text())))
                if [row["release_id"] for row in saved] != [row["release_id"] for row in history[: len(saved)]]:
                    raise RuntimeError("Saved publication history no longer matches the service")
            else:
                write_json(release_path, list(reversed(history)))
            if step % self.args.eval_every == 0 or step == self.args.steps:
                self.evaluate(
                    evaluation, step, release_id=history[step]["release_id"], allow_new=len(history) == step + 1
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:28982")
    parser.add_argument("--scenario", default="opd-math")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume this driver against the same live Reef scenario")
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
