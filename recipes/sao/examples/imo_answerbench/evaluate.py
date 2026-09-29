"""Evaluate a served model on IMOAnswerBench problems with the training grader.

Runs ``--runs`` fresh samples per problem at the training sampling settings
(temperature 1.0, top-p 1.0) against an OpenAI-compatible chat endpoint,
extracts the last ``\\boxed{}`` and scores it with the strict rule from
``harness/grader.py``. One JSON line per sample goes to ``--out``; existing
(problem_idx, run) rows are skipped, so an interrupted evaluation resumes.

Typical use: serve the base weights and the final trained weights with the
same engine, evaluate both on the held-out split, and compare.

    python evaluate.py --problems work/imo_answerbench.jsonl \\
        --indices 0,1,7 --runs 8 --url http://127.0.0.1:30001/v1/chat/completions \\
        --model qwen3 --label sao-final --out work/eval-sao-final.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TextIO

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness.grader import answers_equal, extract_answer

INSTRUCTION_SUFFIX = "\n\nPut your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, help="JSONL with problem_idx, problem, gold")
    parser.add_argument("--indices", required=True, help="comma-separated problem_idx values to evaluate")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--url", required=True, help="OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--model", default="reef")
    parser.add_argument("--label", required=True, help="arm name written into every row")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=61440)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=7200)
    return parser.parse_args()


async def one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    problem: dict,
    run: int,
    out: TextIO,
    lock: asyncio.Lock,
) -> None:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": problem["problem"] + INSTRUCTION_SUFFIX}],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    async with semaphore:
        started = time.time()
        try:
            async with session.post(args.url, json=payload, timeout=aiohttp.ClientTimeout(total=args.timeout_s)) as r:
                body = await r.json()
        except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as error:
            # A failed or malformed request is recorded, not retried; a rerun skips only scored rows.
            async with lock:
                out.write(
                    json.dumps(
                        {
                            "arm": args.label,
                            "problem_idx": problem["problem_idx"],
                            "run": run,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    + "\n"
                )
                out.flush()
            return
    choice = body["choices"][0]
    predicted = extract_answer(choice["message"]["content"] or "")
    score = 1.0 if answers_equal(str(problem["gold"]), predicted) else 0.0
    row = {
        "arm": args.label,
        "problem_idx": problem["problem_idx"],
        "run": run,
        "score": score,
        "predicted": predicted,
        "gold": str(problem["gold"]),
        "completion_tokens": body.get("usage", {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "seconds": round(time.time() - started, 1),
    }
    async with lock:
        out.write(json.dumps(row) + "\n")
        out.flush()
    print(
        f"[{args.label} idx={problem['problem_idx']} run={run}] score={score:.0f} predicted={predicted!r}", flush=True
    )


async def main() -> None:
    args = parse_args()
    with open(args.problems) as handle:
        problems = {int(row["problem_idx"]): row for row in (json.loads(line) for line in handle if line.strip())}
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    out_path = Path(args.out)
    done: set[tuple[int, int]] = set()
    if out_path.exists():
        with open(out_path) as handle:
            for line in handle:
                row = json.loads(line)
                if "score" in row:
                    done.add((row["problem_idx"], row["run"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    with open(out_path, "a") as out:
        async with aiohttp.ClientSession() as session:
            tasks = [
                one(session, semaphore, args, problems[idx], run, out, lock)
                for idx in indices
                for run in range(args.runs)
                if (idx, run) not in done
            ]
            print(f"{len(tasks)} samples to run ({len(done)} already present)", flush=True)
            await asyncio.gather(*tasks)
    with open(out_path) as handle:
        rows = [row for row in (json.loads(line) for line in handle if line.strip()) if "score" in row]
    total = sum(row["score"] for row in rows)
    print(f"{args.label}: {int(total)}/{len(rows)} = {total / max(1, len(rows)):.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
