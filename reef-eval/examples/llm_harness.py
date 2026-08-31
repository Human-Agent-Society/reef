"""A harness that asks a model for candidates and learns from the judge.

The loop that autoresearch is about: propose a solution, submit it, read
the score the judge sends back, and let that score shape the next
proposal. `minimal_harness.py` shows the same integration with random
search, so the difference here is the model and the feedback.

The model is called from the host, and only the submission crosses into
the container. That keeps the task's network policy alone: the container
still reaches nothing but the judge, and your API key never enters it.

Any OpenAI-compatible endpoint works, so this runs against a provider, a
gateway, or a local server:

    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export OPENAI_API_KEY=...
    python examples/llm_harness.py --model deepseek/deepseek-v4-flash

Requires Docker and ``pip install reef-eval[harbor]``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from harbor.agents.base import BaseAgent

from reef_eval import Lab, metrics

TASK = str(
    Path(__file__).parent.parent
    / "tasks"
    / "autoresearch"
    / "first-party"
    / "circle-packing"
)

BRIEF = """Place 3 non-overlapping circles in the unit square, maximizing the
sum of their radii. Circles must lie fully inside the square and must not
overlap; touching is allowed.

Reply with JSON only, no prose, in exactly this shape:
{"circles": [[x, y, r], [x, y, r], [x, y, r]]}"""

# Uploaded into the container once, then run per proposal. The solution
# arrives in $SOLUTION so the command itself stays a fixed string.
SUBMITTER = """
import json, os, urllib.error, urllib.request

request = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=os.environ["SOLUTION"].encode()
)
try:
    print(urllib.request.urlopen(request, timeout=60).read().decode())
except urllib.error.HTTPError as error:
    print(json.dumps({"over_budget": error.code == 429}))
"""


def ask_model(base_url: str, api_key: str, model: str, messages: list[dict]) -> str:
    """One chat completion against an OpenAI-compatible endpoint."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {"model": model, "messages": messages, "max_tokens": 700}
        ).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"]


def parse_solution(reply: str) -> dict | None:
    """Take the first JSON object out of the reply, if there is one."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None


class LLMSearchHarness(BaseAgent):
    """Propose with a model, score with the judge, repeat."""

    def __init__(self, *args, rounds: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds = rounds

    @staticmethod
    def name() -> str:
        return "llm-search"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment) -> None:
        # Nothing to install: the model is called from the host. The only
        # thing the container needs is the submitter.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "submit.py").write_text(SUBMITTER)
            await environment.upload_dir(source_dir=tmp, target_dir="/opt/harness")

    async def _submit(self, environment, solution: dict) -> dict | None:
        """Submit from inside the container, where $JUDGE_URL resolves."""
        result = await environment.exec(
            command="python3 /opt/harness/submit.py",
            env={"SOLUTION": json.dumps(solution)},
        )
        try:
            return json.loads((result.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return None

    async def run(self, instruction, environment, context) -> None:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ["OPENAI_API_KEY"]
        model = (self.model_name or "gpt-5-mini").split("/", 1)[-1]

        messages = [{"role": "user", "content": BRIEF}]
        best = 0.0
        for round_number in range(self.rounds):
            reply = await asyncio.to_thread(
                ask_model, base_url, api_key, model, messages
            )
            solution = parse_solution(reply)
            if solution is None:
                messages.append({"role": "user", "content": "JSON only, please."})
                continue

            scored = await self._submit(environment, solution)
            if scored is None or scored.get("over_budget"):
                break

            score = float(scored["score"])
            best = max(best, float(scored.get("best", score)))
            print(f"round {round_number}: scored {score:.4f}, best {best:.4f}")
            if scored.get("remaining") == 0:
                break

            # The judge's reply is the whole feedback signal, so hand it
            # back verbatim and ask for something better than the best so far.
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        f"The judge scored that {score:.4f}"
                        f" ({scored.get('reason', 'no reason given')})."
                        f" The best so far is {best:.4f}."
                        " Propose a different arrangement that beats it."
                        " JSON only."
                    ),
                },
            ]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--lab", default="runs/llm-harness")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    lab = Lab(args.lab)
    row = await lab.run(
        TASK,
        agent={
            "import_path": "llm_harness:LLMSearchHarness",
            "model_name": args.model,
            "kwargs": {"rounds": args.rounds},
        },
        tags={"harness": "llm-search", "model": args.model},
    )
    if row.tags.get("error"):
        raise SystemExit(f"the episode failed: {row.tags['error']}")
    print("trusted reward:", row.rewards)  # the judge's final score

    trace = lab.df("trace")
    if not trace.empty:
        curve = metrics.anytime(trace)
        print(curve[["t", "score", "best_so_far"]].to_string(index=False))
        print("anytime AUC:", round(metrics.auc(curve), 4))


if __name__ == "__main__":
    asyncio.run(main())
