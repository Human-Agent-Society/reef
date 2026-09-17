"""The two skills of the stream: the reference datasets, their scorers, and the Reef calls.

Shared by the agent container (``stage.py`` drives a stage's training stream)
and the judge container (``score.py`` scores the served model on every
submission). Everything mirrors the reference implementation
(idanshen/Self-Distillation at ``d77573212fa0``): ``main.py`` builds the
prompts and picks the demonstrations, ``eval_science.py`` and
``eval_tooluse.py`` score the test splits.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reef_client import ReefClient

TASKS = ("tooluse", "science")
MODEL = "reef"  # the model name the requests carry; Reef's SGLang serves it
#: The reference's evaluation windows (``eval_tooluse.py``, ``eval_science.py``), greedy.
EVAL_MAX_TOKENS = {"tooluse": 1024, "science": 2048}
#: The reference checkout the container images clone at its pin.
DATA_DIR = Path(os.environ.get("SKILLS_DATA_DIR", "/opt/self-distillation/data"))

SERVICE_URL = os.environ.get("REEF_SERVICE_URL", "http://host.docker.internal:28902")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")
SCENARIO = os.environ.get("REEF_SCENARIO", "skill-stream")

#: The service is gone or rejecting requests; waiting cannot help.
SERVICE_GONE = -1


def load_split(task: str, split: str) -> list[dict[str, Any]]:
    """The reference's ``train_data`` or ``eval_data`` split of a task as plain dicts."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; the stream's tasks are {', '.join(TASKS)}")
    from datasets import load_from_disk

    path = DATA_DIR / f"{task}_data" / f"{split}_data"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the image clones the reference at its pin")
    return [dict(row) for row in load_from_disk(str(path))]


def training_examples(task: str) -> list[tuple[list[dict[str, str]], str]]:
    """``(messages, demonstration)`` per training prompt.

    Science Q&A carries a system message with the answer format and GPT-4o's
    response; Tool Use is a single user message (the tool documentation and
    the request, in the ReAct format) with the dataset's golden response.
    """
    rows = load_split(task, "train")
    if task == "science":
        return [(list(row["messages"]), str(row["output_text"])) for row in rows]
    return [([{"role": "user", "content": row["prompt"]}], "\n".join(row["golden_response"])) for row in rows]


def test_examples(task: str) -> list[tuple[list[dict[str, str]], Any]]:
    """``(messages, gold)`` per test prompt: a letter for Science Q&A, the API call list for Tool Use."""
    rows = load_split(task, "eval")
    if task == "science":
        return [(list(row["prompt"]), row["answer"]) for row in rows]
    return [([{"role": "user", "content": row["prompt"]}], row["golden_answer"]) for row in rows]


def extract_answer(text: str) -> str:
    """The Science Q&A rule: the text inside the last ``<answer>`` tag, stripped."""
    answer = text.split("<answer>")[-1]
    return answer.split("</answer>")[0].strip()


def tool_call(text: str) -> tuple[Counter[str], dict[str, Any]]:
    """The Tool Use rule's reading of a response: the actions named and the merged action inputs."""
    actions = Counter(re.findall(r"Action:\s*(\w+)", text))
    inputs: dict[str, Any] = {}
    for block in re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL):
        try:
            inputs.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    return actions, inputs


def is_correct(task: str, response: str, gold: Any) -> bool:
    """The reference scorer's verdict for one test prompt."""
    if task == "science":
        return extract_answer(response) == gold
    actions, inputs = tool_call(response)
    gold_actions = Counter(item["Action"] for item in gold)
    gold_inputs: dict[str, Any] = {}
    for item in gold:
        try:
            gold_inputs.update(json.loads(item["Action_Input"]))
        except (json.JSONDecodeError, TypeError):
            continue
    return actions == gold_actions and inputs == gold_inputs


def wave_schedule(count: int, epochs: int, prompts_per_step: int, seed: int) -> list[list[int]]:
    """The training order: each epoch a fresh shuffle, cut into steps of ``prompts_per_step`` prompts.

    Steps run across the epoch boundary so every step has the same size (the
    recipe batches exactly that many reports); the prompts left at the very
    end that do not fill a step are dropped.
    """
    generator = random.Random(seed)
    order: list[int] = []
    for _ in range(epochs):
        epoch = list(range(count))
        generator.shuffle(epoch)
        order.extend(epoch)
    return [
        order[start : start + prompts_per_step]
        for start in range(0, len(order) - prompts_per_step + 1, prompts_per_step)
    ]


def make_client(timeout_s: float = 1800.0) -> ReefClient:
    return ReefClient(SERVICE_URL, token=TOKEN, timeout_s=timeout_s)


def ask(
    client: ReefClient, messages: Sequence[dict[str, str]], *, temperature: float, max_tokens: int
) -> tuple[str, str, int]:
    """One chat completion through Reef: the text, its receipt, and its completion token count."""
    response, agent_record_id = client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {"model": MODEL, "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature},
    )
    return response["choices"][0]["message"]["content"], agent_record_id, int(response["usage"]["completion_tokens"])


def ask_all(
    client: ReefClient,
    prompts: Sequence[Sequence[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    concurrency: int,
) -> list[tuple[str, str, int]]:
    """``ask`` for every prompt at once, results in prompt order."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(
            pool.map(lambda messages: ask(client, messages, temperature=temperature, max_tokens=max_tokens), prompts)
        )


def evaluate(client: ReefClient, task: str, *, concurrency: int = 64) -> dict[str, Any]:
    """Accuracy of the served model on a task's test split, decoded greedily as the reference does."""
    examples = test_examples(task)
    started = time.time()
    outputs = ask_all(
        client,
        [messages for messages, _ in examples],
        temperature=0.0,
        max_tokens=EVAL_MAX_TOKENS[task],
        concurrency=concurrency,
    )
    correct = sum(1 for (_, gold), (text, _, _) in zip(examples, outputs, strict=True) if is_correct(task, text, gold))
    return {
        "task": task,
        "accuracy": correct / len(examples),
        "num_correct": correct,
        "num_total": len(examples),
        "mean_completion_tokens": sum(tokens for _, _, tokens in outputs) / len(outputs),
        "elapsed_s": time.time() - started,
    }


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
        time.sleep(5.0)
