"""The Chemistry task of SDPO's generalization sweep: its split, its prompts, its scorer and the Reef calls.

Everything here is the pinned reference's (lasgroup/SDPO at ``7c457fc1b1f6``),
which the task image clones:

- ``datasets/sciknoweval/chemistry/{train,test}.json`` are JSON lines. Each row
  carries the system prompt the authors fix (answer inside ``<answer>`` tags,
  the letter alone), the question with its four options, and the answer letter.
  The split is theirs: 1890 training and 210 test questions.
- ``verl/utils/reward_score/feedback/mcq.py`` scores a response by the letter
  between its last ``<answer>`` tags, compared with the answer letter.
- The sampling windows and decoding are ``verl/trainer/config/user.yaml``'s:
  training samples at temperature 1, evaluation at temperature 0.6 with top-p
  0.95, both with the chat template's thinking switch off.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reef_client import ReefClient

#: The reference checkout the container image clones at its pin.
DATA_DIR = Path(os.environ.get("SDPO_DATA_DIR", "/opt/sdpo-reference/datasets/sciknoweval/chemistry"))
MODEL = "reef"  # the model name the requests carry; Reef's SGLang serves it
#: ``user.yaml``: the prompt and response windows, and the decoding of each phase.
MAX_PROMPT_TOKENS = 2048
MAX_RESPONSE_TOKENS = int(os.environ.get("SDPO_MAX_RESPONSE_TOKENS", "8192"))
TRAIN_TEMPERATURE = 1.0
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95

SERVICE_URL = os.environ.get("REEF_SERVICE_URL", "http://host.docker.internal:28902").rstrip("/")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")
SCENARIO = os.environ.get("REEF_SCENARIO", "sdpo-chemistry")

#: The service is gone or rejecting requests; waiting cannot help.
SERVICE_GONE = -1


def load_split(split: str) -> list[dict[str, Any]]:
    """The reference's ``train`` or ``test`` split as plain dicts, in file order."""
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the image clones the reference at its pin")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """The request the reference builds for one question: its fixed system prompt, then the question."""
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}]


def extract_answer(text: str) -> str:
    """The letter the reference's ``mcq.extract_xml_answer`` reads: the last ``<answer>`` block, stripped."""
    answer = text.split("<answer>")[-1]
    return answer.split("</answer>")[0].strip()


def is_correct(text: str, answer: str) -> bool:
    """The reference's reward: the extracted letter equals the answer letter."""
    return extract_answer(text) == answer


def question_schedule(count: int, epochs: int, per_step: int, seed: int) -> list[list[int]]:
    """Training question indices per step: each epoch a fresh shuffle, chunked into steps.

    ``data.shuffle`` is on in the reference and its trainer runs whole epochs,
    so steps run across the epoch boundary and only the final partial step is
    dropped.
    """
    generator = random.Random(seed)
    order: list[int] = []
    for _ in range(epochs):
        epoch = list(range(count))
        generator.shuffle(epoch)
        order.extend(epoch)
    steps = len(order) // per_step
    return [order[index * per_step : (index + 1) * per_step] for index in range(steps)]


def make_client(timeout_s: float = 3600.0) -> ReefClient:
    return ReefClient(SERVICE_URL, token=TOKEN, timeout_s=timeout_s)


def sample(client: ReefClient, row: dict[str, Any]) -> tuple[str, str]:
    """One recorded training sample: its text and the receipt a report grades."""
    response, receipt = client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages(row),
            "temperature": TRAIN_TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response["choices"][0]["message"]["content"], receipt


def sample_group(client: ReefClient, row: dict[str, Any], rollouts: int) -> list[tuple[str, str]]:
    """``rollouts`` independent recorded samples of one question, in rollout order."""
    with ThreadPoolExecutor(max_workers=rollouts) as pool:
        return list(pool.map(lambda _: sample(client, row), range(rollouts)))


def score_at_n(client: ReefClient, row: dict[str, Any], n: int) -> float:
    """The question's avg@n: ``n`` unrecorded samples at the evaluation decoding, the fraction correct."""
    response = client.inference(
        SCENARIO,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages(row),
            "n": n,
            "temperature": EVAL_TEMPERATURE,
            "top_p": EVAL_TOP_P,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    texts = [choice["message"]["content"] for choice in response["choices"]]
    return sum(1 for text in texts if is_correct(text, row["answer"])) / len(texts)


def evaluate(client: ReefClient, rows: Sequence[dict[str, Any]], *, n: int, concurrency: int) -> dict[str, Any]:
    """avg@n over the test split: each question sampled ``n`` times, then averaged over questions.

    The samples are not recorded, so an evaluation never becomes training data.
    """
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        per_question = list(pool.map(lambda row: score_at_n(client, row, n), rows))
    return {
        "avg_at_n": sum(per_question) / len(per_question),
        "n": n,
        "questions": len(per_question),
        "elapsed_s": round(time.time() - started, 1),
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
