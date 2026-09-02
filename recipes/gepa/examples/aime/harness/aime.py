"""The AIME benchmark as the GEPA method's task side: pins, scorer, feedback, sampler.

Everything the method needs to know about AIME lives here, and nothing here
knows about GEPA: ``evaluate`` and ``feedback`` are the two hooks
``gepa.yaml`` names under ``evolution``, and the mechanism calls them with a
task prompt and an episode result. The answer table is keyed by the prompt
text because that is all the mechanism carries - ``evolution.tasks`` is a
list of strings - so the splits are loaded once and indexed by problem
statement.

The upstream quickstart's scoring rule, its feedback wording, and its epoch
shuffled minibatch order are reproduced here rather than imported: this
example exists to show the Reef method matching GEPA's published numbers, so
a runtime dependency on the package under comparison would defeat it. The
originals are ``gepa.adapters.default_adapter.ContainsAnswerEvaluator`` and
``gepa.strategies.batch_sampler.EpochShuffledBatchSampler`` at the v0.1.2 tag
(commit 92dadfff), MIT licensed, copyright Lakshya A Agrawal and the GEPA
contributors.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from reef.harness.episode import EpisodeResult

# The Pi release the retained record ran and the example still targets; run.py
# refuses any other binary version before it spends anything.
PI_VERSION = "0.84.2"

# Dated snapshots of the quickstart's task and reflection models, so the
# comparison cannot drift under an alias; sampling stays at provider defaults
# as upstream leaves it. The credential is named, not held.
TASK_MODEL = "gpt-4.1-mini-2025-04-14"
REFLECTION_MODEL = "gpt-5-2025-08-07"
OPENAI_BASE_URL = "https://api.openai.com"
API_KEY_ENV = "OPENAI_API_KEY"

# The quickstart's search budget: one search is 150 metric calls, followed by
# two passes over the 150-example test split.
SEARCH_BUDGET = 150

# Pi's default output limit, which it sends as max_completion_tokens on every
# request; the retained record pinned the same value explicitly. Kept so a
# report can name it, not to be set anywhere.
PI_TASK_MAX_TOKENS = 16_384

# The upstream loader does not pin Hugging Face revisions; these do, and the
# split hash catches same-size content drift before any paid call.
AIME_TRAIN_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"
AIME_TEST_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"
AIME_SPLIT_SIZES = {"train": 45, "validation": 45, "test": 150}
AIME_DATASET_SHA256 = "74e81306a9a1debadd64c49a4ab3588615f7bb698b695a59c17c65dd3b895185"

# The quickstart's seed prompt, and the extra node the multi-node variant
# evolves beside it (REEF_GEPA_MULTI=1).
RULES_SEED = "You are a helpful assistant. Answer the question. Put your final answer in the format '### <answer>'"
SKILL_SEED = """# AIME solver

Solve each competition-math problem from first principles. Check the result,
then put only the final value after `###` on the last answer line.
"""
SKILL_SEED_NAME = "aime-solver"

#: Prompt text to expected answer string, and prompt text to the optional
#: context the feedback hook quotes. Filled by :func:`load_aime_splits`; the
#: mechanism only ever passes tasks that came from those splits.
ANSWERS: dict[str, str] = {}
CONTEXTS: dict[str, dict[str, str]] = {}


class AIMEExample(TypedDict, total=False):
    """The subset of GEPA's AIME example schema that Reef consumes."""

    input: str
    answer: str
    additional_context: dict[str, str]


def load_aime_splits() -> tuple[list[AIMEExample], list[AIMEExample], list[AIMEExample]]:
    """The upstream ``examples/aime_math`` splits at pinned revisions, verified before any paid call."""
    from datasets import load_dataset

    source = [
        {
            "input": item["problem"],
            "additional_context": {"solution": item["solution"]},
            "answer": f"### {item['answer']}",
        }
        for item in load_dataset("AI-MO/aimo-validation-aime", "default", split="train", revision=AIME_TRAIN_REVISION)
    ]
    random.Random(0).shuffle(source)
    midpoint = len(source) // 2
    trainset, valset = source[:midpoint], source[midpoint:]
    # The pinned quickstart repeats AIME 2025 five times for its 150-example evaluation.
    testset = [
        {"input": item["problem"], "answer": f"### {item['answer']}"}
        for item in load_dataset("MathArena/aime_2025", "default", split="train", revision=AIME_TEST_REVISION)
    ] * 5
    sizes = {"train": len(trainset), "validation": len(valset), "test": len(testset)}
    if sizes != AIME_SPLIT_SIZES:
        raise RuntimeError(f"upstream AIME split sizes changed: observed {sizes}, expected {AIME_SPLIT_SIZES}")
    digest = dataset_sha256(trainset, valset, testset)
    if digest != AIME_DATASET_SHA256:
        raise RuntimeError(f"upstream AIME content changed: observed SHA-256 {digest}, expected {AIME_DATASET_SHA256}")
    register(trainset, valset, testset)
    return cast(list[AIMEExample], trainset), cast(list[AIMEExample], valset), cast(list[AIMEExample], testset)


def dataset_sha256(trainset, valset, testset) -> str:
    splits = {"train": trainset, "validation": valset, "test": testset}
    return hashlib.sha256((json.dumps(splits, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def register(*splits: Sequence[Mapping[str, Any]]) -> None:
    """Index every split by problem statement, the only key the mechanism carries."""
    for split in splits:
        for example in split:
            ANSWERS[str(example["input"])] = str(example["answer"])
            context = example.get("additional_context")
            if context:
                CONTEXTS[str(example["input"])] = {str(key): str(value) for key, value in context.items()}


def answer_for(task: str) -> str:
    """The expected ``### <answer>`` string for one problem statement."""
    try:
        return ANSWERS[task]
    except KeyError:
        raise RuntimeError(f"no AIME answer is registered for this task: {task[:120]!r}") from None


def evaluate(task: str, result: EpisodeResult) -> float:
    """The quickstart's containment score over one Pi episode.

    A dirty episode - a non-zero exit or a file the adapter did not declare -
    scores zero rather than being read for an answer, so a broken harness can
    never be selected on the strength of stdout it happened to leave behind.
    """
    if result.exit_code != 0 or result.residue:
        return 0.0
    response = final_assistant_text(result.trajectory) or result.stdout
    return 1.0 if answer_for(task) in response else 0.0


def feedback(task: str, output: str, score: float) -> str:
    """``ContainsAnswerEvaluator``'s wording, reproduced without importing it."""
    answer = answer_for(task)
    if score >= 1.0:
        return f"The generated response is correct. The response include the correct answer '{answer}'"
    # ``output`` is part of the hook's contract but not of upstream's wording:
    # the reflection prompt already carries the response beside this text.
    text = (
        f"The generated response is incorrect. The correct answer is '{answer}'. "
        "Ensure that the correct answer is included in the response exactly as it is."
    )
    context = "\n".join(f"{key}: {value}" for key, value in CONTEXTS.get(task, {}).items())
    if context:
        text += f" Here is some additional context that might be helpful:\n{context}"
    return text


def final_assistant_text(trajectory: Sequence[Mapping[str, Any]]) -> str | None:
    """The final assistant text from Pi's wrapped or flat events."""
    for event in reversed(trajectory):
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, Mapping) else event
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part["text"] for part in content if isinstance(part, Mapping) and part.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return None


class Minibatches:
    """GEPA's epoch-shuffled training order, by step index.

    The rule, reproduced from ``EpochShuffledBatchSampler``: shuffle the ids
    once per epoch with a seeded generator whose state carries across epochs,
    pad the shuffled list up to a whole number of minibatches by repeating the
    least frequent id (ties broken by the latest first appearance), and slice
    the step's window out of it. The driver owns the step counter, so an
    interrupted run resumes on the same order it would have drawn.
    """

    def __init__(self, size: int, minibatch_size: int, *, seed: int = 0) -> None:
        if size <= 0 or minibatch_size <= 0:
            raise ValueError("a minibatch sampler needs a non-empty trainset and a positive minibatch size")
        self.size = size
        self.minibatch_size = minibatch_size
        self._rng = random.Random(seed)
        self._order: list[int] = []
        self._epoch = -1

    def ids(self, step: int) -> tuple[int, ...]:
        base = step * self.minibatch_size
        epoch = 0 if self._epoch == -1 else base // max(len(self._order), 1)
        if not self._order or epoch > self._epoch:
            self._epoch = epoch
            self._reshuffle()
        base %= len(self._order)
        return tuple(self._order[base : base + self.minibatch_size])

    def _reshuffle(self) -> None:
        self._order = list(range(self.size))
        self._rng.shuffle(self._order)
        frequencies = Counter(self._order)
        remainder = self.size % self.minibatch_size
        for _ in range(self.minibatch_size - remainder if remainder else 0):
            # ``most_common()[::-1][0]`` upstream: the least frequent id, and
            # among equals the one that appeared last in the shuffled list.
            selected = min(reversed(list(frequencies)), key=lambda identifier: frequencies[identifier])
            self._order.append(selected)
            frequencies[selected] += 1
