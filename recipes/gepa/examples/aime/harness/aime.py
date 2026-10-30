"""The AIME benchmark as the GEPA method's task side: pins, scorer, feedback.

Everything the method needs to know about AIME lives here, and nothing here
knows about GEPA beyond its ``Feedback`` hook: ``gepa.yaml`` names the scorer
as ``evolution.evaluation.scorer`` and the feedback as the method's
``feedback``, and Reef calls them with a task and a run result.
The validation problems are evaluation task records that carry their answers
(``validation_tasks``). The training problems reach the method as recorded
traffic, whose instruction is all a run of them carries, so the splits are
also loaded once and indexed by problem statement for those runs and for the
feedback.

The upstream quickstart's scoring rule and its feedback wording are
reproduced here rather than imported: this example exists to show the Reef
method matching GEPA's published numbers, so a runtime dependency on the
package under comparison would defeat it. The original is
``gepa.adapters.default_adapter.ContainsAnswerEvaluator`` at the v0.1.2 tag
(commit 92dadfff), MIT licensed, copyright Lakshya A Agrawal and the GEPA
contributors. The minibatch order is the method's (``recipes/gepa/archive.py``),
drawn from the same generator as its parent choice, as upstream draws it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from recipes.gepa import Feedback
from reef.evaluation.harness import EvaluationTask, TaskScorer
from reef.harness.tasks.run import TaskResult

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

# The extra node that the multi-node variant (REEF_GEPA_MULTI=1) evolves
# beside the seed's rules.md.
SKILL_SEED = """# AIME solver

Solve each competition-math problem from first principles. Check the result,
then put only the final value after `###` on the last answer line.
"""
SKILL_SEED_NAME = "aime-solver"

#: Prompt text to expected answer string, and prompt text to the optional
#: context the feedback hook quotes. Filled by :func:`load_aime_splits`; Reef
#: only ever passes tasks that came from those splits.
ANSWERS: dict[str, str] = {}
CONTEXTS: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class AIMEScorer(Feedback, TaskScorer):
    """A pickleable snapshot, independent of any worker's module globals.

    The driver registry is only an input at recipe construction time. Later
    registrations do not change a running campaign's scoring or feedback.
    These labels belong to the scorer, never to a run's prompt/files.
    """

    answers: dict[str, str] = field(repr=False)
    contexts: dict[str, dict[str, str]] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", dict(self.answers))
        object.__setattr__(self, "contexts", {task: dict(context) for task, context in self.contexts.items()})

    def answer_for(self, task: str) -> str:
        return _answer_for(self.answers, task)

    def score(self, task: EvaluationTask, result: TaskResult) -> float:
        return _evaluate(task_answer(self.answers, task), result)

    def feedback(self, task: str, output: str, score: float) -> str:
        return _feedback(self.answers, self.contexts, task, output, score)


class AIMEAnswerScorer(TaskScorer):
    """:func:`evaluate` over the splits registered in this process; ``evolution.evaluation.scorer`` names it."""

    def score(self, task: EvaluationTask, result: TaskResult) -> float:
        return _evaluate(task_answer(ANSWERS, task), result)


class AIMEFeedback(Feedback):
    """Feedback over the benchmark splits registered in this process."""

    def feedback(self, task: str, output: str, score: float) -> str:
        return _feedback(ANSWERS, CONTEXTS, task, output, score)


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


def validation_tasks(valset: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The validation split as ``evolution.evaluation.tasks`` records, each with its answer, named by position."""
    return [
        {"id": f"aime-validation-{index:02d}", "instruction": str(example["input"]), "answer": str(example["answer"])}
        for index, example in enumerate(valset)
    ]


def dataset_sha256(trainset, valset, testset) -> str:
    splits = {"train": trainset, "validation": valset, "test": testset}
    return hashlib.sha256((json.dumps(splits, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def register(*splits: Sequence[Mapping[str, Any]]) -> None:
    """Index every split by problem statement, the only key a run carries."""
    for split in splits:
        for example in split:
            ANSWERS[str(example["input"])] = str(example["answer"])
            context = example.get("additional_context")
            if context:
                CONTEXTS[str(example["input"])] = {str(key): str(value) for key, value in context.items()}


def answer_for(task: str) -> str:
    """The expected ``### <answer>`` string for one problem statement."""
    return _answer_for(ANSWERS, task)


def _answer_for(answers: Mapping[str, str], task: str) -> str:
    try:
        return answers[task]
    except KeyError:
        raise RuntimeError(f"no AIME answer is registered for this task: {task[:120]!r}") from None


def task_answer(answers: Mapping[str, str], task: EvaluationTask) -> str:
    """A validation task's own answer; for a training problem's traffic, which carries none, the registered one."""
    return task.answer if task.answer is not None else _answer_for(answers, task.text)


def evaluate(task: str, result: TaskResult) -> float:
    """The quickstart's containment score over one Pi run.

    A dirty run - a non-zero exit or a file the adapter did not declare -
    scores zero rather than being read for an answer, so a broken harness can
    never be selected on the strength of stdout it happened to leave behind.
    """
    return _evaluate(_answer_for(ANSWERS, task), result)


def _evaluate(answer: str, result: TaskResult) -> float:
    if result.exit_code != 0 or result.residue:
        return 0.0
    response = final_assistant_text(result.trajectory) or result.stdout
    return 1.0 if answer in response else 0.0


def feedback(task: str, output: str, score: float) -> str:
    """``ContainsAnswerEvaluator``'s wording, reproduced without importing it."""
    return _feedback(ANSWERS, CONTEXTS, task, output, score)


def _feedback(
    answers: Mapping[str, str], contexts: Mapping[str, Mapping[str, str]], task: str, output: str, score: float
) -> str:
    answer = _answer_for(answers, task)
    if score >= 1.0:
        return f"The generated response is correct. The response include the correct answer '{answer}'"
    # ``output`` is part of the hook's contract but not of upstream's wording:
    # the reflection prompt already carries the response beside this text.
    text = (
        f"The generated response is incorrect. The correct answer is '{answer}'. "
        "Ensure that the correct answer is included in the response exactly as it is."
    )
    context = "\n".join(f"{key}: {value}" for key, value in contexts.get(task, {}).items())
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
