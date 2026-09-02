"""The pinned dataset and the seed compositions."""

from __future__ import annotations

import hashlib
import json
import random
from typing import cast

from .adapter import AIMEExample
from .config import AIME_DATASET_SHA256, AIME_SPLIT_SIZES, AIME_TEST_REVISION, AIME_TRAIN_REVISION

RULES_SEED = "You are a helpful assistant. Answer the question. Put your final answer in the format '### <answer>'"
SKILL_SEED = """# AIME solver

Solve each competition-math problem from first principles. Check the result,
then put only the final value after `###` on the last answer line.
"""
RULES_SEED_CANDIDATE = {"rules": RULES_SEED}
MULTI_NODE_SEED_CANDIDATE = {"rules": RULES_SEED, "skill": SKILL_SEED}


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
    return cast(list[AIMEExample], trainset), cast(list[AIMEExample], valset), cast(list[AIMEExample], testset)


def dataset_sha256(trainset, valset, testset) -> str:
    splits = {"train": trainset, "validation": valset, "test": testset}
    return hashlib.sha256((json.dumps(splits, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
