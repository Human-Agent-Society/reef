"""Pin the public math data and write the OPD prompts and the AIME'24 evaluation set as JSONL."""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

OPD_DATASET = ("zwhe99/DeepMath-103K", "5cf055d1fe3d7a2eb19719ac020211469736ae44")
EVAL_DATASET = ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a")
#: The cookbook's prompt-only cap, in tokens of the model pair's tokenizer.
PROMPT_TOKEN_LIMIT = 1024
#: DeepMath questions that also appear in OpenThoughts3's math subset, found when the pool was first
#: built disjoint from an SFT selection; they stay excluded so the published pool is the measured one.
OPENTHOUGHTS3_DUPLICATE_IDS = frozenset(
    {"100560", "98661", "99164", "80084", "96384", "96992", "102603", "98925", "99462", "97139"}
)
SHUFFLE_SEED = 1


def normalize_question(text: str) -> str:
    """Ignore formatting for exact/whole-question overlap checks, not paraphrases."""
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text).casefold())


def select_prompts(
    prompts: Sequence[dict[str, str]], evaluation_questions: Iterable[str], count: int
) -> list[dict[str, str]]:
    """The first ``count`` prompts in seeded order that repeat neither an evaluation question nor each other."""
    excluded = {normalize_question(question) for question in evaluation_questions}
    seen = set(excluded)
    order = list(prompts)
    random.Random(SHUFFLE_SEED).shuffle(order)
    selected = []
    for row in order:
        question = normalize_question(row["prompt"])
        if row["id"] in OPENTHOUGHTS3_DUPLICATE_IDS or question in seen or any(test in question for test in excluded):
            continue
        seen.add(question)
        selected.append({"id": row["id"], "prompt": row["prompt"]})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} disjoint prompts for a pool of {count}")
    return selected


def prepare(output: Path, tokenizer_path: str, prompt_count: int) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    for name in ("deepmath.jsonl", "aime24.jsonl", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} exists; use a fresh output directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    evaluation = [
        {"id": str(index), "prompt": row["problem"], "answer": str(row["answer"])}
        for index, row in enumerate(load_dataset(EVAL_DATASET[0], revision=EVAL_DATASET[1], split="train"))
    ]
    prompts = [
        {"id": str(index), "prompt": tokenizer.decode(tokenizer.encode(row["question"])[:PROMPT_TOKEN_LIMIT])}
        for index, row in enumerate(load_dataset(OPD_DATASET[0], revision=OPD_DATASET[1], split="train"))
    ]
    selected = select_prompts(prompts, (row["prompt"] for row in evaluation), prompt_count)
    for name, rows in (("deepmath.jsonl", selected), ("aime24.jsonl", evaluation)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "opd": OPD_DATASET,
        "eval": EVAL_DATASET,
        "tokenizer": tokenizer_path,
        "prompt_token_limit": PROMPT_TOKEN_LIMIT,
        "prompt_count": prompt_count,
        "shuffle_seed": SHUFFLE_SEED,
        "excluded_ids": sorted(OPENTHOUGHTS3_DUPLICATE_IDS, key=int),
        "overlap_check": "normalized exact and whole AIME question; not a paraphrase audit",
        "files": {
            path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path in sorted(output.glob("*.jsonl"))
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="The student's tokenizer directory; the teacher shares it.")
    parser.add_argument("--prompts", type=int, default=3840, help="Prompts in the pool: 64 per update.")
    args = parser.parse_args()
    if args.prompts <= 0:
        parser.error("--prompts must be positive")
    prepare(args.output, args.tokenizer, args.prompts)


if __name__ == "__main__":
    main()
