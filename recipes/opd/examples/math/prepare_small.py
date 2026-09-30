"""Select complete short math demonstrations and disjoint prompt-only OPD inputs."""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files

from recipes.opd.examples.math.prepare import SFT_DATASET
from recipes.opd.examples.math.sft import TokenizePair


def file_sha256(path: Path) -> str:
    """Hash source files without loading complete dataset shards into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    """Ignore formatting for exact/whole-question overlap checks, not paraphrases."""
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text).casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/work/data"))
    parser.add_argument("--tokenizer", default="/work/models/Qwen3.5-9B")
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--opd-prompts", type=int, default=1920)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.examples, args.opd_prompts, args.max_length, args.workers) <= 0:
        parser.error("Counts and length must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation = [json.loads(line) for line in (args.source / "aime24.jsonl").read_text().splitlines()]
    excluded = {normalize_question(row["prompt"]) for row in evaluation}
    transform = TokenizePair(args.tokenizer, 10**9)
    shards = sorted(
        name
        for name in list_repo_files(SFT_DATASET[0], repo_type="dataset", revision=SFT_DATASET[1])
        if name.endswith(".parquet")
    )
    random.Random(0).shuffle(shards)
    selected, encoded, used_shards = [], [], []
    seen = set(excluded)
    scanned = 0
    for shard in shards:
        local = hf_hub_download(SFT_DATASET[0], shard, repo_type="dataset", revision=SFT_DATASET[1])
        used_shards.append({"name": shard, "sha256": file_sha256(Path(local))})
        candidates = []
        for batch in pq.ParquetFile(local).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                scanned += 1
                if row["domain"] != "math":
                    continue
                messages = [
                    {"role": "user" if m["from"] == "human" else "assistant", "content": m["value"].strip()}
                    for m in row["conversations"]
                ]
                if [m["role"] for m in messages] != ["user", "assistant"]:
                    continue
                question = normalize_question(messages[0]["content"])
                if question in seen or any(test in question for test in excluded):
                    continue
                candidates.append({"messages": messages, "question": question})
        if not candidates:
            continue
        print(json.dumps({"shard": shard, "math_candidates": len(candidates)}), flush=True)
        tokenized = Dataset.from_list(candidates).map(
            transform,
            batched=True,
            batch_size=16,
            num_proc=args.workers,
            remove_columns=["messages", "question"],
            load_from_cache_file=False,
        )
        for row, tokens in zip(candidates, tokenized, strict=True):
            question = row["question"]
            if question in seen or len(tokens["input_ids"]) > args.max_length:
                continue
            seen.add(question)
            selected.append({"id": hashlib.sha256(question.encode()).hexdigest(), "messages": row["messages"]})
            encoded.append(tokens)
            if len(selected) == args.examples:
                break
        print(json.dumps({"selected": len(selected), "scanned": scanned}), flush=True)
        if len(selected) == args.examples:
            break
    if len(selected) != args.examples:
        raise ValueError("Not enough complete non-overlapping short math demonstrations")
    order = list(range(len(selected)))
    random.Random(0).shuffle(order)
    selected = [selected[i] for i in order]
    encoded = [encoded[i] for i in order]
    prompts = [json.loads(line) for line in (args.source / "deepmath.jsonl").read_text().splitlines()]
    random.Random(1).shuffle(prompts)
    train = []
    for row in prompts:
        question = normalize_question(row["prompt"])
        if question in seen or any(test in question for test in excluded):
            continue
        seen.add(question)
        train.append({"id": row["id"], "prompt": row["prompt"]})
        if len(train) == args.opd_prompts:
            break
    if len(train) != args.opd_prompts:
        raise ValueError("Not enough disjoint OPD prompts")
    for name, rows in [("sft.jsonl", selected), ("deepmath.jsonl", train), ("aime24.jsonl", evaluation)]:
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    Dataset.from_list(encoded).save_to_disk(str(args.output / "tokenized"))
    manifest = {
        "sft_dataset": SFT_DATASET,
        "selection": "seed-0 shuffled pinned shards; first eligible math rows, then seed-0 shuffle",
        "domain": "math",
        "sft_examples": len(selected),
        "opd_prompts": len(train),
        "scanned": scanned,
        "max_length": args.max_length,
        "total_sft_tokens": sum(len(row["input_ids"]) for row in encoded),
        "shards": used_shards,
        "source_manifest": json.loads((args.source / "manifest.json").read_text()),
        "overlap_check": "normalized exact and whole AIME question; not a paraphrase audit",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
