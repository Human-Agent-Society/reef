"""Pin public math data and prepare local JSONL inputs without training labels for OPD."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

SFT_DATASET = ("open-thoughts/OpenThoughts3-1.2M", "61bcf9d4eb38b30295efc2021227a63cc5bb34c8")
OPD_DATASET = ("zwhe99/DeepMath-103K", "5cf055d1fe3d7a2eb19719ac020211469736ae44")
EVAL_DATASET = ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a")


def prepare(output: Path, tokenizer_path: str, sft_examples: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    metadata = {"sft": SFT_DATASET, "opd": OPD_DATASET, "eval": EVAL_DATASET, "sft_examples": sft_examples}
    for name in ("deepmath.jsonl", "aime24.jsonl", "sft.jsonl"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} exists; use a fresh output directory")
    train = load_dataset(OPD_DATASET[0], revision=OPD_DATASET[1], split="train")
    with (output / "deepmath.jsonl").open("w") as handle:
        for index, row in enumerate(train):
            # Match the pinned cookbook's prompt-only 1,024-token cap.
            prompt = tokenizer.decode(tokenizer.encode(row["question"])[:1024])
            handle.write(json.dumps({"id": str(index), "prompt": prompt}, ensure_ascii=False) + "\n")
    evaluation = load_dataset(EVAL_DATASET[0], revision=EVAL_DATASET[1], split="train")
    with (output / "aime24.jsonl").open("w") as handle:
        for index, row in enumerate(evaluation):
            handle.write(
                json.dumps(
                    {"id": str(index), "prompt": row["problem"], "answer": str(row["answer"])}, ensure_ascii=False
                )
                + "\n"
            )
    if sft_examples:
        dataset = load_dataset(SFT_DATASET[0], revision=SFT_DATASET[1], split="train", streaming=True)
        dataset = dataset.shuffle(seed=0, buffer_size=384000).take(sft_examples)
        with (output / "sft.jsonl").open("w") as handle:
            for index, row in enumerate(dataset):
                messages = [
                    {
                        "role": "user" if message["from"] == "human" else "assistant",
                        "content": message["value"].strip(),
                    }
                    for message in row["conversations"]
                ]
                handle.write(json.dumps({"id": str(index), "messages": messages}, ensure_ascii=False) + "\n")
                if (index + 1) % 1000 == 0:
                    print(f"Prepared {index + 1} SFT examples", flush=True)
    metadata["files"] = {}
    for path in sorted(output.glob("*.jsonl")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["files"][path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sft-examples", type=int, default=384000)
    args = parser.parse_args()
    if args.sft_examples < 0:
        parser.error("--sft-examples must be nonnegative")
    prepare(args.output, args.tokenizer, args.sft_examples)


if __name__ == "__main__":
    main()
