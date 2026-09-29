"""Write IMOAnswerBench as the JSONL ``stream.py`` reads: problem_idx, problem, gold.

Run where the Hugging Face hub is reachable; the training node may not be.
"""

from __future__ import annotations

import json
import sys

from datasets import load_dataset


def main(out_path: str) -> None:
    data = load_dataset("OpenEvals/IMO-AnswerBench", split="train")
    with open(out_path, "w") as out:
        for idx, row in enumerate(data):
            out.write(
                json.dumps({"problem_idx": idx, "problem": row["Problem"], "gold": str(row["Short Answer"])}) + "\n"
            )
    print(f"wrote {len(data)} problems to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work/imo_answerbench.jsonl")
