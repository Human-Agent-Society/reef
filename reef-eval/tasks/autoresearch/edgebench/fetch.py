"""Fetch EdgeBench specs from HuggingFace and convert them to Harbor tasks.

    cd tasks/autoresearch/edgebench
    python fetch.py                        # list the 51 tasks
    python fetch.py ann_vector_search_qps  # convert one
    python fetch.py --all                  # convert all

Converted tasks land next to this script and are stock Harbor tasks; see
README.md here for how to run one.

Note: EdgeBench tasks reference prebuilt work/judge images from the EdgeBench
registry; pulling them follows their docs (`sforge pull`). Requires
PyYAML (`python -m pip install pyyaml`).
"""

import json
import sys
import urllib.request
from pathlib import Path

from convert import convert_task

HF = "https://huggingface.co/datasets/ByteDance-Seed/EdgeBench/resolve/main"
OUT = Path(__file__).parent
CACHE = Path(__file__).parent / ".specs"


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        urllib.request.urlretrieve(f"{HF}/{name}", path)
    return path


def list_task_ids() -> list[str]:
    with urllib.request.urlopen(
        "https://huggingface.co/api/datasets/ByteDance-Seed/EdgeBench/tree/main"
    ) as response:
        tree = json.load(response)
    return sorted(
        entry["path"].removesuffix(".json")
        for entry in tree
        if entry["path"].endswith(".json")
    )


def main() -> None:
    task_ids = list_task_ids()
    args = sys.argv[1:]
    if not args:
        print(f"{len(task_ids)} EdgeBench tasks available:")
        print("\n".join(f"  {t}" for t in task_ids))
        print("\nConvert one:  python tasks/autoresearch/edgebench/fetch.py <task_id>")
        return

    wanted = task_ids if args == ["--all"] else args
    benchmark = fetch("BENCHMARK.yaml")
    for task_id in wanted:
        if task_id not in task_ids:
            print(f"unknown task id: {task_id}")
            continue
        task_dir = convert_task(fetch(f"{task_id}.json"), benchmark, OUT)
        print(f"converted: {task_dir}")


if __name__ == "__main__":
    main()
