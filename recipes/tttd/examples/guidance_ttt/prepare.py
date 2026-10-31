"""Fetch pinned benchmark sources and prepare a bootstrap without running candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from harness.config import EXAMPLE_DIR, FRONTIERCS2_TASKS, TASKS

DISCOVER_COMMIT = "6c40e82dab9d5de7416ac873ad5cd3106084aaed"
SIMPLETES_COMMIT = "47d3413da1d85dc24341219d47452d2601e56a57"
FRONTIERCS2_COMMIT = "e989e8c6826585a336fb999db1b87fa327073c2d"


def pinned_checkout(url: str, commit: str, target: Path) -> None:
    created = not target.exists()
    if created:
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(target)], check=True)
    actual = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True)
    if not created and actual.returncode == 0 and actual.stdout.strip() == commit:
        if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"]):
            raise RuntimeError(f"benchmark checkout has modified tracked files: {target}")
        return
    # Only populate the empty checkout that this command just created.
    if (target / "LICENSE").exists() or (target / "README.md").exists():
        raise RuntimeError(f"{target} is not at {commit}; use a fresh destination")
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", commit], check=True)


def prepare_frontiercs2(task: str, state_dir: Path) -> None:
    """Pin FrontierCS 2, then write the starting file and the executor's repository context."""
    checkout = EXAMPLE_DIR / "work" / "FrontierCS-2"
    # Task images carry the large files, so the checkout skips Git LFS content.
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    pinned_checkout("https://github.com/FrontierCS/FrontierCS-2.git", FRONTIERCS2_COMMIT, checkout)
    spec = json.loads((EXAMPLE_DIR / "judges/frontiercs2_tasks.json").read_text())[task]
    (directory,) = (checkout / "tasks/frontier-cs-2.0-demo/problems").glob(f"*/{task}")
    repo = directory / "agent-base/frontier/task/repo"
    context = ["Files of the task repository that the candidate file works with. They are fixed.\n"]
    for name in spec["context_files"]:
        fence = "cpp" if Path(name).suffix in {".h", ".hpp", ".cpp", ".cc"} else "python"
        context.append(f"## `{name}`\n\n```{fence}\n{(repo / name).read_text().rstrip()}\n```\n")
    if spec["context_files"]:
        (state_dir / "execution_context.md").write_text("\n".join(context))
    target = state_dir / f"bootstrap{Path(spec['editable_file']).suffix}"
    if "bootstrap_summary" not in spec:
        print(
            f"{task}: the shipped {spec['editable_file']} is a stub; write a bootstrap with bootstrap_frontiercs2.py"
        )
        return
    source = (repo / spec["editable_file"]).read_text()
    if target.exists() and target.read_text() != source:
        raise RuntimeError(f"refusing to replace a different bootstrap: {target}")
    target.write_text(source)
    target.with_suffix(".md").write_text(spec["bootstrap_summary"] + "\n")
    print(f"Prepared {target}. The judge reads --problems {directory.parent.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS + FRONTIERCS2_TASKS)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    state_dir = Path(
        args.state_dir or os.environ.get("GUIDANCE_STATE_DIR") or EXAMPLE_DIR / "work" / args.task
    ).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    if args.task in FRONTIERCS2_TASKS:
        prepare_frontiercs2(args.task, state_dir)
        return
    if args.task == "polyomino_packing":
        print("Polyomino bootstrap is bundled; start the pinned FrontierCS judge described in README.md")
        return
    if args.task == "lasso_path":
        checkout = state_dir / "SimpleTES"
        pinned_checkout("https://github.com/wq-will/SimpleTES.git", SIMPLETES_COMMIT, checkout)
        source = checkout / "datasets/numerical_tasks/lasso_path/init_program.py"
        expected = "778c81aba5a75534614f4f454d57536117e53b9eb175a0af474d4d6b6cc1d6ae"
    else:
        checkout = state_dir / "discover"
        pinned_checkout("https://github.com/test-time-training/discover.git", DISCOVER_COMMIT, checkout)
        if args.task == "ahc058":
            source = checkout / "results/algorithm-design/ahc058.cpp"
            expected = "32d7cc7272fec19f4dbd35b4d04717ce78b8b89b1c593a2537f902af212ee19c"
        else:
            source = checkout / "results/kernel-engineering/trimul.py"
            expected = None
    if not source.is_file():
        raise FileNotFoundError(f"pinned bootstrap is missing: {source}")
    if expected and hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"bootstrap checksum mismatch: {source}")
    target = state_dir / f"bootstrap{source.suffix}"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError(f"refusing to replace a different bootstrap: {target}")
    shutil.copyfile(source, target)
    shutil.copyfile(EXAMPLE_DIR / "harbor" / args.task / "bootstrap-summary.md", target.with_suffix(".md"))
    print(f"Prepared {target}. The harness will score it before search.")


if __name__ == "__main__":
    main()
