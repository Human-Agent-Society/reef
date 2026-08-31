"""Generate FrontierCS Harbor tasks into this folder, both tracks.

Wraps the official Frontier-CS → Harbor adapters (MIT). One sample per
track is vendored here; this script fetches their repo and generates any
of the others. Problem ids pick the track: numeric ids are the 1.0
algorithmic track (172 open-ended competitive-programming problems),
names are the 2.0 track (20 open research problems).

    cd tasks/autoresearch/frontier-cs
    python fetch.py                      # list problem ids
    python fetch.py --problems 1 17      # judge test data for the algorithmic track
    python fetch.py 1 17                 # algorithmic track
    python fetch.py erdos_unit_distance  # 2.0 track
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/FrontierCS/Frontier-CS.git"
PUBLISHED_JUDGE = "yanagiorigami/frontier-cs-harbor-judge:latest"


def _generate(src: Path, module: str, pythonpath: Path, argv: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *argv],
        check=True,
        env={"PYTHONPATH": str(pythonpath), "PATH": "/usr/bin:/bin"},
    )


def _fetch_problems(src: Path, ids: list[str]) -> None:
    """Copy the algorithmic judge's test data out of the upstream checkout.

    The judge mounts it through $FRONTIER_CS_ALGORITHMIC_PATH. All 188
    problems are 2.6 GB, so take the ones you mean to run.
    """
    root = HERE / ".problems"
    dest = root / "problems"
    artifacts = root / "artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    for problem_id in ids:
        source = src / "algorithmic" / "problems" / problem_id
        if not source.is_dir():
            raise SystemExit(f"no upstream problem {problem_id}")
        target = dest / problem_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        # Compose reads a .env beside the compose file, and Harbor points it
        # at that directory, so writing one here is what saves every caller
        # from exporting the same two paths by hand.
        env_file = HERE / f"frontier-cs-algorithm-{problem_id}" / "environment" / ".env"
        if env_file.parent.is_dir():
            env_file.write_text(
                f"FRONTIER_CS_ALGORITHMIC_PATH={root}\n"
                f"HOST_ARTIFACTS_PATH={artifacts}\n"
            )
    print(f"{len(ids)} problem(s) -> {dest}")
    print("their tasks are ready to run; nothing to export")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="frontiercs-") as tmp:
        src = Path(tmp) / "Frontier-CS"
        print(f"cloning {REPO} ...")
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", REPO, str(src)], check=True
        )
        algorithmic = sorted(
            (
                p.name
                for p in (src / "algorithmic" / "problems").iterdir()
                if p.is_dir()
            ),
            key=lambda n: (len(n), n),
        )
        research = sorted(
            p.parent.name for p in (src / "2.0" / "problems").glob("*/evaluator.py")
        )
        if len(sys.argv) < 2:
            print(f"{len(algorithmic)} algorithmic-track problems (numeric ids):")
            print("  " + " ".join(algorithmic))
            print(f"\n{len(research)} 2.0-track problems:")
            print("\n".join(f"  {p}" for p in research))
            print("\nGenerate:  python fetch.py <id> [...] (from this folder)")
            return

        if sys.argv[1] == "--problems":
            _fetch_problems(src, sys.argv[2:] or algorithmic)
            return

        wanted = sys.argv[1:]
        unknown = set(wanted) - set(algorithmic) - set(research)
        if unknown:
            raise SystemExit(f"unknown problem ids: {sorted(unknown)}")

        numeric = [w for w in wanted if w in set(algorithmic)]
        named = [w for w in wanted if w in set(research)]
        if numeric:
            _generate(
                src,
                "frontier_cs_algorithm.main",
                src / "adapters" / "frontier-cs-algorithm" / "src",
                [
                    "--source",
                    str(src),
                    "--output-dir",
                    str(HERE),
                    "--task-ids",
                    *numeric,
                    "--overwrite",
                    "--judge-docker-image",
                    PUBLISHED_JUDGE,
                ],
            )
        if named:
            _generate(
                src,
                "frontier_cs_2_0.main",
                src / "adapters" / "frontier-cs-2.0" / "src",
                [
                    "--source",
                    str(src),
                    "--output-dir",
                    str(HERE),
                    "--task-ids",
                    *named,
                    "--overwrite",
                ],
            )
        print(f"generated into {HERE}: {wanted}")


if __name__ == "__main__":
    main()
