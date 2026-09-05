"""Copy a serve file's recipe sections where the recipe registry reads them.

configs/serve.yaml (or configs/serve-native.yaml, the argument run.sh passes
for the native variant) carries both the deployment (`reef:`, `services:`)
and the harness_evolve recipe itself. The registry loads named recipes from
REEF_RECIPE_CONFIG_DIR, so run.sh drops the four recipe sections plus optional
execution selectors and executor profiles there before starting Reef, and the
task list beside them for run.py.
"""

import argparse
import json
from pathlib import Path

import yaml

RECIPE_SECTIONS = ("implementation", "model", "evolution", "data")


def materialize(serve: Path, work: Path) -> None:
    """Write ``work/recipes/harness_evolve.yaml`` and ``work/tasks.json`` from ``serve``."""
    config = yaml.safe_load(serve.read_text())
    recipe = {key: config[key] for key in RECIPE_SECTIONS}
    recipe.update({key: config[key] for key in ("execution", "executors") if key in config})
    (work / "recipes").mkdir(parents=True, exist_ok=True)
    (work / "recipes" / "harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    (work / "tasks.json").write_text(json.dumps(recipe["evolution"]["tasks"], indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy a serve file's recipe sections where the recipe registry reads them."
    )
    parser.add_argument("serve", nargs="?", default="configs/serve.yaml", type=Path, help="the serve file to read")
    parser.add_argument("--work", default="work", type=Path, help="the state directory run.sh and run.py share")
    args = parser.parse_args(argv)
    materialize(args.serve, args.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
