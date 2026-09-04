"""Copy the serve file's recipe sections where the recipe registry reads them.

serve.yaml (or serve-native.yaml, the argument run.sh passes for the native
variant) carries both the deployment (`reef:`, `services:`) and the
harness_evolve recipe itself. The registry loads named recipes from
REEF_RECIPE_CONFIG_DIR, so run.sh drops those four sections there before
starting Reef, and the task list beside them for run.py.
"""

import json
import sys
from pathlib import Path

import yaml

RECIPE_SECTIONS = ("implementation", "model", "evolution", "data")

serve = Path(sys.argv[1] if len(sys.argv) > 1 else "serve.yaml")
config = yaml.safe_load(serve.read_text())
recipe = {key: config[key] for key in RECIPE_SECTIONS}

Path("work/recipes/harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
Path("work/tasks.json").write_text(json.dumps(recipe["evolution"]["tasks"], indent=2) + "\n")
