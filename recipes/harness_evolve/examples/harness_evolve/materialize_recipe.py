"""Copy serve.yaml's recipe sections where the recipe registry reads them.

serve.yaml carries both the deployment (`reef:`, `services:`) and the
harness_evolve recipe itself. The registry loads named recipes from
REEF_RECIPE_CONFIG_DIR, so run.sh drops those four sections there before
starting Reef, and the task list beside them for run.py.
"""

import json
from pathlib import Path

import yaml

RECIPE_SECTIONS = ("kind", "model", "evolution", "data")

config = yaml.safe_load(Path("serve.yaml").read_text())
recipe = {key: config[key] for key in RECIPE_SECTIONS}

Path("work/recipes/harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
Path("work/tasks.json").write_text(json.dumps(recipe["evolution"]["tasks"], indent=2) + "\n")
