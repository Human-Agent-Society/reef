"""Start the tutorial's harness deployment; keep serving until interrupted."""

import os
import sys
from pathlib import Path

import yaml


def main():
    tutorial = Path(__file__).resolve().parent
    work = Path(os.environ.get("REEF_HARNESS_WORK_DIR", tutorial / "work/deployment")).resolve()
    recipes = work / "recipes"
    recipes.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((tutorial / "serve.yaml").read_text())
    settings = config["reef"]
    settings["upstream_url"] = os.environ.get("REEF_UPSTREAM_URL", settings["upstream_url"])
    model = os.environ.get("REEF_UPSTREAM_MODEL", settings["upstream_model"])
    settings["upstream_model"] = model
    config["model"]["path"] = model
    settings["port"] = int(os.environ.get("REEF_PORT", "8901"))
    settings["token"] = "${REEF_TOKEN}"
    for key, name in {
        "agent_record_dir": "agent-record",
        "artifact_repository": "artifacts.git",
        "artifact_work_dir": "artifact-work",
        "artifact_cache_dir": "artifact-cache",
    }.items():
        settings[key] = str(work / name)
    config["run_dir"] = str(work / "stack")
    recipe = {key: config[key] for key in ("implementation", "model", "evolution", "data")}
    (recipes / "harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    deployment = work / "serve.yaml"
    deployment.write_text(yaml.safe_dump(config, sort_keys=False))

    env = dict(os.environ)
    env.setdefault("REEF_TOKEN", "reef-local")
    env["REEF_RECIPE_CONFIG_DIR"] = str(recipes)
    env["PYTHONPATH"] = os.pathsep.join([str(tutorial.parent.parent), str(tutorial), env.get("PYTHONPATH", "")])
    os.chdir(tutorial)
    os.execvpe(sys.executable, [sys.executable, "-m", "reef", "serve", "-c", str(deployment)], env)


if __name__ == "__main__":
    main()
