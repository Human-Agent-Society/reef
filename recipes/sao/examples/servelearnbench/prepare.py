"""Write a 128k YaRN configuration view of a Qwen3 checkpoint, linking its weights.

SGLang serves the view with YaRN, and Reef's Megatron model provider reads
the same ``rope_scaling`` to install the matching rotary embedding for the
actor and the critic.
"""

import argparse
import json
from pathlib import Path

CONTEXT_LENGTH = 131072
ORIGINAL_CONTEXT = 32768
SCALING_FACTOR = 4.0


def prepare_model(source: Path, destination: Path) -> None:
    """Link immutable model files and write a separate 128k configuration."""
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("The Retail long-context recipe requires a dense Qwen3 checkpoint")
    if destination.exists():
        raise ValueError("Use a fresh destination for the long-context model view")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Start from an unscaled Qwen3 checkpoint")
    theta = rope.get("rope_theta", config.get("rope_theta", 1000000.0))
    scaling = {
        "rope_type": "yarn",
        "factor": SCALING_FACTOR,
        "original_max_position_embeddings": ORIGINAL_CONTEXT,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    config.pop("rope_parameters", None)
    config["rope_theta"] = theta
    config["rope_scaling"] = scaling
    config["max_position_embeddings"] = CONTEXT_LENGTH
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.name != "config.json":
            (destination / path.name).symlink_to(path)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    prepare_model(arguments.source, arguments.destination)
