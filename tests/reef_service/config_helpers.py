"""Views of shipped examples through the production config translation paths."""

import os
from pathlib import Path
from unittest.mock import patch

from reef.service.deploy.config import load_config
from reef.service.deploy.orchestrator import resolve_deployment_config


def load_deployment(path):
    return resolve_deployment_config(load_config(path, interpolate_env=False), None, path)[0]


def deployment_layout(config):
    """Inspect generated example topology using test provider values when absent."""
    with patch.dict(
        os.environ,
        {
            "REEF_UPSTREAM_URL": os.environ.get("REEF_UPSTREAM_URL") or "http://127.0.0.1:8000",
            "REEF_UPSTREAM_MODEL": os.environ.get("REEF_UPSTREAM_MODEL") or "test-model",
        },
    ):
        return resolve_deployment_config(config, None, Path(__file__).resolve().parents[2] / "<example>")[0]


def load_harness_deployment(path):
    from reef.recipe.config import recipe_config_from_mapping

    raw = load_config(path)
    config = load_deployment(path)
    return {**config, **recipe_config_from_mapping(raw)}
