"""Selected training integrations own deployment topology and runtime construction."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from reef.service.deploy.config import DeployConfigError
from reef.train.deployment import TrainingDeployment

# Built-ins are references, so discovery does not import execution dependencies.
_BUILTINS = {"slime": "reef.train.slime_backend.launch:SlimeDeployment"}


def training_deployment_for(name: str | None) -> TrainingDeployment:
    """Load one built-in, installed entry point, or module:class deployment definition."""
    selected = name or "slime"
    reference = _BUILTINS.get(selected, selected)
    if ":" not in reference:
        matches = tuple(entry_points(group="reef.training_backends", name=selected))
        if len(matches) != 1:
            reason = "unknown" if not matches else "ambiguous"
            raise DeployConfigError(
                f"{reason} training backend {selected!r}; install its integration or use module:class"
            )
        reference = matches[0].value
    module, _, attribute = reference.partition(":")
    try:
        definition = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise DeployConfigError(f"cannot load training backend {selected!r}: {exc}") from exc
    if not isinstance(definition, type) or not issubclass(definition, TrainingDeployment):
        raise DeployConfigError("training backend must name a TrainingDeployment class")
    return definition()
