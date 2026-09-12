"""The recipe contract and its machinery — what a method implements.

A recipe is *how records and their feedback become a new version of the
weights or the harness*. This package holds everything a method binds to:

- ``base`` — ``Recipe`` (the default record-only recipe and base contract:
  ``build``, ``build_artifact_validator``, ``build_surface``, the
  ``CheckpointStrategy``) and ``WeightTrainingRecipe`` with its
  ``WeightTrainingSpec`` (step preparer, loss family, data processor).
- ``checkpoint_strategy`` — policy for which committed steps need durable artifacts.
- ``cordis`` — ``CordisRecipe`` assembles the harness evolution backend.
- ``registry`` — dotted class resolution (``recipe_class_for``) and
  ``build_named_recipe`` for a deployment preset.
- ``config_fields`` — one dataclass field as the whole configuration surface
  for one setting (YAML key, env fallback, typed parser).
- ``config`` — recipe-config YAML loading; ``errors`` — the error family.

Candidate evaluation is part of this contract, not a separate subsystem:
every ``Recipe`` carries ``candidate_evaluation``, built from the top-level
``evaluation`` section of its config, and the trainer runs it between
prepare and settle. Shared candidate contracts live in ``reef.core.evaluation``; built-in policies
and configuration live in ``reef.train.evaluation``. The recipe chooses and
configures the policy.

Learning methods live outside the core package. The repository's sibling
``recipes`` tree contains cookbook implementations, but Reef neither imports
them at boot nor ships them in its wheel. Deployments select one explicitly by
its dotted ``package.module:ClassName`` reference.

The recipe decides; it does not execute or deliver. ``build`` returns a
``Trainer``, ``build_artifact_validator`` an admission policy,
``build_surface`` serving capabilities, and the scenario binds them with the
runtime as peers. Scenario coordination consumes recipes. Recipes depend on ``train``, ``harness``,
``surface``, ``storage``, and ``runtime``; none imports this package. Concrete
weight backends are loaded only when a weight-training recipe builds them.
"""

from reef.recipe.base import Recipe, WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config import load_recipe_config
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for

__all__ = [
    "Recipe",
    "RecipeConfigError",
    "WeightTrainingRecipe",
    "WeightTrainingSpec",
    "build_named_recipe",
    "build_recipe",
    "config_field",
    "load_recipe_config",
    "recipe_class_for",
]
