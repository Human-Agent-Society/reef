"""The recipe contract and its machinery — what a method implements.

A recipe is *how records and their feedback become a new version of the
weights or the harness*. This package holds everything a method binds to:

- ``base`` — ``Recipe`` (the default record-only kind and base contract:
  ``build``, ``build_artifact_validator``, ``build_surface``, the
  ``CheckpointStrategy``) and ``WeightTrainingRecipe`` with its
  ``WeightTrainingSpec`` (step preparer, loss family, data processor).
- ``registry`` — the kind table (``register_kind``, ``recipe_class_for``),
  ``build_named_recipe`` for the recipe a deployment names, and
  ``RecipeRegistry``, the closed instance table scenarios bind to.
- ``config_fields`` — one dataclass field as the whole configuration surface
  for one setting (YAML key, env fallback, typed parser).
- ``config`` — recipe-config YAML loading; ``errors`` — the error family.

Candidate evaluation is part of this contract, not a separate subsystem:
every ``Recipe`` carries ``candidate_evaluation``, built from the top-level
``evaluation`` section of its config, and the trainer runs it between
prepare and settle. Its types (``CandidateEvaluator``, ``CandidateSelector``,
``CandidateEvaluationPlugin``, ``AlwaysSelect``) live in
``reef.train.evaluation`` because the trainer, backends, and runtime bind to
them too — the recipe is what chooses and configures them.

The bundled methods are packages under the sibling ``recipes`` tree
(``recipes.sao``, ``recipes.tttd``, …); ``reef/__init__`` imports them last,
which registers their kinds. Nothing else in ``reef`` imports a method package.

The recipe decides; it does not execute or deliver. ``build`` returns a
``Trainer``, ``build_artifact_validator`` an admission policy,
``build_surface`` serving capabilities, and the scenario binds them with the
runtime as peers. The dependency points down only — this package imports
``scenario``, ``train``, ``surface``, and ``runtime``, never the reverse,
and never ``reef.train.slime_backend`` at module scope.
"""

from reef.recipe.base import Recipe, WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config import load_recipe_config
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError, ScenarioRecipeConflict, ScenarioRecipeError, UnknownScenarioRecipe
from reef.recipe.registry import RecipeRegistry, build_named_recipe, build_recipe, recipe_class_for, register_kind

__all__ = [
    "Recipe",
    "RecipeConfigError",
    "RecipeRegistry",
    "ScenarioRecipeConflict",
    "ScenarioRecipeError",
    "UnknownScenarioRecipe",
    "WeightTrainingRecipe",
    "WeightTrainingSpec",
    "build_named_recipe",
    "build_recipe",
    "config_field",
    "load_recipe_config",
    "recipe_class_for",
    "register_kind",
]
