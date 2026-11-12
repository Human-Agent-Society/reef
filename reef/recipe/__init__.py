"""The recipe contract and its machinery — what a method implements.

A recipe is *how records and their feedback become a new version of the
weights or the harness*. This package holds everything a method binds to:

- ``base`` — ``Recipe`` (the default record-only recipe and base contract:
  its stages, its report type and ``build_components``) and
  ``WeightTrainingRecipe``, with its default job parameters.
- ``harness`` — ``HarnessEvolutionRecipe``, the recipe contract every harness
  evolution method extends; it assembles ``reef.train.harness_backend``.
- ``composite`` — ``CompositeRecipe`` binds one recipe per snapshot component
  and runs their trainers side by side in one scenario.
- ``loading`` — the one loader: ``load_recipe`` makes the recipe value of a
  class from its config, ``build_recipe`` from a class reference and
  ``build_named_recipe`` from a deployment preset. A recipe class declares
  its configuration and does not load itself.
- ``registry`` — dotted class resolution (``recipe_class_for``).
- ``config_fields`` — one dataclass field as the whole configuration surface
  for one setting (YAML key, env fallback, shared typed parser),
  ``section_field`` for a field that holds a parsed config section, and
  ``reader_field`` for a field that a ``ConfigReader`` reads from the config.
- ``config`` — recipe-config YAML loading; ``errors`` — the error family.

The activation rule is part of this contract, not a separate subsystem:
every ``Recipe`` carries ``activation``, built from the top-level
``activation`` section of its config, and the scenario asks it whether an
artifact that training makes serves. The contract lives in
``reef.core.activation``; the built-in rules live in ``reef.evaluation``.

Supervised fine-tuning (``sft``) is bundled; other learning
methods live outside the core package. The repository's sibling
``recipes`` tree contains cookbook implementations, but Reef neither imports
them at boot nor ships them in its wheel. Deployments select one explicitly by
its dotted ``package.module:ClassName`` reference.

The recipe decides; it does not execute or deliver. It builds no trainer: the
scenario factory builds each scenario's trainers from the recipe's values
(``reef.scenario.factory.read_trainer_specs``). ``build_components`` returns
the snapshot's components with their kinds, admission checks and client file
trees, and the scenario binds them with the runtime as peers. Scenario coordination consumes recipes. Recipes depend on ``train``, ``harness``,
``artifact``, ``storage``, and ``runtime``; none imports this package. Concrete
weight backends are loaded only when a weight-training recipe builds them.
"""

from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.composite import CompositeRecipe
from reef.recipe.config import load_recipe_config
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.loading import build_named_recipe, build_recipe, load_recipe
from reef.recipe.registry import recipe_class_for

__all__ = [
    "CompositeRecipe",
    "Recipe",
    "RecipeConfigError",
    "WeightTrainingRecipe",
    "build_named_recipe",
    "build_recipe",
    "config_field",
    "load_recipe",
    "load_recipe_config",
    "recipe_class_for",
]
