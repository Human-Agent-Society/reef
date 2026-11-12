"""Reefine's recipe: the harness evolution recipe with Reefine's defaults.

``ReefineRecipe`` answers a person's training requests, one job per
request (``training_mode: manual``), with ``ReefineMethod``. It only sets
defaults: it merges them into the deployment's ``evolution`` section and
builds nothing. The defaults are the method, the seed beside this module, and
an evaluation that runs the candidate alone on the deployment's tasks, scored
by ``AnswerScorer``, and keeps it when every task meets the floor. On the
health task the floor checks that the tree still works, not that the
requested change does; the job's design and review and the person judge that.
By default, a person activates a tree that changes an extension (the
``activation`` rule's ``review_kinds``).

``evolution.agent`` says where the coding agent that answers a request on
pi runs (``reef.train.harness_backend.agent.settings``); Reefine adds
``trial_timeout_s`` to it, the limit of each trial, and runs its agent on the
harness's own binary unless the section names another. ``multimodal`` is
Reefine's own key: the gateway for images, embeddings, speech and decisions
(``recipes.beta.reefine.multimodal``). The recipe gives both to the method as
its config (``evolution.method.config``); a deployment that names
``HarnessEvolutionRecipe`` with ``ReefineMethod`` writes them there itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recipes.beta.reefine.answer import EXTENSION_ADAPTER
from recipes.beta.reefine.method import DEFAULT_TRIAL_TIMEOUT_S, ReefineMethod
from recipes.beta.reefine.multimodal import MultimodalSettings
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.core.plugins import PluginConfigError
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.train.harness_backend import Method

#: The first tree of a new Reefine scenario: the starter skill.
SEED_DIRECTORY = Path(__file__).with_name("seed")
#: Reefine's defaults for the ``evolution`` section and its ``evaluation`` section; the deployment's keys win.
EVOLUTION_DEFAULTS: Mapping[str, Any] = {
    "method": f"{ReefineMethod.__module__}:{ReefineMethod.__qualname__}",
    "seed": str(SEED_DIRECTORY),
}
#: The parts whose change waits for a person when the adapter's tree has them: an extension runs in the harness's
#: process.
REVIEWED_PARTS = ("extension",)
EVALUATION_DEFAULTS: Mapping[str, Any] = {
    "policy": "floor",
    "scorer": f"{AnswerScorer.__module__}:{AnswerScorer.__qualname__}",
}


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(HarnessEvolutionRecipe):
    """Refine a harness once per instruction; a person activates a tree that changes an extension.

    Values: the fields of ``HarnessEvolutionRecipe``, read with Reefine's
    defaults merged in; the recipe adds no field that is not configuration.
    ``evolution.method`` names ``ReefineMethod`` unless the deployment names
    another method; its config holds ``trial_timeout_s`` (from
    ``evolution.agent.trial_timeout_s``) and ``multimodal`` (from
    ``evolution.multimodal``). ``evolution.agent`` is ``None`` off pi, where
    no agent answers requests.
    Purpose: a deployment names it as its recipe, for Reefine's defaults.
    Override ``evolution.evaluation.policy`` to compare scores against the
    head tree, and ``training-mode`` to learn from reports too.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        section = settings.get("evolution")
        if not isinstance(section, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        evaluation = section.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            raise RecipeConfigError("evolution.evaluation must be a mapping")
        adapter = str(section.get("adapter", EXTENSION_ADAPTER))
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        evaluation = {**EVALUATION_DEFAULTS, **evaluation}
        tasks = evaluation.get("tasks")
        if descriptor.takes_task_directory and isinstance(tasks, Sequence) and not isinstance(tasks, str):
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            evaluation["tasks"] = [
                HEALTH_TASK_DIRECTORY if str(task).startswith("[health]") else task for task in tasks
            ]
        own = {key: value for key, value in section.items() if key not in ("agent", "multimodal")}
        # The agent answers requests on pi alone; on another adapter the served model does, so no agent is
        # built, jailed or warned about. On pi it runs unless the section is null, on the harness's own binary.
        agent = section.get("agent", {}) if adapter == EXTENSION_ADAPTER else None
        trial_timeout_s: Any = DEFAULT_TRIAL_TIMEOUT_S
        if agent is not None:
            if not isinstance(agent, Mapping):
                raise RecipeConfigError("evolution.agent must be a mapping")
            if agent.get("adapter", EXTENSION_ADAPTER) != EXTENSION_ADAPTER:
                raise RecipeConfigError(f"Reefine's agent is {EXTENSION_ADAPTER}: evolution.agent.adapter must be it")
            trial_timeout_s = agent.get("trial_timeout_s", DEFAULT_TRIAL_TIMEOUT_S)
            if (
                isinstance(trial_timeout_s, bool)
                or not isinstance(trial_timeout_s, (int, float))
                or trial_timeout_s <= 0
            ):
                raise RecipeConfigError("evolution.agent.trial_timeout_s must be a positive number")
            binary = section.get("binary")
            agent = {**({"binary": binary} if binary else {}), **agent}
        multimodal = section.get("multimodal")
        try:
            # Read now, so a deployment with a bad section refuses to start; the method reads it again.
            MultimodalSettings.from_config(multimodal, values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        try:
            method = Method.reference(own.get("method", EVOLUTION_DEFAULTS["method"]), setting="evolution.method")
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc
        if issubclass(method.plugin_class, ReefineMethod):
            # Reefine's own keys of the section are the method's config; keys that the deployment wrote there win.
            method_config = {"trial_timeout_s": float(trial_timeout_s), **method.config}
            if multimodal is not None:
                method_config = {"multimodal": dict(multimodal), **method_config}
            own["method"] = {
                "module": f"{method.plugin_class.__module__}:{method.plugin_class.__qualname__}",
                "config": method_config,
            }
        reviewed = [part for part in REVIEWED_PARTS if part in descriptor.layout.parts]
        return super()._recipe_kwargs(
            {
                **settings,
                "evolution": {**EVOLUTION_DEFAULTS, **own, "agent": agent, "evaluation": evaluation},
                # An extension runs in the harness's process with the user's privileges, so by default a person
                # activates a tree that changes one.
                "activation": settings.get("activation", {"module": "always", "config": {"review_kinds": reviewed}}),
            },
            values,
        )


__all__ = ["SEED_DIRECTORY", "ReefineRecipe"]
