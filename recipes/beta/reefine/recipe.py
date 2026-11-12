"""Reefine's recipe: the harness evolution recipe with Reefine's defaults.

``ReefineRecipe`` answers a person's training requests, one job per
request (``training_mode: manual``), with ``ReefineMethod``. It only sets
defaults and builds nothing. ``section_defaults`` declares the defaults that
the loader merges under the deployment's ``evolution`` section: the method,
the seed beside this module, an agent that runs unless the section is null,
and an evaluation that runs the candidate alone on the deployment's tasks,
scored by ``AnswerScorer``, and keeps it when every task meets the floor. On
the health task the floor checks that the tree still works, not that the
requested change does; the job's design and review and the person judge that.
The defaults that depend on the adapter are set when the recipe is made: by
default, a person activates a tree that changes an extension (the
``activation`` rule's ``review_kinds``).

``evolution.agent`` says where the coding agent that answers a request on
pi runs (``reef.train.harness_backend.agent.settings``); Reefine runs its
agent on the harness's own binary unless the section names another.
``ReefineMethod`` reads its own two settings from ``evolution.method.config``:
``trial_timeout_s``, the limit of each trial, and ``multimodal``, the gateway
for images, embeddings, speech and decisions
(``recipes.beta.reefine.multimodal``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from recipes.beta.reefine.answer import EXTENSION_ADAPTER
from recipes.beta.reefine.method import ReefineMethod
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.core.plugins import PluginConfigError
from reef.evaluation.activation import activation_reference
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe

#: The first tree of a new Reefine scenario: the starter skill.
SEED_DIRECTORY = Path(__file__).with_name("seed")
#: The parts whose change waits for a person when the adapter's tree has them: an extension runs in the harness's
#: process.
REVIEWED_PARTS = ("extension",)


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(HarnessEvolutionRecipe):
    """Refine a harness once per instruction; a person activates a tree that changes an extension.

    Values: the fields of ``HarnessEvolutionRecipe``; the recipe adds none.
    A recipe that the loader makes has Reefine's ``section_defaults`` under
    the deployment's keys. Every value also meets these conditions, which
    ``__post_init__`` sets: on an adapter that plays task directories, a
    ``[health]`` evaluation task is the health task directory
    (``HEALTH_TASK_DIRECTORY``). ``evolution.agent`` is ``None`` off pi, where
    no agent answers requests; on pi its adapter is pi, and its ``binary`` is
    ``evolution.binary`` unless the section names one. ``activation`` is
    never ``None``: without an ``activation`` section it is the ``always``
    rule whose ``review_kinds`` are the ``REVIEWED_PARTS`` of the adapter's
    tree.
    Purpose: a deployment names it as its recipe, for Reefine's defaults.
    Override ``evolution.evaluation.policy`` to compare scores against the
    head tree, and ``training-mode`` to learn from reports too.
    """

    #: The deployment's keys win. The agent runs unless ``evolution.agent`` is null.
    section_defaults: ClassVar[Mapping[str, Any]] = {
        "evolution": {
            "method": {"module": f"{ReefineMethod.__module__}:{ReefineMethod.__qualname__}", "config": {}},
            "seed": str(SEED_DIRECTORY),
            "agent": {},
            "evaluation": {
                "policy": "floor",
                "scorer": f"{AnswerScorer.__module__}:{AnswerScorer.__qualname__}",
            },
        }
    }

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")

    def __post_init__(self) -> None:
        evolution, evaluation = self.evolution, self.evaluation
        try:
            descriptor = get_adapter(evolution.adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        if descriptor.takes_task_directory:
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            tasks = tuple(HEALTH_TASK_DIRECTORY if task.startswith("[health]") else task for task in evaluation.tasks)
            if tasks != evaluation.tasks:
                object.__setattr__(self, "evaluation", replace(evaluation, tasks=tasks))
        # The agent answers requests on pi alone; on another adapter the served model does, so no agent is
        # built, jailed or warned about. On pi it runs unless the section is null, on the harness's own binary.
        agent = evolution.agent if evolution.adapter == EXTENSION_ADAPTER else None
        if agent is not None:
            if not isinstance(agent, Mapping):
                raise RecipeConfigError("evolution.agent must be a mapping")
            if agent.get("adapter", EXTENSION_ADAPTER) != EXTENSION_ADAPTER:
                raise RecipeConfigError(f"Reefine's agent is {EXTENSION_ADAPTER}: evolution.agent.adapter must be it")
            if "trial_timeout_s" in agent:
                raise RecipeConfigError(
                    "evolution.agent has no trial_timeout_s: the limit of a trial is the method's setting "
                    "evolution.method.config.trial_timeout_s"
                )
            if evolution.binary and "binary" not in agent:
                agent = {"binary": evolution.binary, **agent}
        if agent != evolution.agent:
            object.__setattr__(self, "evolution", replace(evolution, agent=agent))
        if self.activation is None:
            # An extension runs in the harness's process with the user's privileges, so by default a person
            # activates a tree that changes one.
            reviewed = [part for part in REVIEWED_PARTS if part in descriptor.layout.parts]
            try:
                rule = activation_reference({"module": "always", "config": {"review_kinds": reviewed}})
            except PluginConfigError as exc:
                raise RecipeConfigError(str(exc)) from exc
            object.__setattr__(self, "activation", rule)
        super().__post_init__()


__all__ = ["SEED_DIRECTORY", "ReefineRecipe"]
