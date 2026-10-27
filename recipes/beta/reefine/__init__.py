"""Reefine (beta): harness refinement from a user's training instructions.

The parts:

``recipe``
    ``ReefineRecipe``, the harness evolution recipe with Reefine's defaults:
    the method, the seed, the floor evaluation, the agent's isolation
    (``evolution.agent``) and the multimodal gateway.
``method``
    ``ReefineMethod``: a step with a request answers it, a step without one
    learns one skill from its failing traces.
``answer``
    The text answer: the served model writes the design and the files, and
    reviews them. Its prompts are the templates in ``prompts/``, and each
    adapter's facts are in ``harness_facts/``.
``agent``
    The agent answer: a pi session changes the step's ``harness/``, checks
    and tries the change, and stops. ``agent_gateway`` and ``trial`` serve
    it, and ``multimodal`` relays the provider calls of the harness and its
    trials.
``scorer``
    ``AnswerScorer``, the health task's scorer.

The example serve config (``examples/configs/serve.yaml``) supplies the health task; a deployment with
tasks of its own also supplies its own ``evolution.evaluation.scorer``.
"""

from __future__ import annotations

from recipes.beta.reefine.method import ReefineMethod
from recipes.beta.reefine.recipe import ReefineRecipe
from recipes.beta.reefine.scorer import AnswerScorer

__all__ = ["AnswerScorer", "ReefineMethod", "ReefineRecipe"]
