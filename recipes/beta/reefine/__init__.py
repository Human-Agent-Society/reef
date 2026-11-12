"""Reefine (beta): harness refinement from a user's training requests.

The parts:

``recipe``
    ``ReefineRecipe``, the harness evolution recipe with Reefine's defaults:
    the method and its config, the seed, the floor evaluation and the
    agent's isolation (``evolution.agent``). It holds configuration only.
``method``
    ``ReefineMethod``: a job with a request answers it, a job without one
    learns one skill from its failing traces.
``answer``
    The text answer: the served model writes the design and the files, and
    reviews them. Its prompts are the templates in ``prompts/``, and each
    adapter's facts are in ``harness_facts/``.
``agent``
    The agent answer: a pi session changes the job's ``harness/``, checks
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
