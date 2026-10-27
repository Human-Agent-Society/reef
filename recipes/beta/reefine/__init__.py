"""Reefine (beta): harness refinement from a user's training instructions.

The parts, in the standard method layout of ``reef.recipe.harness``:

``recipe``
    ``ReefineRecipe``, the harness evolution recipe with Reefine's defaults.
``proposer``
    ``TextProposer``: the served model answers a request with files, design
    first, and reviews them; without a request it learns one skill from the
    step's failing reports.
``agent``
    ``AgentProposer``: a pi session answers a request in an editable copy of
    the tree, checks and tries the change, and hands it back; it uses the
    text proposer where no agent runs. ``agent_gateway`` and ``trial`` serve
    it, and ``multimodal`` relays the provider calls of the harness and its
    trials.
``scorer``
    ``AnswerScorer``, the health task's scorer.

The example serve config (``examples/configs/serve.yaml``) supplies the health task; a deployment with
tasks of its own also supplies its own ``evolution.evaluate`` scorer.
"""

from __future__ import annotations

from recipes.beta.reefine.agent import AgentProposer
from recipes.beta.reefine.proposer import TextProposer
from recipes.beta.reefine.recipe import ReefineRecipe
from recipes.beta.reefine.scorer import AnswerScorer

__all__ = ["AgentProposer", "AnswerScorer", "ReefineRecipe", "TextProposer"]
