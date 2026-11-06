"""The CORAL weight-training recipe: what a deployment names in ``serve.yaml``.

Binds the CORAL stages (``CoralSamples`` and ``CoralJobData`` in
:mod:`recipes.beta.coral.processor`) to the grouped
relative-reward training machinery. CORAL sibling groups have the same shape
as TTT-Discover steps — comparison sets of scored rollouts from one shared
starting point — so the recipe reuses the ``tttd`` training objective (grouped
adaptive-entropic leave-one-out advantages) and the ``tttd`` Slime loss
family rather than duplicating either. The recipe's only method setting is the
group barrier, ``group_size`` in ``pipeline.record2sample.config``.

Deployment reference: ``recipes.beta.coral.recipe:CoralRecipe``. Run from
the repository checkout, which also provides the ``tttd`` registrations this
recipe reuses; ``reef serve`` finds it beside the example config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The tttd training objective and loss-family reference live in the repository
# cookbook; importing them is what registers both names this recipe binds.
import recipes.tttd  # noqa: F401  (registration side effect)
from recipes.beta.coral.processor import CoralJobData, CoralSamples
from reef.recipe.base import WeightTrainingRecipe
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class CoralRecipe(WeightTrainingRecipe):
    """Train on CORAL sibling groups with the ``tttd`` objective.

    The stages are ``CoralSamples`` and ``CoralJobData``.
    """

    name: str = "coral"

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {kind: {"objective": "tttd"} for kind in SAMPLE_KINDS}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=CoralSamples, sample2job=CoralJobData)
