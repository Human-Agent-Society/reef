"""The CORAL weight-training recipe: what a deployment names in ``serve.yaml``.

Binds the CORAL stages (``CoralRecord2Record`` here, and ``CoralTasks`` and
``CoralBatches`` in :mod:`recipes.beta.coral.processor`) to the grouped
relative-reward training machinery. CORAL sibling groups have the same shape
as TTT-Discover steps — comparison sets of scored rollouts from one shared
starting point — so the recipe reuses the ``tttd`` training objective (grouped
adaptive-entropic leave-one-out advantages) and the ``tttd`` Slime loss
family rather than duplicating either. The recipe's only method setting is the
group barrier, ``group_size`` in ``pipeline.record2record.config``.

Deployment reference: ``recipes.beta.coral.recipe:CoralRecipe``. Run from
the repository checkout, which also provides the ``tttd`` registrations this
recipe reuses; ``reef serve`` finds it beside the example config.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Self

# The tttd training objective and loss-family reference live in the repository
# cookbook; importing them is what registers both names this recipe binds.
import recipes.tttd  # noqa: F401  (registration side effect)
from recipes.beta.coral.processor import CoralBatches, CoralTasks, attempt_release_ids, sibling_group
from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.record2record import GroupDecision, ReportContext, ReportGrouping, ReportRecord2Record
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec

logger = logging.getLogger(__name__)


class CoralGrouping(ReportGrouping):
    """The siblings of one parent commit form one group, and each attempt's commit hash is its slot.

    A group is ready once ``group_size`` siblings are present. A group whose
    attempts span releases is discarded and listed in ``discarded_groups``.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.mixed_release_groups: dict[str, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return sibling_group(context)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {release_id for member in members for release_id in attempt_release_ids(member)}
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self.mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(members),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self.mixed_release_groups.items())
            ],
        }


@dataclass(frozen=True)
class CoralGroupSettings(PluginSettings):
    """The config of ``CoralRecord2Record``."""

    group_size: int = config_option(
        4,
        help=(
            "The scored siblings of one parent commit that complete its group. "
            "A parent with fewer scored children never trains."
        ),
    )

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")


class CoralRecord2Record(ReportRecord2Record):
    """CORAL's record stage: sibling groups in the order that they close; a terminal report owns its sources.

    Config: ``group_size`` (``CoralGroupSettings``). CORAL sibling counts are
    dynamic, so the barrier is a recipe setting, not a CORAL invariant.
    """

    def __init__(self, context: PipelineContext, settings: CoralGroupSettings) -> None:
        super().__init__(context, CoralGrouping(settings.group_size), exclusive_sources=True)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, CoralGroupSettings.parse(config, plugin_class=cls))


@dataclass(frozen=True, kw_only=True)
class CoralRecipe(WeightTrainingRecipe):
    """Train on CORAL sibling groups with the ``tttd`` objective.

    The stages are ``CoralRecord2Record``, ``CoralTasks``, and ``CoralBatches``.
    """

    name: str = "coral"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="tttd",
            # Sibling groups train as one optimizer step, as TTTD's grid does.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=CoralRecord2Record, record2task=CoralTasks, task2batch=CoralBatches)
