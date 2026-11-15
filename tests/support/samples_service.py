"""The samples service's step, for tests that drive trainers and scenarios without a samples worker.

In Reef, the samples service makes a trainer's samples from its new records (``Trainer.produce_samples``); a
trainer's own cycle only reads the sample store. A test that steps a trainer or a scenario by hand calls these
before the cycle, as the samples worker would.
"""

from __future__ import annotations

from reef.record2sample.staleness import ServedArtifacts
from reef.scenario.scenario import Scenario
from reef.train.trainer import Trainer


def make_samples(owner: Scenario | Trainer, served: ServedArtifacts | None = None) -> bool:
    """Make the samples of every new record of ``owner``'s trainers, and judge them as the samples service does;
    return whether background work still runs. A trainer's samples are judged against ``served`` (none is stale
    without it); a scenario's against the artifacts that its commit history served."""
    if isinstance(owner, Trainer):
        return owner.produce_samples(served=served).background
    running = False
    for component_trainer in owner.trainers:
        running = owner.produce_samples(component_trainer.component).background or running
    return running
