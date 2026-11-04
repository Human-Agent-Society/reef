"""The samples service's step, for tests that drive trainers and scenarios without a samples worker.

In Reef, the samples service makes a trainer's samples from its new records (``Trainer.produce_samples``); a
trainer's own cycle only reads the sample store. A test that steps a trainer or a scenario by hand calls these
before the cycle, as the samples worker would.
"""

from __future__ import annotations

from reef.scenario.scenario import Scenario
from reef.train.trainer import Trainer


def make_samples(owner: Scenario | Trainer) -> bool:
    """Make the samples of every new record of ``owner``'s trainers; return whether background work still runs."""
    trainers = [owner] if isinstance(owner, Trainer) else [item.trainer for item in owner.trainers]
    running = False
    for trainer in trainers:
        running = trainer.produce_samples().background or running
    return running
