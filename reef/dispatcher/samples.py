"""The samples service on the scenarios that this process holds: it makes their trainers' samples.

``LocalSampleProducers`` makes the samples of a trainer of a scenario that this process has loaded. After it stores
them, it wakes the trainer's worker, which builds the next job from the sample store.
"""

from __future__ import annotations

from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.record2sample.service import SampleProducers
from reef.scenario.registry import ScenarioRegistry


class LocalSampleProducers(SampleProducers):
    """Makes samples with the trainers of the scenarios that ``registry`` holds loaded in this process.

    Values: ``registry`` holds the scenarios; ``local_workers`` runs the trainers' workers, which it wakes after it
    stores samples.

    Purpose: the dispatcher gives it to the samples workers that it runs in this process.
    """

    def __init__(self, registry: ScenarioRegistry, local_workers: LocalTrainingWorkers) -> None:
        self.registry = registry
        self.local_workers = local_workers

    def produce(self, scenario: str, component: str) -> bool:
        current = self.registry.get_loaded(scenario)
        if current is None:
            return False
        background = False
        for component_trainer in current.trainers:
            if component_trainer.component == component and component_trainer.trainer.trains:
                background = component_trainer.trainer.produce_samples()
        self.local_workers.wake(current)
        return background


__all__ = ["LocalSampleProducers"]
