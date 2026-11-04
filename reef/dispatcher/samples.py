"""The samples service on the scenarios of this process: it makes their trainers' samples.

``LocalSampleProducers`` makes the samples of a trainer of a scenario with this process's copy of the scenario. When
the process runs control, it uses the loaded scenario and then wakes the trainer's worker, which builds the next job
from the sample store. When control runs in another process, it loads the scenario on demand, and control learns of
the samples from the samples queue.
"""

from __future__ import annotations

from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.record2sample.service import ProducedSamples, SampleProducers
from reef.scenario.registry import ScenarioRegistry


class LocalSampleProducers(SampleProducers):
    """Makes samples with the trainers of the scenarios that ``registry`` holds loaded in this process.

    Values: ``registry`` holds the scenarios; ``local_workers`` runs the trainers' workers, which it wakes after it
    stores samples, or is None when control runs in another process.

    Purpose: the dispatcher gives it to the samples workers that it runs in this process.
    """

    def __init__(self, registry: ScenarioRegistry, local_workers: LocalTrainingWorkers | None) -> None:
        self.registry = registry
        self.local_workers = local_workers

    def produce(self, scenario: str, component: str) -> ProducedSamples:
        if self.local_workers is not None:
            current = self.registry.get_loaded(scenario)
        else:
            current = self.registry.get_or_create(scenario, allow_implicit_creation=False)
        if current is None:
            # Control asks again when the scenario loads; a scenario that does not exist has no samples.
            return ProducedSamples(changed=False, background=False)
        produced = ProducedSamples(changed=False, background=False)
        for component_trainer in current.trainers:
            if component_trainer.component == component and component_trainer.trainer.trains:
                produced = component_trainer.trainer.produce_samples()
        # Only new input gives the trainer's worker something to build; a selection that brought none leaves it.
        if self.local_workers is not None and produced.changed:
            self.local_workers.wake(current)
        return produced


__all__ = ["LocalSampleProducers"]
