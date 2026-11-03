"""The samples service on the scenarios of this process: it makes their trainers' samples and runs their requested
task-generation rounds.

``LocalSampleProducers`` makes the samples of a trainer of a scenario with this process's copy of the scenario. When
the process runs control, it uses the loaded scenario and then wakes the trainer's worker, which builds the next job
from the sample store. When control runs in another process, it loads the scenario on demand, selects the training
mode that control stored, and control learns of the samples from the samples queue.

``LocalTaskGenerations`` gives a requested round to the trainer of this process's copy of the scenario, loaded on
demand, and asks for a selection of the trainer, so that the ``SamplesWorker`` starts the round and stores its tasks.
"""

from __future__ import annotations

from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.record2sample.requested_generation import RequestedRoundResult
from reef.record2sample.service import ProducedSamples, SampleProducers, TaskGenerations, request_selection
from reef.scenario.registry import ScenarioRegistry
from reef.storage.work_queue import WorkQueue


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
            # Control changes the mode in its own copy and stores it; this copy follows at each selection.
            self.registry.apply_stored_training_mode(scenario)
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


class LocalTaskGenerations(TaskGenerations):
    """Gives requested task-generation rounds to the trainers of the scenarios that ``registry`` holds.

    Values: ``registry`` holds the scenarios, loaded on demand; ``queue`` is the work queue, where it asks for a
    selection of a trainer whose stage took a new round.

    Purpose: the dispatcher gives it to the task-generation worker that it runs in this process.
    """

    def __init__(self, registry: ScenarioRegistry, queue: WorkQueue) -> None:
        self.registry = registry
        self.queue = queue

    def advance_request(
        self, scenario: str, component: str, request_id: str, report_ids: tuple[str, ...]
    ) -> RequestedRoundResult | None:
        current = self.registry.get_or_create(scenario, allow_implicit_creation=False)
        if current is None:
            raise ValueError(f"unknown scenario {scenario!r}")
        trainers = [entry.trainer for entry in current.trainers if entry.component == component]
        if not trainers:
            raise ValueError(f"scenario {scenario!r} has no trainer for component {component!r}")
        (trainer,) = trainers
        # A stage that a reload or an unload rebuilt holds no request: it takes the round again and continues the
        # same generation from the task store.
        if trainer.request_task_generation(request_id, report_ids):
            request_selection(self.queue, scenario, component)
        return trainer.take_task_generation_result(request_id)


__all__ = ["LocalSampleProducers", "LocalTaskGenerations"]
