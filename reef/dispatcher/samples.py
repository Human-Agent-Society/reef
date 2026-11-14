"""The samples service on the scenarios of this process: it makes their trainers' samples.

``LocalSampleProducers`` makes the samples of a trainer of a scenario with this process's copy of the scenario. When
the process runs control, it uses the loaded scenario and then wakes the trainer's worker, which builds the next job
from the sample store. When control runs in another process, it loads the scenario on demand, and loads it again when
control changed the scenario's ``record2sample_config``. Control learns of the samples from the samples queue.
"""

from __future__ import annotations

from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.execution.workers import ItemFailedError
from reef.record2sample.service import ProducedSamples, SampleProducers, Selection
from reef.scenario.registry import ScenarioRegistry
from reef.train.job_queue import SCENARIO_DELETED_ERROR


class UndeletedSelection(Selection):
    """``selection`` of a trainer of ``scenario``, whose check before each write also checks that a delete did not
    mark the scenario (``ScenarioRegistry.was_deleted``).

    Values: ``selection`` is the attempt that runs; ``registry`` reads the mark; ``scenario`` is the selection's
    scenario. ``task_id`` is the task ID of ``selection``.

    Purpose: ``LocalSampleProducers`` gives it to the trainer, so a selection that runs past a delete's mark writes
    nothing after it: ``check_lease`` raises ``ItemFailedError`` with ``SCENARIO_DELETED_ERROR``, and the item fails
    for good. A write that passed its check before the mark lands before control archives the storage, because
    control waits for the samples service to answer its ``delete`` selection, which runs after this one.
    """

    def __init__(self, selection: Selection, registry: ScenarioRegistry, scenario: str) -> None:
        super().__init__(selection.task_id)
        self.selection = selection
        self.registry = registry
        self.scenario = scenario

    def progress(self, records_read: int, samples_created: int) -> None:
        self.selection.progress(records_read, samples_created)

    def check_lease(self) -> None:
        self.selection.check_lease()
        if self.registry.was_deleted(self.scenario):
            raise ItemFailedError(SCENARIO_DELETED_ERROR)

    def requested_by(self, trigger: str) -> bool:
        return self.selection.requested_by(trigger)


class LocalSampleProducers(SampleProducers):
    """Makes samples with the trainers of the scenarios that ``registry`` holds loaded in this process.

    Values: ``registry`` holds the scenarios; ``local_workers`` runs the trainers' workers, which it wakes after it
    stores samples, or is None when control runs in another process.

    Purpose: the dispatcher gives it to the samples workers that it runs in this process.
    """

    def __init__(self, registry: ScenarioRegistry, local_workers: LocalTrainingWorkers | None) -> None:
        self.registry = registry
        self.local_workers = local_workers

    def produce(self, scenario: str, component: str, selection: Selection) -> ProducedSamples:
        if self.local_workers is not None:
            current = self.registry.get_loaded(scenario)
        elif selection.requested_by("delete") and self.registry.drop_deleted(scenario):
            # Control's cleanup of the deleted scenario asks: this process dropped its instance and its backend.
            current = None
        else:
            current = self.registry.get_or_create(scenario, allow_implicit_creation=False)
            if current is not None:
                stored = self.registry.stored_settings(scenario).record2sample_config
                if stored != current.record2sample_config:
                    # Control changed the stages' config after this process loaded the scenario.
                    current = self.registry.reload(scenario)
        if current is None:
            # Control asks again when the scenario loads; a scenario that does not exist has no samples.
            return ProducedSamples(changed=False, background=False)
        produced = ProducedSamples(changed=False, background=False)
        for component_trainer in current.trainers:
            if component_trainer.component == component and component_trainer.trainer.trains:
                produced = current.produce_samples(component, UndeletedSelection(selection, self.registry, scenario))
        # Only new input gives the trainer's worker something to build; a selection that brought none leaves it.
        if self.local_workers is not None and produced.changed:
            self.local_workers.wake(current)
        return produced


__all__ = ["LocalSampleProducers"]
