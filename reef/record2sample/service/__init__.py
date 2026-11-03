"""The samples service: it turns a scenario's new records into samples through the ``samples`` work queue.

``worker`` holds the queue protocol (``request_selection``), the ``SampleProducers`` contract and the
``SamplesWorker``. ``task_generation`` holds the items of requested task-generation rounds
(``file_task_generation``), the ``TaskGenerations`` contract and the ``TaskGenerationWorker``. Like the generator
service (``reef.record2sample.generator``), it is a service around the record-to-sample contracts, not one of them.
"""

from reef.record2sample.service.task_generation import (
    TASK_GENERATION_KIND,
    TaskGenerations,
    TaskGenerationWorker,
    file_task_generation,
    task_generation_item,
)
from reef.record2sample.service.worker import (
    BACKGROUND_POLL_SECONDS,
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    ProducedSamples,
    SampleProducers,
    SamplesWorker,
    request_selection,
    selection_key,
)

__all__ = [
    "BACKGROUND_POLL_SECONDS",
    "RECORD_SELECTION_KIND",
    "SAMPLES_QUEUE",
    "TASK_GENERATION_KIND",
    "ProducedSamples",
    "SampleProducers",
    "SamplesWorker",
    "TaskGenerationWorker",
    "TaskGenerations",
    "file_task_generation",
    "request_selection",
    "selection_key",
    "task_generation_item",
]
