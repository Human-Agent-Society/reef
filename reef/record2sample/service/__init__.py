"""The samples service: it turns a scenario's new records into samples through the ``samples`` work queue.

``worker`` holds the queue protocol (``request_selection``), the ``SampleProducers`` contract and the
``SamplesWorker``. Like the generator service (``reef.record2sample.generator``), it is a service around the
record-to-sample contracts, not one of them.
"""

from reef.record2sample.service.worker import (
    BACKGROUND_POLL_SECONDS,
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    SampleProducers,
    SamplesWorker,
    request_selection,
    selection_key,
)

__all__ = [
    "BACKGROUND_POLL_SECONDS",
    "RECORD_SELECTION_KIND",
    "SAMPLES_QUEUE",
    "SampleProducers",
    "SamplesWorker",
    "request_selection",
    "selection_key",
]
