"""The work queues of a deployment's data (RFC #1028): the data service's queue, and control's sync queue.

- ``DATA_QUEUE``: the data service takes its items. A ``record_copy`` item copies the new records of one scenario
  to record items, an ``artifact_copy`` item makes the artifact items of one scenario, a ``data_import`` item
  imports the files at a URI, and a ``data_export`` item writes items to an archive at a URI.
- ``DATA_SYNC_QUEUE``: control takes its items. A ``data_sync`` item syncs one loaded scenario with the items of
  its owner, after the data service wrote them.

Each queue has one service that takes its items, so a worker of one service never waits behind the other's work.
"""

from __future__ import annotations

#: The queue of the data service.
DATA_QUEUE = "data"
#: The queue of control's data syncs.
DATA_SYNC_QUEUE = "data_sync"

#: The ``DATA_QUEUE`` kind that copies the new records of one scenario to record items.
RECORD_COPY_KIND = "record_copy"
#: The ``DATA_QUEUE`` kind that makes the artifact items of one scenario's artifacts.
ARTIFACT_COPY_KIND = "artifact_copy"
#: The ``DATA_QUEUE`` kind that imports the files at a URI as items of one owner.
DATA_IMPORT_KIND = "data_import"
#: The ``DATA_QUEUE`` kind that writes items of one owner to a tar.gz at a URI.
DATA_EXPORT_KIND = "data_export"
#: The ``DATA_SYNC_QUEUE`` kind that syncs one scenario with the items of its owner.
DATA_SYNC_KIND = "data_sync"

__all__ = [
    "ARTIFACT_COPY_KIND",
    "DATA_EXPORT_KIND",
    "DATA_IMPORT_KIND",
    "DATA_QUEUE",
    "DATA_SYNC_KIND",
    "DATA_SYNC_QUEUE",
    "RECORD_COPY_KIND",
]
