"""The data store of a deployment: the data items of every owner, and the files and queues beside them (RFC #1028).

- ``store`` defines the data store (``DataStore``), its values and errors, and the store in process memory.
- ``item_kinds`` holds the kinds of data items and the rules of each kind's content.
- ``tasks`` holds the contracts of the published Harbor tasks of a scenario, the task limits and the task file
  store. ``task_collections`` holds benchmarks and collections of task items.
- ``artifact_items`` holds the artifacts of a workspace as items.
- ``segments`` holds segment files, which keep the large content of items outside the database.
- ``job_data`` holds job data directories: the data of a training job as read-only files.
- ``queue`` names the work queues of the data, and ``sync`` asks control to sync a scenario after a data write.

These modules import ``reef.core``, ``reef.storage.coordination`` and each other. They import no SQL code.

Import each module directly; this package exports nothing.
"""
