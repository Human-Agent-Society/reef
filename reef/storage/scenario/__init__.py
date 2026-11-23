"""The stored state of one scenario: the storage contracts, and the stores that combine or wrap them.

- ``store`` defines the scenario store (``ScenarioStore``), the scenario storage that opens it
  (``ScenarioStorage``), and the scenario's settings.
- ``records`` defines the record store, its results, its errors and the retention limits. ``record_items``
  implements it over the data store: each record is a record item of the scenario's owner.
- ``commits`` holds the commit record, the registration metadata and their persisted formats.
  ``commit_history`` defines the commit history and combines one with any record store.
- ``samples`` holds the training samples of a scenario and its training state.
- ``evaluations`` defines the store of the artifact evaluations that clients ask a scenario for.
- ``observer`` wraps a scenario storage to report accepted records and commits to an observer.
- ``task_import`` wraps a scenario storage, so that each scenario that opens gets its owner, its record items and
  the deployment's imported Harbor tasks.

These modules import ``reef.core``, ``reef.storage.data`` and each other. They import no SQL code.

Import each module directly; this package exports nothing.
"""
