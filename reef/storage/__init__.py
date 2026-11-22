"""The database and the work queues: the storage contracts, and their SQLite and PostgreSQL implementations.

- ``scenario`` holds the contracts of one scenario's state: its records, commit history, training samples and
  evaluations, and the scenario storage that opens them.
- ``data`` holds the contract of the data store, which keeps the data items of the whole deployment.
- ``coordination`` holds what connects the services: the work queues, ``served_models`` and the control lease.
- ``sql`` implements the three in SQLite and PostgreSQL. Database-specific code stays there.
- ``durable_files`` writes JSON files atomically and creates directories durably.

``sql`` imports the other three. ``scenario`` imports ``data``, and ``data`` imports ``coordination``. No import goes
the other way, and the contracts import no concrete adapter. Nothing in this package imports scenario coordination,
recipes, runtimes, or training. Application assembly selects a scenario storage and gives it to scenario
coordination. Records, commits, samples, generated tasks and scenario settings all go through that storage.

Import each module directly; this package exports nothing.
"""
