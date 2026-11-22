Scenario and record store API
=============================

The storage contracts that hold a scenario's commits, settings and records: ``ScenarioStorage``, its session ``ScenarioStore``, the commit record, and ``RecordStore`` with its audit reads and retention.

`Task and data stores <python-api-data-stores.rst>`__ gives the stores of tasks, task files, data items and segments. `Storage <../user-guide/storage.rst>`__ shows how a deployment selects each backend.

Scenario stores
---------------

Three abstract bases hold a scenario's state. A storage implementation subclasses them explicitly and overrides every abstract member.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Base
     - Module
     - Role
   * - ``RecordStore``
     - ``reef.storage.scenario.records``
     - Appends, reads and audits records (`Record storage and audit`_).
   * - ``ScenarioStore``
     - ``reef.storage.scenario.store``
     - One scenario's session: a ``RecordStore`` with the committed scenario state. It settles each step and its record progress together.
   * - ``ScenarioStorage``
     - ``reef.storage.scenario.store``
     - Opens the sessions and keeps all per-scenario state.

``reef.storage.scenario.store`` also exports ``ScenarioSettings`` and ``ScenarioStoreConflictError``. The bundled implementations are ``ItemRecordStore``, ``CommitHistoryScenarioStore``, ``SQLiteScenarioStorage`` and ``PostgresScenarioStorage``.

An existing ``EvaluationStore`` supports unkeyed evaluations. To support start IDs, implement ``KeyedEvaluationStore`` from ``reef.storage.scenario.evaluations``. Reserve the ID, save the evaluation, and save its pending work in one transaction.

.. code:: python

   from pathlib import Path

   from reef.control.dispatcher import Dispatcher
   from reef.storage.sql.coordination import memory_coordination
   from reef.storage.sql.sqlite import SQLiteScenarioStorage

   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   dispatcher = Dispatcher(
       recipe,
       repository_backend_factory,
       scenario_storage=storage,
       coordination=memory_coordination(),
   )

``Dispatcher`` and ``build_default_dispatcher`` require both arguments:

- ``scenario_storage``: the dispatcher passes it to scenario creation, closes it on shutdown, and never selects a storage implementation itself. Deployment assembly builds the storage that ``storage.record_backend`` names: SQLite, PostgreSQL, or a ``ScenarioStorage`` plugin.
- ``coordination``: the store of the services' work queues, served models and control lease. One process that runs every service and keeps no queue state across restarts passes ``memory_coordination()``. Deployment assembly opens the shared store that ``storage.coordination_backend`` names.

Direct ``Scenario`` construction requires ``store=...``, one session that a ``ScenarioStorage`` opened. The supplied trainer must use that store's ``records``. Read committed history through ``scenario.store.history()`` and the persistence capability through ``scenario.store.durable``.

The storage contract
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files, data=None)``
     - Class method that builds the storage that a deployment names in ``storage.record_backend``. ``config`` is the plugin's own mapping. ``directory`` is ``storage.agent_record_dir``. ``task_files`` is the ``TaskFileStore`` of the deployment (`Task file stores <python-api-data-stores.rst#task-file-stores>`__). The storage owns it: ``archive`` and ``close`` also apply to it. ``data`` is the deployment's data store, which keeps each scenario's records as record items. The storage shares it and does not close it.
   * - ``durable``
     - Whether new sessions and settings survive a process restart.
   * - ``open(scenario)``
     - Open an independently owned session (`The session contract`_). Raise ``ScenarioDeletedError`` for a scenario that ``archive`` archived, so a late record makes no new store of a deleted scenario.
   * - ``read_settings(scenario)``
     - The scenario's ``ScenarioSettings``: ``training_mode``, ``artifact_evaluation``, ``activation`` (the scenario's own activation rule), ``owner``, ``serving_config``, ``serves`` and ``record2sample_config``. Defaults for a scenario with none.
   * - ``write_settings(scenario, settings)``
     - Replace the settings durably. The caller serializes writers of one scenario. Raise ``ScenarioDeletedError`` for a scenario that ``archive`` archived.
   * - ``sample_store(scenario, session)``
     - The data items that the scenario trains on, with the training state of its open ``session``, as a ``SampleStore``. The default returns ``None``: a storage without a data store has none, and its trainers keep their samples in memory.
   * - ``archive(scenario)``
     - Retire the scenario's records, commits, tasks and settings, and remove the scenario from the uses of its owner's items. Return the archive locations. A repeated call changes nothing more. A scenario ID is not used again after its delete.
   * - ``close()``
     - Release shared resources; repeated calls are safe.

The bundled storages:

- ``SQLiteScenarioStorage()`` without a directory keeps records, commits and settings in memory. With a directory, each scenario has one SQLite database file for its records, commits, settings, tasks and uploaded trajectories.
- ``PostgresScenarioStorage(url, directory, schema=...)`` keeps commits (the ``scenario_commit`` table) and settings (the ``scenario_setting`` table) in the schema. Install ``reef-infra[postgres]`` for the PostgreSQL driver. A deployment selects it with ``storage.record_backend: postgres`` (`Choose the storage backend <../user-guide/storage.rst#choose-the-storage-backend>`__).

The session contract
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``records``
     - A ``RecordStore`` implementation that preserves append deduplication, ordered replay, scenario isolation, and audit reads.
   * - ``durable``
     - Whether committed history survives a session or process restart. A durable store requires a repository backend that supports staged snapshots.
   * - ``tasks``
     - The ``TaskStorage`` that opens the task stores of the scenario (`Task stores <python-api-data-stores.rst#task-stores>`__). The stores close with the session and survive restarts when ``durable``.
   * - ``training_state``
     - The ``TrainingState`` of the scenario: the train counts of its samples, the number of committed jobs of each trainer that trained each one, and the records that each trainer's stages consumed. It closes with the session and survives restarts when ``durable``.
   * - ``evaluations``
     - The ``EvaluationStore`` of the artifact evaluations that clients asked the scenario for. It closes with the session and survives restarts when ``durable``.
   * - ``use_snapshots(snapshots)``
     - Locate the snapshot of each record that ``history`` reads. The default does nothing, for a store that holds the snapshots itself.
   * - ``history()``
     - ``CommitRecord`` values in commit number order: training steps (including inactive artifacts), activations, and new training starts. This is authoritative after an ambiguous commit error.
   * - ``training_run_position()``
     - The commit number of the last ``train_from`` commit and the count of training commits after it, for experiment run numbering.
   * - ``append_commit(*, expected_commit_number, commit)``
     - Persist the ``CommitRecord`` and its consumption progress, and return the canonical accepted record. Its ``commit_number`` must equal ``expected_commit_number + 1`` (`Commits`_).
   * - ``recover(*, checkpoint)``
     - Accept a checkpoint ``CommitRecord`` (``None`` at initial registration), reconcile it with history, and return the head ``CommitRecord``, or ``None`` for a fresh scenario.
   * - ``close()``
     - Release the session's record and commit resources; repeated calls are safe. Each session closes after its trainer, also when construction or recovery fails.

Commits
~~~~~~~

A ``CommitRecord`` carries the operation, component selection, private storage locator, algorithm state, record watermark, consumed IDs, checkpoint and pending flags, metrics and training job identity. The operations include ``training``, ``activate``, ``train_from``, ``import``, ``sync``, ``serve``, ``stop_serving`` and ``restore``:

- A selection maps component names directly to artifact IDs.
- ``activate`` and ``serve`` record a complete selection without publishing another artifact snapshot. ``activate`` and ``train_from`` carry no trainer state.
- ``sync`` selects its new artifact for serving and training in one commit.
- ``restore`` records a complete prior active selection and leaves training starts unchanged.

``append_commit`` must validate the current commit number atomically:

- Two different commits prepared from the same commit number cannot both succeed.
- An identical retry returns the original record, even after later commits. Only ``recorded_at`` is excluded from the comparison.
- A new commit with a stale expected commit number, or a retry with conflicting content, raises ``ScenarioStoreConflictError`` without advancing the store.
- A failure after the append can leave the commit stored. Retry the exact commit or recover the session; do not infer rollback from an exception.

``CommitHistoryScenarioStore`` takes a ``RecordStore``, a ``TaskStorage`` (``tasks=``) and a ``CommitHistory``. The bundled record stores give them from ``task_storage()`` and ``commit_history()``. The store appends each commit, and the train count of each sample that the commit lists, in one write transaction of the scenario's database. Direct writes through a ``CommitHistory`` bypass this contract. Recovery uses committed ``consumed_ids`` and separate skipped-batch receipts to rebuild processor memory without repeating work. It does not modify stored bodies.

Artifact bytes and backend head movement remain outside ``ScenarioStore``. ``ScenarioCommitter`` owns their order around store settlement, including inactive artifacts, activation, new training starts, and checkpoint reconciliation (`Commit ordering <../advanced_topics/state-model.rst#commit-ordering>`__). Step validation does not provide distributed leases or authorize multiple active training writers.

Record storage and audit
------------------------

``reef.storage.scenario.records`` defines ``RecordStore``, the record result and error types, and ``RecordRetention``. ``reef.RecordStore`` is the same class.

``reef.storage.scenario.record_items.ItemRecordStore`` is the ``RecordStore`` of one scenario over the data store. It appends each record as its record item, and reads the record items that the scenario made, in item sequence order. ``SQLiteScenarioStorage`` and ``PostgresScenarioStorage`` give one to each session. Both take a data store (``data=``). Without one, the SQLite storage keeps its records in ``<directory>/data``, and the PostgreSQL storage in the schema ``<schema>_data``.

An older database keeps its records in the tables ``agent_record``, ``record_node``, ``consumed_agent_record``, ``record_consumption`` and ``record_eviction``. The storage drops these tables when the record item of the newest record of each active scenario is in the data store: the PostgreSQL storage when it starts, the SQLite storage at the first open of each scenario. Otherwise it keeps them and logs a warning: run Reef 83a6f3ab once to copy the records.

Appending
~~~~~~~~~

``RecordStore.append_many(items)`` atomically appends an ordered batch from one scenario and returns an ``AppendResult`` for each input. Conflicts roll back the batch. Identical retries and evicted records follow ``append_result`` semantics. SQLite and PostgreSQL implement this with one write transaction. Other adapters must implement atomic batch append to support ``POST /reef/records/batch``; the default raises ``NotImplementedError`` before writing.

Training commits persist consumption progress without changing record visibility or deleting bodies.

Reading
~~~~~~~

``read``, ``replay``, ``replay_page`` and ``count`` read all retained records.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Result
   * - ``read_for_audit(scenario, agent_record_id)``
     - A ``StoredRecord``, or ``None`` if no body is retained in that scenario.
   * - ``audit_page(scenario, *, after_sequence=0, limit=256, request_type=None)``
     - A bounded tuple of ``StoredRecord`` entries, in append order, including consumed bodies, of ``request_type`` when given. Advance the cursor with the last entry's ``sequence``.
   * - ``loss(scenario)``
     - Durable ``RecordLoss`` totals: ``record_count``, ``body_bytes``, ``first_sequence`` and ``last_sequence`` of the record items that retention deleted (``DataStore.record_loss``). Counts include consumed and unconsumed records.
   * - ``trajectory(scenario, agent_record_id)``
     - The record IDs of the trajectory that the record ends, oldest first: the earlier calls with the same API format and tools whose messages the record's messages extend, one per prefix, and then the record. A retry with the same messages replaces the call that it retries. A call that retention deleted is left out. A record without a message list, or a store that keeps no trajectories, gives a trajectory of one call; ``()`` for a record that is not stored.

``StoredRecord`` contains ``sequence`` and ``item`` (the original ``AgentRecord``).

- Reading a record does not change a consumer's progress.
- A missing body was purged or never stored; the read API does not say which.
- Use the commit history's per-step ``consumed_ids`` to determine consumption, including intentional skips. It does not prove that every named record produced a model update.

For example, inspect one trace without making it available to training again:

.. code:: python

   entry = scenario.records.read_for_audit(scenario.name, record_id)
   if entry is not None:
       payload = entry.item.payload
       references = entry.item.references

Retention
~~~~~~~~~

The HTTP service runs background retention at startup and every 60 seconds. It deletes the oldest record items, across owners, only when they hold more than a shared 20 GiB budget (``DataStore.prune_records``). The budget counts each record item's row and the trajectory nodes that only it needs. It counts and deletes only the record items whose ``origin.source`` is ``inference``; it keeps the calls of evaluation rollouts. `Configuration <configuration.rst>`__ gives the limits.

Retention logs nothing per record. A trainer reports the loss of its scenario in its metrics and status. Concurrent writes can exceed the budget between sweeps, and indexes and other metadata also need disk space.

An embedded deployment has two options:

- ``dispatcher.prune_record_archives(RecordRetention(max_bytes=20 * 1024**3))`` runs one sweep over the dispatcher's data store and serializes it with scenario archival. ``Dispatcher`` construction starts no maintenance task.
- ``create_app(dispatcher, record_retention=RecordRetention(...))`` enables the service's background retention.
