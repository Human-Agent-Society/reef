Choose where a deployment stores its data
=========================================

Use local stores for a single-host deployment. Use shared storage when services
run on several hosts. Keep every required store persistent before running training.

Changing a backend does not migrate data. Keep previous stores readable or
use a new deployment directory/schema. See `storage settings
<../reference/configuration-settings.rst#storage>`__ for backend and plugin syntax.

Choose the storage backend
--------------------------

``sqlite`` stores each scenario's records, commits and settings under
``storage.agent-record-dir``. It supports services sharing files on one host.
Use PostgreSQL for services on several hosts:

.. code:: bash

   uv pip install 'reef-infra[postgres]'

.. code:: yaml

   storage:
     record-backend: postgres
     record-database-url: ${REEF_RECORD_DATABASE_URL}
     record-database-schema: reef_records

Set the URL to an existing PostgreSQL 14 or newer database.
``postgresql://`` and ``postgresql+psycopg://`` URLs are accepted.
The database role needs permission to create the schema and tables on first use.
Use distinct record schemas for independent deployments.

Coordination tables use ``<record schema>_coordination``. A custom record backend
must explicitly select ``coordination-backend: sqlite`` or ``postgres``.
All services must reach those tables. See `Split services <split-services.rst>`__.

Choose the artifact backend
---------------------------

The default ``git_lfs`` backend stores snapshots at
``storage.artifact-repository`` and requires Git LFS. A custom
``storage.artifact-backend`` must provide its own configuration; do not also
set a nondefault ``artifact-repository``.

To publish Git LFS snapshots to a private Hugging Face repository:

.. code:: yaml

   storage:
     artifact-backend:
       module: reef.artifact.git_lfs:GitLFSRepositoryBackendFactory
       config:
         repository: https://huggingface.co/org/reef-artifacts
         refs: branches-and-tags
         token_env: HF_TOKEN

Set ``HF_TOKEN`` to a token with repository write access.
``branches-and-tags`` uses refs accepted by the Hub. The default ``reef`` layout
uses ``refs/reef/``. An existing repository cannot switch layouts.
Reef reads its local clone and pushes each published snapshot.
``hf://org/reef-artifacts@<revision>`` identifies a Git revision, not a Reef artifact ID.

Local repositories remove unreferenced LFS content hourly or during scenario
cleanup. Published, pending and archived snapshots retain their content.
Remote servers retain uploads until their own cleanup; stop all writers first.
Scenario-ref archival is supported only for local repositories.

Shared local repositories require filesystem locks that coordinate every host.
Reef uses ``flock`` and cannot detect unsafe mounts.

Choose the task file store
--------------------------

Harbor task files default to ``storage.task-file-dir`` (``.reef/task-files``).
Archived files move under ``archived/``. For remote storage, select a
``storage.task-file-backend`` plugin; it can use ``task-file-dir`` as a cache.
See the `TaskFileStore contract <../../reef/storage/data/tasks.py>`__.

Choose the data store
---------------------

The data store holds records, trajectories, tasks and their metadata.
Each item belongs to an owner; its metadata identifies the scenarios using it.
Archiving a scenario does not delete these items.

Without ``storage.data-store``:

- SQLite uses ``<agent-record-dir>/data/data-store.sqlite3``.
- PostgreSQL uses schema ``reef_data`` in ``storage.record-database-url``.
- A plugin record backend uses the SQLite data store.

Deployments using the same PostgreSQL database share ``reef_data`` by default,
even when their record schemas differ. Choose another schema for separate items:

.. code:: yaml

   storage:
     data-store:
       module: reef.storage.sql.data:PostgresDataStore
       config:
         database_url: ${REEF_DATA_DATABASE_URL}
         schema: reef_data_acme

Record retention
~~~~~~~~~~~~~~~~

``storage.agent-record-retention-max-bytes`` limits stored record bodies across
all owners and scenarios, including archived scenarios. Deployments sharing a
data store share this budget.

This is not a disk quota. Indexes need extra space, and writes can exceed the
budget between sweeps. SQLite reuses freed pages without shrinking its file.

Evictions appear in ``processor`` status as ``record_data_incomplete``,
``evicted_record_count`` and ``evicted_body_bytes``.
Reef skips pending reports whose referenced records were removed and logs a warning.
See `Status <../reference/http-api-status.rst#status>`__.

Keep large content in segment files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``storage.segment-backend: local`` to move large content out of SQL into
``<agent-record-dir>/segments``. Multi-host deployments need a shared
`SegmentStore plugin <../../reef/storage/data/segments.py>`__.

Keep that backend configured: disabling it makes migrated content unreadable.
Unused segments are deleted after 24 hours.

Back up a deployment
--------------------

Stop the deployment's writers before taking a consistent backup.
Include every configured persistent store:

.. list-table::
   :header-rows: 1

   * - State
     - Back up
   * - SQLite scenario, data and coordination stores
     - ``storage.agent-record-dir`` and any separately configured data-store files.
   * - PostgreSQL stores
     - Scenario, coordination and data schemas, including custom databases or schemas.
   * - Artifacts
     - The Git repository and its LFS objects, or the plugin's storage.
   * - Task and segment files
     - Their local directories or remote stores.
   * - Backend and recipe state
     - Training checkpoints, ``state-dir`` and harness ``evolution.state_dir``, where configured.

Reef can rebuild artifact working copies and caches from the artifact repository.
They do not replace a repository backup. Remote checkpoints referenced by Tinker
artifacts must remain available in the originating account and service.

Upgrade from a Reef that wrote older formats
--------------------------------------------

Stop every deployment process before upgrading. Keep a backup of the previous stores.

.. list-table::
   :header-rows: 1

   * - Store
     - Upgrade behavior
   * - Data store
     - Upgrades tables in one transaction. Older Reef processes then refuse them. A newer-than-supported version fails without changes.
   * - Commits and scenario metadata
     - No migration. This version reads ``reef-commit/9`` and ``reef-scenario/8`` only.
   * - Record and coordination stores
     - No migration. An unsupported PostgreSQL record schema prevents startup.

For unsupported older formats, use a new deployment directory and PostgreSQL
schema, then recreate scenarios. Reef does not delete the previous data.
