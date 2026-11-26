Choose where a deployment stores its data
=========================================

Use local stores for a single-host deployment. Use shared storage when services
run on several hosts. Keep every required store persistent before running training.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Store
     - Setting
     - Default
   * - Scenario state
     - ``storage.record-backend``
     - SQLite under ``storage.agent-record-dir``.
   * - Work queues and service coordination
     - ``storage.coordination-backend``
     - Follows a built-in record backend.
   * - Artifacts
     - ``storage.artifact-backend``
     - Git LFS at ``storage.artifact-repository``.
   * - Harbor task files
     - ``storage.task-file-backend``
     - Local files under ``storage.task-file-dir``.
   * - Data items
     - ``storage.data-store``
     - Follows the record backend.
   * - Large item content
     - ``storage.segment-backend``
     - Off; content stays in the data store.

Changing a backend does not migrate data. Use a new deployment directory or
schema, or keep the previous store readable. Plugins use a dotted class reference
or a mapping with ``module`` and ``config``.

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

Scenario settings survive restarts. Read them with
``GET /reef/scenarios/{scenario}/settings`` and update them through the scenario API.
An unsupported stored training mode falls back to the recipe default with a warning.

Choose the artifact backend
---------------------------

The default ``git_lfs`` backend needs Git and Git LFS. It stores snapshots in
``storage.artifact-repository``. A custom backend reads its own configuration:

.. code:: yaml

   storage:
     artifact-backend:
       module: my_package.artifacts:ObjectStoreArtifactBackend
       config:
         bucket: reef-snapshots

Do not also set ``storage.artifact-repository`` for a custom backend.
Reef refuses that combination. Plugins implement
``ConfigurableRepositoryBackendFactory``; see ``reef.artifact.repository``.

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

**Discarded content.** Local repositories remove unreferenced LFS objects during
the hourly work sweep, or immediately during scenario cleanup.
Snapshots, pending snapshots and archived scenarios retain their objects.
Discarded cache entries remain until the next sweep.

Remote Git servers retain uploaded objects. Stop every writer before using the
server's cleanup tools. On the Hub, the relevant operations are
``HfApi.list_lfs_files`` and ``HfApi.permanently_delete_lfs_files``.
Archiving a deleted scenario's ref is supported only for local repositories.

A shared local repository needs filesystem locks that coordinate every host.
Reef uses ``flock`` on ``reef-objects.lock`` and does not detect unsafe mounts.
Do not use a mount whose locks are local to one host.

Choose the task file store
--------------------------

Harbor task files can be text or binary. The default store uses
``storage.task-file-dir`` (default ``.reef/task-files``).
Archived scenario files move under its ``archived/`` directory.

For a remote store, configure a ``TaskFileStore`` plugin:

.. code:: yaml

   storage:
     task-file-backend:
       module: my_package.tasks:ObjectStoreTaskFileStore
       config:
         bucket: reef-tasks
     task-file-dir: .reef/task-files

A remote plugin can use ``task-file-dir`` as its local cache.
The extension contract is in ``reef.storage.data.tasks.TaskFileStore``.

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

Set ``storage.segment-backend`` to move large record and trajectory content
out of the SQL database. The database retains references to segment files.

- Trajectory writes of at least 1 MiB go directly to a segment.
- Every five minutes, the data service moves or combines content when an owner
  has at least 64 MiB waiting.
- Segments unused for 24 hours are deleted.

``local`` uses ``<agent-record-dir>/segments`` on one host.
For several hosts, configure a shared ``SegmentStore`` plugin:

.. code:: yaml

   storage:
     segment-backend:
       module: my_package.segments:ObjectStoreSegmentStore
       config:
         bucket: reef-data

Keep this backend configured after enabling it. Without it, reads of migrated
content fail. See ``reef.storage.data.segments.SegmentStore`` for the plugin contract.

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
