Choose where a deployment stores its data
=========================================

Use local stores on one host and shared stores across hosts. Keep them persistent
before training. Changing a backend does not migrate existing data.
See `storage settings <../reference/configuration-settings.rst#storage>`__ for
all paths and plugin options.

Choose the storage backend
--------------------------

SQLite is the default. It stores scenario state under ``storage.agent-record-dir``
and supports services sharing files on one host. For several hosts, use PostgreSQL:

.. code:: bash

   uv pip install 'reef-infra[postgres]'

.. code:: yaml

   storage:
     record-backend: postgres
     record-database-url: ${REEF_RECORD_DATABASE_URL}
     record-database-schema: reef_records

Use PostgreSQL 14+ and a role that can create schemas and tables.
Coordination uses ``<record schema>_coordination``. A custom record backend must
explicitly select a SQLite or PostgreSQL coordination backend.

Choose the data store
---------------------

The data store holds records, trajectories, tasks and metadata. By default:

- SQLite uses ``<agent-record-dir>/data/data-store.sqlite3``.
- PostgreSQL uses schema ``reef_data`` in the record database.
- Custom record backends use SQLite for data.

Different record schemas still share ``reef_data``. To isolate deployments,
configure ``storage.data-store`` with another database or schema:

.. code:: yaml

   storage:
     data-store:
       module: reef.storage.sql.data:PostgresDataStore
       config:
         database_url: ${REEF_DATA_DATABASE_URL}
         schema: reef_data_acme

All services must also reach task and segment files. Defaults are
``storage.task-file-dir`` and, when enabled, ``<agent-record-dir>/segments``.
Use shared-store plugins across hosts. Keep segment storage configured after
migration; disabling it makes migrated content unreadable.

Choose the artifact backend
---------------------------

The default ``git_lfs`` backend needs Git LFS. For a private Hugging Face repository:

.. code:: yaml

   storage:
     artifact-backend:
       module: reef.artifact.git_lfs:GitLFSRepositoryBackendFactory
       config:
         repository: https://huggingface.co/org/reef-artifacts
         refs: branches-and-tags
         token_env: HF_TOKEN

Supply a write token through ``HF_TOKEN``. Existing repositories cannot change
ref layouts. Shared local repositories need filesystem locks across all writers;
Reef uses ``flock`` and cannot detect unsafe mounts.

Record retention
----------------

``storage.agent-record-retention-max-bytes`` limits record bodies across all owners
and scenarios in a data store. It is not a disk quota: indexes and delayed
cleanup need extra space. SQLite reuses pages without shrinking the file.

Evictions appear in the ``pipeline`` object of the status. Reef skips pending reports whose referenced
records were removed and logs a warning. Keep separate backups for longer retention.

Back up a deployment
--------------------

Stop all writers before taking a consistent backup. Include:

- Scenario, data and coordination databases, including custom schemas.
- The artifact repository and Git LFS objects, or plugin storage.
- Task and segment files.
- Training checkpoints, backend ``state-dir`` and harness ``evolution.state_dir``.

Artifact working copies and caches are replaceable; repository storage is not.
Tinker artifacts also require their remote checkpoints and originating service.

Upgrade from a Reef that wrote older formats
--------------------------------------------

Stop every process and back up all stores before upgrading.
The data store upgrades transactionally; older processes then refuse it.
A newer-than-supported schema fails without changes.

Commits and scenario metadata have no migration. This version reads
``reef-commit/9`` and ``reef-scenario/8`` only. Unsupported record or coordination
schemas also fail. Use a new deployment directory/schema for unsupported formats,
then recreate scenarios. Reef keeps the old data.
