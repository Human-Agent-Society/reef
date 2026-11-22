Choose where a deployment stores its data
=========================================

Select the backend of each store, back the stores up, and move to a new Reef version.

:For: whoever sets up or maintains a Reef deployment
:Before you start: a deployment config (`Configuration <../reference/configuration.rst>`__)
:You will have: the stores of a deployment, their backends, and the backup and upgrade rules

A deployment keeps its state in five stores. Each has a default under ``.reef/`` and accepts a plugin:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Store
     - What it keeps
     - Setting
     - Default
   * - Scenario storage
     - The records, commits and settings of each scenario
     - ``storage.record-backend``
     - ``sqlite`` under ``storage.agent-record-dir``
   * - Artifact backend
     - The snapshots of each scenario
     - ``storage.artifact-backend``
     - ``git_lfs`` in ``storage.artifact-repository``
   * - Task file store
     - The files of Harbor task directories
     - ``storage.task-file-backend``
     - ``local`` in ``storage.task-file-dir``
   * - Data store
     - The data items of each owner
     - ``storage.data-store``
     - Follows the record backend
   * - Segment store
     - Large content of the data store, when set
     - ``storage.segment-backend``
     - Off

Put every store on persistent storage. With all backends, each scenario needs one Reef writer. A change from one backend to another moves no data: use a new deployment directory or schema for a new backend, or keep the old store readable.

A plugin is named as ``package.module:ClassName``, or as a mapping with ``module`` and ``config``. Reef makes it with the class method ``from_config``.

Choose the storage backend
--------------------------

The storage backend keeps the records, commits and settings of each scenario. The settings are the model configuration, the selected training mode, the automatic evaluation settings and the activation rule. You set them with ``POST /reef/scenarios/{scenario}/update`` and read them with ``GET /reef/scenarios/{scenario}/settings``. After a restart, the scenario uses its stored settings. If the recipe does not support the stored training mode, the scenario uses the default mode and Reef logs a warning.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``storage.record-backend``
     - Where scenario state is kept
   * - ``sqlite`` (default)
     - One SQLite database per scenario under ``agent_record_dir``.
   * - ``postgres``
     - Tables in one PostgreSQL schema. ``agent_record_dir`` holds no scenario state.
   * - a plugin reference
     - The ``ScenarioStorage`` subclass that the reference names.

**PostgreSQL.** Install the driver with ``uv pip install 'reef-infra[postgres]'`` (or ``uv pip install -e '.[postgres]'`` from a checkout), then configure:

.. code-block:: yaml

   storage:
     record-backend: postgres
     record-database-url: ${REEF_RECORD_DATABASE_URL}
     record-database-schema: reef_records
     agent-record-dir: .reef/agent-record

- Set ``REEF_RECORD_DATABASE_URL`` to a connection URL, such as ``postgresql://reef:password@db.example.com/reef?sslmode=require``. ``postgresql+psycopg://`` URLs are also accepted.
- The database must exist. Reef creates its tables in the selected schema, so the role needs permission to create that schema and its tables at the first start.
- Give independent deployments distinct schemas (default: ``reef_records``).
- PostgreSQL 14 and newer are supported.
- Archived scenarios stay in PostgreSQL under their old storage identifier, and a recreated scenario gets a new one. The retention settings apply to all active and archived stores of the schema, and preserve active records and retry hashes.

**A plugin.** Subclass ``reef.storage.scenario.ScenarioStorage`` and implement all abstract methods, including ``from_config``:

.. code-block:: yaml

   storage:
     record-backend:
       module: my_package.storage:ObjectStoreScenarioStorage
       config:
         bucket: reef-state
     coordination-backend: sqlite
     agent-record-dir: .reef/agent-record

Reef calls ``from_config(config, directory=...)`` with the ``config`` mapping and ``agent_record_dir``. A plugin holds no tables of the services, so name where they share them: ``coordination-backend: sqlite`` for processes on one host, or ``postgres`` with ``record-database-url`` for several hosts. The `Python storage contract <../reference/python-api-scenario-stores.rst#scenario-stores>`__ gives the necessary behavior. An embedded deployment gives a ``ScenarioStorage`` instance to ``Dispatcher(..., scenario_storage=...)`` instead.

Choose the artifact backend
---------------------------

The artifact backend stores the snapshots of each scenario. The default, ``git_lfs``, keeps them in the Git LFS repository that ``storage.artifact-repository`` names. Git and Git LFS must be installed.

**A plugin.** Name a subclass of ``reef.artifact.ConfigurableRepositoryBackendFactory`` in ``storage.artifact-backend``:

.. code-block:: yaml

   storage:
     artifact-backend:
       module: my_package.artifacts:ObjectStoreArtifactBackend
       config:
         bucket: reef-snapshots

Reef calls ``from_config(config, context=...)``. The `Python artifact backend contract <../reference/python-api-artifacts.rst#artifact-backends>`__ gives the necessary behavior. With a plugin backend, Reef needs no Git LFS tools. Do not set ``storage.artifact-repository``: Reef refuses that configuration, because the plugin reads its own config.

**Git LFS on Hugging Face.** Give the built-in factory with its options. Reef still reads every snapshot from its local clone, and pushes each snapshot to the Hub as it publishes it:

.. code-block:: yaml

   storage:
     artifact-backend:
       module: reef.artifact.git_lfs:GitLFSRepositoryBackendFactory
       config:
         repository: https://huggingface.co/org/reef-artifacts
         refs: branches-and-tags
         token_env: HF_TOKEN

- ``refs: branches-and-tags`` keeps each snapshot on a tag ``reef/releases/<revision>``, and the refs that move on branches under ``reef/``, because the Hub accepts no other refs. The default, ``reef``, keeps every ref under ``refs/reef/``. A repository keeps the layout it started with, and Reef refuses the other.
- ``hf://org/reef-artifacts@<revision>`` names a stored snapshot by its Git commit revision, for a serving runtime or an import. Reef's artifact IDs are independent of these revisions.
- ``token_env`` names the variable of a token that can write the repository. Reef sends it to the repository's host only, as an HTTP header, so it appears in no URL, file or log. Keep the repository private unless you mean to publish what Reef trains.
- Archiving a deleted scenario's ref works with a local repository only.

**Space of discarded checkpoints.** The training service stores each checkpoint in the repository before a commit takes it. Reef discards a checkpoint after its commit, when the sweep deletes its ended job, and when its scenario is deleted. What happens to its Git LFS objects depends on where the repository is:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Repository
     - Git LFS objects of discarded content
   * - Local (a path or a ``file://`` URL)
     - The discard deletes the ref. Control's sweep of ended work, once an hour, removes each object that no ref references. The cleanup of a deleted scenario does the same at once. Snapshots, pending snapshots and archived scenarios keep their objects. A cache entry of discarded content goes one sweep later, because another service that shares the cache directory can still read it.
   * - A remote Git server or the Hugging Face Hub
     - The objects stay on the remote, and Reef's clone in the work directory keeps a copy of each object that it uploaded. To free the remote's space, stop every Reef service that writes the repository, then remove the unreferenced objects with the server's own tools. On the Hub, use ``HfApi.list_lfs_files`` and ``HfApi.permanently_delete_lfs_files`` of ``huggingface_hub``.

.. warning::

   Share a local repository between hosts only on a filesystem whose ``flock`` reaches every host. The processes coordinate the object removal with ``flock`` on ``reef-objects.lock`` in the repository. This is safe on one host with any local filesystem, and across hosts on a Linux NFS mount, such as Amazon EFS. On SMB or a FUSE mount of object storage, the lock does not reach other hosts. Reef does not detect the filesystem.

Choose the task file store
--------------------------

A recipe that generates tasks saves each published task as a Harbor task directory. The scenario storage keeps the state of the tasks. The task file store keeps the files of the task directories, which can be text or binary.

The default, ``local``, keeps the files in ``storage.task-file-dir`` (default: ``.reef/task-files``). Archiving a scenario moves its files under ``archived/`` in that directory.

**A plugin.** Name a subclass of ``reef.storage.tasks.TaskFileStore`` in ``storage.task-file-backend``:

.. code-block:: yaml

   storage:
     task-file-backend:
       module: my_package.tasks:ObjectStoreTaskFileStore
       config:
         bucket: reef-tasks
     task-file-dir: .reef/task-files

Reef calls ``from_config(config, directory=...)`` with the ``config`` mapping and ``storage.task-file-dir``, which a remote store can use as a local cache. The `Python task file store contract <../reference/python-api-data-stores.rst#task-file-stores>`__ gives the necessary behavior.

Choose the data store
---------------------

The data store keeps the deployment's data items with their metadata: the trajectories and tasks that clients create, and a record item for each inference call and report (`Data items <../reference/http-api-data-items.rst#http-api-data-items>`__). Each item belongs to an owner, and its metadata names the scenarios that use it. The task file store keeps the files of task items.

The data store grows until clients delete items: archiving a scenario does not delete its items, and record items stay when record retention removes the records from a scenario's record store.

Without ``storage.data-store``, the store follows ``storage.record-backend``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Record backend
     - Data store
   * - ``sqlite``
     - ``<agent-record-dir>/data/data-store.sqlite3``
   * - ``postgres``
     - The schema ``reef_data`` in ``storage.record-database-url``. Deployments that use one database share their items.
   * - a plugin
     - As ``sqlite``.

To use another database or schema, or your own store, name a subclass of ``reef.storage.data.DataStore``:

.. code-block:: yaml

   storage:
     data-store:
       module: reef.storage.sql_data:PostgresDataStore
       config:
         database_url: ${REEF_DATA_DATABASE_URL}
         schema: reef_data

Keep large content in segment files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data store's database grows with each trajectory and each record item. With ``storage.segment-backend``, a SQL data store keeps that content in segment files instead. A segment is one JSON Lines file per owner. The database keeps the place of each line, and a read fetches only the lines that it needs.

- A write of new trajectory items with at least 1 MiB of content, such as an upload, goes to one segment at once.
- Every 5 minutes, the data service moves the content that the database still holds into segments: the messages of record items, the trajectory items of smaller writes, and trajectory items from before you set the backend. It moves an owner's content when at least 64 MiB of it waits, oldest first, in segments of at least 64 MiB.
- In the same run, the data service merges segments of less than 1 MiB into segments of at least 64 MiB, when an owner has at least 64 MiB in them.
- A segment that no item uses for 24 hours is deleted.

``local`` keeps the segments in ``<agent-record-dir>/segments``. Only the services of one host can share this directory. To share segments between hosts, name a subclass of ``reef.storage.segments.SegmentStore``, for example a store in object storage:

.. code-block:: yaml

   storage:
     segment-backend:
       module: my_package.segments:ObjectStoreSegmentStore
       config:
         bucket: reef-data

Reef calls ``from_config(config, directory=...)`` with ``<agent-record-dir>/segments``, which a remote store can use as a local cache. The `Python segment store contract <../reference/python-api-data-stores.rst#segment-stores>`__ gives the necessary behavior.

.. warning::

   Keep the backend set after you set it. A data store without it cannot read content that it moved, and it raises an error for such an item.

Back up a deployment
--------------------

A backup holds the scenario state, the artifact repository and the task file store. Reef rebuilds ``artifact_work_dir`` and ``artifact_cache_dir`` from the repository, so they need no backup.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Store
     - What to back up
   * - Scenario storage, ``sqlite``
     - ``agent_record_dir``, copied while the service is stopped
   * - Scenario storage, ``postgres``
     - A dump of the configured schema
   * - Scenario storage, a plugin
     - The plugin's own store
   * - Artifact backend
     - The artifact repository (``storage.artifact_repository``); for a remote repository, the Git remote; for a plugin, the plugin's own store
   * - Task file store
     - ``task_file_dir`` with the ``local`` store, or the plugin's own store

With PostgreSQL, a remote artifact repository and a remote task file store, a new host with the same configuration continues every scenario from the database and the two remote stores.

Some recipes have their own directories, such as the harness evolution recipe's ``evolution.state_dir``, which holds each scenario's history and step records. These directories are not scenario storage. Keep them on a persistent volume.

Upgrade from a Reef that wrote older formats
--------------------------------------------

Stop every process of the deployment before you start a new Reef version.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Store
     - On an older format
   * - Data store
     - Upgrades its own tables in one transaction, before it reads an item. It records their version in the table ``data_schema_version``. A process of the earlier Reef then fails on the upgraded tables. A database of a later version stops the start with ``a later Reef wrote them``, and Reef changes nothing.
   * - Commit records and scenario metadata
     - Not upgraded. This Reef reads ``reef-commit/9`` and ``reef-scenario/8`` only. A scenario with an older format does not load, and its error names the format.
   * - Record database and coordination store
     - Not upgraded. A PostgreSQL record schema of another version stops the start with ``unsupported PostgreSQL record schema version``.

Reef does not change or delete older data. To move from a Reef that wrote older formats of the stores that do not upgrade, use a new deployment directory and, for PostgreSQL, a new schema. Then create the scenarios again.
