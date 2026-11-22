Task and data store API
=======================

The storage contracts under a scenario's tasks and a deployment's data: ``TaskStore``, ``TaskFileStore``, ``DataStore`` and ``SegmentStore``, with the task digest and the task limits.

`Storage <../user-guide/storage.rst>`__ shows how a deployment selects each store. `Scenario and record stores <python-api-scenario-stores.rst>`__ gives the storage that opens them.

Task stores
-----------

A ``TaskStore`` (``reef.storage.data.tasks``) holds published Harbor tasks, each in the ``train`` or ``eval`` split. A record-to-sample stage that generates tasks saves them in a task store before any rollout, so generation and rollout recover independently after a restart.

To open a store, call ``task_storage.open(name)``. A pipeline stage gets ``task_storage`` from ``PipelineContext.task_storage``, and each record-to-sample stage uses its own ``name``. For a composite recipe, each component's ``TaskStorage`` opens ``<component>/<name>``, so two components never share a store.

A store holds generations. A generation holds its source record IDs and its candidates. A task is a Harbor task directory: the database keeps the state of the store and the digest of each directory, and the task file store keeps the files, text or binary, under that digest. Each ``TaskCandidate`` holds the task ``name``, the ``files_digest`` of its directory, and its outcome:

- A published candidate has a ``split``, ``train`` or ``eval``. The stage chooses the split, and the store keeps it.
- A rejected candidate has an ``error`` that gives the reason.
- A pending candidate has neither. After a restart, the stage validates the candidates in ``GenerationState.pending_indices``.

The published tasks are the candidates with a split. A ``StoredTask`` holds the task ``name``, its ``split`` and its ``files_digest``.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``begin_generation(*, generation, source_record_ids)``
     - Create the checkpoint of a generation. A retry with the same source record IDs returns the checkpoint.
   * - ``write_candidates(*, generation, directories)``
     - Save a copy of each candidate task directory once, before validation or rollouts. The name of a candidate is the name of its directory.
   * - ``publish_task(*, generation, index, split)``
     - Publish one candidate and return its ``StoredTask``. Each task name is published once. A later candidate with the same files must give the same split; the call then returns the published task.
   * - ``reject_task(*, generation, index, reason)``
     - Save the rejection of one candidate.
   * - ``complete_generation(*, generation)``
     - Mark the generation complete. Each candidate must have an outcome.
   * - ``generation(generation)``, ``generations()``
     - Return one checkpoint or all checkpoints.
   * - ``tasks(*, split=None)``
     - Return the published tasks in generation and candidate order, each name once. Give a ``split`` to return only the tasks of that split.
   * - ``path(files_digest)``
     - Return the local directory of a candidate, a published task or a ``write_files`` call, for example for ``read_harbor_task`` or ``TaskRolloutRunner.run``. Do not change it.
   * - ``task_files(files_digest)``
     - Return the ``TaskFiles`` that names that directory in the task file store. A job data directory names a task by it. The call does not check that the store holds the digest.
   * - ``write_document(*, key, document)``, ``documents()``
     - Save a text document under a non-empty key, or return every document by key. Each key holds one document; another document under a used key raises ``TaskStoreError``. The store does not read a document: the stage that writes the store keeps its own state about its tasks here.
   * - ``write_state_document(*, key, document)``, ``delete_state_document(*, key)``, ``state_document(key)``, ``state_document_keys(*, prefix="")``
     - Save, remove or read a state document, or return the keys that start with ``prefix`` in increasing string order. A state document is text under a non-empty key. Unlike a document, its key can be written again and removed. The two kinds of document have separate keys. The stage keeps there the state that changes in place: the generated-task stage gives it to its task generator as ``GeneratorState``.
   * - ``write_files(directory)``
     - Save a copy of a directory in the task files of the scenario and return the digest of its files, which ``path`` takes. The call makes no candidate and changes no generation.

Write behavior:

- Each write is durable when it returns. An identical retry of a write changes nothing.
- A write that conflicts with the stored state raises ``TaskStoreError``.
- If a task name is published with other files or in the other split, the store raises ``TaskConflictError``, a subclass of ``TaskStoreError``.
- The caller serializes the writers of one store.

SQLite keeps the task tables in the database file of the scenario. PostgreSQL keeps them in the deployment schema, with a key for each store generation. ``archive`` retires the task tables and the task files with the records and commits. Record retention does not delete tasks.

The scenario's imported tasks are not a task store. They are task items that the scenario uses, and each trainer reads them from ``PipelineContext.samples`` (a ``SampleStore``). The service takes a task item in the ``train`` split of a scenario (``POST /reef/namespaces/{owner}/data/tasks``) only when a trainer of the scenario trains imported tasks: its sample-to-job stage sets the class attribute ``accepts_imported_tasks``. An upload to the ``eval`` split is open to every recipe, because an evaluation runs those tasks.

Task file stores
----------------

A ``TaskFileStore`` (``reef.storage.data.tasks``) keeps the directories of the published tasks, addressed by the digest of their files. The digest covers the directory name and each file's path, bytes, and executable flag, so a repeated save of the same files changes nothing. The flag is set if any executable bit is set. Existing digests still address their stored directories. The store can reuse a legacy digest when the files and their executable flags match. Use the digest that the store returns to read a saved directory.

A deployment selects the store in ``storage.task_file_backend`` (`Choose the task file store <../user-guide/storage.rst#choose-the-task-file-store>`__). ``LocalTaskFileStore`` is the built-in store. An embedded caller can give a store to ``SQLiteScenarioStorage(..., task_files=...)`` or ``PostgresScenarioStorage(..., task_files=...)``. Without one, the storage uses a ``LocalTaskFileStore`` in ``task-files`` under its directory.

A store keeps its files in partitions: one for each scenario, and ``data/<owner>`` for the task items of each data owner. A client can write a task directory into a partition itself, in the layout of the store, and then create the task item by reference (`Create task items <http-api-data-upload.rst#create-task-items>`__). The layout belongs to each store.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``from_config(config, *, directory)``
     - Class method that builds the store that a deployment names. ``directory`` is ``storage.task_file_dir``. A remote store can use it as a local cache.
   * - ``write(scenario, source)``
     - Save a copy of the task directory ``source`` and return the digest. The copy is durable when the method returns. Raise ``TaskStoreError`` for a symbolic link or a special file.
   * - ``path(scenario, digest)``
     - Return a local directory with the files of ``digest``. The directory has the name of the saved directory. Raise ``TaskStoreError`` if the store does not hold the digest.
   * - ``contains(partition, digest, name)``
     - Return whether ``partition`` holds a complete task directory named ``name`` with the files of ``digest``. A directory that a client writes is complete when the client has written its last part. Do not read the files. Raise ``ValueError`` for a digest that is not 64 lowercase hexadecimal digits.
   * - ``archive(scenario)``
     - Retire the files of the scenario and return the archive locations.
   * - ``close()``
     - Release the resources of the store. A second call does nothing.

The module also gives the digest and the limits that every client uses for a task directory:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Name
     - Contract
   * - ``files_digest(source)``
     - Return the SHA-256 hex digest of the local task directory ``source``: its name and each file's path, bytes, and executable flag. Directories add nothing. Raise ``TaskStoreError`` for a symbolic link or a special file.
   * - ``TaskDirectoryDigest(directory_name)``
     - Compute ``files_digest`` from files given one at a time. Call ``add_file(relative_path, size_bytes, mode)`` for each file, then ``update(chunk)`` until the file's bytes are given, then ``hexdigest()``. The files come in increasing ``task_file_order``. A file out of order, or a file with more or fewer bytes than its size, raises ``TaskStoreError``.
   * - ``FilesDigest(directory_name)``
     - Compute the legacy digest without executable flags.
   * - ``task_file_order(relative_path)``
     - Return the sort key of a file path relative to the task directory: the tuple of its parts, so ``a/b`` comes before ``a-b``. Raise ``ValueError`` for an empty part, ``.`` or ``..``.
   * - ``task_limit_errors(entries)``
     - Return a ``TaskLimitError`` for each limit that a task breaks, from its ``(relative_path, size_bytes)`` entries (``None`` as the size of a directory). Each error has a ``code``, the ``path`` (``""`` for the task directory) and an English ``message``.

The codes of ``task_limit_errors`` are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Code
     - Limit
   * - ``file_too_large``
     - A file over ``MAX_TASK_FILE_BYTES``, 500 GB.
   * - ``too_many_entries``
     - A directory with more than ``MAX_TASK_DIRECTORY_ENTRIES``, 10,000, entries.
   * - ``too_many_files``
     - More than ``MAX_TASK_FILES``, 100,000, files.

Data stores
-----------

A ``DataStore`` (``reef.storage.data.store``) keeps the deployment's data items. An item is ``(id, content, metadata)``:

- Its content never changes: the JSON object of a trajectory or a record, or the directory of a task, which the task file store keeps in the owner's partition.
- Its metadata is a JSON object. Reef reads ``kind`` (``trajectory``, ``task`` or ``record``), ``owner``, ``scenarios`` (scenario name to ``train`` or ``eval``) and, for a record or artifact, ``origin`` (``{scenario, artifact_id}``). The keys in ``FIXED_KEYS`` (``kind``, ``owner``, ``origin`` and ``evaluation_id``) never change.

A deployment selects the store in ``storage.data_store``. ``SQLiteDataStore`` and ``PostgresDataStore`` (``reef.storage.sql.data``) and ``MemoryDataStore`` are built in.

- Every operation names one owner and reads or writes only its items.
- Every store links the record items of each owner into trajectories.
- The SQL stores keep only the messages that each item adds; reads return the whole content. With a segment store, the SQL stores keep the content of trajectory items and record messages in segments (`Segment stores`_).
- The store checks the form of the metadata. The service checks that ``scenarios`` names only scenarios of the owner.

The table gives the members that a caller uses most. A plugin implements every abstract member of the class, including the scenario uses, holds, collections and job data directories that ``reef/storage/data/store.py`` describes.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files, segments, job_directory_ttl_seconds=...)``
     - Class method that builds the store that a deployment names. ``directory`` is ``storage.agent_record_dir``. The caller owns ``task_files``. ``segments`` is the ``storage.segment_backend`` store or ``None``; the data store owns it and closes it. ``job_directory_ttl_seconds`` is ``storage.job_directory_ttl_s``.
   * - ``create(owner, items, *, generation=None)``
     - Stores ``NewItem`` values, all or none. The same ID and content again changes nothing; other content is ``ItemConflictError``. Bad metadata is ``ValueError``. A task's content is a local task directory, which the store copies into the owner's task file partition, or a ``TaskFilesReference(digest)`` (``reef.storage.data.item_kinds``) to a directory that the partition holds. A reference that it does not hold is ``TaskFilesNotFoundError``.
   * - ``read(owner, item_id)``
     - The ``StoredItem``, or ``None``.
   * - ``query(owner, *, kind, scenario, after, limit, content=True, ...)``
     - The owner's items after the sequence ``after``, in arrival order, at most ``limit`` (1 to 1000), of ``kind`` and used by ``scenario`` when given. With ``content=False``, each item's ``content`` is ``None`` and the store does not read it. The method takes more optional filters; see its signature.
   * - ``update(owner, item_ids, patch)``
     - Applies one JSON merge patch to each item's metadata, all or none. Changing a fixed key is ``ValueError``; an unknown ID is ``ItemNotFoundError``.
   * - ``delete(owner, item_id)``
     - Removes the item; ``False`` for an unknown ID. The store keeps it as a ``DeletedItem`` (the item without content, and ``deleted_at``), one for each ID; a second delete of the ID replaces the earlier one. A deleted item is in no read, query, set, use or record link, and ``create`` can use its ID again.
   * - ``deleted_items(owner, *, kind=None, before=None, limit=1000)``
     - The owner's ``DeletedItem`` values, of ``kind`` when given, by item sequence newest first, below ``before`` when given.
   * - ``read_deleted(owner, item_ids)``
     - The ``DeletedItem`` of each given ID that has one, in the order given.
   * - ``restore(owner, item_ids)``
     - Makes the deleted items items again, all or none, with their sequence, metadata and content, and gives them the uses that their metadata gives. Each use gets a new use sequence. An ID without a deleted item is ``ItemNotFoundError``; an ID that the owner stores is ``ItemConflictError``.
   * - ``purge(owner, item_id)``
     - Removes the deleted item for good; ``False`` when there is none.
   * - ``purge_deleted(time)``
     - Purges each deleted item, across owners, deleted before ``time``, and returns their number.
   * - ``task_path(item)``
     - A local directory with the files of a task item.
   * - ``record_links(owner, item_ids)``
     - The ``RecordLink`` of each given record item that has a message list: ``parent``, the nearest ancestor in its trajectory that is still stored (``None`` for a root), and ``entry_count``. A trajectory ends at a call that no stored call names as parent; its calls are ``reef.core.trajectory_index.trajectory_calls`` of the path from the root. Reports and unknown IDs are left out.
   * - ``groups(owner, *, kind, key, scenario=None, before=None, limit=1000)``
     - The ``ItemGroup`` values of the owner's items of ``kind`` by the string under the metadata ``key``: ``value`` (``None`` without a string), ``item_count``, ``first_sequence``, ``last_sequence`` and the first item's ``metadata``. Newest first, below ``before`` when given. The default reads every page of ``query``; the SQL stores group in SQL.
   * - ``maintain()``
     - The store's periodic work. The data service calls it every 5 minutes. It changes no item. The default deletes the job data directories that no job used for the time to live, and purges the items deleted 30 days ago or earlier (``DELETED_ITEM_TTL_SECONDS``).
   * - ``close()``
     - Releases the store's resources; a second call does nothing.

Segment stores
--------------

A ``SegmentStore`` (``reef.storage.data.segments``) keeps segments: immutable JSON Lines files, each in one partition. The SQL data stores use one partition per owner. A deployment selects the store in ``storage.segment_backend`` (`Keep large content in segment files <../user-guide/storage.rst#keep-large-content-in-segment-files>`__). ``LocalSegmentStore`` is the built-in store.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``from_config(config, *, directory)``
     - Class method that builds the store that a deployment names. ``directory`` is ``<storage.agent_record_dir>/segments``. A remote store can use it as a local cache.
   * - ``write(partition, segment_id, data)``
     - Store the bytes ``data``. The segment is durable when the method returns. The data store never writes one ID twice.
   * - ``read(partition, lines)``
     - The bytes of each ``SegmentLine`` (``segment_id``, ``offset``, ``length``), in order. Raise ``SegmentNotFoundError`` for a missing segment or a line past its end.
   * - ``delete(partition, segment_id)``
     - Remove the segment. A missing segment is not an error.
   * - ``read_directory_uri(key)``
     - The URI of the directory ``key``, or ``None`` when the store does not hold all of it. A key is ``/``-separated names. The data store keeps each job data directory as ``job-data/<key>``.
   * - ``write_directory(key, source)``
     - Keep a copy of the files under ``source`` as the directory ``key``, and return its URI. A write of a key that the store holds keeps the first copy.
   * - ``open_directory(uri)``
     - A local directory with the files of the directory at ``uri``. A remote store copies them into its local cache.
   * - ``delete_directory(key)``
     - Remove the directory. A missing directory is not an error.
   * - ``close()``
     - Release the resources of the store.
