HTTP API: create and transfer data items
========================================

Upload trajectories and Harbor tasks as `data items <http-api-data-items.rst>`__, replace the files of a task, and import or export items at a URI.

Create trajectory items
-----------------------

Send trajectories as JSON Lines to ``POST /reef/namespaces/{owner}/data/trajectories`` with ``Content-Type: application/x-ndjson`` (or ``application/jsonl``):

- Each line is one ATIF document. Reef skips blank lines.
- A document must have an ``ATIF-v`` ``schema_version``, an ``agent`` object, one or more steps, and a ``session_id``, which is the item's ID.
- One request holds 1 to 1000 trajectories with distinct ``session_id`` values, in at most 10 MiB.
- The query parameter ``metadata`` is the JSON metadata of every trajectory of the request. Its ``kind`` is ``trajectory``.

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/trajectories" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/x-ndjson' \
     --data-binary @trajectories.jsonl

The response has status 201: ``{"owner", "items": [{"id", "sequence", "digest", "metadata"}], "synced"}``. Reef stores all of the items in one transaction, or none. An identical retry changes nothing.

Before Reef stores the items, it checks each scenario that they name. The scenario must have the owner, and its recipe must train on the items:

- A trajectory needs a recipe that accepts uploaded trajectories. Reef does not read the trajectories for this check. The recipe turns each one into a training sample when it reads it, and skips one that cannot train with a warning. The SFT recipe trains every agent turn, except the steps marked ``is_copied_context: true`` (`SFT <../user-guide/recipes/sft.rst#train-on-whole-conversations>`__), and skips a trajectory without such a turn. The training component renders the trajectory with the model's chat template when a job trains it, and leaves out one that cannot render.
- A task in the ``train`` split needs a recipe that trains imported tasks (``Sample2Job.accepts_imported_tasks``), such as ``sft``, the generated-task recipe, or harness evolution that runs offline (`Offline runs on uploaded tasks <../user-guide/evolve-your-harness.rst#offline-runs-on-uploaded-tasks>`__). The ``eval`` split takes tasks for any recipe, because an `artifact evaluation <http-api-evaluation.rst>`__ runs them.

Reef checks a scenario that does not exist yet when the scenario opens.

Reef returns these errors and stores nothing:

- 400 for a line that is not an ATIF document; the error names the line.
- 400 for an item that a named scenario cannot train on; the error names the item.
- 400 for a trajectory or a record item with an image, also one that names no scenario, when the deployment's recipes include a weight-training recipe, because training renders text only. The error names the item.
- 409 for a known ID with other content.

``POST /reef/namespaces/{owner}/data`` refuses a trajectory; it creates record and artifact items only.

Create task items
-----------------

``POST /reef/namespaces/{owner}/data/tasks`` creates one task item for each Harbor task directory. The task name is the item's ID: a lowercase letter or a digit, then lowercase letters, digits, ``.``, ``_`` and ``-``. A request holds 1 to 1000 tasks. The request has one of two forms:

- An archive (``Content-Type: application/gzip``). The data service unpacks the archive and copies each task directory into the task file store.
- A reference (``Content-Type: application/json``). The client wrote each task directory into the task file store itself, and the data service copies no file.

Every task must be within these limits on its files. The module ``reef.storage.tasks`` defines them, so a client can check a task before it sends it (``task_limit_errors``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Limit
     - Value
   * - Size of one file
     - 500 GB (``MAX_TASK_FILE_BYTES``)
   * - Entries, files and directories, directly in one directory
     - 10,000 (``MAX_TASK_DIRECTORY_ENTRIES``)
   * - Files in one task
     - 100,000 (``MAX_TASK_FILES``)
   * - Size of one task
     - no limit

**Archive form.** Each top-level entry of the gzip-compressed tar archive is one task directory, named by its task name. The query parameter ``metadata`` is the JSON metadata of every task of the archive:

.. code:: bash

   tar -czf tasks.tar.gz -C /data/benchmark/tasks sum-391 sort-17
   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/gzip' \
     --data-binary @tasks.tar.gz

The setting ``data.max-task-archive-bytes`` limits the body (default 512 MiB; see `The data section <configuration-service.rst#the-data-section>`__). The files of one archive unpack to at most 8 GiB; create larger tasks in the reference form. Harbor's task loader must accept each task directory.

**Reference form.** First write each task directory into the owner's partition of the task file store, in the layout of the deployment's ``TaskFileStore`` (``storage.task-file-backend``). The partition of the owner ``<owner>`` is ``data/<owner>``, and the directory has the task name as its name. Compute its digest with ``reef.storage.tasks.FilesDigest``, or with ``files_digest`` for a local directory. Then send the task names and digests as JSON. ``metadata`` is optional, as in the archive form:

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"tasks": [{"name": "sum-391", "digest": "<files digest>"}],
          "metadata": {"scenarios": {"my-agent": "train"}}}'

For each task, the task file store must hold a complete directory with that name and digest in the owner's partition (``TaskFileStore.contains``). The item then gets that digest. The data service does not run Harbor's task loader on a reference: the client that wrote the files checks them.

Both forms return the same response as for trajectory items, with the items in name order. Reef returns these errors and stores nothing:

- 400 for an archive that is not tar.gz, a link or a device in it, a path outside its task directories, a file at the top level, a task over a limit of the table above, or a directory that Harbor's task loader refuses;
- 400 for a reference body that is not ``{"tasks": [{"name", "digest"}], "metadata"}``, or a digest that is not 64 lowercase hexadecimal digits;
- 400 for a name that is not a task name, more than 1000 tasks, or metadata that is not a JSON object of kind ``task``;
- 404 for a reference to a task directory that the owner's partition does not hold;
- 409 for a task name that the owner holds with other files. To change the files of a task, replace them (below);
- 413 for an archive body larger than ``data.max-task-archive-bytes``;
- 415 for another content type.

Replace a task's files
----------------------

``PUT /reef/namespaces/{owner}/data/tasks/{id}`` gives the task item ``id`` new files. The new task directory has the item's ID as its name. The request has the same two forms as a create:

- An archive of that one directory (``Content-Type: application/gzip``):

  .. code:: bash

     tar -czf sum-391.tar.gz -C /data/benchmark/tasks sum-391
     curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks/sum-391" \
       -X PUT \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/gzip' \
       --data-binary @sum-391.tar.gz

- A reference (``Content-Type: application/json``) to a directory that the client wrote into the owner's partition of the task file store: ``{"digest": "<files digest>"}``.

The item keeps its ID, metadata and splits. The same files again change nothing. After a change:

- Each evaluation attempt reads the task's current files. A new evaluation, and a retry of a started one, run the new files. An attempt that already runs finishes with the old files, so one evaluation can hold scores of both.
- A trainer of a scenario that uses the task reads it again.

The response is ``{"owner", "item", "synced"}``, without content. Reef returns these errors and changes nothing:

- 400 for an archive of no task or of several, a directory with another name, a directory that Harbor's task loader refuses, a reference body that is not ``{"digest"}``, or an item that is not a task;
- 404 for an unknown item, or a reference to a directory that the owner's partition does not hold;
- 413 and 415 as for a create.

Import and export items
-----------------------

An import stores the files at a URI as items of the owner. An export writes items of the owner to a tar.gz at a URI. Each runs in the background on a data service worker:

- ``POST /reef/namespaces/{owner}/data/imports`` and ``POST /reef/namespaces/{owner}/data/exports`` start one and return 202 with its status. A body that is not valid returns 400.
- ``GET /reef/namespaces/{owner}/data/imports/{id}`` and ``GET /reef/namespaces/{owner}/data/exports/{id}`` read the status. An unknown ID, or the ID of another owner's transfer, returns 404.

The status is ``{"id", "kind", "state", "request", "progress", "result", "error"}``:

- ``kind`` is ``data_import`` or ``data_export``.
- ``state`` is ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``.
- ``request`` is the body that started the transfer.
- ``progress`` is ``{"items"}`` after a trajectory import stores its first transaction: the count of items stored so far. It is ``null`` before that, and for a task import and an export. A retry stores the same transactions again, which changes no item, and ``progress`` does not go back: it changes only when the retry stores more than the attempts before.
- ``result`` is set when the transfer finished: ``{"items"}``, the item count, for an import, and ``{"items", "bytes"}`` for an export.
- ``error`` tells why a run failed.

An import body is ``{"uri", "metadata"}``. Every item gets ``metadata``, whose ``kind`` is ``task`` or ``trajectory``. The data service reads the URI with the readers of `artifact imports <http-api-artifacts.rst#import-an-artifact>`__: ``hf://``, ``s3://``, ``file://`` and the schemes that ``storage.artifact-sources`` adds. It finds the items as follows:

- ``task``: when the URI's root holds ``task.toml``, the root is one task, named after the last segment of the URI's path. Otherwise each directory directly under the root that holds ``task.toml`` is one task, named after the directory.
- ``trajectory``: each ``*.jsonl`` file and each gzip ``*.jsonl.gz`` file at the root holds one ATIF document a line, as the upload takes them. Each document's ``session_id`` is its item ID. Reef reads no other file. It reads the files in name order, in two passes:

  1. Reef checks every line. A bad line, or a ``session_id`` that an earlier line has, fails the import, and Reef stores no item.
  2. Reef stores the items one transaction at a time, each of at most 1000 items and about 32 MiB of lines. A check that only the data store makes, such as a conflict with an item that the owner stores already, stops the import. The items of the earlier transactions stay.

The import checks and stores the items as the create routes do. It does not wait for the syncs of the scenarios that it touched.

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/namespaces/my-agent/data/imports \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"uri": "s3://my-bucket/benchmark/tasks/",
          "metadata": {"kind": "task", "scenarios": {"my-agent": "eval"}}}'

An export body is ``{"uri"}`` and one of these selections:

- ``ids``: 1 to 1000 item IDs. An unknown ID fails the export.
- ``kind`` and ``scenario``, each optional: the items that match, as for the list route. With neither, the export writes every item of the owner.

The archive holds ``manifest.jsonl``, one line ``{"id", "kind", "metadata", "path"}`` for each item, in export order. The content of a trajectory or a record is at ``items/<n>.json``, and the files of a task are under ``items/<n>/``, where ``<n>`` counts the items from 0. The export writes these URIs:

- ``file:///absolute/path``: a new file on the data service's machine. The parent directory must exist. When the file exists, the export fails.
- ``s3://bucket/key``: one object, written with boto3's standard credentials. It needs the ``s3`` extra (``uv pip install 'reef-infra[s3]'``).

Another scheme fails the export.
