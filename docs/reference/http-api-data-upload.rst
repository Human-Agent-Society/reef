HTTP API: create and transfer data items
========================================

Upload trajectories and Harbor tasks, replace task files, or transfer `data items <http-api-data-items.rst>`__ through a URI.

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

Before storing items, Reef checks their scenario assignments. Each scenario must belong to the owner and support the item kind:

- Trajectories require a recipe that accepts uploads. This check does not read trajectory content. When processing it, the recipe skips unusable trajectories with a warning.
- SFT trains every agent turn except ``is_copied_context: true`` steps, and skips trajectories without trainable turns. See `SFT <../user-guide/recipes/sft.rst#train-on-whole-conversations>`__.
- Training renders trajectories with the model's chat template. It skips trajectories that cannot render.
- Train tasks require support for imported tasks (``Sample2Job.accepts_imported_tasks``). Examples include SFT, generated tasks, and `offline harness evolution <../user-guide/evolve-your-harness.rst#offline-runs-on-uploaded-tasks>`__.
- Eval tasks work with any recipe through `artifact evaluations <http-api-evaluation.rst>`__.

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

Tasks must meet these limits. Use ``reef.storage.data.tasks.task_limit_errors`` to check them before upload:

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

The setting ``data.max-task-archive-bytes`` limits the body (default 512 MiB; see `Settings reference <configuration-settings.rst#storage>`__). The files of one archive unpack to at most 8 GiB; create larger tasks in the reference form. Harbor's task loader must accept each task directory.

**Reference form.** Write task directories into the owner's task-store partition, ``data/<owner>``, using the deployment's ``TaskFileStore`` layout (``storage.task-file-backend``). Name each directory after its task. Compute the digest with ``reef.storage.data.tasks.TaskDirectoryDigest``, or ``files_digest`` for local directories. Send names and digests as JSON, with optional ``metadata``:

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"tasks": [{"name": "sum-391", "digest": "<files digest>"}],
          "metadata": {"scenarios": {"my-agent": "train"}}}'

The owner's partition must contain each complete directory with the supplied name and digest (``TaskFileStore.contains``). The item uses that digest. Clients must validate referenced tasks: data does not run Harbor's loader on them.

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

- By default, each evaluation attempt reads the task's current files. A new evaluation, and a retry of a started one, run the new files. An attempt that already runs finishes with the old files, so one evaluation can hold scores of both.
- With ``evaluation.pin_task_files`` enabled, every attempt uses the files that Reef selected when it accepted the evaluation. Start a new evaluation to score the replacement files.
- A trainer of a scenario that uses the task reads it again.

The response is ``{"owner", "item", "synced"}``, without content. Reef returns these errors and changes nothing:

- 400 for an archive of no task or of several, a directory with another name, a directory that Harbor's task loader refuses, a reference body that is not ``{"digest"}``, or an item that is not a task;
- 404 for an unknown item, or a reference to a directory that the owner's partition does not hold;
- 413 and 415 as for a create.

Import and export items
-----------------------

Imports read files at a URI into the owner's data store. Exports write the owner's items to a tar.gz. Both run in the background:

- ``POST /reef/namespaces/{owner}/data/imports`` and ``POST /reef/namespaces/{owner}/data/exports`` start one and return 202 with its status. A body that is not valid returns 400.
- ``GET /reef/namespaces/{owner}/data/imports/{id}`` and ``GET /reef/namespaces/{owner}/data/exports/{id}`` read the status. An unknown ID, or the ID of another owner's transfer, returns 404.

The status is ``{"id", "kind", "state", "request", "progress", "result", "error"}``:

- ``kind`` is ``data_import`` or ``data_export``.
- ``state`` is ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``.
- ``request`` is the body that started the transfer.
- ``progress`` is ``{"items"}``, the stored count, after a trajectory import's first transaction. It is ``null`` beforehand and for task imports and exports. Retries preserve items and advance progress only after exceeding the previous count.
- ``result`` is set when the transfer finished: ``{"items"}``, the item count, for an import, and ``{"items", "bytes"}`` for an export.
- ``error`` tells why a run failed.

An import body is ``{"uri", "metadata"}``. All items receive the supplied metadata, with ``kind`` set to ``task`` or ``trajectory``. Supported URIs match `artifact imports <http-api-artifacts.rst#import-an-artifact>`__: ``hf://``, ``s3://``, ``file://`` and schemes configured in ``storage.artifact-sources``.

Reef discovers items by kind:

- ``task``: when the URI's root holds ``task.toml``, the root is one task, named after the last segment of the URI's path. Otherwise each directory directly under the root that holds ``task.toml`` is one task, named after the directory.
- ``trajectory``: each ``*.jsonl`` file and each gzip ``*.jsonl.gz`` file at the root holds one ATIF document a line, as the upload takes them. Each document's ``session_id`` is its item ID. Reef reads no other file. It reads the files in name order, in two passes:

  1. Reef checks every line. A bad line, or a ``session_id`` that an earlier line has, fails the import, and Reef stores no item.
  2. Reef stores transactions of at most 1000 items and about 32 MiB of lines. A data-store error, such as an existing-item conflict, stops the import. Earlier transactions remain stored.

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
