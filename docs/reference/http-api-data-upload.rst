HTTP API: create and transfer data items
========================================

Upload trajectories or Harbor tasks, replace task files, or transfer
`data items <http-api-data-items.rst>`__ through a URI.

Create trajectory items
-----------------------

``POST /reef/namespaces/{owner}/data/trajectories`` accepts JSON Lines with
``Content-Type: application/x-ndjson`` or ``application/jsonl``:

- Each nonblank line contains one ATIF document: an ``ATIF-v``
  ``schema_version``, ``agent`` object, nonempty ``steps``, and ``session_id``.
- ``session_id`` becomes the item ID. IDs must be distinct within the request.
- A request holds 1–1000 trajectories in at most 10 MiB.
- The ``metadata`` query parameter supplies JSON metadata for every item.
  Its ``kind`` is ``trajectory``.

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/trajectories" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/x-ndjson' \
     --data-binary @trajectories.jsonl

This example stores trajectories without assigning them to a scenario.
Success returns 201 with
``{"owner", "items": [{"id", "sequence", "digest", "metadata"}], "synced"}``.
The write is atomic: all items are stored, or none.
An identical retry changes nothing.

Assigned scenarios must belong to the owner and accept uploaded trajectories.
Reef checks nonexistent scenarios when they open. Content need not be
trainable: recipes skip unusable trajectories with a warning. See
`SFT trajectory training <../user-guide/recipes/sft.rst#train-on-whole-conversations>`__.

Errors store nothing:

- 400: invalid ATIF, metadata, or scenario assignment. Invalid-line errors name the line.
- 400: images in trajectory or record items when the deployment includes weight training,
  even without scenario assignments. The error names the item.
- 409: an existing ID has different content.
- 413: the body exceeds 10 MiB; 415: unsupported content type.

``POST /reef/namespaces/{owner}/data`` accepts record and artifact items only.
Use the trajectory route for ATIF.

Create task items
-----------------

``POST /reef/namespaces/{owner}/data/tasks`` accepts 1–1000 Harbor tasks.
Each task name becomes its item ID and must match ``[a-z0-9][a-z0-9._-]*``.
Choose one request form:

- **Archive**, ``Content-Type: application/gzip``: a tar.gz containing task
  directories at its top level. The ``metadata`` query parameter applies to
  every task. Reef validates each directory with Harbor and copies its files.
- **Reference**, ``Content-Type: application/json``: names and digests of
  directories already in the owner's task store. Reef copies no files and
  does not run Harbor validation; the client must validate them.

Archive example:

.. code:: bash

   tar -czf tasks.tar.gz -C /data/benchmark/tasks sum-391 sort-17
   curl -f http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks \
     -H "Authorization: Bearer $REEF_TOKEN" -H 'Content-Type: application/gzip' \
     --data-binary @tasks.tar.gz

``data.max-task-archive-bytes`` limits the body to 512 MiB by default. Unpacked files cannot exceed 8 GiB per archive.
Use references for larger tasks. See `Storage settings
<configuration-settings.rst#storage>`__.

Task limits apply to both forms:

.. list-table::
   :header-rows: 1

   * - Limit
     - Value
   * - One file
     - 500 GB
   * - Direct entries in one directory
     - 10,000 files and directories
   * - Files per task
     - 100,000
   * - Total task size
     - No limit

For references, write directories under ``data/<owner>`` using the
``storage.task-file-backend`` layout. Name each directory after its task.
Compute digests with ``reef.storage.data.tasks.TaskDirectoryDigest`` or
``files_digest`` for local directories. Digests contain 64 lowercase hexadecimal
characters. Each complete directory must exist under its supplied name and digest.

Reference body, with optional ``metadata``:

.. code:: json

   {
     "tasks": [{"name": "sum-391", "digest": "<files digest>"}],
     "metadata": {"scenarios": {"my-agent": "train"}}
   }

Metadata must be a JSON object; its ``kind`` defaults to ``task``.
Training assignments require a recipe that accepts imported tasks;
evaluation assignments work with any recipe through
`artifact evaluations <http-api-evaluation.rst>`__.

Both forms return 201 with the trajectory-create response shape, sorted by
name. Failed creates store no items:

- 400: invalid body, name, digest, metadata, assignment, task count, or task limits.
- 400: malformed tar.gz, top-level files, links, devices, paths outside task
  directories, excessive unpacked size, or a task Harbor rejects.
- 404: the referenced directory is absent from the owner's partition.
- 409: an existing task name has different files; use replacement below.
- 413: the archive exceeds ``data.max-task-archive-bytes``.
- 415: unsupported content type.

Replace a task's files
----------------------

``PUT /reef/namespaces/{owner}/data/tasks/{id}`` accepts either:

- a gzip archive containing exactly one task directory named ``{id}``; or
- JSON ``{"digest": "<files digest>"}`` referencing that directory in the owner's task store.

Use the create route's content types and limits.
Success returns 200 with ``{"owner", "item", "synced"}``, without content.
The item keeps its ID, metadata, and splits. Identical files change nothing.

By default, new evaluation attempts and retries read the replacement files.
Running attempts finish with their old files, so one evaluation can contain
scores from both versions. With ``evaluation.pin_task_files``, every attempt
uses the files selected when Reef accepted the evaluation. Start a new
evaluation to score replacements. A trainer using the task reads it again.

Errors change nothing:

- 400: invalid archive or reference, wrong directory name, Harbor rejection,
  or an item that is not a task.
- 404: unknown item or missing referenced directory.
- 413 and 415: the same size and content-type errors as create.

Import and export items
-----------------------

Transfers run in the background:

- ``POST /reef/namespaces/{owner}/data/imports`` and
  ``POST /reef/namespaces/{owner}/data/exports`` return 202 with transfer status.
  Invalid request bodies return 400.
- ``GET /reef/namespaces/{owner}/data/imports/{id}`` and
  ``GET /reef/namespaces/{owner}/data/exports/{id}`` read status.
  Unknown IDs and other owners' transfers return 404.

Status fields are ``id``, ``kind``, ``state``, ``request``, ``progress``,
``result``, and ``error``:

- ``kind``: ``data_import`` or ``data_export``.
- ``state``: ``queued``, ``assigned``, ``running``, ``finished``, or ``failed``.
- ``request``: the submitted body; ``error``: the failure reason.
- ``progress``: ``{"items"}`` after a trajectory import's first transaction;
  otherwise null. Retries preserve items and advance only beyond the previous count.
- ``result``: ``{"items"}`` for a completed import or ``{"items", "bytes"}``
  for a completed export.

**Import.** Supply ``uri`` and metadata with kind ``task`` or ``trajectory``:

.. code:: json

   {
     "uri": "s3://my-bucket/benchmark/tasks/",
     "metadata": {"kind": "task", "scenarios": {"my-agent": "eval"}}
   }

Supported sources are ``hf://``, ``s3://``, ``file://``, and schemes configured
in ``storage.artifact-sources``, as for
`artifact imports <http-api-artifacts.rst#import-an-artifact>`__.
All imported items receive the metadata.

- **Tasks:** a root containing ``task.toml`` is one task, named after the
  URI's last path segment. Otherwise, immediate child directories containing
  ``task.toml`` become tasks named after those directories.
- **Trajectories:** Reef reads only root-level ``*.jsonl`` and ``*.jsonl.gz``
  files, in name order. Each document's ``session_id`` becomes its item ID.

Trajectory imports first validate every line and reject duplicate IDs.
Failure in this pass stores nothing. The second pass commits batches of at
most 1000 items, ending earlier after about ``32 * 1024 * 1024`` line characters.
Validation or store errors stop this pass; earlier transactions remain stored.
Imports apply the create checks and do not wait for scenario synchronization.

**Export.** Supply ``uri`` and either:

- ``ids``: 1–1000 item IDs; an unknown ID fails the export; or
- optional ``kind`` and ``scenario`` filters. With neither, export every item.

The tar.gz contains ``manifest.jsonl`` entries
``{"id", "kind", "metadata", "path"}`` in export order.
Content goes to ``items/<n>.json``; task files go under ``items/<n>/``.
Numbering starts at 0.

Destinations are:

- ``file:///absolute/path``: a new file on the data service machine.
  The parent directory must exist; an existing file causes failure.
- ``s3://bucket/key``: one object using boto3's standard credentials.
  Install support with ``uv pip install 'reef-infra[s3]'``.

Other destination schemes fail the export.
