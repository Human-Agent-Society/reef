HTTP API: data items
====================

Read, update and download items under ``/reef/namespaces/{owner}/data``.
Send these routes to the data service; control does not serve them in a split
deployment. See `uploads and transfers <http-api-data-upload.rst>`__ for creation.

Items have content and JSON metadata. Create another item to replace content,
or use the dedicated `task-file replacement
<http-api-data-upload.rst#replace-a-task-s-files>`__ route. Metadata can contain
custom keys; Reef reads these:

- ``kind``: ``trajectory``, ``task``, ``record`` or ``artifact``.
- ``owner``: namespace owner, shared by assigned scenarios.
- ``scenarios``: scenario→``train``/``eval``. Trajectories use train; artifacts use neither.
- ``origin``: originating scenario/artifact.
- ``benchmarks``/``trajectory_sets``: named memberships.

``kind``, ``owner``, ``origin`` and ``evaluation_id`` cannot change.
A scenario's owner defaults to its name. Held sets can assign items without
changing metadata; see `sets and collections <http-api-data-sets.rst>`__.

Assignments must match recipe capabilities. Reassignment preserves training counts;
archiving removes assignments, not items. Responses include ``synced`` after waiting
up to 30 seconds for control; false means synchronization remains pending.
Unknown items return 404; deployments without a data store return 503.

Record items
------------

Inference calls and reports are record items; training requests are not.
For a scenario's own record:

- ID: ``<scenario>/<agent_record_id>``.
- Content: ``{agent_record_id, request_type, payload, created_at, references, artifact_id}``.
- Metadata: ``{kind: record, scenarios: {<scenario>: train}, origin: {scenario, artifact_id}}``.
  The artifact ID can be null.

Share records by adding scenarios of the same owner. References become
``<origin scenario>/<record ID>``; assign reports with their inferences.
Receiving recipes determine eligibility. The origin assignment cannot be removed.
Removing other assignments stops new reads, but loaded data can still train.

To upload record items, use ``POST .../data`` with
``{"items": [{"id": "...", "content": {...}, "metadata": {...}}]}``.
The content fields above are required, ``request_type`` must be ``inference``
or ``report``, and the ID must match the origin and record ID. Invalid items
return 400. See `Artifact items`_ for the request format.

Evaluation records and verifier trajectories never train and reject scenario
assignments. Query ``kind=record&group_key=evaluation_id&group_value=<id>``,
or use `evaluation call routes <http-api-evaluation-calls.rst>`__.

Artifact items
--------------

Artifact items reference files without moving them. Their content is:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - ``source_uri``
     - Importable file directory, or null. Git LFS provides no URI.
   * - ``kind``
     - Artifact kind, such as ``model`` or ``model_adapter``, or null.
   * - ``component``
     - Originating scenario component, or null for uploaded items.

Scenario-created items use ``<scenario>/<artifact_id>`` and
``origin: {scenario, artifact_id}``. Uploaded items omit ``origin``.

Register external files with ``POST /reef/namespaces/{owner}/data``:

.. code:: json

   {"items": [{"id": "checkpoint-1",
               "content": {"source_uri": "s3://bucket/checkpoint/", "kind": "model", "component": null},
               "metadata": {"kind": "artifact"}}]}

Use no ``origin``. This route accepts 1–1000 record or artifact items and returns
``{owner, items, synced}`` with status 201, without content. Trajectories and tasks
use their `upload routes <http-api-data-upload.rst>`__.

Import an item into a scenario of the same owner with
``POST /reef/scenarios/{scenario}/artifacts`` and ``{"item": "<id>"}``.
A missing ``source_uri`` returns 400. The imported artifact keeps its identity;
reimporting returns it without reading files again. An uploaded item's first
use sets immutable ``metadata.artifact_id``.
A new scenario can also `start from an item
<http-api-scenarios.rst#create-a-scenario>`__.

Read, query, update and delete items
------------------------------------

**List:** ``GET /reef/namespaces/{owner}/data`` returns ``{owner, items, next}``.
Each item has ``id``, ``sequence``, ``digest`` and ``metadata``.

.. list-table::
   :header-rows: 1

   * - Query
     - Behavior
   * - ``kind``, ``scenario``
     - Filter by kind or scenario use.
   * - ``benchmark``, ``trajectory_set``, ``id_prefix``
     - Filter by set membership or ID prefix.
   * - ``group_key``, ``group_value``
     - Match a string metadata value. Omitted value matches missing/nonstring values. Value without key returns 400.
   * - ``after``
     - Forward from this sequence, exclusive; default 0.
   * - ``before``
     - Backward from this sequence, exclusive, newest first.
   * - ``limit``
     - 1–1000, default 1000.
   * - ``content``
     - 1 includes content; default 0 omits it. Trajectory pages can be large.

Continue with ``next`` as ``after`` or ``before`` until it is null.
A full final page can require one more request, returning an empty page.

**Groups:** ``GET .../data/groups/{key}?kind=&scenario=&before=&limit=`` returns
``{owner, groups, next}``. ``kind`` is required; keys use 1–64 letters/digits/underscores.
Rows contain value, item count, first/last sequence and first-item metadata.
Missing/nonstring values group under null. Groups sort newest first; ``scenario``
restricts membership and counts. Read members through ``group_key``/``group_value``.

**Sets:** ``GET .../data/sets/{kind}`` lists nonempty ``benchmark`` or
``trajectory_set`` sets as ``{owner, sets, next}``. It uses the same pagination,
counts and metadata as groups; read members through the corresponding item filter.

**Read:** ``GET .../data/{id}`` returns ``{owner, id, sequence, digest, metadata, content}``.
Task content is null.

**Patch:** ``POST .../data/metadata`` takes ``{ids: [...], patch: {...}}``
for 1–1000 items. It applies a JSON merge patch (RFC 7386) atomically.
Null removes a key; for example, ``{"scenarios": {"my-agent": null}}`` removes
an assignment. ``{"scenarios": {"my-agent": "eval"}}`` moves tasks to evaluation.
Assignments undergo creation checks; immutable-field changes return 400.
The response is ``{owner, items, synced}``, without content.

**Delete:** ``DELETE .../data/{id}`` returns ``{owner, id, deleted: true, synced}``.
The item leaves new data and record reads. Previously loaded training data can
remain in memory. Deleting a scenario-created artifact item returns 400 while
the scenario exists. Items remain `restorable for 30 days <#restore-deleted-items>`__.

**Scenario tasks:** ``GET /reef/scenarios/{scenario}/tasks`` returns
``{scenario, tasks: [{name, split, benchmark, files_digest, trained}]}`` in use order.
``benchmark`` names the held benchmark, or null for a direct assignment.
Multiple uses produce multiple rows. ``trained`` becomes true after any
scenario trainer commits a task job using it.

IDs can contain ``/``, but GET reserves IDs ending in ``/archive`` and IDs
matching ``imports/<x>`` or ``exports/<x>`` for one-segment ``<x>``.

Restore deleted items
---------------------

Deleted items leave active queries, sets and scenarios. Their IDs can be reused.
They remain recoverable for 30 days:

.. list-table::
   :header-rows: 1

   * - Route
     - Result
   * - ``GET /reef/namespaces/{owner}/deleted-items?kind=&before=&limit=``
     - ``{owner, items, next}``, newest sequence first. Each item adds epoch-second ``deleted_at`` and ``purge_at``; content is omitted.
   * - ``POST /reef/namespaces/{owner}/deleted-items/restore``
     - Body ``{"ids": [...]}``, 1–1000 IDs. Restores all or none; returns ``{owner, items, synced}`` without content.
   * - ``DELETE /reef/namespaces/{owner}/deleted-items/{id}``
     - Permanently purges the item; returns ``{owner, id, purged: true}``.

Listing uses the item-list pagination limits. Restore preserves sequence,
metadata, content and set membership, and validates assignments again.
Missing deleted items return 404; an active item reusing the ID causes 409.
Archiving a scenario also removes its deleted-item assignments.

Forward pagination does not revisit restored sequences already passed.

Download an item
----------------

``GET /reef/namespaces/{owner}/data/{id}/archive`` downloads:

- Tasks: ``<name>.tar.gz`` (``application/gzip``), containing the ``<name>`` directory.
- Other items: ``<name>.json`` (``application/json``), containing their content.

``name`` is the ID's last segment. Use `Signed download URLs`_ to grant temporary
access without sending the service token.

Signed download URLs
--------------------

Add ``?signed=1`` to the archive route or
``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``.
The response is ``{url, expires_at}``: a same-service path/query and its
expiry in Unix seconds. ``expires_in`` accepts 1–86400 seconds, default 3600;
other values return 400.

Send the returned URL with GET and no Authorization header.
The signature permits only that download, including its ``path`` and
``component`` queries. Changed paths/queries, expiry or revoking the signing
token return 401. A deployment without tokens returns an unsigned URL.
