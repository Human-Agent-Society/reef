HTTP API: harness trees and pages
=================================

Pull the harness tree of a scenario, commit a person's harness files, and open the page of a step or of a request.

A weight-training scenario needs no client action: keep calling the same inference endpoint, and it serves the active selection.

For a scenario whose artifact is the harness tree, the client pulls. It fetches the tree of the active selection with ``GET /reef/harness``, writes the files where its agent reads them, binds the agent's model to this Reef, and runs the agent on that snapshot. It then reports against the receipts of that snapshot. ``reef-client harness`` is that client (see `Use a published tree <../user-guide/evolve-your-harness.rst#use-a-published-tree>`__).

In the split layout, the harness service answers ``GET /reef/harness``, ``POST /reef/harness/import``, ``GET /reef/harness/artifacts`` and ``GET /reef/harness/adapters``. Control answers the other harness routes (`Run the services in separate processes <../user-guide/split-services.rst>`__).

Harness artifacts
-----------------

+--------------------------------+---------------------------------------------------------------+
| Route                          | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | ``{artifact_id, parent_artifact_id, files, evaluation,        |
|                                | requires}``, plus ``components`` (component name to           |
|                                | artifact ID) for several components;                          |
|                                | ``x-reef-artifact-id`` identifies the tree                    |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/artifacts``| ``{scenario, artifacts}``, oldest first: the creation and     |
|                                | every training step of the harness component, also a step     |
|                                | that published nothing (``checkpoint: false``). Each row has  |
|                                | ``artifact_id``, ``component``, ``active`` (true on exactly   |
|                                | one row, the active artifact) and ``activated`` (true when    |
|                                | the row's artifact has been active at some time); each        |
|                                | training row carries the evaluation metrics of its step and   |
|                                | ``page_path``, its step's page with the query a browser opens |
|                                | it by                                                         |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/adapters`` | ``{adapters}`` — every harness adapter this process resolves, |
|                                | each with ``name``, ``binary``, ``trajectory_format``,        |
|                                | ``model_bindings``, the pinned ``install`` spec, ``roots``    |
|                                | (each render root to its ``variable`` and ``default``         |
|                                | directory on a person's machine) and ``files`` (the render    |
|                                | paths of the harness's files, ``{name}`` for one path         |
|                                | segment); both are empty for an adapter that does not sync    |
+--------------------------------+---------------------------------------------------------------+
| ``POST /reef/harness/import``  | ``{artifact_id, parent_artifact_id, changed, merged, files,   |
|                                | ignored}``, with ``x-reef-artifact-id``; see                  |
|                                | `Importing a harness`_                                        |
+--------------------------------+---------------------------------------------------------------+

The manifest and catalog reads take ``x-reef-scenario`` and never create a scenario. An unknown scenario or one without a harness tree answers HTTP 404.

``?form=`` chooses what ``files`` holds, path to text. Another value returns 400:

- ``rendered`` (the default): the snapshot rendered for the scenario's adapter. These are the paths that the agent reads, with the adapter's enforced config defaults merged in (see `Rendering <../developer-guide/harness-adapters.rst#rendering>`__).
- ``source``: the source tree that the snapshot stores. These are the tree paths that the adapter's parts take, the ones that a patch names (see `Tree parts <../developer-guide/harness-adapters.rst#tree-parts>`__).
- ``client``: the snapshot as a layer over a person's own configuration, for an install at ``root``, bound to ``<service_url>/reef/scenarios/<scenario>``. Both query parameters are required: ``service_url`` is the address at which the client reaches Reef, and ``root`` is the directory in which the client installs the files. An adapter without a ``layer`` returns 400. The binding asks for the served model, and its key is the literal ``REEF_TOKEN_PLACEHOLDER``, which the client replaces with its own token. This form adds:

  - ``run``: the command that runs the agent on the files (``binary``, ``args`` and ``env``; ``env`` may hold the placeholder too);
  - ``register`` and ``unregister``: the agent's own commands that record the install and remove it, each a list of argument lists, empty for most agents;
  - ``install``: the agent's pinned install spec;
  - ``synced``: the render templates of the files that the client sends back, with ``{name}`` for one path segment.

In the ``rendered`` and ``source`` forms, a config file may hold ``{root}``: write the directory that holds the render roots there. These two forms carry no endpoint and no credential: the client writes its own model binding where its agent reads it.

A new scenario's starting artifact holds ``evolution.seed``, so a client can pull before any step publishes. A recipe whose seed is empty returns 404 until a step publishes a tree. Use ``?artifact_id=`` to request a specific harness artifact, in any form. An unknown or unrestorable artifact returns 404.

``requires`` describes what the harness needs where it runs. It merges the ``training_request.requires`` items from the artifact's training history by name, with the newest definition winning. A later request that adds no items does not remove earlier requirements. One request's cap of 8 does not bound the combined list. Reef runs no check: the client checks each item before it runs the artifact. `Requirements of a change <http-api-training.rst#requirements-of-a-change>`__ gives the shape.

Catalog and manifest reads do not wait for an evolve step's method or evaluation runs: they serve the existing artifacts while a candidate is prepared. A manifest's artifact and evaluation metrics always come from the same artifact.

Importing a harness
-------------------

``POST /reef/harness/import`` commits the harness files of a person's agent as a sync, with ``x-reef-scenario``. A sync is a commit operation of its own: the merged tree becomes a new artifact, and the operation selects it for serving and as the harness trainer's start.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"files": {"plugin/rules.md": "Run the tests first.\n"},
          "base_artifact_id": "artifact-12"}' \
     "$REEF_URL/reef/harness/import"

- ``files`` maps each file's render path (``roots`` and ``files`` of ``GET /reef/harness/adapters``) to its text: every harness file of the install, changed or not. Reef maps each file back to the tree path of the part that renders there. A path that no part renders to is ignored and listed in ``ignored``. A file of the base that ``files`` lacks is one the person removed. Config files are not read: the model binding owns them.
- ``base_artifact_id`` is required: the artifact the client installed.
- The head here is the active selection, not the training start: a sync serves its result at once, so it never carries a kept tree that the activation rule left inactive.
- Reef puts the client's changes since the base on the head, file by file. A file that only the client changed takes the client's text; a file that only the head changed keeps the head's. A file that both changed merges as its part's ``merge`` says: ``lines`` keeps each side's changed lines, and ``file`` keeps neither. What cannot merge is HTTP 409, which names the file at its render path. An empty file counts as no file. The answer's ``merged`` is true when the head had moved on from the base.
- The merged tree must pass the adapter's validation, as a method's candidate does; a problem is HTTP 400 naming it. Reef does not evaluate the tree.
- A sync that changes nothing answers ``changed: false`` and commits nothing. Otherwise the commit's operation is ``sync``, and its metrics carry ``sync`` (with ``base_artifact_id``), ``patch`` and ``patch_parts``. The snapshot serves at once, without the activation rule.
- A sync while a step holds job data is HTTP 409; send it again when the step ends. So is a sync that another commit overtook.

Harness requests
----------------

A client submits a person's request for a harness change through ``POST /reef/train`` (`Manual training <http-api-training.rst#manual-training>`__), with ``text``, ``session`` and the ``artifact_id`` that it runs. The scenario must be in ``manual`` mode; a scenario in ``auto`` refuses the request, and the client shows that error and leaves the mode switch to the person. A request needs no inference receipts.

The trainer runs the method's step on the request, then evaluates and publishes under the same policy as automatic evolution. The snapshot catalog shows the commit metrics ``training_request``, ``method_summary`` and ``method_notes``.

Inactive artifacts
------------------

The deployment's activation rule can leave a new tree inactive. An example is a tree whose patch changes a file of a kind listed in the ``always`` rule's ``review_kinds``, or of a kind that is always reviewed (native's ``native_loop``). The step creates an artifact but does not activate it:

- Its row carries ``active: false`` and ``activated: false``.
- The manifest keeps serving the active selection, and ``?artifact_id=`` pulls the inactive tree for a trial run.
- To make it serve, send ``{"served": {"<artifact_id>": 100}}`` with the row's ``artifact_id`` to ``POST /reef/scenarios/{scenario}/update`` (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__).

``GET /reef/harness/artifacts`` lists only the harness component's artifacts. A weights update does not create another harness artifact. Each row's ``parent_artifact_id`` identifies the artifact from which it derives, when known. Every listed artifact can be requested through ``?artifact_id=``.

Version page
------------

``GET /reef/harness/artifacts/{step}/page`` answers one self-contained HTML page (``text/html``) for one catalog row. ``step`` is the row's position in ``GET /reef/harness/artifacts``, oldest first, with the creation row as 0. A rejected step creates no artifact, so the step number identifies the page independently of its artifact.

The page shows:

- why the step ran (the request, or a failure in the job data);
- ``method_summary``, the patch against the artifact that the step ran on, the evaluation's numbers, ``method_notes`` and ``requires``;
- links to the parent step and to later training or sync operations;
- ``Active selection`` for the selected artifact, and ``Waiting for activation`` for a step whose artifact was never activated (``activated: false``).

The row itself is in a ``<script type="application/json">`` block at the end of the page. A step outside the catalog, or one that is not a number of at most nine digits, returns 404.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/artifacts/3/page" > harness-step-3.html

Request page
------------

``GET /reef/harness/requests/{record_id}/page`` answers one self-contained HTML page (``text/html``, ``Cache-Control: no-store``) for a harness request filed through ``POST /reef/train``. ``record_id`` is the ``agent_record_id`` of the request's response, and its ``page_path`` links this page.

- Until a step settles the request, the page reloads every five seconds and shows the request's state and the step's activity (its latest 80 lines).
- The catalog row whose ``metrics.training_request.id`` is the record ID settles the page. It then shows the result, the next action (``GET /reef/harness`` for an active selection, the update request that serves an inactive one), ``method_summary``, the patch with each file's kind of the harness tree (from the row's ``patch_parts``), and ``method_notes``.
- An unknown ID, or one that is not a training request, returns 404.

``GET /reef/harness/requests/{record_id}/progress`` answers the same state as JSON (``Cache-Control: no-store``) for a client that polls:

- ``request_id`` and ``settled``;
- ``step``: the step the row landed as once it settles, else ``null``;
- ``state``: ``queued`` (no step has taken the request), ``proposing`` (the method makes the candidate), ``evaluating`` (the candidate's rollouts run), ``running`` (the trainer holds the request and reports no phase), or the settled row's result, ``failed`` included;
- ``meaning``: the words the page prints beside the state, ``null`` once settled;
- while a step holds the request: ``started_at``, ``runs_total``, ``step_record`` and ``activity``, the activity lines oldest first, each ``{at, kind, text}`` with ``failed: true`` on a failed line. ``kind`` is ``model`` for a model call, ``evaluate`` for the method's rollouts, or a kind that the method notes itself.

This route reads the headers alone; a ``?key=`` is HTTP 401. An unknown id is HTTP 404.

Page links
----------

A browser sends no header, so the two pages also take the scenario and a credential as query parameters, ``?scenario=<name>&key=<page key>``; a header wins when present. Clients never build these links: ``POST /reef/train`` answers ``page_path`` for the request page, and ``GET /reef/harness/artifacts`` one per row for its step's page. A client prints its service URL followed by that path.

.. code:: text

   $REEF_URL/reef/harness/requests/<record_id>/page?scenario=<scenario>&key=<page key>

- The path holds a page key only when the request presented a service token. An evaluation token gets none, and with authentication off the query names the scenario alone.
- The key is an HMAC SHA-256 of the scenario, keyed by the token's digest. It opens these two pages of that one scenario and no other route, and the token cannot be read back from it. So a client may print the links where a session's model reads them.
- A request whose ``x-reef-scenario`` header names another scenario returns 401.
- Reef never reads the token itself from a query: ``?token=`` returns 401 on every route, and ``?key=`` returns 401 on every route but the two pages.

Retained step files
-------------------

``GET /reef/harness/artifacts/{step}/records`` returns the raw file inventory of the same catalog step's record directory, authenticated and scoped to the scenario like the version page:

- Without a query, the response is ``{"status": "retained", "files": [{"path": "calls.json", "bytes": 123}]}``.
- With ``?path=calls.json`` (or another inventory path, such as ``output/notes.json`` or ``material/traces/0/trace.json``), it is ``{"status": "retained", "path": "...", "text": "..."}``.
- A step whose row names no record directory returns ``status: not_recorded``, and an absent directory returns ``status: missing``. Both carry an empty files list.
- All successful reads use ``Cache-Control: no-store``.

Reef reads only the step directory that the selected catalog row names, under the scenario's ``steps/`` directory:

- It rejects absolute paths, traversal and symlinks.
- It exposes only JSON and JSONL files. An inventory holds at most 1000 filesystem entries, and a file at most 4 MiB. Past these limits it returns 400; it never truncates.
- A missing file or catalog step returns 404.

The service does not interpret model replies or agent events. ``calls.json`` holds the recorded messages and replies of the method's model calls. The evaluation runs are in the scenario's history, not in the step directory, so this route does not serve them.

Reads copy no record into another store and change no retention. Step files remain separate from the capacity eviction of online record bodies.
