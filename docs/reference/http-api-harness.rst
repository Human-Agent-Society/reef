HTTP API: harness trees and pages
=================================

Pull harness trees, sync local files, and inspect training steps or requests.

``reef-client harness`` installs the active tree, binds its model to Reef and reports inference receipts (`Use a published tree <../user-guide/evolve-your-harness.rst#use-a-published-tree>`__). In split deployments, harness serves the four routes below; control serves the remaining routes.

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

Manifest and catalog reads require ``x-reef-scenario``. Unknown scenarios or missing harness trees return 404; reads never create scenarios.

``?form=`` selects the ``files`` mapping, from paths to text. Unsupported values return 400:

- ``rendered`` (default): agent paths with enforced adapter defaults (`Rendering <../developer-guide/harness-adapters.rst#rendering>`__).
- ``source``: stored tree paths used by patches (`Tree parts <../developer-guide/harness-adapters.rst#tree-parts>`__).
- ``client``: a layer over local config. Required queries ``service_url`` and ``root`` select Reef's client-accessible address and install directory. The model binds to ``<service_url>/reef/scenarios/<scenario>``. Replace ``REEF_TOKEN_PLACEHOLDER`` with the client's token. Adapters without ``layer`` return 400. Additional fields:

  - ``run``: ``{binary, args, env}``; ``env`` can also contain the token placeholder.
  - ``register``, ``unregister``: install-registration and removal commands as argument lists, usually empty.
  - ``install``: pinned installation spec.
  - ``synced``: upload path templates, with ``{name}`` matching one segment.

For ``rendered`` and ``source``, replace ``{root}`` with the render-root directory and supply your own model binding; neither includes endpoints or credentials.

Initially, the tree is ``evolution.seed``; empty seeds return 404 until publication. ``?artifact_id=`` selects any restorable artifact in any form; unknown or unrestorable IDs return 404.

``requires`` merges history by name, newest definition winning. Empty requests remove nothing; the combined list can exceed 8. Clients check requirements before execution (`Requirement fields <http-api-training.rst#requirements-of-a-change>`__).

Reads serve existing artifacts during evolution. Manifest artifacts and evaluation metrics always match.

Importing a harness
-------------------

``POST /reef/harness/import`` syncs local files to ``x-reef-scenario``. The merged artifact becomes active and the harness trainer's start.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"files": {"plugin/rules.md": "Run the tests first.\n"},
          "base_artifact_id": "artifact-12"}' \
     "$REEF_URL/reef/harness/import"

- ``files`` maps every installed harness render path to text, including unchanged files. Adapter ``roots`` and ``files`` define paths. Missing base files mean deletions; unknown paths appear in ``ignored``. Config files belong to model binding and are ignored.
- ``base_artifact_id`` is required: the artifact the client installed.
- Merge against the active tree, not an inactive training start. One-sided changes survive. Conflicts use the part's ``merge`` policy: ``lines`` combines changes; ``file`` rejects conflicts. Unmergeable paths return 409. Empty files count as absent.
- ``merged`` is true when the active tree advanced beyond the base. Invalid merged trees return 400; evaluation does not run.
- No changes returns ``changed: false`` without a commit. Otherwise ``sync`` commits ``sync.base_artifact_id``, ``patch`` and ``patch_parts`` metrics and bypasses activation rules.
- Held job data or an intervening commit returns 409; retry afterward.

Harness requests
----------------

Submit harness changes through ``POST /reef/train`` with ``text``, ``session`` and the currently used ``artifact_id``. Requests work in both ``auto`` and ``manual`` modes and need no inference receipts. See `Manual training <http-api-training.rst#manual-training>`__.

Normal evaluation and publication apply. Catalog metrics include ``training_request``, ``method_summary`` and ``method_notes``.

Inactive artifacts
------------------

Activation can hold trees for review, including ``review_kinds`` and always-reviewed kinds such as ``native_loop``:

- Its row carries ``active: false`` and ``activated: false``.
- The manifest keeps serving the active selection, and ``?artifact_id=`` pulls the inactive tree for a trial run.
- To make it serve, send ``{"served": {"<artifact_id>": 100}}`` with the row's ``artifact_id`` to ``POST /reef/scenarios/{scenario}/update`` (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__).

The catalog excludes weight updates. ``parent_artifact_id`` identifies derivation when known; every listed artifact accepts ``?artifact_id=``.

Version page
------------

``GET /reef/harness/artifacts/{step}/page`` returns self-contained ``text/html``. ``step`` indexes the catalog, oldest first, with creation at 0. Rejected steps also have pages.

The page shows:

- why the step ran (the request, or a failure in the job data);
- ``method_summary``, the patch against the artifact that the step ran on, the evaluation's numbers, ``method_notes`` and ``requires``;
- links to the parent step and to later training or sync operations;
- ``Active selection`` for the selected artifact, and ``Waiting for activation`` for a step whose artifact was never activated (``activated: false``).

The row is embedded in ``<script type="application/json">``. Unknown steps or values other than numbers of at most nine digits return 404.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/artifacts/3/page" > harness-step-3.html

Request page
------------

``GET /reef/harness/requests/{record_id}/page`` returns ``text/html`` with ``Cache-Control: no-store``. Use the training response's ``agent_record_id`` and ``page_path``.

- Unsettled pages reload every five seconds, showing state and the latest 80 activity lines.
- The matching ``metrics.training_request.id`` settles the page. It shows result, next action, ``method_summary``, patch file kinds from ``patch_parts``, and ``method_notes``. Active results link to ``GET /reef/harness``; inactive results offer the serving update.
- An unknown ID, or one that is not a training request, returns 404.

``GET /reef/harness/requests/{record_id}/progress`` answers the same state as JSON (``Cache-Control: no-store``) for a client that polls:

- ``request_id`` and ``settled``;
- ``step``: the step the row landed as once it settles, else ``null``;
- ``state``: ``queued``, ``proposing``, ``evaluating``, ``running`` (no reported phase), or the settled result, including ``failed``;
- ``meaning``: the words the page prints beside the state, ``null`` once settled;
- During a step: ``started_at``, ``runs_total``, ``step_record`` and oldest-first ``activity`` entries ``{at, kind, text}``, plus ``failed: true`` for failures. Kinds are ``model``, ``evaluate`` or method-defined.

This route reads the headers alone; a ``?key=`` is HTTP 401. An unknown id is HTTP 404.

Page links
----------

For browser links, both pages accept ``?scenario=<name>&key=<page key>``. Headers take precedence. Use the returned ``page_path`` from ``POST /reef/train`` or each ``GET /reef/harness/artifacts`` row. Append it to the service URL:

.. code:: text

   $REEF_URL/reef/harness/requests/<record_id>/page?scenario=<scenario>&key=<page key>

- The path holds a page key only when the request presented a service token. An evaluation token gets none, and with authentication off the query names the scenario alone.
- Page keys are HMAC SHA-256 values scoped to these two pages of one scenario. They cannot reveal the service token and may be shown to the session's model.
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

``calls.json`` contains raw method model exchanges. Evaluation runs live in scenario history and are excluded. Reads copy nothing and change no retention; online-record capacity eviction does not remove step files.
