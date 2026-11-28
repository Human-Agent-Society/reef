HTTP API: harness trees and pages
=================================

Pull a harness tree, merge local changes, and inspect evolution results.
Use the service token and ``x-reef-scenario``. Unknown scenarios or missing trees
return 404; reads never create scenarios.

Harness artifacts
-----------------

``GET /reef/harness`` returns ``{artifact_id, parent_artifact_id, files,
evaluation, requires}``, plus component IDs for composed scenarios.
``x-reef-artifact-id`` identifies the returned tree.

- ``?form=rendered`` (default): agent paths with adapter defaults.
- ``?form=source``: stored paths for patches.
- ``?form=client``: a layer over local configuration. Requires ``service_url``
  and ``root`` queries. Replace ``REEF_TOKEN_PLACEHOLDER`` with the client's token.
  Returns ``run`` (binary/args/env), ``register``, ``unregister``, ``install``
  and ``synced`` path templates. Unsupported adapters return 400.

For rendered/source forms, replace ``{root}`` and supply your model binding;
these forms contain no endpoint or credentials. Unsupported forms return 400.
``?artifact_id=`` selects a restorable artifact, including an inactive one.
Unknown/unrestorable IDs return 404. Empty seeds return 404 until publication.

``GET /reef/harness/artifacts`` returns ``{scenario, artifacts}``, oldest first.
Rows include artifact/component IDs, ``active``, ``activated``, metrics and
``page_path``. Unpublished training steps have ``checkpoint: false``.
To activate a tree, update the scenario with ``{"served": {"<id>": 100}}``.

``GET /reef/harness/adapters`` lists resolved adapters: name, binary,
trajectory format, model bindings, pinned install spec, render roots and file paths.
``requires`` combines historical requirements by name; newest definitions win.
Clients check them before execution.

Importing a harness
-------------------

``POST /reef/harness/import`` merges installed files into the active tree,
then makes the result active and the trainer's start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"files": {"plugin/rules.md": "Run the tests first.\n"},
          "base_artifact_id": "artifact-12"}' \
     "$REEF_URL/reef/harness/import"

Send every installed render path, including unchanged files. Missing base files
mean deletion; empty files count as absent. Unknown paths and model-binding
config appear in ``ignored``. ``base_artifact_id`` is the installed artifact.

The response is ``{artifact_id, parent_artifact_id, changed, merged, files, ignored}``.
``merged`` means the active tree advanced beyond the base. No changes create no commit.
One-sided edits survive; conflicts follow the part's line/file merge policy.
Unmergeable paths or concurrent commits return 409. Invalid merged trees return 400.
Sync bypasses evaluation and activation rules.

Submit later changes with `training requests <http-api-training.rst#manual-training>`__.
Normal evaluation/publication rules apply to those requests.

Version page
------------

``GET /reef/harness/artifacts/{step}/page`` returns self-contained HTML with
the request, patch, metrics and result. ``step`` indexes the catalog from 0;
rejected steps also have pages. Unknown steps return 404.

``GET /reef/harness/artifacts/{step}/records`` lists retained JSON/JSONL files.
``?path=`` reads one as ``{status, path, text}``. Missing directories report
``missing``; steps without records report ``not_recorded``. Missing files return 404.
Paths must remain inside the step directory; symlinks are rejected.
Inventories exceeding 1000 entries or files exceeding 4 MiB return 400.
Reads use ``Cache-Control: no-store`` and do not change retention.

Request page
------------

``GET /reef/harness/requests/{record_id}/page`` shows a training request's
state, recent activity and final result. Use ``agent_record_id`` from the
training response; unknown/non-training IDs return 404.

``GET /reef/harness/requests/{record_id}/progress`` returns JSON:
``request_id``, ``settled``, ``step``, ``state``, ``meaning`` and activity.
Unsettled states include ``queued``, ``proposing``, ``evaluating`` and ``running``.
Both routes use ``Cache-Control: no-store``. Progress requires headers;
``?key=`` returns 401.

Page links
----------

Use the returned ``page_path`` for browser access. The two HTML pages accept
``?scenario=<name>&key=<page key>``; headers take precedence.
Keys permit only those pages in one scenario and reveal no service token.
A different scenario header returns 401. Evaluation tokens receive no page key.
With authentication disabled, links need only the scenario.
``?token=`` is always rejected; ``?key=`` is rejected outside the two HTML pages.
