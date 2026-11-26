HTTP API: artifacts
===================

List, download and import immutable component artifacts. ``artifact_id`` is opaque; component updates preserve other components' IDs. Activation and serving record operations without creating artifacts or repository revisions. See `Artifact <glossary.rst#artifact>`__.

List the artifacts
------------------

``GET /reef/scenarios/{scenario}/artifacts`` returns ``{scenario, artifacts}``, newest first:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``artifact_id``
     - The artifact.
   * - ``component``
     - The component whose content it is.
   * - ``parent_artifact_id``
     - The artifact that this one derives from, or null.
   * - ``operation``
     - ``creation``, ``training``, ``import``, ``sync``, or ``recovery`` when restored without a durable commit log.
   * - ``recorded_at``
     - The commit time, ``null`` for the creation artifact.
   * - ``active``
     - True for the active artifact of each component.
   * - ``activated``
     - True when the artifact has been active at some time.
   * - ``served``
     - True when the scenario serves the artifact beside the active artifact (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__).
   * - ``metrics``, ``record_progress``
     - Training rows only: the step's metrics and the record high-water mark that the step consumed.
   * - ``source``
     - Import rows: ``{uri, version}`` that the import read. The creation row: ``{kind: "default", inputs}`` or ``{kind: "item", item, uri, version}``.

Parents are scenario-local: training uses its start; sync uses the previous active artifact; creation and ordinary imports have none. Imports keep ``source`` and optional ``source.item``, never cross-scenario parent links. Non-artifact steps and selection changes appear only in `operation history <http-api-records.rst#record-and-commit-history>`__.

Files of an artifact
--------------------

Data serves ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``: ``{scenario, artifact_id, files}``, with ``{path, bytes, url, url_expires_at}`` in path order.

- Signed ``url`` values download directly from storage without tokens until ``url_expires_at`` (Unix seconds). ``expires_in``: 1–86400 seconds, default 3600. Prefer these for large files.
- Unsupported backends, including ``git_lfs``, return null URL fields; data serves downloads.

``path`` downloads one file; multi-component scenarios require ``component``. Unlisted artifacts or files return 404. ``signed=1`` returns a `signed URL <http-api-data-items.rst#signed-download-urls>`__ for the list or file.

Import an artifact
------------------

``POST /reef/scenarios/{scenario}/artifacts?component=<name>`` imports a directory. Omit ``component`` only for a scenario with one component. Send either:

- ``{"source": "<uri>"}`` to read a directory by URI.
- ``{"item": "<id>"}`` to read the ``source_uri`` of an `artifact item <http-api-data-items.rst#artifact-items>`__ owned by the scenario's owner.

Layout validation precedes copying. Sources:

- ``hf://org/model@revision``: Hugging Face snapshot; full 40-digit commit SHAs allow weight references below.
- ``s3://bucket/prefix/``: prefix objects, excluding folder markers ending in ``/``.
- ``file:///absolute/path``: service-local directory, percent-encoded as ``Path.as_uri()``; files are copied.
- A scheme that ``storage.artifact-sources`` adds.

Upload local files to an accessible source first:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/adapters/run-7/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

HTTP 201 returns ``{scenario, component, artifact_id}`` after durable storage; allow long timeouts. URI imports create new IDs even for identical content; item imports preserve IDs. The ``import`` operation changes no serving/training selection and runs no activation rule.

**Hugging Face references.** Full-SHA weight imports store ``hf-reference.json``, using Hub metadata, small files and weight checksums without downloading weights. Other components copy files. Keep repositories available for serving/training and configure private-repository authentication, such as ``HF_TOKEN``. Unavailable revisions return 400.

**Layout.** Imports require the backend/configuration's snapshot layout; components without one reject imports.

+-----------------------------+------------------------------------------------+------------------+
| Component                   | The directory holds                            | Other files      |
+=============================+================================================+==================+
| Full weights (Slime)        | ``config.json`` and ``*.safetensors`` or       | allowed          |
|                             | ``*.bin`` at the root                          |                  |
+-----------------------------+------------------------------------------------+------------------+
| LoRA adapter (Slime)        | ``adapter/adapter_config.json`` and            | allowed          |
|                             | ``adapter/adapter_model.safetensors`` or       |                  |
|                             | ``.bin``; ``r`` is at most the configured      |                  |
|                             | rank, and ``target_modules`` is a list within  |                  |
|                             | the configured modules                         |                  |
+-----------------------------+------------------------------------------------+------------------+
| Tinker                      | ``tinker-checkpoint.json`` of the configured   | allowed          |
|                             | model, rank and service; its ``state_path``    |                  |
|                             | resolves on that service as a LoRA of that     |                  |
|                             | model and rank                                 |                  |
+-----------------------------+------------------------------------------------+------------------+
| Harness                     | a source tree whose files are UTF-8 text and   | refused: a path  |
|                             | pass validation for the adapter, the files a   | off the harness  |
|                             | harness artifact stores                        | layout fails     |
|                             |                                                | validation       |
+-----------------------------+------------------------------------------------+------------------+

**Base roots.** Slime/Tinker accept ``base.json`` alone, for example ``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``. Inputs must match the creation row's ``source``; mismatched model/rank/service returns 400 naming differences. These artifacts serve/train the base without adapters and are created for new scenarios (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 201    | the artifact is imported                                           |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body or URI; a URI of an unknown scheme or of nothing  |
|        | to read; an unknown ``component``, or ``component`` is missing in  |
|        | a scenario with several components; the component takes no         |
|        | import (Reef reads nothing for these three); the directory does    |
|        | not have the component's layout, or its admission check refuses it |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario; Reef reads nothing then                          |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
+--------+--------------------------------------------------------------------+
| 503    | the source failed, for example S3 refused the read                 |
+--------+--------------------------------------------------------------------+
