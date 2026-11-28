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

- Slime full weights: root ``config.json`` and safetensors/bin weights.
- Slime LoRA: ``adapter/adapter_config.json`` plus adapter safetensors/bin;
  rank and target modules must fit configuration.
- Tinker: ``tinker-checkpoint.json`` resolving to the configured model, rank and service.
- Harness: UTF-8 source files passing the adapter's layout validation.

**Base roots.** Slime/Tinker accept ``base.json`` alone, for example ``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``. Inputs must match the creation row's ``source``; mismatched model/rank/service returns 400 naming differences. These artifacts serve/train the base without adapters and are created for new scenarios (`Artifacts <../user-guide/operate.rst#read-the-artifacts>`__).

Invalid bodies, layouts or sources return 400; unknown scenarios return 404.
Pending training commits return 409; source-service failures return 503.
Unknown/unsupported components fail before reading the source.
