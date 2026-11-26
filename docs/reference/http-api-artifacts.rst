HTTP API: artifacts
===================

List the artifacts of a scenario, read their files, and import a directory as a new artifact.

An artifact is an immutable component output, such as checkpoint weights or a harness tree. Training, import and harness sync create artifacts. Each component starts with an artifact from the default configuration or a selected artifact item.

Treat ``artifact_id`` as an opaque string. Do not derive a repository revision or component name from it. Updating one component leaves the other components' IDs unchanged. Activating or serving existing artifacts records operations and changes selections. Neither action creates an artifact or repository revision.

List the artifacts
------------------

``GET /reef/scenarios/{scenario}/artifacts`` answers ``{scenario, artifacts}``, **newest first**. Each row is one artifact of one component:

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
     - ``creation`` for the creation artifact, ``training``, ``import``, ``sync``, or ``recovery`` for the artifact that a store without a durable commit log served at a restart.
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

``parent_artifact_id`` names a parent in the same scenario. For training, it is the component's training start. For harness sync, it is the previously active artifact. Creation and ordinary imports have no parent. Imports retain their source in ``source``, including ``source.item`` for artifact items. Importing from another scenario creates no cross-scenario parent link.

Only operations that create artifacts appear here. Read `operation history <http-api-records.rst#record-and-commit-history>`__ for skipped or rejected steps, job submissions and completions, activations, training-start changes, and serving changes.

Files of an artifact
--------------------

``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` answers ``{scenario, artifact_id, files}``: each file of the artifact as ``{path, bytes, url, url_expires_at}``, in path order. The data service answers this route.

- If the backend supports signed URLs, ``url`` downloads the file directly from storage without a Reef token. It expires at ``url_expires_at`` (Unix seconds). Set its lifetime with ``?expires_in=``: 1–86400 seconds, default 3600. Use direct downloads for large artifacts.
- Otherwise, both URL fields are null and the data service serves the files. This applies to ``git_lfs``.

With ``?path=<path>``, the route downloads that file. In a scenario of several components, name the component with ``?component=``. An artifact that the scenario does not list, or a path that is not one of its files, returns 404. ``?signed=1`` returns a `signed download URL <http-api-data-items.rst#signed-download-urls>`__ of the list or the file.

Import an artifact
------------------

``POST /reef/scenarios/{scenario}/artifacts?component=<name>`` imports a directory. Omit ``component`` only for a scenario with one component. Send either:

- ``{"source": "<uri>"}`` to read a directory by URI.
- ``{"item": "<id>"}`` to read the ``source_uri`` of an `artifact item <http-api-data-items.rst#artifact-items>`__ owned by the scenario's owner.

Reef validates the source layout before copying it. Supported sources are:

- ``hf://org/model@revision``: a Hugging Face snapshot. With a full 40-digit commit SHA as ``revision``, a component that holds model weights keeps a reference instead of a copy (see below).
- ``s3://bucket/prefix/``: every object under the prefix. Empty folder markers, keys that end in ``/``, are skipped.
- ``file:///absolute/path``: a directory on the service's file system, percent encoded as ``Path.as_uri()`` writes it. Its files are copied.
- A scheme that ``storage.artifact-sources`` adds.

To import a directory from your own machine, first put it in one of these places, such as an S3 prefix.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/adapters/run-7/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

The response is HTTP 201 with ``{scenario, component, artifact_id}`` after durable storage completes. Allow a long client timeout for large imports. URI imports get new artifact IDs, even for identical content. Artifact-item imports preserve the item's artifact ID.

Reef records an ``import`` operation. Import does not change serving selections or training starts, or run the activation rule. Select the artifact explicitly to serve or train from it.

**Hugging Face references.** A model-weight import with a full commit SHA stores ``hf-reference.json`` instead of downloading the weights. Reef checks the layout through Hub metadata and small files, and stores the Hub's weight-file checksums. Other components, including harnesses, store copies.

Keep the Hub repository available while Reef serves or trains the referenced artifact. For private repositories, configure standard Hugging Face authentication, such as ``HF_TOKEN``, on the service. An unavailable revision returns 400.

**Layout.** Match the snapshot layout required by the component's training backend and configuration. Components without a snapshot layout do not accept imports.

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

**Base roots.** Slime and Tinker weight components also accept a directory containing only ``base.json``. It identifies the deployment's default start, for example ``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``.

The inputs must match ``source`` on the scenario's ``creation`` row. A different model, rank or service returns 400 and lists the mismatched inputs. Serving or training from this artifact starts at the base model without an adapter. Reef creates a base root for each new Slime or Tinker scenario; see `Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__.

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
