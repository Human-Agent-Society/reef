HTTP API: artifacts
===================

List the artifacts of a scenario, read their files, and import a directory as a new artifact.

An artifact is one immutable output of a component, such as checkpoint weights or a harness tree. Training, import and harness sync create artifacts. Scenario creation makes each component's starting artifact from the default configuration or a selected artifact item.

``artifact_id`` is the only public artifact identity. It is an opaque string: callers must not derive a repository revision or component name from it. An unchanged component keeps its artifact ID through another component's update. Activation and serving existing artifacts append operations and change scenario selections; they create no artifacts or repository revisions.

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

``parent_artifact_id`` describes derivation within the scenario. A training output follows that component's training start; a harness sync follows its previously active artifact. Creation and ordinary imports have no local parent, so the field is null. The import source remains in ``source``, including ``source.item`` for an artifact item. An import from another scenario creates no parent link across scenarios.

The list holds artifacts only. A training step that made no artifact, such as a rejected or skipped step, a job submission or a job's end, has no row; ``GET /reef/scenarios/{scenario}/operations`` holds its record (see `Record and commit history <http-api-records.rst#record-and-commit-history>`__). Activations, train-from operations and serves have no rows either.

Files of an artifact
--------------------

``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` answers ``{scenario, artifact_id, files}``: each file of the artifact as ``{path, bytes, url, url_expires_at}``, in path order. The data service answers this route.

- When the repository backend signs URLs of its files, ``url`` opens the file in the backend's storage, without a Reef token, until ``url_expires_at`` (Unix seconds). ``?expires_in=`` sets the life of these URLs in seconds: 1 to 86400, 3600 by default. The bytes of a download through these URLs do not pass through Reef, so use them for a large artifact.
- With a backend that signs no URLs, such as ``git_lfs``, both fields are null, and the data service serves the files itself.

With ``?path=<path>``, the route downloads that file. In a scenario of several components, name the component with ``?component=``. An artifact that the scenario does not list, or a path that is not one of its files, returns 404. ``?signed=1`` returns a `signed download URL <http-api-data-items.rst#signed-download-urls>`__ of the list or the file.

Import an artifact
------------------

``POST /reef/scenarios/{scenario}/artifacts?component=<name>`` makes a new artifact from the directory that a URI names. Leave out ``component`` when the scenario has one component. The body is ``{"source": "<uri>"}``, or ``{"item": "<id>"}`` for an `artifact item <http-api-data-items.rst#artifact-items>`__ of the scenario's owner, whose ``source_uri`` Reef reads. Reef checks the layout below on the source, and then copies the directory from one of these URIs:

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

The answer is HTTP 201 with ``{scenario, component, artifact_id}``. It comes after the artifact is durable, so give a large import a long client timeout. Each import creates a new artifact, even when another artifact contains the same files. Storage can reuse equal content without changing either ID.

Reef appends an ``import`` operation. Serving selections and training starts stay unchanged, and the activation rule does not run. Serve the artifact or train from it as from any other artifact.

**Hugging Face references.** An import of ``hf://org/model@<commit>`` with a full commit SHA into a component of model weights stores only a reference, ``hf-reference.json``. Any other component, such as a harness, stores a copy. Reef checks the layout from the revision's file list and its small files, and takes each weight file's SHA-256 from the Hub, so it downloads no weights. The repo must stay on the Hub while Reef serves or trains the artifact. A private repo needs a token that the service reads in Hugging Face's standard way, such as ``HF_TOKEN``. A revision that does not resolve for that token returns 400.

**Layout.** The directory must have the layout of the snapshots that the scenario's component makes, which the training backend and its configuration decide. A component without a layout, such as the records of a trainer that serves nothing, takes no import.

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

**Base roots.** A weight component of the Slime or the Tinker backend also takes a base root: a directory that holds ``base.json`` alone. The file names a deployment's default start by its inputs, for example ``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``, and the artifact stands for the base model with no adapter (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__). Reef accepts it only when the inputs equal those of this deployment's default start, which the ``source`` of a scenario's ``creation`` row shows. Another base model, rank or service returns 400 and names each input that differs. Serving or training from such an artifact starts at the base model. A new scenario of these backends starts from a base root that Reef writes (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).

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
