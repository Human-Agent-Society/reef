HTTP API: inference, feedback, and artifacts
============================================

Reef supports OpenAI Chat Completions (``/v1/chat/completions``), OpenAI Responses (``/v1/responses``), and Anthropic Messages (``/v1/messages``). Send provider-format request bodies. Use ``/reef/*`` routes for feedback, data, scenarios, artifacts and status.

Request bodies can be up to 64 MiB. Record imports have a 10 MiB limit.

Start with the `inference and feedback quickstart <../getting-started/quickstart.rst>`__. For artifact workflows, see `harness evolution <../user-guide/evolve-your-harness.rst>`__ or `model training <../user-guide/evolve-your-model.rst>`__.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Pages
-----

Each page covers one API area. Services below expose its routes in a `split deployment <../user-guide/split-services.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 75 25

   * - Page
     - Service
   * - `Inference and feedback <http-api-inference.rst>`__
     - inference, control
   * - `Scenario routing <http-api-routing.rst>`__
     - inference
   * - `Records and operation history <http-api-records.rst>`__
     - control, data
   * - `Data items <http-api-data-items.rst>`__
     - data
   * - `Create and transfer data items <http-api-data-upload.rst>`__
     - data
   * - `Benchmarks, collections and trajectory sets <http-api-data-sets.rst>`__
     - data
   * - `Scenarios and their settings <http-api-scenarios.rst>`__
     - control
   * - `What a scenario serves <http-api-serving.rst>`__
     - control
   * - `Artifacts <http-api-artifacts.rst>`__
     - control, data
   * - `Training requests and jobs <http-api-training.rst>`__
     - control, data
   * - `Samples <http-api-samples.rst>`__
     - control, data
   * - `Evaluate an artifact <http-api-evaluation.rst>`__
     - control, data
   * - `Calls and trajectories of an evaluation <http-api-evaluation-calls.rst>`__
     - control, data
   * - `Harness trees and pages <http-api-harness.rst>`__
     - harness, control
   * - `Status, status codes and browser access <http-api-status.rst>`__
     - control

Routes
------

Routes grouped by their reference page. Routed deployments expose fewer routes (`Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__).

Inference and feedback routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Inference and feedback <http-api-inference.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /v1/chat/completions``
     - OpenAI-format inference
   * - ``POST /v1/responses``
     - OpenAI Responses-format inference
   * - ``POST /v1/messages``
     - Anthropic-format inference
   * - ``POST /v1/messages/count_tokens``
     - count request tokens; recorded like any inference
   * - ``POST /reef/scenarios/{scenario}/v1/{route}``
     - inference with the scenario in the path
   * - ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``
     - inference by the active model; keeps no record
   * - ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}``
     - the same, for a candidate of one component
   * - ``POST /reef/report``
     - submit feedback about one or more trajectories
   * - ``GET /reef/scenarios/{scenario}/inference/status``
     - startup state of a served artifact
   * - ``POST /reef/scenarios/{scenario}/redeploy``
     - deploy the served engines again

Record routes
~~~~~~~~~~~~~

`Records and operation history <http-api-records.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /reef/records``
     - import one existing inference or report
   * - ``POST /reef/records/batch``
     - atomically import a batch of records
   * - ``GET /reef/scenarios/{scenario}/records``
     - retained record metadata
   * - ``GET /reef/scenarios/{scenario}/commits``
     - compatibility route for operation history
   * - ``GET /reef/scenarios/{scenario}/operations``
     - paginated operation history and selections
   * - ``GET /reef/scenarios/{scenario}/records/{record_id}``
     - one retained record and its trace payload

Data routes
~~~~~~~~~~~

`Data items <http-api-data-items.rst>`__, `uploads and transfers <http-api-data-upload.rst>`__, `sets and collections <http-api-data-sets.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /reef/namespaces/{owner}/data``
     - create record and artifact items
   * - ``GET /reef/namespaces/{owner}/data``
     - the owner's items, in arrival order
   * - ``GET /reef/namespaces/{owner}/data/groups/{key}``
     - the owner's items grouped by a metadata value
   * - ``GET /reef/namespaces/{owner}/data/sets/{kind}``
     - the owner's benchmarks or trajectory sets
   * - ``GET /reef/namespaces/{owner}/data/{id}``
     - one item, with its content
   * - ``POST /reef/namespaces/{owner}/data/metadata``
     - patch the metadata of items
   * - ``DELETE /reef/namespaces/{owner}/data/{id}``
     - delete one item; restore it for 30 days
   * - ``GET /reef/namespaces/{owner}/deleted-items``
     - the owner's deleted items, newest first
   * - ``POST /reef/namespaces/{owner}/deleted-items/restore``
     - restore deleted items
   * - ``DELETE /reef/namespaces/{owner}/deleted-items/{id}``
     - purge one deleted item for good
   * - ``GET /reef/namespaces/{owner}/data/{id}/archive``
     - download one item: a task as tar.gz, else JSON
   * - ``GET /reef/scenarios/{scenario}/tasks``
     - the tasks the scenario uses, which trained
   * - ``POST /reef/namespaces/{owner}/data/trajectories``
     - create trajectory items from JSON Lines
   * - ``POST /reef/namespaces/{owner}/data/tasks``
     - create task items from a tar.gz, or by reference
   * - ``PUT /reef/namespaces/{owner}/data/tasks/{id}``
     - replace a task item's files, or by reference
   * - ``POST /reef/namespaces/{owner}/data/imports``
     - import the files at a URI as items
   * - ``GET /reef/namespaces/{owner}/data/imports/{id}``
     - the status of one import
   * - ``POST /reef/namespaces/{owner}/data/exports``
     - export items to a tar.gz at a URI
   * - ``GET /reef/namespaces/{owner}/data/exports/{id}``
     - the status of one export
   * - ``GET /reef/namespaces/{owner}/collections``
     - the owner's collections of benchmarks
   * - ``PUT /reef/namespaces/{owner}/collections/{name}``
     - create or replace one collection
   * - ``GET /reef/namespaces/{owner}/collections/{name}``
     - one collection, with its task counts
   * - ``DELETE /reef/namespaces/{owner}/collections/{name}``
     - delete one collection
   * - ``POST /reef/namespaces/{owner}/collections/{name}/scenarios``
     - make a scenario hold a collection's benchmarks
   * - ``GET /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``
     - the benchmarks that a scenario holds
   * - ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``
     - make a scenario hold a benchmark in a split
   * - ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``
     - end a scenario's hold of a benchmark
   * - ``GET /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets``
     - the trajectory sets that a scenario holds
   * - ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``
     - make a scenario hold a trajectory set
   * - ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``
     - end a scenario's hold of a trajectory set

Scenario routes
~~~~~~~~~~~~~~~

`Scenarios and settings <http-api-scenarios.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``GET /reef/scenarios``
     - scenario selections and harness adapter
   * - ``POST /reef/scenarios``
     - create a scenario explicitly
   * - ``POST /reef/scenarios/{scenario}/update``
     - update a scenario's settings and what it serves
   * - ``GET /reef/scenarios/{scenario}/settings``
     - read a scenario's settings
   * - ``GET /reef/scenarios/{scenario}/contract``
     - what this scenario accepts
   * - ``DELETE /reef/scenarios/{scenario}``
     - delete a scenario; the sweep archives its state

Serving and artifact routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Serving <http-api-serving.rst>`__ and `artifacts <http-api-artifacts.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``
     - inference by one served artifact, recorded
   * - ``POST /reef/scenarios/{scenario}/restore``
     - restore the selection of an earlier commit
   * - ``GET /reef/scenarios/{scenario}/artifacts``
     - ``{scenario, artifacts}``, newest first
   * - ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``
     - an artifact's file list, or one file
   * - ``POST /reef/scenarios/{scenario}/artifacts``
     - import a directory as a new artifact

Training and sample routes
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Training <http-api-training.rst>`__ and `samples <http-api-samples.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /reef/train``
     - enqueue one training request
   * - ``POST /reef/scenarios/{scenario}/train-from``
     - set the artifact that a trainer starts from
   * - ``GET /reef/scenarios/{scenario}/jobs``
     - the scenario's training jobs in the jobs queue
   * - ``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel``
     - cancel one training job of the scenario
   * - ``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel``
     - cancel one training request of the scenario
   * - ``GET /reef/scenarios/{scenario}/samples``
     - the scenario's training samples, in use order
   * - ``GET /reef/scenarios/{scenario}/sample-processing``
     - the scenario's sample processing tasks and counts
   * - ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}``
     - one sample processing task of the scenario
   * - ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs``
     - the log of one attempt of a task
   * - ``GET /reef/scenarios/{scenario}/next-job``
     - the samples that the next automatic job trains

Evaluation routes
~~~~~~~~~~~~~~~~~

`Evaluations <http-api-evaluation.rst>`__ and `evaluation calls <http-api-evaluation-calls.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``POST /reef/scenarios/{scenario}/evaluations``
     - evaluate one artifact of a snapshot
   * - ``GET /reef/scenarios/{scenario}/evaluation-capabilities``
     - check support for evaluation start IDs
   * - ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``
     - cancel one evaluation
   * - ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``
     - run the failed rollouts again
   * - ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``
     - run the cancelled and the unstarted rollouts
   * - ``GET /reef/scenarios/{scenario}/evaluations``
     - artifact evaluations, newest first
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}``
     - one evaluation: its state and result
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts``
     - the rollouts of one evaluation, as they run
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls``
     - the model calls of one rollout, without bodies
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls/{attempt}/{index}``
     - one model call of a rollout, with its bodies
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories``
     - the ATIF trajectories of each attempt of a rollout
   * - ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``
     - the trajectories of an evaluation, as NDJSON

Harness routes
~~~~~~~~~~~~~~

+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| Route                                               | Response                                      | Documented in                                                      |
+=====================================================+===============================================+====================================================================+
| ``GET /reef/harness``                               | the harness tree of the active selection      | `Harness artifacts <http-api-harness.rst#harness-artifacts>`__     |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``POST /reef/harness/import``                       | merge and commit a person's harness files     | `Importing a harness <http-api-harness.rst#importing-a-harness>`__ |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/artifacts``                     | harness artifact catalog, oldest first        | `Harness artifacts <http-api-harness.rst#harness-artifacts>`__     |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/page``         | the HTML page of one catalog step             | `Version page <http-api-harness.rst#version-page>`__               |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/page``     | the HTML page of one harness request          | `Request page <http-api-harness.rst#request-page>`__               |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/progress`` | the same state as JSON, for a polling client  | `Request page <http-api-harness.rst#request-page>`__               |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/records``      | retained raw step file inventory or file body | `Retained step files <http-api-harness.rst#retained-step-files>`__ |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+
| ``GET /reef/harness/adapters``                      | every harness adapter this process resolves   | `Harness artifacts <http-api-harness.rst#harness-artifacts>`__     |
+-----------------------------------------------------+-----------------------------------------------+--------------------------------------------------------------------+

Status routes
~~~~~~~~~~~~~

`Status <http-api-status.rst>`__.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Route
     - Result
   * - ``GET /healthz``
     - liveness of the process; needs no token
   * - ``GET /reef/status``
     - training, serving, and storage state

Headers
-------

+-----------------------------------+---------------------------------------------------------+
| Header                            | Required for                                            |
+===================================+=========================================================+
| ``x-reef-scenario``               | inference, report, harness manifest, artifacts and      |
|                                   | import. Names the workload a record belongs to.         |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz`` and, on a routed    |
|                                   | inference service, ``GET /readyz``, when auth is        |
|                                   | configured; with no Authorization header, ``x-api-key`` |
|                                   | carries the token. The evaluation routes also accept    |
|                                   | the token the service issues the recipe at startup.     |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-artifact-id``            | response only: the artifact that answered inference or  |
|                                   | supplied the harness tree. To pull a specific harness   |
|                                   | artifact, use ``?artifact_id=``.                        |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: context stored under             |
|                                   | ``metadata.tags``; see `Client artifact tags`_ below.   |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-rollout: 1``             | optional on inference: a call of a rollout that Reef    |
|                                   | runs itself. It is served like any other call, but no   |
|                                   | record is kept. The response has no ``training`` block; |
|                                   | ``x-reef-artifact-id`` names the answering artifact.    |
|                                   | The call must not stream (HTTP 400). Only the service   |
|                                   | token opens it.                                         |
+-----------------------------------+---------------------------------------------------------+

Client artifact tags
~~~~~~~~~~~~~~~~~~~~

``x-reef-tag-artifact_id`` identifies the native client's installed harness artifact. Reef stores it in ``metadata.tags`` for processors to read. It changes neither routing nor the record's ``artifact_id``. The response header ``x-reef-artifact-id`` identifies the responding model artifact.
