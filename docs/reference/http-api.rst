HTTP API: inference, feedback, and artifacts
============================================

Reef serves the provider's own inference routes: OpenAI Chat Completions at ``/v1/chat/completions``, OpenAI Responses at ``/v1/responses``, and Anthropic at ``/v1/messages``. It forwards each request to the runtime unchanged. It adds ``/reef/*`` routes for feedback, data, scenarios, artifacts, and status.

The service accepts request bodies up to 64 MiB, so coding agents can send large contexts. The record import routes accept bodies up to 10 MiB.

For a complete request, receipt, and feedback example, start with the `inference and feedback quickstart <../getting-started/quickstart.rst>`__. To put the snapshot routes into practice, follow the `agent harness tutorial <../user-guide/evolve-your-harness.rst>`__ or the `model weight training guide <../user-guide/evolve-your-model.rst>`__.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Pages
-----

Each page covers one part of the API. The service column names the service that answers most of the page's routes when the services run in separate processes (`Run the services in separate processes <../user-guide/split-services.rst>`__).

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Page
     - Service
     - Holds
   * - `Inference and feedback <http-api-inference.rst>`__
     - inference
     - Inference routes, startup status, receipts, the report body
   * - `Scenario routing <http-api-routing.rst>`__
     - inference
     - Routed external-model requests and the managed routing lifecycle
   * - `Records and operation history <http-api-records.rst>`__
     - control
     - Record imports, record reads, operation history
   * - `Data items <http-api-data-items.rst>`__
     - data
     - The item model; record, evaluation and artifact items; read, patch, delete, restore and download
   * - `Create and transfer data items <http-api-data-upload.rst>`__
     - data
     - Create trajectory and task items, replace a task's files, import and export
   * - `Benchmarks, collections and trajectory sets <http-api-data-sets.rst>`__
     - data
     - Benchmarks, collections, trajectory sets, and the sets that a scenario holds
   * - `Scenarios and their settings <http-api-scenarios.rst>`__
     - control
     - Create and delete scenarios; training mode, activation, serving config and the other settings
   * - `What a scenario serves <http-api-serving.rst>`__
     - control
     - The ``served`` mapping, traffic split, one served artifact, serve nothing, restore
   * - `Artifacts <http-api-artifacts.rst>`__
     - control, data
     - Artifact list, artifact files, artifact import
   * - `Training requests and jobs <http-api-training.rst>`__
     - control
     - Training requests (``POST /reef/train``), training jobs, train-from
   * - `Samples <http-api-samples.rst>`__
     - samples
     - Sample list, next job, sample processing tasks
   * - `Evaluate an artifact <http-api-evaluation.rst>`__
     - evaluation
     - Start, read, cancel, retry and resume an artifact evaluation; automatic evaluation
   * - `Calls and trajectories of an evaluation <http-api-evaluation-calls.rst>`__
     - evaluation
     - Model calls, verifier calls and trajectories of a rollout
   * - `Harness trees and pages <http-api-harness.rst>`__
     - harness
     - Harness tree, catalog, adapters, sync, step and request pages
   * - `Status, status codes and browser access <http-api-status.rst>`__
     - control
     - ``GET /reef/status``, status codes, browser consoles

Routes
------

Every route of the service, grouped by the page that documents it. A routed deployment serves fewer routes (`Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__).

Inference and feedback routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| Route                                                                            | Response                                          | Documented in                                                    |
+==================================================================================+===================================================+==================================================================+
| ``POST /v1/chat/completions``                                                    | OpenAI-format inference                           | `Request <http-api-inference.rst#request>`__                     |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /v1/responses``                                                           | OpenAI Responses-format inference                 | `Request <http-api-inference.rst#request>`__                     |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /v1/messages``                                                            | Anthropic-format inference                        | `Request <http-api-inference.rst#request>`__                     |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /v1/messages/count_tokens``                                               | count request tokens; recorded like any inference | `Request <http-api-inference.rst#request>`__                     |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/v1/{route}``                                   | inference with the scenario in the path           | `Request <http-api-inference.rst#request>`__                     |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``                        | inference by the active model; keeps no record    | `Evaluation routes <http-api-inference.rst#evaluation-routes>`__ |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}`` | the same, for a candidate of one component        | `Evaluation routes <http-api-inference.rst#evaluation-routes>`__ |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /reef/report``                                                            | submit feedback about one or more trajectories    | `Report <http-api-inference.rst#report>`__                       |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/inference/status``                              | startup state of a served artifact                | `Startup status <http-api-inference.rst#startup-status>`__       |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/redeploy``                                     | deploy the served engines again                   | `Startup status <http-api-inference.rst#startup-status>`__       |
+----------------------------------------------------------------------------------+---------------------------------------------------+------------------------------------------------------------------+

Record routes
~~~~~~~~~~~~~

+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| Route                                                  | Response                                   | Documented in                                                                                            |
+========================================================+============================================+==========================================================================================================+
| ``POST /reef/records``                                 | import one existing inference or report    | `Import records for cold-start learning <http-api-records.rst#import-records-for-cold-start-learning>`__ |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``POST /reef/records/batch``                           | atomically import a batch of records       | `Batch imports and resume <http-api-records.rst#batch-imports-and-resume>`__                             |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records``             | retained record metadata                   | `Record and commit history <http-api-records.rst#record-and-commit-history>`__                           |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/commits``             | compatibility route for operation history  | `Record and commit history <http-api-records.rst#record-and-commit-history>`__                           |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/operations``          | paginated operation history and selections | `Record and commit history <http-api-records.rst#record-and-commit-history>`__                           |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records/{record_id}`` | one retained record and its trace payload  | `Record and commit history <http-api-records.rst#record-and-commit-history>`__                           |
+--------------------------------------------------------+--------------------------------------------+----------------------------------------------------------------------------------------------------------+

Data routes
~~~~~~~~~~~

+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| Route                                                                           | Response                                         | Documented in                                                                                         |
+=================================================================================+==================================================+=======================================================================================================+
| ``POST /reef/namespaces/{owner}/data``                                          | create record and artifact items                 | `Record items <http-api-data-items.rst#record-items>`__                                               |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data``                                           | the owner's items, in arrival order              | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/groups/{key}``                              | the owner's items grouped by a metadata value    | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/sets/{kind}``                               | the owner's benchmarks or trajectory sets        | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}``                                      | one item, with its content                       | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/metadata``                                 | patch the metadata of items                      | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/data/{id}``                                   | delete one item; restore it for 30 days          | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/deleted-items``                                  | the owner's deleted items, newest first          | `Restore deleted items <http-api-data-items.rst#restore-deleted-items>`__                             |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/deleted-items/restore``                         | restore deleted items                            | `Restore deleted items <http-api-data-items.rst#restore-deleted-items>`__                             |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/deleted-items/{id}``                          | purge one deleted item for good                  | `Restore deleted items <http-api-data-items.rst#restore-deleted-items>`__                             |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}/archive``                              | download one item: a task as tar.gz, else JSON   | `Download an item <http-api-data-items.rst#download-an-item>`__                                       |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/tasks``                                        | the tasks the scenario uses, which trained       | `Read, query, update and delete items <http-api-data-items.rst#read-query-update-and-delete-items>`__ |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/trajectories``                             | create trajectory items from JSON Lines          | `Create trajectory items <http-api-data-upload.rst#create-trajectory-items>`__                        |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/tasks``                                    | create task items from a tar.gz, or by reference | `Create task items <http-api-data-upload.rst#create-task-items>`__                                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/data/tasks/{id}``                                | replace a task item's files, or by reference     | `Replace a task's files <http-api-data-upload.rst#replace-a-task-s-files>`__                          |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/imports``                                  | import the files at a URI as items               | `Import and export items <http-api-data-upload.rst#import-and-export-items>`__                        |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/imports/{id}``                              | the status of one import                         | `Import and export items <http-api-data-upload.rst#import-and-export-items>`__                        |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/exports``                                  | export items to a tar.gz at a URI                | `Import and export items <http-api-data-upload.rst#import-and-export-items>`__                        |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/exports/{id}``                              | the status of one export                         | `Import and export items <http-api-data-upload.rst#import-and-export-items>`__                        |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/collections``                                    | the owner's collections of benchmarks            | `Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/collections/{name}``                             | create or replace one collection                 | `Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/collections/{name}``                             | one collection, with its task counts             | `Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/collections/{name}``                          | delete one collection                            | `Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``POST /reef/namespaces/{owner}/collections/{name}/scenarios``                  | make a scenario hold a collection's benchmarks   | `Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__                    |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``                | the benchmarks that a scenario holds             | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``         | make a scenario hold a benchmark in a split      | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``      | end a scenario's hold of a benchmark             | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``GET /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets``           | the trajectory sets that a scenario holds        | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``    | make a scenario hold a trajectory set            | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}`` | end a scenario's hold of a trajectory set        | `The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__            |
+---------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------+

Scenario routes
~~~~~~~~~~~~~~~

+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| Route                                       | Response                                        | Documented in                                                        |
+=============================================+=================================================+======================================================================+
| ``GET /reef/scenarios``                     | scenario selections and harness adapter         | `Scenario routes <http-api-scenarios.rst#scenario-routes>`__         |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| ``POST /reef/scenarios``                    | create a scenario explicitly                    | `Create a scenario <http-api-scenarios.rst#create-a-scenario>`__     |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/update``  | update a scenario's settings and what it serves | `Scenario updates <http-api-scenarios.rst#scenario-updates>`__       |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/settings`` | read a scenario's settings                      | `Read the settings <http-api-scenarios.rst#read-the-settings>`__     |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | what this scenario accepts                      | `Scenario routes <http-api-scenarios.rst#scenario-routes>`__         |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``       | delete a scenario; the sweep archives its state | `Deleting a scenario <http-api-scenarios.rst#deleting-a-scenario>`__ |
+---------------------------------------------+-------------------------------------------------+----------------------------------------------------------------------+

Serving and artifact routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+
| Route                                                                  | Response                                   | Documented in                                                                        |
+========================================================================+============================================+======================================================================================+
| ``POST /reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}`` | inference by one served artifact, recorded | `Call one served artifact <http-api-serving.rst#call-one-served-artifact>`__         |
+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/restore``                            | restore the selection of an earlier commit | `Restore a scenario selection <http-api-serving.rst#restore-a-scenario-selection>`__ |
+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/artifacts``                           | ``{scenario, artifacts}``, newest first    | `List the artifacts <http-api-artifacts.rst#list-the-artifacts>`__                   |
+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``       | an artifact's file list, or one file       | `Files of an artifact <http-api-artifacts.rst#files-of-an-artifact>`__               |
+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/artifacts``                          | import a directory as a new artifact       | `Import an artifact <http-api-artifacts.rst#import-an-artifact>`__                   |
+------------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------+

Training and sample routes
~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| Route                                                               | Response                                          | Documented in                                                                          |
+=====================================================================+===================================================+========================================================================================+
| ``POST /reef/train``                                                | enqueue one training request                      | `Manual training <http-api-training.rst#manual-training>`__                            |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/train-from``                      | set the artifact that a trainer starts from       | `Train from an artifact <http-api-training.rst#train-from-an-artifact>`__              |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/jobs``                             | the scenario's training jobs in the jobs queue    | `Training jobs <http-api-training.rst#training-jobs>`__                                |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel``            | cancel one training job of the scenario           | `Training jobs <http-api-training.rst#training-jobs>`__                                |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel``    | cancel one training request of the scenario       | `Training jobs <http-api-training.rst#training-jobs>`__                                |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/samples``                          | the scenario's training samples, in use order     | `Choose the samples that train <http-api-samples.rst#choose-the-samples-that-train>`__ |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing``                | the scenario's sample processing tasks and counts | `Sample processing tasks <http-api-samples.rst#sample-processing-tasks>`__             |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}``      | one sample processing task of the scenario        | `Sample processing tasks <http-api-samples.rst#sample-processing-tasks>`__             |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` | the log of one attempt of a task                  | `Sample processing tasks <http-api-samples.rst#sample-processing-tasks>`__             |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/next-job``                         | the samples that the next automatic job trains    | `Choose the samples that train <http-api-samples.rst#choose-the-samples-that-train>`__ |
+---------------------------------------------------------------------+---------------------------------------------------+----------------------------------------------------------------------------------------+

Evaluation routes
~~~~~~~~~~~~~~~~~

+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| Route                                                                                                                 | Response                                           | Documented in                                                                                   |
+=======================================================================================================================+====================================================+=================================================================================================+
| ``POST /reef/scenarios/{scenario}/evaluations``                                                                       | evaluate one artifact of a snapshot                | `Start an evaluation <http-api-evaluation.rst#start-an-evaluation>`__                           |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``                                                | cancel one evaluation                              | `Cancel, retry and resume <http-api-evaluation.rst#cancel-retry-and-resume>`__                  |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``                                                 | run the failed rollouts again                      | `Cancel, retry and resume <http-api-evaluation.rst#cancel-retry-and-resume>`__                  |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``                                                | run the cancelled and the unstarted rollouts       | `Cancel, retry and resume <http-api-evaluation.rst#cancel-retry-and-resume>`__                  |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations``                                                                        | artifact evaluations, newest first                 | `Read an evaluation <http-api-evaluation.rst#read-an-evaluation>`__                             |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}``                                                        | one evaluation: its state and result               | `Read an evaluation <http-api-evaluation.rst#read-an-evaluation>`__                             |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts``                                               | the rollouts of one evaluation, as they run        | `Read an evaluation <http-api-evaluation.rst#read-an-evaluation>`__                             |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls``                   | the model calls of one rollout, without bodies     | `Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__           |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` | one model call of a rollout, with its bodies       | `Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__           |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories``            | the ATIF trajectories of each attempt of a rollout | `Trajectories of an evaluation <http-api-evaluation-calls.rst#trajectories-of-an-evaluation>`__ |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``                                           | the trajectories of an evaluation, as NDJSON       | `Trajectories of an evaluation <http-api-evaluation-calls.rst#trajectories-of-an-evaluation>`__ |
+-----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------+-------------------------------------------------------------------------------------------------+

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

+----------------------+-------------------------------------------+-----------------------------------------+
| Route                | Response                                  | Documented in                           |
+======================+===========================================+=========================================+
| ``GET /healthz``     | liveness of the process; needs no token   | `Health <http-api-status.rst#health>`__ |
+----------------------+-------------------------------------------+-----------------------------------------+
| ``GET /reef/status`` | training, serving, and storage state      | `Status <http-api-status.rst#status>`__ |
+----------------------+-------------------------------------------+-----------------------------------------+

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

``x-reef-tag-artifact_id`` names the harness artifact that the native client installed. Reef stores it under ``metadata.tags`` like any other tag, so a processor can read which harness artifact made the call. The tag does not change which artifact serves the call, or the record's ``artifact_id``, which names the artifact that answered. It is separate from ``x-reef-artifact-id``, the response header that names the model artifact.
