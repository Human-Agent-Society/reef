HTTP API: inference, feedback, and artifacts
============================================

Send provider-format bodies to OpenAI Chat Completions, OpenAI Responses or
Anthropic Messages routes. Use ``/reef/*`` for Reef operations.
Bodies allow 64 MiB; record imports allow 10 MiB.
See the `quickstart <../getting-started/quickstart.rst>`__ for a first request.

Routes
------

Routes grouped by their reference page. Routed deployments expose fewer routes (`Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__).

Inference and feedback routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Inference and feedback <http-api-inference.rst>`__.

- ``POST /v1/chat/completions``
- ``POST /v1/responses``
- ``POST /v1/messages``
- ``POST /v1/messages/count_tokens``
- ``POST /reef/scenarios/{scenario}/v1/{route}``
- ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``
- ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}``
- ``POST /reef/report``
- ``GET /reef/scenarios/{scenario}/inference/status``
- ``POST /reef/scenarios/{scenario}/redeploy``

Record routes
~~~~~~~~~~~~~

`Records and operation history <http-api-records.rst>`__.

- ``POST /reef/records``
- ``POST /reef/records/batch``
- ``GET /reef/scenarios/{scenario}/records``
- ``GET /reef/scenarios/{scenario}/commits``
- ``GET /reef/scenarios/{scenario}/operations``
- ``GET /reef/scenarios/{scenario}/records/{record_id}``

Data routes
~~~~~~~~~~~

`Data items <http-api-data-items.rst>`__, `uploads and transfers <http-api-data-upload.rst>`__, `sets and collections <http-api-data-sets.rst>`__.

- ``POST /reef/namespaces/{owner}/data``
- ``GET /reef/namespaces/{owner}/data``
- ``GET /reef/namespaces/{owner}/data/groups/{key}``
- ``GET /reef/namespaces/{owner}/data/sets/{kind}``
- ``GET /reef/namespaces/{owner}/data/{id}``
- ``POST /reef/namespaces/{owner}/data/metadata``
- ``DELETE /reef/namespaces/{owner}/data/{id}``
- ``GET /reef/namespaces/{owner}/deleted-items``
- ``POST /reef/namespaces/{owner}/deleted-items/restore``
- ``DELETE /reef/namespaces/{owner}/deleted-items/{id}``
- ``GET /reef/namespaces/{owner}/data/{id}/archive``
- ``GET /reef/scenarios/{scenario}/tasks``
- ``POST /reef/namespaces/{owner}/data/trajectories``
- ``POST /reef/namespaces/{owner}/data/tasks``
- ``PUT /reef/namespaces/{owner}/data/tasks/{id}``
- ``POST /reef/namespaces/{owner}/data/imports``
- ``GET /reef/namespaces/{owner}/data/imports/{id}``
- ``POST /reef/namespaces/{owner}/data/exports``
- ``GET /reef/namespaces/{owner}/data/exports/{id}``
- ``GET /reef/namespaces/{owner}/collections``
- ``PUT /reef/namespaces/{owner}/collections/{name}``
- ``GET /reef/namespaces/{owner}/collections/{name}``
- ``DELETE /reef/namespaces/{owner}/collections/{name}``
- ``POST /reef/namespaces/{owner}/collections/{name}/scenarios``
- ``GET /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``
- ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``
- ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``
- ``GET /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets``
- ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``
- ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``

Scenario routes
~~~~~~~~~~~~~~~

`Scenarios and settings <http-api-scenarios.rst>`__.

- ``GET /reef/scenarios``
- ``POST /reef/scenarios``
- ``POST /reef/scenarios/{scenario}/update``
- ``GET /reef/scenarios/{scenario}/settings``
- ``GET /reef/scenarios/{scenario}/contract``
- ``DELETE /reef/scenarios/{scenario}``

Serving and artifact routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Serving <http-api-serving.rst>`__ and `artifacts <http-api-artifacts.rst>`__.

- ``POST /reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``
- ``POST /reef/scenarios/{scenario}/restore``
- ``GET /reef/scenarios/{scenario}/artifacts``
- ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``
- ``POST /reef/scenarios/{scenario}/artifacts``

Training and sample routes
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Training <http-api-training.rst>`__ and `samples <http-api-samples.rst>`__.

- ``POST /reef/train``
- ``POST /reef/scenarios/{scenario}/train-from``
- ``GET /reef/scenarios/{scenario}/jobs``
- ``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel``
- ``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel``
- ``GET /reef/scenarios/{scenario}/samples``
- ``GET /reef/scenarios/{scenario}/sample-processing``
- ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}``
- ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs``
- ``GET /reef/scenarios/{scenario}/next-job``

Evaluation routes
~~~~~~~~~~~~~~~~~

`Evaluations <http-api-evaluation.rst>`__ and `evaluation calls <http-api-evaluation-calls.rst>`__.

- ``POST /reef/scenarios/{scenario}/evaluations``
- ``GET /reef/scenarios/{scenario}/evaluation-capabilities``
- ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``
- ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``
- ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``
- ``GET /reef/scenarios/{scenario}/evaluations``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls/{attempt}/{index}``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories``
- ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``

Harness routes
~~~~~~~~~~~~~~

`Harness trees and pages <http-api-harness.rst>`__.

- ``GET /reef/harness``
- ``POST /reef/harness/import``
- ``GET /reef/harness/artifacts``
- ``GET /reef/harness/requests/{record_id}/page``
- ``GET /reef/harness/requests/{record_id}/progress``
- ``GET /reef/harness/artifacts/{step}/records``
- ``GET /reef/harness/adapters``

+---------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/page`` |
+---------------------------------------------+

Status routes
~~~~~~~~~~~~~

`Status <http-api-status.rst>`__.

- ``GET /healthz``
- ``GET /reef/status``

Headers
-------

- ``Authorization: Bearer <token>``: required when authentication is configured,
  except for health/readiness. Without Authorization, ``x-api-key`` carries the token.
  Evaluation routes also accept the recipe's startup token.
- ``x-reef-scenario``: required on inference, reports, harness routes and record
  imports unless the scenario is in the path.
- ``x-reef-artifact-id``: response-only identity of the answering model or harness tree.
- ``x-reef-rollout: 1``: service-token inference without records or training tensors.
  Streaming returns 400.
- ``x-reef-record-training: 1``: the record keeps the call's exact sample as ``response.training``; the response does not change.
  A streaming or record-less call returns 400, as does one whose ``temperature``, ``top_p`` or ``top_k`` neither the request nor the scenario names.
  The engine must be SGLang or vLLM chat on a route that names one weight load, without ``min_p``, penalties or logit bias; otherwise 400.

Client artifact tags
~~~~~~~~~~~~~~~~~~~~

``x-reef-tag-<name>`` stores context in ``metadata.tags``. In particular,
``x-reef-tag-artifact_id`` identifies the client's installed harness.
Tags change neither routing nor the responding model's artifact ID.
