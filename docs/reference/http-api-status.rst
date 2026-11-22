HTTP API: status, status codes and browser access
=================================================

Read what a deployment trains and serves, look up the cause of a status code, and let a browser console call the API.

Health
------

``GET /healthz`` answers ``{"ok": true}`` from a running process and needs no token. It is a liveness check: on a managed routed deployment it does not show that the deployment accepts inference (`Managed routing lifecycle <http-api-routing.rst#managed-routing-lifecycle>`__). An inference service with the external router lifecycle also answers ``GET /readyz`` without a token.

Status
------

``GET /reef/status`` reports what trains and what serves:

.. code:: json

   {
     "error": null,
     "loaded_scenarios": 1,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "commit_number": 3,
         "artifact_head_sync": {"state": "synchronized", "error": null},
         "last_training_commit": {
           "commit_number": 3,
           "recorded_at": 1756400000.0,
           "metrics": {"selected": false, "selection": {"reason": "candidate lost"}}
         },
         "served_artifact_id": "5e1c...",
         "job_data_ready": false,
         "training_mode": "auto",
         "pipeline": {"...": "..."},
         "inference_admission": {"...": "..."},
         "artifacts": {"weights": {"active": "5e1c...", "training_start": "5e1c..."}},
         "training_jobs": [{"job_id": "9c41...", "start": "/checkpoints/iter_0000004", "checkpoints_handled": 0,
                            "train_from": null, "backend": "slime"}],
         "finished_jobs": 2
       }
     },
     "serving": {"...": "..."}
   }

Top-level fields:

- ``error`` and ``preload_errors`` report asynchronous training and preload failures.
- ``loaded_scenarios`` counts the scenarios loaded in this process.
- ``serving`` is runtime-wide and keyed by recipe name. The runtime of a weight-training recipe that serves per-scenario adapters reports the engines' shared adapter residency there, under ``adapters``. For a composite recipe, ``adapters`` is under the name of the weight component.

Fields of each scenario:

- ``commit_number`` is the number of the scenario's newest commit.
- ``last_training_commit`` holds the commit number of the latest durable training commit, its commit time, and its recipe-owned metrics. It is ``null`` before the first training commit. An activation, a train-from, an import or a sync advances ``commit_number`` without replacing it.
- ``served_artifact_id`` is the ``artifact_id`` of the artifact that answers the scenario's requests now, the one that ``served_models`` names: the active artifact of the weight trainer's component, or of the only component. It is null before inference publishes a route.
- ``job_data_ready`` says whether the training pipeline has a job's data waiting.
- ``training_mode`` is the selected mode.
- ``artifacts`` gives two ``artifact_id`` values for each component: its active artifact and its trainer's training start. A value is ``null`` when Reef does not know it. Both values start at the creation artifact.
- ``pipeline`` includes ``ready_samples``, the untrained samples that the next automatic job takes, from recorded traffic and uploaded trajectories; ``job_samples``, the samples that automatic job data needs (``pipeline.sample2job.config.job_samples``); and ``buffer_size``, the most samples that one automatic job takes, beside the rest of a group. A training pipeline that takes requests also reports ``pending_requests``.
- ``processor`` is the older name of ``pipeline``. The answer holds the same object under both keys, in each scenario and in each entry of ``components``, so that a caller that reads ``processor`` continues to work. Read ``pipeline`` in new code.

A scenario that trains weights also reports its training jobs:

- ``training_jobs`` lists the jobs that the scenario's training component runs, in the order of their submission. The list is empty while no job runs. Each job holds:

  - ``job_id`` and ``backend``, the backend that runs the job;
  - ``start``, which the job trains from: a checkpoint's path as the component reads it, or ``snapshot:`` and a JSON object ``{revision, root}`` that names the snapshot of the scenario;
  - ``checkpoints_handled``, the number of the job's saved checkpoints that Reef has committed;
  - ``train_from``, the artifact that the job's request named as its start, or ``null`` for a job that continues the training start.

- ``finished_jobs`` counts the jobs of the scenario that ended, finished or failed.
- A runtime that serves each scenario its own adapter adds ``adapter_runtime_load_id``: the runtime load ID of the scenario's served adapter, or ``null`` while the scenario samples the base model.

A scenario whose recipe runs one trainer per recipe component adds a ``components`` object. For each recipe component, it holds the component's own ``job_data_ready``, ``training_mode``, ``pipeline`` status and ``last_training_commit``. The scenario-wide ``job_data_ready``, ``training_mode`` and ``pipeline`` then describe the first trainer that trains, else the first trainer. The scenario-wide ``last_training_commit`` is the latest training commit of any recipe component.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | malformed body, a missing or empty ``x-reef-scenario`` on a |
|        | scenario-scoped route, a report violating the recipe's      |
|        | declared schema, or a ``served`` mapping or a train-from    |
|        | naming an artifact that Reef cannot serve or train from     |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token, or ``x-api-key`` when no     |
|        | Authorization header is sent; the two harness pages also    |
|        | read ``?key=`` (see Request page)                           |
+--------+-------------------------------------------------------------+
| 403    | a routed request for an unconfigured Scenario or a pin to a |
|        | target outside it (see Routed external-model requests), a   |
|        | console origin, method or headers that are not allowed, or  |
|        | an engine's own 403; an unaccepted token is 401, and        |
|        | per-scenario authorization belongs to the gateway in front  |
|        | of Reef                                                     |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (on inference, on ``POST /reef/train``,    |
|        | or with implicit creation off), a scenario that serves no   |
|        | model, unknown snapshot or artifact, unknown adapter, no    |
|        | configured harness recipe, or a scenario that serves no     |
|        | files                                                       |
+--------+-------------------------------------------------------------+
| 409    | a record id resent with different content, a ``served``     |
|        | mapping or a train-from while a training result waits for   |
|        | its commit, or an engine that aborted the request on both   |
|        | attempts                                                    |
+--------+-------------------------------------------------------------+
| 4xx    | an engine's own 4xx, with its message, such as for a prompt |
|        | that fills the model's context window: an engine's 400,     |
|        | 401, 403, 404, 408, 409, 422 or 429 passes through as sent, |
|        | and any other engine 4xx becomes 400                        |
+--------+-------------------------------------------------------------+
| 502    | the engine failed with a 5xx, answered with no JSON object, |
|        | or could not be reached; the body carries the engine's      |
|        | status and message for a 5xx, and the engine's URL and the  |
|        | connection error for an engine that could not be reached    |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, no loaded scenario is    |
|        | idle at ``reef.max_loaded_scenarios`` (with                 |
|        | ``Retry-After``), ``served_models`` cannot be read, the     |
|        | record store does not take an inference record, the engine  |
|        | did not serve the selected weights on either attempt, or a  |
|        | retry found no row for its artifact (with ``Retry-After``)  |
+--------+-------------------------------------------------------------+
| 504    | the engine did not answer within ``inference.timeout-s``    |
+--------+-------------------------------------------------------------+

Browser consoles
----------------

A browser console sends requests to this HTTP API directly, so the browser must be able to open the service URL. To let a console connect, add ``reef.console-origins`` to the serve YAML. List each trusted console origin explicitly, with no path, trailing slash, credentials, or wildcard:

.. code-block:: yaml

   reef:
     console-origins:
       - "https://api.reefinfra.ai"
       - "http://localhost:3000"

Restart the service after you change the configuration.

- Without this option, Reef adds no CORS headers. With it, Reef rejects unlisted browser origins before a route runs.
- CORS preflight requests from listed origins need no service token. Actual requests keep the configured Bearer authentication.
- Browser requests may use GET, POST or DELETE with the ``Authorization``, ``Content-Type`` and ``x-reef-scenario`` headers. CORS enables no cookies.
- CORS headers also cover errors and streamed responses, and expose ``x-reef-agent-record-id`` and ``x-reef-artifact-id`` to the browser.
- Clients without an ``Origin`` header keep their behavior.
- A browser can also require local network permission or HTTPS. An origin that Reef allows does not override browser policy.

A connected console acts with the permissions of the service token, on the scenarios of this service. The CORS connection does not create a cloud deployment or upload history.
