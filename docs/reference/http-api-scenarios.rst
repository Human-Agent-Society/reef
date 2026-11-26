HTTP API: scenarios and their settings
======================================

A scenario isolates a workload's records, trainer and artifact history.
Create it before sending inference requests:

.. code:: bash

   curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "hello-reef"}'

- Unknown scenarios return 404 on inference.
- Reports and imported records can create their named scenario.
  Harness installation can create a generated name.
  ``reef.allow_implicit_scenario_creation: false`` disables this behavior; unknown scenarios return 404.
- Every scenario uses the deployment's recipe, components, runtime and report schema.
  A request naming a different recipe returns 409.

Creation beyond ``reef.max_scenarios`` returns 409.
Loading beyond ``reef.max_loaded_scenarios`` returns 503 with ``Retry-After``
when no loaded scenario is idle.
See `Scenario limits <../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__.

Create a scenario
-----------------

``POST /reef/scenarios`` accepts these fields. Unknown keys return 400.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``name``
     - Required nonempty string.
   * - ``owner``
     - Data-item owner; defaults to the scenario name.
       It can change only while still the default and no trajectory/task items are in use; otherwise, 400.
   * - ``artifact_id``
     - Starting artifact item in that owner's store; defaults to the deployment's starting artifact.
   * - ``component``
     - Component receiving the item. Requires ``artifact_id``; omit for a single component.
   * - ``serves``
     - Boolean, default ``true``. ``false`` starts no engine and makes inference return 404
       until an update sets ``served``.

Starting artifacts become both active selections and training starts.
Their rows use ``operation: creation``; ``source`` identifies the item or default configuration.
Other components use their default starts.

For an owner-specific starting artifact, send:

.. code:: json

   {"name": "hello-reef-2", "owner": "acme", "artifact_id": "hello-reef/<artifact_id>"}

Reef validates items under the `artifact import rules <http-api-artifacts.rst#import-an-artifact>`__.
Import failures register no scenario: invalid layouts return 400; missing items return 404.

New scenarios return 201 with ``{scenario, owner, components}``.
``components`` maps names to selected artifact IDs; single-component responses also include ``artifact_id``.
Existing scenarios return 200 with current selections, ignoring ``artifact_id``, ``component`` and ``serves``.

Scenario routes
---------------

- ``GET /reef/scenarios`` lists known scenarios and loaded selections.
  Harness entries include ``adapter``, even while loading.
- ``GET /reef/scenarios/{scenario}/contract`` returns
  ``{scenario, processor, required_request_types, training_mode, status, parameters_schema}``.
  Mode and status are live. Parameter schemas are keyed by sample kind.
- ``DELETE /reef/scenarios/{scenario}`` archives a scenario.
- See `Training jobs <http-api-training.rst#training-jobs>`__ for job listing and cancellation.

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` returns ``{scenario, archived}``.
``archived`` names the repository ref under ``refs/reef/archived/``.
Remote repositories return 501 without changes; archiving requires a local repository.

After deletion:

- Reads, inference and repeated deletion return 404.
- Reusing the name, explicitly or implicitly, returns 409.
- In-flight work cannot commit or store samples. Training jobs are cancelled and their output discarded.
- Published snapshots remain reachable through the archived ref.

Background cleanup archives records, commits and recipe directories, and removes
service instances and caches. Recipe-directory cleanup waits for the training service.
Resident adapters remain until normal eviction.
Cleanup failure leaves the scenario deleted; retries run after an hour and after restart.

Scenario updates
----------------

``POST /reef/scenarios/{scenario}/update`` accepts one or more settings:

- ``training_mode``: ``auto`` or ``manual``.
- ``artifact_evaluation``: `automatic evaluation settings <http-api-evaluation.rst#automatic-evaluation>`__.
- ``activation``: the rule for activating new artifacts.
- ``serving_config``: per-component serving overrides.
- ``served``: `artifacts and traffic weights <http-api-serving.rst#set-what-a-scenario-serves>`__.
  ``component`` is valid only with ``served``.
- ``record2sample_config``: per-component sample-stage overrides.

The response includes the scenario and changed settings.
Unknown scenarios return 404; invalid or unknown fields return 400.
Settings survive reloads and restarts.

**Training mode.** ``auto`` (default) schedules jobs when ``job_samples`` samples are available;
``manual`` trains only requested samples. For example:

.. code:: json

   {"training_mode": "manual"}

The response is ``{"scenario": "agents", "training_mode": "manual"}`` with HTTP 200.
Reserved or running jobs remain unchanged. Queued requests train in either mode.
Switching back to ``auto`` makes previously untrained samples eligible,
including samples collected in manual mode.
Unsupported modes return 501 without changing state.
See `Manual training <http-api-training.rst#manual-training>`__.

Weight-training scenarios can share a deployment only when its backend and runtime
isolate their weights. Otherwise, use separate deployments.

**Activation.** Override the deployment's rule with a built-in rule:

.. code:: json

   {"activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}}}

``always`` accepts ``review_kinds``; ``never`` takes no config;
``evaluated`` requires ``min_mean_score``.

- ``null`` restores the deployment's rule. The response contains the stored override.
- Custom class references return 400; configure them in the deployment.
- ``evaluated`` requires ``artifact_evaluation``. Reef validates and stores this pair together;
  removing required evaluation settings returns 400.
- Changing the rule activates nothing immediately. It applies to later outputs
  and, for ``evaluated``, evaluations that finish afterward.

See `Activation configuration <configuration-recipe.rst#the-activation-section>`__.

Serving config
--------------

``serving_config`` changes runtime fields without restarting or activating artifacts:

.. code:: json

   {"serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}}}

Fields merge into stored overrides. ``null`` restores the deployment's value;
omitted fields stay unchanged. The response includes stored and effective values.

Only the served component accepts fields. Its runtime and handler define them:

- Token-native handlers support ``temperature``, ``top_p`` and ``top_k``.
  Runtimes can add engine fields, such as replica count.
- Unknown components, fields or invalid values return 400 without storing the change.
- Startup-only fields, such as GPU type, remain deployment configuration.

Changes apply within seconds: request fields affect subsequent requests;
engine fields affect current and future engines.
After deployment changes, unsupported stored fields log warnings and use deployment defaults.

Record-to-sample config
-----------------------

``record2sample_config`` overrides sample-stage settings by component.
Keys replace deployment ``pipeline.record2sample.config`` values; the stage class stays unchanged:

.. code:: json

   {"record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}}}

The request replaces the entire stored mapping; ``{}`` restores deployment configuration.
The response includes the stored mapping.

Reef rebuilds stages and replays records after committed progress.
Separate samples services reload before their next selection.
Components without trainers or invalid settings return 400, preserving the loaded scenario and stored configuration.

Read the settings
-----------------

``GET /reef/scenarios/{scenario}/settings`` returns the current settings.
Example response, with default evaluation fields omitted:

.. code:: json

   {
     "scenario": "agents",
     "training_mode": "auto",
     "artifact_evaluation": {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2},
     "activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}},
     "effective_activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}},
     "serving_config": {"weights": {"temperature": 0.6}},
     "effective_serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}},
     "served": {"weights": {"artifact:a": 95, "artifact:b": 5}},
     "record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}},
     "job_parameters": {"trajectory": {"objective": "sft", "save_interval": 50}}
   }

- ``training_mode`` is the current mode.
- ``artifact_evaluation`` and ``activation`` are stored settings, or null.
- ``effective_activation`` resolves the override or deployment rule.
  Custom deployment rules appear as ``package.module:ClassName``.
- ``serving_config`` and ``record2sample_config`` are stored mappings, empty when unset.
- ``effective_serving_config`` resolves each field from the override or deployment.
- ``served`` maps components to artifacts and traffic weights. Without a split,
  the active artifact has weight 100; other artifacts have 0.
  It also receives the weights of split artifacts no longer served.
- ``job_parameters`` maps sample kinds to recipe parameters.
  Request parameters merge into them; kinds without defaults are absent.

Unknown scenarios return 404.
