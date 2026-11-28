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
See `Scenario limits <configuration-settings.rst#service>`__.

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
  ``{scenario, sample2job, required_request_types, training_mode, status, parameters_schema}``.
  ``sample2job`` names the recipe's sample-to-job stage. ``processor`` is the older name of the key and holds the same value.
  Mode and status are live. Parameter schemas are keyed by sample kind.
- ``DELETE /reef/scenarios/{scenario}`` archives a scenario.
- See `Training jobs <http-api-training.rst#training-jobs>`__ for job listing and cancellation.

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` returns ``{scenario, archived}``.
``archived`` names the repository ref under ``refs/reef/archived/``.
Remote repositories return 501 without changes; archiving requires a local repository.

Reads and repeated deletion return 404; reusing the name returns 409.
Running work cannot commit, and training jobs are cancelled.
Published snapshots remain under the archived ref. Background cleanup retries failures.

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

``manual`` starts jobs only from requests; ``auto`` also uses accumulated samples.
Changes leave reserved/running jobs unchanged. Requests work in both modes.
Unsupported modes return 501.

``activation`` accepts built-ins: ``always`` with optional ``review_kinds``,
``never``, or ``evaluated`` with ``min_mean_score``. Null restores deployment settings.
``evaluated`` requires ``artifact_evaluation``; Reef validates the pair together.
Custom classes belong in deployment configuration. Changes affect later outputs
and finishing evaluations, never immediate activation.

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

``record2sample_config`` maps component names to overrides of
``pipeline.record2sample.config``; it does not change stage classes.
The request replaces the whole stored mapping; ``{}`` restores deployment defaults.
Stages reload and replay records after committed progress.
Invalid settings or components without trainers return 400 without changes.

Read the settings
-----------------

``GET /reef/scenarios/{scenario}/settings`` returns stored ``training_mode``,
``artifact_evaluation``, ``activation``, ``serving_config`` and ``record2sample_config``.
``effective_activation`` and ``effective_serving_config`` resolve deployment defaults.
``served`` maps components to artifact traffic weights; ``job_parameters`` maps
sample kinds to recipe defaults. Unknown scenarios return 404.
