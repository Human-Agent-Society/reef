HTTP API: scenarios and their settings
======================================

A scenario isolates one workload's records, trainer and snapshot chain. Create it with ``POST /reef/scenarios`` before sending inference requests:

- An inference request never creates a scenario: an unknown scenario answers 404.
- The first report or imported record with a new ``x-reef-scenario`` creates the scenario. A harness install without ``x-reef-scenario`` creates one with a generated name. With ``reef.allow_implicit_scenario_creation: false``, these requests return 404 for an unknown scenario instead.
- Every scenario uses the deployment's configured recipe, and requests never select one. The scenario fixes the recipe's components, runtime, inference backend and optional report schema. A request that names a different recipe with the same scenario returns 409.

.. code:: bash

   curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "hello-reef"}'

A deployment can limit its scenarios (`Limit the scenarios a service holds <../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__):

- A request that would create a scenario past ``reef.max_scenarios`` returns 409.
- A request that must load a scenario returns 503 with a ``Retry-After`` header when the service already holds ``reef.max_loaded_scenarios`` loaded scenarios, none of them idle.

Create a scenario
-----------------

``POST /reef/scenarios`` takes the object ``{"name", "owner"?, "artifact_id"?, "component"?, "serves"?}``. Any other key returns 400, and the message names the valid keys:

- ``name``: a non-empty string.
- ``owner``: the owner whose `data items <http-api-data-items.rst>`__ the scenario uses, by default the scenario's name. Reef sets it when it creates the scenario. A later call with another ``owner`` returns 400 for a scenario that already uses data items.
- ``artifact_id``: an `artifact item <http-api-data-items.rst#artifact-items>`__ of the owner, the new scenario's starting artifact. Without it, Reef uses the deployment's default start (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).
- ``component``: the component that holds the item. It needs ``artifact_id``. Leave it out when the scenario has one component.
- ``serves``: a boolean, true by default. With ``false``, the new scenario serves nothing: Reef starts no engine for it, and its inference requests answer HTTP 404. It serves after an update sets ``served`` (`Serve nothing <http-api-serving.rst#serve-nothing>`__). An existing scenario ignores ``serves``.

The starting artifacts are the initial active selections and training starts. Their artifact rows have ``operation: creation``, and ``source`` names the selected item or the default configuration.

With ``artifact_id``, Reef reads and validates the item's files using the `artifact import <http-api-artifacts.rst#import-an-artifact>`__ rules. Other components use their default starts. Import failures register no scenario and return the import's error. Examples include 400 for invalid layouts and 404 for items absent from the owner's store.

The response is ``{scenario, owner, components}``. ``components`` maps component names to selected artifact IDs. Single-component scenarios also return ``artifact_id``.

New scenarios return 201. Existing scenarios return 200 with current selections, ignoring the requested ``artifact_id`` and ``component``.

.. code:: bash

   curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "hello-reef-2", "owner": "acme", "artifact_id": "hello-reef/<artifact_id>"}'

Scenario routes
---------------

- ``GET /reef/scenarios`` lists every known scenario, its selections once loaded and, for a harness recipe, the ``adapter`` that its tree is rendered for, also while the scenario loads.
- ``GET /reef/scenarios/{scenario}/contract`` returns ``{scenario, processor, required_request_types, training_mode, status, parameters_schema}``. ``training_mode`` and ``status`` are live values. ``parameters_schema`` maps a sample kind to the JSON Schema of a job's parameters, for each kind whose schema the training component declares.
- ``DELETE /reef/scenarios/{scenario}`` deletes the scenario (`Deleting a scenario`_).

- `Training jobs <http-api-training.rst#training-jobs>`__ gives the routes that list the scenario's training jobs and cancel a job or a training request.

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` archives the repository ref under ``refs/reef/archived/``. It returns ``{"scenario", "archived"}``, where ``archived`` names that ref. Remote artifact repositories return 501 without changes; only local repositories support archiving.

After deletion:

- A read of the scenario and a second delete answer 404.
- A create with the ID, and a record that would create the scenario implicitly, answer 409: a scenario ID is never used again.
- The inference service answers the scenario 404.
- A local step in flight for the scenario ends without a commit.
- The training service runs no job of the scenario. When it takes such a job, or stores for one, it cancels the job and discards what the job stored.

Control immediately starts background cleanup:

- It cancels the scenario's training jobs, discards its stored content, and moves its record store and commit log under ``<agent_record_dir>/archived/``. The training service moves the recipe's own directories (a harness scenario's history and step records) under an ``archived/`` sibling; this waits while no training service runs.
- A sample selection that runs past the delete writes no sample.
- Each other service drops its instance, its clone and its cached snapshots of the scenario within seconds.
- Every snapshot that the scenario published stays reachable under the archived ref.
- On a runtime that serves per-scenario adapters, the scenario's adapter stays resident in the serving engines until the adapter residency evicts it.

Cleanup failures leave the scenario deleted. Control retries after an hour and after restarts.

Scenario updates
----------------

``POST /reef/scenarios/{scenario}/update`` updates an existing scenario. The body holds one or more of these settings, and Reef rejects other fields:

- ``training_mode``: the mode of the data processor, below.
- ``artifact_evaluation``: the settings that Reef evaluates each new artifact with (`Automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__).
- ``activation``: the scenario's activation rule, below.
- ``serving_config``: the scenario's serving fields (`Serving config`_).
- ``served``: the artifacts that the scenario serves of one component, each with its weight (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__). ``component`` names that component, and is valid only with ``served``.
- ``record2sample_config``: the scenario's record-to-sample config keys (`Record-to-sample config`_).

The response includes the scenario name and changed settings. Unknown scenarios return 404 without implicit creation; invalid payloads return 400. Settings survive reloads and restarts.

**Training mode.** The values are ``auto`` and ``manual``. ``auto`` starts a job whenever ``job_samples`` samples wait. ``manual`` starts one only for a request, on the samples that it names (`Manual training <http-api-training.rst#manual-training>`__).

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.

- The change selects later job data. Job data that is already reserved or running completes in its original mode.
- After a switch back to ``auto``, automatic jobs train the samples that no job trained, including those collected in ``manual``.
- A queued request trains in both modes, so a request that is not yet read when the mode changes still trains.
- A processor that does not support the requested mode returns 501 without changing its state.

A Reef process runs at most one scenario that trains full weights, so preparation, remote execution, and commit never interleave. It may run any number of scenarios that produce no updates or that update text artifacts in process. Record acceptance never waits for artifact evolution.

**Activation.** ``activation`` overrides the deployment's `activation rule <configuration-recipe.rst#the-activation-section>`__ for this scenario. ``module`` selects a built-in rule:

.. code:: json

   {"activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}}}

- ``always``: ``config`` takes ``review_kinds``.
- ``never``: there is no ``config``.
- ``evaluated``: ``config`` takes ``min_mean_score``.

These rules apply to the setting:

- The response holds the stored rule. Send ``"activation": null`` to follow the deployment's rule again.
- The setting names only built-in rules. Reef answers 400 for a ``package.module:ClassName`` rule. Set such a rule in the deployment's ``activation`` section.
- ``evaluated`` requires ``artifact_evaluation``. Reef validates both settings together and stores both or neither. Missing or cleared evaluation settings under ``evaluated`` return 400.
- Changing the rule alone activates nothing. It applies to later training outputs. Under ``evaluated``, evaluations finishing after the change can activate artifacts whose ``mean_score`` reaches ``min_mean_score``.

Serving config
--------------

``serving_config`` changes serving settings without restarting or activating artifacts. It maps component names to overrides:

.. code:: json

   {"serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}}}

Reef merges fields into the stored config. ``null`` removes an override, restoring the deployment's value. Omitted fields keep their stored values. The response includes ``serving_config`` and ``effective_serving_config`` (`Read the settings`_).

Only the component that the inference runtime serves has serving fields. The runtime and its request handler define the fields and check their values:

- The token-native chat handlers have ``temperature``, ``top_p`` and ``top_k`` (`native backend options <configuration-inference.rst#native-backend-options>`__). A runtime can add engine fields, such as a number of replicas.
- Reef answers 400 and stores nothing for a field that they do not have, for a value that a field does not take, or for another component.
- A field that takes effect only when an engine starts, such as a GPU type, stays in the deployment configuration.

Inference applies changes within seconds. Request fields affect subsequent requests; engine fields affect current and future engines. Reef warns about stored fields the deployment no longer accepts and ignores them. Those fields use deployment defaults until updated.

Record-to-sample config
-----------------------

``record2sample_config`` overrides each trainer's record-to-sample configuration by component name. Keys replace the deployment's `pipeline.record2sample.config <configuration-recipe.rst#the-pipeline-section>`__ values. The stage class stays unchanged:

.. code:: json

   {"record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}}}

The request replaces the stored mapping as a whole. Send ``{}`` to use the deployment's config again. The response holds the stored ``record2sample_config``.

Reef reloads the scenario to rebuild stages and replay records after committed progress. Separate samples services reload before their next selection. Components without trainers or rejected stage configurations return 400, preserving the stored setting and loaded scenario.

Read the settings
-----------------

``GET /reef/scenarios/{scenario}/settings`` returns the settings of an existing scenario:

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

- ``training_mode`` is what the scenario runs.
- ``artifact_evaluation`` and ``activation`` are the stored settings, or null.
- ``effective_activation`` is the rule that applies: the scenario's ``activation``, else the deployment's. When the deployment names a rule class of its own, ``module`` is the ``package.module:ClassName`` of that class.
- ``serving_config`` is the stored `serving config <#serving-config>`__, ``{}`` when the scenario sets no field.
- ``served`` maps each component to the artifacts that the scenario serves of it, each with its weight (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__). Without a traffic split, the active artifact has 100 and each other served artifact has 0. With a split, the active artifact also has the weight of each artifact of the split that the scenario no longer serves, because it answers their requests.
- ``effective_serving_config`` maps the served component to each of its serving fields that has a value: the scenario's value, else the deployment's.
- ``record2sample_config`` is the stored `record-to-sample config <#record-to-sample-config>`__, ``{}`` when the scenario sets no key.
- ``job_parameters`` maps a sample kind (``trajectory`` or ``task``) to the recipe's parameters of a job of that kind. A training request's ``parameters`` merge into them. A kind that the recipe gives no parameters is absent.

An unknown scenario returns ``404``.
