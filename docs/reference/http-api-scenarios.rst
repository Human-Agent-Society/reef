HTTP API: scenarios and their settings
======================================

A scenario isolates the records, trainer, and snapshot chain for one workload. Create it with ``POST /reef/scenarios`` before you send inference requests to it:

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

With ``artifact_id``, Reef reads and checks the item's files as ``POST /reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}`` does (`Import an artifact <http-api-artifacts.rst#import-an-artifact>`__), by reference or as a copy. Each other component uses its default start. When Reef cannot read the item, or the check refuses it, Reef registers no scenario and answers with the status of the refusal, such as 400 for a directory without the component's layout or 404 for an item that the owner does not store.

The answer is ``{scenario, owner, components}``. ``components`` maps each component name to its selected ``artifact_id``. A scenario with one component also returns that value as ``artifact_id``. A new scenario returns 201. An existing scenario returns 200 with its current selections and ignores the requested ``artifact_id`` and ``component``.

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

``DELETE /reef/scenarios/{scenario}`` renames the scenario's ref in the artifact repository into ``refs/reef/archived/`` and returns ``{"scenario", "archived"}``, where ``archived`` names the archived ref. Only a local artifact repository can be archived; a remote one answers 501 and changes nothing. From then on the scenario does not exist:

- A read of the scenario and a second delete answer 404.
- A create with the ID, and a record that would create the scenario implicitly, answer 409: a scenario ID is never used again.
- The inference service answers the scenario 404.
- A local step in flight for the scenario ends without a commit.
- The training service runs no job of the scenario. When it takes such a job, or stores for one, it cancels the job and discards what the job stored.

Control then removes the scenario's state in the background, at once:

- It cancels the scenario's training jobs, discards its stored content, and moves its record store and commit log under ``<agent_record_dir>/archived/``. The training service moves the recipe's own directories (a harness scenario's history and step records) under an ``archived/`` sibling; this waits while no training service runs.
- A sample selection that runs past the delete writes no sample.
- Each other service drops its instance, its clone and its cached snapshots of the scenario within seconds.
- Every snapshot that the scenario published stays reachable under the archived ref.
- On a runtime that serves per-scenario adapters, the scenario's adapter stays resident in the serving engines until the adapter residency evicts it.

When a cleanup step fails, for example because the artifact store does not answer, the scenario stays deleted. Control tries again an hour later, and after a restart.

Scenario updates
----------------

``POST /reef/scenarios/{scenario}/update`` updates an existing scenario. The body holds one or more of these settings, and Reef rejects other fields:

- ``training_mode``: the mode of the training pipeline, below.
- ``artifact_evaluation``: the settings that Reef evaluates each new artifact with (`Automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__).
- ``activation``: the scenario's activation rule, below.
- ``serving_config``: the scenario's serving fields (`Serving config`_).
- ``served``: the artifacts that the scenario serves of one component, each with its weight (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__). ``component`` names that component, and is valid only with ``served``.
- ``record2sample_config``: the scenario's record-to-sample config keys (`Record-to-sample config`_).

The response holds the scenario name and each setting that the request changed. An unknown scenario returns 404 without implicit creation, and an invalid payload returns 400. A reload and a restart keep the settings.

**Training mode.** The values are ``auto`` and ``manual``. ``auto`` starts a job whenever ``job_samples`` samples wait. ``manual`` starts one only for a request, on the samples that it names (`Manual training <http-api-training.rst#manual-training>`__).

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.

- The change selects later job data. Job data that is already reserved or running completes in its original mode.
- After a switch back to ``auto``, automatic jobs train the samples that no job trained, including those collected in ``manual``.
- Accepted requests wait for ``manual``, including requests not yet read when the mode changes to ``auto``.
- When the sample-to-job stage does not support the requested mode, Reef returns 501 and changes no state.

A Reef process runs at most one scenario that trains full weights, so preparation, remote execution, and commit never interleave. It may run any number of scenarios that produce no updates or that update text artifacts in process. Record acceptance never waits for artifact evolution.

**Activation.** The ``activation`` setting selects the rule that decides whether Reef activates each new artifact of this scenario. It replaces the deployment's `activation section <configuration-recipe.rst#the-activation-section>`__ for this scenario. ``module`` names one of the built-in rules:

.. code:: json

   {"activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}}}

- ``always``: ``config`` takes ``review_kinds``.
- ``never``: there is no ``config``.
- ``evaluated``: ``config`` takes ``min_mean_score``.

These rules apply to the setting:

- The response holds the stored rule. Send ``"activation": null`` to follow the deployment's rule again.
- The setting names only built-in rules. Reef answers 400 for a ``package.module:ClassName`` rule. Set such a rule in the deployment's ``activation`` section.
- The ``evaluated`` rule needs ``artifact_evaluation``. Reef checks the two settings together, and stores both or neither. It answers 400 when, after the request, the rule that applies is ``evaluated`` and the scenario has no ``artifact_evaluation``. This includes a request that clears ``artifact_evaluation`` under ``evaluated``.
- A new rule applies to the artifacts that training makes after the change. It never activates an existing artifact. Under ``evaluated``, Reef activates an artifact when an evaluation of it finishes after the change with a ``mean_score`` of at least ``min_mean_score``.

Serving config
--------------

The ``serving_config`` setting changes how the scenario is served while it serves, without a restart or an activation. It maps a component name to the fields that the scenario sets for that component:

.. code:: json

   {"serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}}}

Reef merges the request into the stored config field by field. ``null`` removes a field, so the field takes the deployment's value again. A field that the request leaves out keeps its stored value. The response holds the stored ``serving_config`` and the ``effective_serving_config`` (`Read the settings`_).

Only the component that the inference runtime serves has serving fields. The runtime and its request handler define the fields and check their values:

- The token-native chat handlers have ``temperature``, ``top_p`` and ``top_k`` (`native backend options <configuration-inference.rst#native-backend-options>`__). A runtime can add engine fields, such as a number of replicas.
- Reef answers 400 and stores nothing for a field that they do not have, for a value that a field does not take, or for another component.
- A field that takes effect only when an engine starts, such as a GPU type, stays in the deployment configuration.

Inference applies a change within a few seconds: the request fields to each later request, and the runtime's engine fields to the engine that serves the scenario and to each engine that starts for it later. Reef ignores, with a warning, a stored field that the deployment no longer accepts. The scenario then serves with the deployment's values until a person changes the config.

Record-to-sample config
-----------------------

The ``record2sample_config`` setting gives one scenario its own config for the record-to-sample stage of each trainer. It maps a component name to config keys. Each key replaces the same key of the deployment's `pipeline.record2sample.config <configuration-recipe.rst#the-pipeline-section>`__ for that component's trainer. The stage class stays the deployment's:

.. code:: json

   {"record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}}}

The request replaces the stored mapping as a whole. Send ``{}`` to use the deployment's config again. The response holds the stored ``record2sample_config``.

Reef applies a change by loading the scenario again, as a restart does: each trainer builds its stage with the new config and reads the records after its committed progress again. A samples service in another process loads the scenario again before its next selection. Reef answers 400, and keeps the stored setting and the loaded scenario, when the mapping names a component that has no trainer, or when a stage refuses its config.

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
