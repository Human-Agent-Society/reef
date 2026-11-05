Scenario components
===================

A recipe declares the components that every snapshot of a scenario binds, such
as ``weights``, ``harness`` or ``config``. For each component, it says what
kind of artifact the component holds, how Reef admits and imports it, and
which files a client pulls. The scenario reads this declaration. The inference
runtime loads the component of a weight kind, and a client pulls the component
that has a file tree. Nothing in this declaration changes an inference
request: the inference service forwards each request to the engine that
``served_models`` names (see `Run the services in separate processes
<../user-guide/operate.rst#run-the-services-in-separate-processes>`__).

:For: recipe authors who declare a new component, and anyone reading ``reef/artifact/components.py``
:Before you start: the ``ScenarioComponents`` fields from `Python API <../reference/python-api.rst#scenario-components>`__
:You will have: what each field does, where Reef reads it, what the built-in recipes declare, and how to declare your own

Scope
-----

A recipe's components own the serving-side facts of its artifacts:

- the admission check that runs before a component is published or activated;
- the layout that an imported artifact must meet;
- the kind of artifact, which decides whether the inference runtime loads it;
- the file tree that a client pulls.

They do not own producing or selecting a new artifact, storage or version
identity, commit ordering, HTTP routing, or how a runtime loads weights. Those
live in ``train/``, ``artifact/``, ``scenario/``, ``service/``, and
``runtime/`` respectively.

Components
----------

``Recipe.build_components(scenario)`` returns a ``ScenarioComponents``. Its
``components`` field maps each component name to a ``ComponentSpec``. The
default is empty: the scenario records traffic only. At most one component
has a weight kind (``model`` or ``model_adapter``), and at most one has a
file tree.

Each selected artifact keeps its own ``artifact_id``. ``Artifact.component``
provides the view of one component. A snapshot of at most one component is
flat; a snapshot of several components has one directory per component. That
view does not create a new artifact identity.

Publication admits the changed component and carries the others as previously
admitted. Activation validates the target component and changes the complete
scenario selection atomically. It does not publish a repository snapshot. The
runtime loads a snapshot only when its loaded component changed. See the
`state model <../advanced_topics/state-model.rst>`__ for ordering and recovery.

Fields and where Reef reads them
--------------------------------

+----------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``ComponentSpec`` field          | Read in                             | Meaning                                                              |
+==================================+=====================================+======================================================================+
| ``validator``                    | ``scenario/committer.py``,          | Admit a component before it is published or activated. Raising       |
|                                  | ``scenario/factory.py``             | rejects the step or the activation. The default accepts anything.    |
+----------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``layout``                       | ``scenario/committer.py``,          | Check an imported directory against the component's snapshot layout  |
|                                  | ``scenario/factory.py``             | before admission. Without ``layout``, the component takes no import. |
+----------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``files``                        | ``service/request_service.py``      | Read the files a client pulls from the frozen release. Without a     |
|                                  |                                     | component with ``files``, harness file routes reject the scenario.   |
+----------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``kind``                         | ``scenario/binding.py``,            | The ``ArtifactKind`` of the component. A ``model`` or                |
|                                  | ``scenario/scenario.py``,           | ``model_adapter`` component loads into the inference runtime         |
|                                  | ``service/artifact_evaluations.py`` | (``activate_snapshot``). A kind also makes the component a side of an|
|                                  |                                     | artifact evaluation. ``None`` declares no kind.                      |
+----------------------------------+-------------------------------------+----------------------------------------------------------------------+

``activate_snapshot`` (``reef.scenario.binding``) calls the runtime's
``InferenceRuntime.activate_checkpoint`` for the loaded component. The inference
worker calls it for each selection that control queues, after the commit that
makes the snapshot active. Without a runtime or a loaded component, moving the
head is enough.

The scenario owns lifecycle ordering and admission; the service owns
transport. The components supply only the facts at these sites.

Built-in recipes
----------------

+-------------------------------+-------------+---------------------------------+----------------------------+------------------+
| Recipe                        | Component   | Kind                            | Layout                     | Files            |
+===============================+=============+=================================+============================+==================+
| ``Recipe``                    | none        |                                 |                            |                  |
+-------------------------------+-------------+---------------------------------+----------------------------+------------------+
| ``WeightTrainingRecipe``      | ``weights`` | the runtime's                   | ``TrainingArtifactLayout`` | none             |
|                               |             | ``served_weight_kind``          |                            |                  |
+-------------------------------+-------------+---------------------------------+----------------------------+------------------+
| ``HarnessEvolutionRecipe``    | ``harness`` | ``harness_plugin``              | ``HarnessLayout``          | ``HarnessFiles`` |
+-------------------------------+-------------+---------------------------------+----------------------------+------------------+
| ``CompositeRecipe``           | one per     | each recipe's                   | each recipe's              | each recipe's    |
|                               | recipe      |                                 |                            |                  |
+-------------------------------+-------------+---------------------------------+----------------------------+------------------+

**Weights.** ``InferenceRuntime.activate_checkpoint`` makes the runtime serve
a snapshot. It returns the runtime load ID that serves the snapshot: a new load
ID for a runtime that loads weights, or the release ID for one that loads
nothing, such as a proxy. The runtime loads only published weights; no trainer
changes what the engines serve. The runtime load ID stays inside the runtime:
each record names the artifact that answered by its ``artifact_id`` (see
`HTTP API <../reference/http-api.rst#artifact-of-a-recorded-response>`__).

**Per-scenario adapters.** When the runtime serves one PEFT adapter per
scenario (``InferenceRuntime.per_scenario_adapters``), ``activate_snapshot``
passes the scenario name to ``activate_checkpoint``. The runtime serves the
scenario's adapter revision under its own model name
(``InferenceRuntime.served_model_name``), and control writes that name to the
scenario's row in ``served_models``, so every request reaches that revision.
After a restart, each scenario's snapshot loads again under a new runtime load
ID. ``adapter_name(scenario, runtime_load_id)`` in ``reef.inference.residency``
names a revision on the engine, and ``parse_adapter_name`` inverts it, so a
recorded ``lora_path`` names exactly one (scenario, revision). The serving
runtime owns one ``AdapterResidencyManager``, which keeps at most
``adapter_capacity`` adapters loaded. It evicts a leaked adapter first, then
the oldest-activated adapter that is no scenario's current one.

**Harness file trees.** ``HarnessFiles`` (``reef.harness.sync``) serves each
snapshot of the ``harness`` component. A snapshot stores the harness source
tree, and a read renders it for the descriptor's adapter without a model
binding, so the client gets the files its agent reads and binds its own model.
``GET /reef/harness?form=source`` reads the stored source tree instead (see
`Rendering <harness-adapters.rst#rendering>`__).

Declaring components
--------------------

A recipe returns a ``ScenarioComponents`` from ``build_components``:

.. code:: python

   from reef.artifact import ArtifactValidator
   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree


   class PromptValidator(ArtifactValidator):
       def validate(self, artifact):
           validate_prompt_artifact(artifact)


   class PromptRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           prompt = ComponentSpec(validator=PromptValidator(), files=TextFileTree())
           return ScenarioComponents({"prompt": prompt})

A recipe whose scenario serves two components declares both. A step then
publishes one of them, naming it in ``TrainStepResult.component``, and the
committer carries the other forward from the previous checkpoint:

.. code:: python

   class AgentRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           return ScenarioComponents(
               {
                   "weights": ComponentSpec(kind="model_adapter"),
                   "harness": ComponentSpec(files=TextFileTree()),
               }
           )

The root snapshot of such a scenario keeps one directory per component
(``weights/``, ``harness/``). ``CompositeRecipe`` (``reef.recipe.composite``)
builds exactly this from existing flat recipes, one per component, and runs
each one's trainer as its own worker; see `configuration
<../reference/configuration.rst#recipe-configuration>`__ for its layout.

Remote weight snapshots
-----------------------

An inference runtime implements ``activate_checkpoint(artifact)``. The
saved-weight serving runtimes (``reef.inference.serving``) load the snapshot's
files into the engines. A proxy runtime loads nothing and returns the release
ID. An inference runtime that serves immutable remote snapshots validates the
checkpoint and binds its remote sampler. Tinker's inference runtime uses this
to serve the authoritative artifact before the scenario serves requests. The
training component does not read the head: the next job starts from the
trainer's committed start, and a ``train-from`` imports the snapshot into the
component with ``import_start``.
