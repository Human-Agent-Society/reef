Surfaces
========

A surface is how a scenario serves the release it has frozen. Its
consumers are an inference runtime, a provider request, or a harness client
pulling files, and the one invariant is version fidelity: the consumer
observes exactly that version, and Reef records the request and response
against it.

.. page::
   :for: recipe authors deciding what a served artifact can do, and anyone reading ``reef/surface/``
   :needs: the ``Surface`` fields from `Python API <../reference/python-api.rst#surface>`__
   :outcome: which call site invokes each capability, what the bundled surfaces bind, and how to build one

Scope
-----

A surface owns the serving-side behavior of an artifact:

- admitting a component before it is published or restored;
- loading a durable artifact into a runtime and recovering the serving head;
- preparing inference requests and verifying provider responses;
- exposing a versioned file tree to a client.

It does not own producing or selecting a new artifact, storage or version
identity, commit ordering, HTTP routing, or runtime internals such as workers,
GPU placement, or adapter caches. Those live in ``train/``, ``artifact/``,
``scenario/``, ``service/``, and ``runtime/`` respectively.

Components
----------

A release binds named components: ``weights``, ``harness``, ``skills``, or
whatever a recipe declares. ``Surface.components`` maps each name to a
``ComponentSurface`` holding that component's ``validator``, ``loader``,
``inference`` hooks, and ``files`` tree. Every shipped recipe declares one
component, so its releases keep the flat layout: the whole artifact is the
component, and nothing about serving changes. A recipe that declares several
components serves a release whose files sit in one directory per component,
described by a manifest in the release metadata; ``Artifact.component(name)``
returns that component's view, and the surface routes each capability to
it. See `state model <../advanced_topics/state-model.rst>`__ for how such a
release is committed and rolled back.

At most one component may load into a runtime and at most one may expose a
client-pulled file tree. Inference hooks of several components run in
declaration order, and their leases are released together.

Capabilities and call sites
---------------------------

``ComponentSurface`` is a frozen composition of optional capabilities;
``None`` means the component does not support one. Callers inspect fields,
never types, and a recipe never subclasses ``Surface``. The scenario reaches
the loader through ``Surface.validate``, ``recover``, ``load``, and
``activate``, which route to the loaded component; the service reads
``Surface.inference`` and ``Surface.files``, which route to their
components. Every call site for each capability is listed here;
``activate`` alone is invoked from more than one module, once per lifecycle
event that makes a version servable:

+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| Capability                                        | Called from                         | Meaning                                                              |
+===================================================+=====================================+======================================================================+
| ``validator.validate(artifact)``                  | ``scenario/committer.py``           | Admit a component before it is published or restored. Raising        |
|                                                   |                                     | rejects the step or the rollback; the default accepts anything.      |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``loader.recover(current, checkpoint, runtime)``  | ``scenario/factory.py``             | Choose the head the runtime can still serve after startup. Without   |
|                                                   |                                     | a loader, recovery uses the durable checkpoint.                      |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``loader.load(artifact, runtime)``                | ``scenario/committer.py``           | Load a durable rollback target before the rollback commit becomes    |
|                                                   |                                     | authoritative. Without a loader, moving the head is sufficient. A    |
|                                                   |                                     | composed release loads only when its loaded component changed.       |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``inference.prepare_request(...)``                | ``service/request_service.py``      | Address or inject the frozen artifact before forwarding. The         |
|                                                   |                                     | returned request is both forwarded and recorded.                     |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``inference.verify_response(...)``                | ``service/request_service.py``      | Verify the completed provider response before recording it. Raising  |
|                                                   |                                     | rejects it, so a mismatched exchange never enters training data.     |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``files.read_files(artifact)``                    | ``service/request_service.py``      | Read the file manifest for the frozen snapshot. Without ``files``,   |
|                                                   |                                     | harness file routes reject the scenario before materialization.      |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``loader.activate(artifact, runtime, source=...)``| ``scenario/factory.py``,            | Optional ``ArtifactActivator``. Make a final version servable: after |
|                                                   | ``scenario/committer.py``           | recovery, and after a publication or rollback minted its version but |
|                                                   |                                     | before the commit record makes it the head. A composed release       |
|                                                   |                                     | activates only when its loaded component changed.                    |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``inference.begin_request(artifact, path)``       | ``service/request_service.py``      | Optional ``LeasingInferenceHooks``. Hold serving state for one       |
|                                                   |                                     | attempt; the lease is released when the attempt ends.                |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+

The scenario owns lifecycle ordering and admission; the service owns
transport. A surface supplies only consumer-facing behavior at these sites.

Built-in surfaces
-----------------

+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| Surface                                | Component   | Validator          | Loader                  | Inference                        | Files            |
+========================================+=============+====================+=========================+==================================+==================+
| ``Surface()``                          | none        | none               | none                    | none                             | none             |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| ``create_weight_surface()``            | ``weights`` | optional, accepts  | ``WeightLoader``        | ``WeightInferenceHooks``         | none             |
|                                        |             | any by default     |                         |                                  |                  |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| ``create_weight_surface(scenario=...)``| ``weights`` | as above           | ``WeightLoader``        | ``WeightInferenceHooks``         | none             |
|                                        |             |                    |                         | selecting that scenario's        |                  |
|                                        |             |                    |                         | adapter revision                 |                  |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| ``create_harness_surface()``           | ``harness`` | accepts any        | none                    | none                             | ``TextFileTree`` |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| ``create_skill_surface(...)``          | ``skills``  | ``SkillValidator`` | none                    | optional ``SkillInferenceHooks`` | ``TextFileTree`` |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+
| ``create_config_surface()``            | ``config``  | ``ConfigValidator``| none                    | ``ConfigInferenceHooks``         | none             |
+----------------------------------------+-------------+--------------------+-------------------------+----------------------------------+------------------+

**Weights.** ``WeightLoader`` probes whether an in-memory live runtime load ID
survived a restart and restores durable checkpoints during rollback.
``WeightInferenceHooks`` asks the engine to report its serving runtime load ID
and verifies that metadata before Reef records the exchange. The training
runtime activates newly trained weights inside its own durable job
transaction; the surface does not repeat that step.

**Per-scenario adapters.** When a runtime trains one LoRA adapter per
scenario, ``create_weight_surface(scenario=...)`` puts that scenario's frozen
adapter revision on every request. ``reef.surface.adapter`` holds only the
naming contract (``adapter_name(scenario, runtime_load_id)`` and its inverse), so a
recorded ``lora_path`` names exactly one (scenario, revision). Residency is
engine-global, not per surface: the training bridge owns one
``AdapterResidencyManager`` per engine, and the surface never touches it.

**Harness file trees.** ``create_harness_surface()`` exposes the artifact's
UTF-8 text files through ``TextFileTree``, excluding repository bookkeeping
and binary files. Paths and text are otherwise unchanged, because the harness
client owns how the tree is installed and interpreted.

**Configuration.** ``create_config_surface()`` serves the ``config``
component: one ``config.json`` object whose ``request_defaults`` fill
provider request fields the caller left unset (``temperature``,
``max_tokens``, ...). A plain key applies to every generation route. A key
that names a route path (``"/v1/responses"``) holds that route's own
fields and wins over the plain ones, since the API dialects spell the same
setting differently; ``/v1/messages/count_tokens`` takes only its own entry,
because a token count carries no generation fields. A change of defaults is
a release like any other, frozen per request and recorded against the
release that served it. The service applies it; it exposes no pulled tree,
so it composes beside a harness component.

**Skill trees.** ``create_skill_surface()`` adds optional server-side
injection to the same ``TextFileTree``. A ``SkillLayer`` owns one top-level
directory and validates it; a layer that also implements
``RequestSkillLayer`` can prepare inference requests. Pull-only layers expose
no request method, so the surface has no inference capability and never
materializes the artifact on the inference path. The same layers form the
component's ``SkillValidator``, its admission policy.

Building a surface
------------------

A recipe returns a ``Surface`` from ``build_surface``:

.. code:: python

   class PromptHooks:
       def prepare_request(self, artifact, path, request):
           prompt = read_prompt(artifact)
           return {**request, "messages": [prompt, *request["messages"]]}

       def verify_response(self, artifact, path, response):
           return None


   class PromptValidator:
       def validate(self, artifact):
           validate_prompt_artifact(artifact)


   class PromptRecipe(Recipe):
       def build_surface(self, scenario: str) -> Surface:
           return Surface(
               components={
                   "prompt": ComponentSurface(
                       validator=PromptValidator(), inference=PromptHooks(), files=TextFileTree()
                   )
               }
           )

A recipe whose scenario serves two components declares both; a step then
publishes one of them, naming it in ``TrainStepResult.component``, and the
committer carries the other forward from the previous checkpoint:

.. code:: python

   class AgentRecipe(Recipe):
       def build_surface(self, scenario: str) -> Surface:
           return Surface(
               components={
                   "weights": ComponentSurface(loader=WeightLoader(), inference=WeightInferenceHooks()),
                   "harness": ComponentSurface(files=TextFileTree()),
               }
           )

The base artifact of such a scenario keeps one directory per component
(``weights/``, ``harness/``), and every step must publish a durable
checkpoint: a live weight release names only an engine load, so it cannot
carry the other components. ``CompositeRecipe`` (``reef.recipe.composite``)
builds exactly this from existing flat recipes, one per component, and runs
each one's trainer as its own worker; see `configuration
<../reference/configuration.rst#recipe-configuration>`__ for its layout.

Add a small factory function when a composition is reused. Add a new
capability protocol only when no existing call site can express the consumer
interaction. The package depends only on ``artifact/`` and ``core/``; it sees
runtimes structurally, through the ``ServingRuntime`` and ``WeightRuntime``
protocols, and never imports a concrete one.

Remote weight snapshots
------------------------

``WeightLoader.activate`` calls the inference runtime's
``activate_checkpoint(artifact)`` when a materializable head becomes
available, including startup recovery and rollback. ``WeightRuntime`` binds
nothing by default, since a local engine already serves what the artifact
names. An inference runtime that serves immutable remote snapshots overrides
it to validate the checkpoint and bind its remote sampler and training state.
Tinker's inference runtime uses this to restore the authoritative artifact
before the scenario serves requests; its training runtime learns the same
head through ``TrainingRuntime.commit_candidate`` and rollback's
``restore_checkpoint``. The surface layer never imports the concrete SDK.
