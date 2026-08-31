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

- loading a durable artifact into a runtime and recovering the serving head;
- preparing inference requests and verifying provider responses;
- exposing a versioned file tree to a client.

It does not own admission (whether an artifact may be published), producing
or selecting a new artifact, storage or version identity, commit ordering,
HTTP routing, or runtime internals such as workers, GPU placement, or adapter
caches. Those live in ``train/``, ``artifact/``, ``scenario/``,
``service/``, and ``runtime/`` respectively.

Admission is deliberately adjacent rather than a field on ``Surface``. A
recipe selects it through ``build_artifact_validator()``, the scenario factory
freezes it on ``ScenarioBinding.artifact_validator``, and the commit protocol
runs it before publication and rollback, so the decision stays in the commit
path instead of looking like a serving capability.

Capabilities and call sites
---------------------------

``Surface`` is a frozen composition of optional capabilities; ``None`` means
the scenario does not support one. Callers inspect fields, never types, and a
recipe never subclasses ``Surface``. Each capability has exactly one place it
is called from:

+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| Capability                                        | Called from                         | Meaning                                                              |
+===================================================+=====================================+======================================================================+
| ``loader.recover(current, checkpoint, runtime)``  | ``scenario/factory.py``             | Choose the head the runtime can still serve after startup. Without   |
|                                                   |                                     | a loader, recovery uses the durable checkpoint.                      |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``loader.load(artifact, runtime)``                | ``scenario/commit_protocol.py``     | Load a durable rollback target before the rollback commit becomes    |
|                                                   |                                     | authoritative. Without a loader, moving the head is sufficient.      |
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
|                                                   | ``scenario/commit_protocol.py``     | recovery, and after a publication or rollback minted its version but |
|                                                   |                                     | before the commit record makes it the head.                          |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+
| ``inference.begin_request(artifact, path)``       | ``service/request_service.py``      | Optional ``LeasingInferenceHooks``. Hold serving state for one       |
|                                                   |                                     | attempt; the lease is released when the attempt ends.                |
+---------------------------------------------------+-------------------------------------+----------------------------------------------------------------------+

The scenario owns lifecycle ordering and admission; the service owns
transport. A surface supplies only consumer-facing behavior at these sites.

Built-in surfaces
-----------------

+----------------------------------------+-------------------------+----------------------------------+------------------+
| Surface                                | Loader                  | Inference                        | Files            |
+========================================+=========================+==================================+==================+
| ``Surface()``                          | none                    | none                             | none             |
+----------------------------------------+-------------------------+----------------------------------+------------------+
| ``create_weight_surface()``            | ``WeightLoader``        | ``WeightInferenceHooks``         | none             |
+----------------------------------------+-------------------------+----------------------------------+------------------+
| ``create_weight_surface(scenario=...)``| ``WeightLoader``        | ``WeightInferenceHooks``         | none             |
|                                        |                         | selecting that scenario's        |                  |
|                                        |                         | adapter revision                 |                  |
+----------------------------------------+-------------------------+----------------------------------+------------------+
| ``create_harness_surface()``           | none                    | none                             | ``TextFileTree`` |
+----------------------------------------+-------------------------+----------------------------------+------------------+
| ``create_skill_surface(...)``          | none                    | optional ``SkillInferenceHooks`` | ``TextFileTree`` |
+----------------------------------------+-------------------------+----------------------------------+------------------+

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

**Skill trees.** ``create_skill_surface()`` adds optional server-side
injection to the same ``TextFileTree``. A ``SkillLayer`` owns one top-level
directory and validates it; a layer that also implements
``RequestSkillLayer`` can prepare inference requests. Pull-only layers expose
no request method, so the surface has no inference capability and never
materializes the artifact on the inference path. A skill recipe binds
``SkillValidator`` separately, as its admission policy.

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
           return Surface(inference=PromptHooks(), files=TextFileTree())

       def build_artifact_validator(self) -> PromptValidator:
           return PromptValidator()

Add a small factory function when a composition is reused. Add a new
capability protocol only when no existing call site can express the consumer
interaction. The package depends only on ``artifact/`` and ``core/``; it sees
runtimes structurally, through the ``ServingRuntime`` and ``WeightRuntime``
protocols, and never imports a concrete one.
