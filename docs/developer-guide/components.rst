Scenario components
===================

A recipe declares the components that every snapshot of a scenario binds,
such as ``weights`` or ``harness``. For each component it says what kind of
artifact the component holds, how Reef admits and imports it, and which
files a client pulls. This page defines the declaration and says which
service reads each field.

:For: recipe authors who declare a component other than ``weights`` or ``harness``
:Before you start: the plug-in map in `Write a recipe <write-a-recipe.rst>`__
:You will have: a ``build_components`` override, and the admission check of your artifacts

Most recipes need no override: ``WeightTrainingRecipe`` declares ``weights``
and ``HarnessEvolutionRecipe`` declares ``harness``.

The declaration
---------------

``Recipe.build_components(scenario)`` returns a ``ScenarioComponents``
(``reef.artifact.components``). Its ``components`` field maps each component
name to a ``ComponentSpec``. The result must depend on the recipe's
configuration alone. Every service that loads a scenario calls it, and the
loader calls it once when the deployment starts.

- ``validator`` (read by ``training`` and ``control``): an
  ``ArtifactValidator`` that admits an artifact of the component before Reef
  stores it (`Admission checks`_). The default accepts every artifact.
- ``layout`` (read by ``control``): an ``ArtifactLayout`` that checks an
  imported directory before admission. Without ``layout``, the component
  takes no import.
- ``files`` (read by ``harness``): a ``FileTree`` that reads the files a
  client pulls from a snapshot. Without a component with ``files``, the
  harness file routes refuse the scenario.
- ``kind`` (read by ``inference`` and ``control``): the ``ArtifactKind``.
  The inference service loads a ``model`` or ``model_adapter`` component
  into the runtime. A kind also makes the component a side of an artifact
  evaluation. ``None`` declares no kind.

Constraints, checked when the value is built (``ValueError``):

- at most one component has a weight kind (``model`` or ``model_adapter``);
- at most one component has ``files``.

The default declares no component: the scenario records traffic only.

Built-in recipes
----------------

+----------------------------+-------------+--------------------------+----------------------------+------------------+
| Recipe                     | Component   | Kind                     | Layout                     | Files            |
+============================+=============+==========================+============================+==================+
| ``Recipe``                 | none        |                          |                            |                  |
+----------------------------+-------------+--------------------------+----------------------------+------------------+
| ``WeightTrainingRecipe``   | ``weights`` | the runtime's            | ``TrainingArtifactLayout`` | none             |
|                            |             | ``served_weight_kind``   |                            |                  |
+----------------------------+-------------+--------------------------+----------------------------+------------------+
| ``HarnessEvolutionRecipe`` | ``harness`` | ``harness_plugin``       | ``HarnessLayout``          | ``HarnessFiles`` |
+----------------------------+-------------+--------------------------+----------------------------+------------------+
| ``CompositeRecipe``        | one per     | each recipe's            | each recipe's              | each recipe's    |
|                            | recipe      |                          |                            |                  |
+----------------------------+-------------+--------------------------+----------------------------+------------------+

The kind and the layout of ``weights`` are not configuration: the runtime
and the training backend give them. ``HarnessFiles`` renders the stored
source tree for the descriptor's adapter on each read, so the client gets
the files that its agent reads (see `Rendering
<harness-adapters.rst#rendering>`__).

Declare components
------------------

.. code:: python

   from reef.artifact import ArtifactValidator
   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
   from reef.recipe import Recipe


   class PromptValidator(ArtifactValidator):
       def validate(self, artifact):
           validate_prompt_artifact(artifact)


   class PromptRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           prompt = ComponentSpec(validator=PromptValidator(), files=TextFileTree())
           return ScenarioComponents({"prompt": prompt})

A recipe whose scenario serves two components declares both:

.. code:: python

   class AgentRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           return ScenarioComponents(
               {
                   "weights": ComponentSpec(kind="model_adapter"),
                   "harness": ComponentSpec(files=TextFileTree()),
               }
           )

- A training step publishes one component and names it in
  ``TrainingCommit.component``. Reef carries the other components forward
  from the previous snapshot.
- A snapshot of at most one component is flat. A snapshot of several
  components has one directory per component (``weights/``, ``harness/``).
- Each artifact keeps its own ``artifact_id``.
- ``CompositeRecipe`` (``reef.recipe.composite``) builds such a declaration
  from a mapping of component name to recipe, and runs each recipe's trainer
  as its own worker (see `Recipe configuration
  <../reference/configuration-recipe.rst#recipe-configuration>`__).

Admission checks
----------------

A validator reads the files of one artifact of its component, in a local
directory. It raises ``ValueError`` or a ``ReefError`` that names what does
not hold. The service that holds the files runs it once, before Reef stores
the artifact:

- The ``training`` service checks each checkpoint of a training job. A
  refusal fails the job with the validator's message, and Reef does not
  store the checkpoint. The samples of the job count as trained.
- The ``control`` service checks an import, a person's synced tree, and the
  artifact item that starts a scenario. A refusal returns HTTP 400.

Activation, restore, serving and a train-from run no check, so a check
applies only to artifacts stored after it was configured. A snapshot of
several components has no check of its own: Reef checks each component
alone.

What components do not own
--------------------------

Components declare facts. They do not produce or select artifacts, store
them, order commits, or load weights. Activation changes the scenario's
complete selection atomically and runs no check. How the inference runtime
loads a ``model`` or ``model_adapter`` component, including one adapter per
scenario, is the runtime's contract: see `What a runtime implements
<serving-runtimes.rst#what-a-runtime-implements>`__ and `Serving saved weights
<serving-runtimes.rst#serving-saved-weights>`__. The `state model
<../advanced_topics/state-model.rst>`__ defines ordering and recovery.
