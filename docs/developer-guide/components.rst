Scenario components
===================

Components declare the artifacts a scenario holds, imports and exposes to
clients. Most recipes use the built-in ``weights`` or ``harness`` component.
Override ``Recipe.build_components`` only when those declarations do not fit.

The declaration
---------------

``build_components(scenario)`` returns ``ScenarioComponents`` from
``reef.artifact.components``. Its ``components`` mapping binds names to
``ComponentSpec`` values. Build the declaration from configuration alone:
every service that loads the scenario reads it.

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Field
     - Contract
     - Default
   * - ``validator``
     - Admit an artifact before Reef stores it.
     - Accept all
   * - ``layout``
     - Check and import an artifact directory.
     - No imports
   * - ``files``
     - Read the file tree a harness client pulls.
     - No file routes
   * - ``kind``
     - Declare the artifact kind. ``model`` and ``model_adapter`` load into inference.
     - No kind

At most one component can load weights, and at most one can expose ``files``.
Invalid declarations raise ``ValueError``.

Built-in recipes
----------------

- ``Recipe`` declares no components; it records traffic.
- ``WeightTrainingRecipe`` declares ``weights``, with the runtime's weight
  kind and the backend's training layout.
- ``HarnessEvolutionRecipe`` declares ``harness``, with ``HarnessLayout``
  and ``HarnessFiles``. File reads render the source for the selected adapter.
- ``CompositeRecipe`` combines the declarations of its component recipes.

See `Rendering <harness-adapters.rst#rendering>`__ for harness file behavior.

Declare components
------------------

For a component that exposes text files:

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
   from reef.recipe import Recipe


   class PromptRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           return ScenarioComponents({"prompt": ComponentSpec(files=TextFileTree())})

Add an ``ArtifactValidator`` to reject invalid artifacts and an
``ArtifactLayout`` to accept imports.

Use ``CompositeRecipe`` for multiple components. Each recipe has its own
trainer, and each training commit names the component it publishes.
Reef carries the other components forward. Each artifact retains its own ID.

Snapshots with zero or one component are flat. Multi-component snapshots
have one directory per component. See `Recipe configuration
<../reference/configuration-recipe.rst#recipe-configuration>`__.

Admission checks
----------------

An ``ArtifactValidator`` reads one component's local artifact directory.
Raise ``ValueError`` or ``ReefError`` with the reason for refusal.

- Training checks each checkpoint before storing it. Refusal fails the job;
  its samples still count as trained.
- Control checks imports, synced trees and scenario starting artifacts.
  Refusal returns HTTP 400.

Activation, restore, serving and train-from operations do not rerun validation.
A new validator therefore applies only to subsequently stored artifacts.
Multi-component snapshots require per-component checks; they cannot have a
separate snapshot validator.

What components do not own
--------------------------

Components declare artifact contracts. Training produces artifacts, control
commits and selects them, and inference loads weights. See
`Serving runtimes <serving-runtimes.rst>`__ and the
`state model <../advanced_topics/state-model.rst>`__.
