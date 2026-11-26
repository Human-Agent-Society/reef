Scenario components
===================

Components declare the artifacts a scenario stores and exposes.
Use the built-in ``weights`` or ``harness`` component unless you need a
custom ``Recipe.build_components`` declaration.

The declaration
---------------

``build_components(scenario)`` returns ``ScenarioComponents``: a mapping of
names to ``ComponentSpec`` values. Build it from configuration alone;
every service loading the scenario reads it.

.. list-table::
   :header-rows: 1

   * - Field
     - Purpose
   * - ``validator``
     - Admit artifacts before storage; default accepts all.
   * - ``layout``
     - Check and import directories; default disables imports.
   * - ``files``
     - Expose a client file tree; default disables file routes.
   * - ``kind``
     - Identify artifacts; ``model`` and ``model_adapter`` load into inference.

At most one component can load weights, and one can expose files.
Invalid declarations raise ``ValueError``.

Built-in recipes
----------------

``Recipe`` declares none. ``WeightTrainingRecipe`` declares ``weights``.
``HarnessEvolutionRecipe`` declares ``harness`` with adapter-specific file
`rendering <harness-adapters.rst#rendering>`__.
``CompositeRecipe`` combines component recipes.

Declare components
------------------

To expose text files:

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
   from reef.recipe import Recipe


   class PromptRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           return ScenarioComponents({"prompt": ComponentSpec(files=TextFileTree())})

Add ``ArtifactValidator`` and ``ArtifactLayout`` implementations for validation
and imports. See the `component contracts <../../reef/artifact/components.py>`__.

Composite snapshots have a directory per component. Each commit replaces
one component and preserves the others, including their artifact IDs.
Snapshots with zero or one component are flat.

Admission checks
----------------

An ``ArtifactValidator`` reads one component directory.
Raise ``ValueError`` or ``ReefError`` to refuse it:

- Invalid checkpoints fail their job; samples still count as trained.
- Invalid imports, synced trees and starting artifacts return HTTP 400.

Previously stored artifacts are not revalidated during activation, restore,
serving or train-from. Multi-component snapshots need per-component checks.

What components do not own
--------------------------

Training produces artifacts, control commits and selects them, and inference
loads weights. Components only declare their artifact contracts.
