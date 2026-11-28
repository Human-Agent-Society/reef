Scenario components
===================

Components declare artifacts and their import, validation and client-file contracts.
Most recipes use built-in ``weights`` or ``harness`` components.

The declaration
---------------

Override ``Recipe.build_components(scenario)`` to return ``ScenarioComponents``.
Build declarations from configuration alone; every service loads them.
Each ``ComponentSpec`` can provide:

- ``validator``: admit artifacts; the default accepts all.
- ``layout``: import directories; omitted means no imports.
- ``files``: expose a client file tree; omitted disables file routes.
- ``kind``: artifact kind; ``model`` and ``model_adapter`` load into inference.

At most one component loads weights and one exposes files.
Invalid declarations raise ``ValueError``.

Declare components
------------------

This recipe exposes text files:

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
   from reef.recipe import Recipe


   class PromptRecipe(Recipe):
       def build_components(self, scenario: str) -> ScenarioComponents:
           return ScenarioComponents({"prompt": ComponentSpec(files=TextFileTree())})

Add ``ArtifactValidator`` and ``ArtifactLayout`` for validation and imports.
See the `component contracts <../../reef/artifact/components.py>`__.
Composite commits replace one component while preserving the others and their IDs.

Admission checks
----------------

Validators read one component directory and raise ``ValueError`` or ``ReefError`` on refusal.
Invalid checkpoints fail their job; samples remain consumed.
Invalid imports, synced trees and starting artifacts return HTTP 400.
Previously stored artifacts are not revalidated during activation, restore, serving or train-from.
