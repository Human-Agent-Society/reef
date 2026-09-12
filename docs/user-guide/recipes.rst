Choose a recipe for agent learning
==================================

A recipe is picked along two axes: the **task type** your workload is, and
**what the recipe evolves**, model weights or the agent harness. Each task type
below names the standard benchmarks its examples have measured. Recipes that
evolve model weights need GPUs and the training stack in `Train model weights
from agent feedback <evolve-your-model.rst>`__; harness recipes need only a
model endpoint.

Reefine ships with ``reef-infra``; the other implementations live in the
repository's ``recipes/`` cookbook and do not ship in the Reef wheel.
``recipes/basic/`` is the record-only starting stack and stays outside the
catalog, and beta recipes join it once they publish learning results. The root
`README <../../README.md#recipes-and-examples>`__ and `recipes/README.md
<../../recipes/README.md>`__ present the same catalog.

Scientific discovery
--------------------

One hard problem, repeated attempts, and a measurable objective. The recipe
trains on the attempts it generates itself, at test time.

Measured benchmarks: TriMul (Guidance-TTT); circle packing (n = 26 and 32)
and Erdős minimum overlap (TTT-Discover).

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example and results
   * - TTT-Discover
     - model weights
     - ``recipes/tttd/``
     - `TTT-Discover <recipes/tttd.rst>`__
     - `example <../../recipes/tttd/examples/tttd/README.md>`__ · `results <../../recipes/tttd/examples/tttd/README.md#formal-8x64-results>`__
   * - Guidance-TTT
     - guidance-model weights; the executor stays frozen
     - ``recipes/tttd/``
     - `Guidance-TTT <../../recipes/tttd/examples/guidance_ttt/README.md>`__
     - `example <../../recipes/tttd/examples/guidance_ttt/README.md>`__ · `results <../../recipes/tttd/examples/guidance_ttt/results/README.md>`__

No recipe evolves the harness for this task type yet. CORAL TTT targets it and
is in beta; see the beta recipes below.

Continual learning on a task stream
-----------------------------------

Independent tasks, each scored by a verifier. The recipe learns from the
feedback on each task as the stream goes by.

Measured benchmarks: AIME 2025 (GEPA), three IMOAnswerBench problems (SAO),
and the Terminal-Bench 30-task hard subset (Meta-Harness). Reefine grades
three fixed coding tasks rather than a standard benchmark.

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example and results
   * - SAO
     - model weights
     - ``recipes/sao/``
     - `SAO <recipes/sao.rst>`__
     - `example <../../recipes/sao/examples/sao/README.md>`__ · `results <../../recipes/sao/examples/sao/README.md#results>`__
   * - GEPA
     - harness tree: rules, skills, and agent commands
     - ``recipes/gepa/``
     - `GEPA <recipes/gepa.rst>`__
     - `example <../../recipes/gepa/examples/aime/README.md>`__ · `results <../../recipes/gepa/examples/aime/README.md#the-validation-contract>`__
   * - Meta-Harness
     - harness: complete compositions
     - ``recipes/meta_harness/``
     - `Meta-Harness <../../recipes/meta_harness/README.md>`__
     - `results <../../recipes/meta_harness/RESULTS.md>`__
   * - Reefine
     - harness: skills, rules, agent commands, and pi extensions
     - ``reef/recipe/reefine/``
     - `Reefine <recipes/reefine.rst>`__
     - `example <../../tutorials/reefine/README.md>`__ · `results <../../tutorials/reefine/README.md#runs>`__

Learning from usage
-------------------

Real interaction with no explicit score, or delayed feedback. The recipe reads
the signal out of the traffic it already serves.

Measured benchmarks: the OpenClaw-RL simulated-student homework stream,
72 GSM8K sessions (OpenClaw-RL).

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example and results
   * - OpenClaw-RL
     - model weights
     - ``recipes/openclawrl/``
     - `OpenClaw-RL <recipes/openclawrl.rst>`__
     - `example <../../recipes/openclawrl/examples/openclawrl/README.md>`__ · `results <../../recipes/openclawrl/examples/openclawrl/README.md#results>`__
   * - SkillClaw
     - harness skill pool
     - ``recipes/skillclaw/``
     - `SkillClaw <recipes/skillclaw.rst>`__
     - `example <../../recipes/skillclaw/README.md>`__ · `results <../../recipes/skillclaw/README.md#the-2026-08-29-results-glm-53-flash-preliminary>`__

How a recipe is selected
------------------------

A deployment serves exactly one recipe, named by ``recipe.implementation`` in its config.
Every scenario it creates uses that recipe. Requests never name a recipe, and
scenario snapshots do not store one. The scenario header is the only routing a
caller provides. The artifact repository is therefore deployment-owned: do not
point deployments configured with different recipes at the same repository.

.. code:: yaml

   schema-version: 2
   recipe:
     implementation: recipes.sao.recipe:SAORecipe
     config:
       batch-size: 1

``recipe.implementation`` accepts the core value ``recipe``, a dotted class, or a preset.
Reefine ships in Reef; other learning methods are imported only when selected. The ``recipes/`` tree in
this repository is a cookbook; installed method packages work the same way.
`Configuration <../reference/configuration.rst#recipe-configuration>`__
describes each spelling.

Every recipe has a checkpoint strategy, defaulting to ``EveryNVersions(1)``.
``checkpoint_every_n_versions`` is the shorter spelling in deployment YAML.

Run the recipe you chose
------------------------

Start with the `inference and feedback quickstart
<../getting-started/quickstart.rst>`__ if you have not sent traffic through
Reef yet. For a harness recipe, follow `Evolve agent prompts, rules, and skills
<evolve-your-harness.rst>`__. For a weight recipe, follow `Train model weights
from agent feedback <evolve-your-model.rst>`__. Each recipe page above provides
its own configuration and example.

Built-in harness refinement
---------------------------

`Reefine <recipes/reefine.rst>`__ turns user instructions into harness updates
and ships with ``reef-infra``. Start it with ``reef serve --recipe reefine``
and a configured model endpoint.

Beta recipes
------------

`CORAL TTT <../../recipes/beta/coral/README.md>`__ and its
`coral_demo <../../recipes/beta/coral/examples/coral_demo/>`__ example are
**beta** until complete, reproducible learning results are published. Both
live under ``recipes/beta/coral/``. Integration and smoke tests validate the
wiring; they do not establish learning performance.
