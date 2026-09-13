Choose a recipe for agent learning
==================================

Pick a recipe by the **task type** of your workload and by **what it
evolves**, model weights or the agent harness. Weight recipes need GPUs and
the training stack in `Train model weights from agent feedback
<evolve-your-model.rst>`__, while harness recipes need only a model endpoint.

Reefine ships with ``reef-infra``, and the other implementations live in the
repository's ``recipes/`` cookbook and do not ship in the Reef wheel.
``recipes/basic/`` is the record-only starting stack and stays outside the
catalog, and beta recipes join it once they publish learning results. The root
`README <../../README.md#recipes-and-examples>`__ and `recipes/README.md
<../../recipes/README.md>`__ show the same catalog.

Scientific discovery
--------------------

One hard problem with a measurable objective, where the recipe makes repeated
attempts and trains on those attempts at test time.

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example
   * - TTT-Discover
     - model weights
     - ``recipes/tttd/``
     - `TTT-Discover <recipes/tttd.rst>`__
     - `TTT-Discover on circle packing and Erdős minimum overlap <../../recipes/tttd/examples/tttd/README.md>`__
   * - Guidance-TTT
     - guidance-model weights; the executor stays frozen
     - ``recipes/tttd/``
     - `Guidance-TTT <../../recipes/tttd/examples/guidance_ttt/README.md>`__
     - `Guidance-TTT on TriMul <../../recipes/tttd/examples/guidance_ttt/README.md>`__

Continual learning on a task stream
-----------------------------------

A stream of independent tasks that a verifier scores one by one, so the recipe
learns from each score before the next task arrives.

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example
   * - SAO
     - model weights
     - ``recipes/sao/``
     - `SAO <recipes/sao.rst>`__
     - `SAO on IMOAnswerBench <../../recipes/sao/examples/sao/README.md>`__
   * - GEPA
     - harness tree: rules, skills, and agent commands
     - ``recipes/gepa/``
     - `GEPA <recipes/gepa.rst>`__
     - `GEPA on AIME 2025 <../../recipes/gepa/examples/aime/README.md>`__
   * - Meta-Harness
     - harness: complete compositions
     - ``recipes/meta_harness/``
     - `Meta-Harness <../../recipes/meta_harness/README.md>`__
     - Meta-Harness on Terminal-Bench:
       `example <../../recipes/meta_harness/examples/terminal_bench/README.md>`__,
       `results <../../recipes/meta_harness/RESULTS.md>`__

Learning from usage
-------------------

Real interaction where no one reports a score or the feedback arrives late, so
the recipe reads the signal out of the traffic it already serves.

.. list-table::
   :header-rows: 1

   * - Recipe
     - Evolves
     - Code
     - Docs
     - Example
   * - OpenClaw-RL
     - model weights
     - ``recipes/openclawrl/``
     - `OpenClaw-RL <recipes/openclawrl.rst>`__
     - `OpenClaw-RL on the GSM8K homework stream <../../recipes/openclawrl/examples/openclawrl/README.md>`__
   * - SkillClaw
     - harness skill pool
     - ``recipes/skillclaw/``
     - `SkillClaw <recipes/skillclaw.rst>`__
     - `SkillClaw on WildClawBench <../../recipes/skillclaw/README.md>`__
   * - Reefine
     - harness: skills, rules, agent commands, and pi extensions
     - ``reef/recipe/reefine/``
     - `Reefine <recipes/reefine.rst>`__
     - `Reefine on reef-pi <../../tutorials/reefine/README.md>`__

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
Reefine ships in Reef, and other learning methods are imported only when selected. The ``recipes/`` tree in
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
wiring but do not establish learning performance.
