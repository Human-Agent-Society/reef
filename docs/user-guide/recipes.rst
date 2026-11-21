Choose a recipe for agent learning
==================================

Pick a recipe by the **task type** of your workload and by **what it
evolves**, model weights or the agent harness. Weight recipes need GPUs and
the training stack in `Train model weights from agent feedback
<evolve-your-model.rst>`__, while harness recipes need only a model endpoint.

The implementations live in the repository's ``recipes/`` cookbook and do
not ship in the Reef wheel. Reef ships `SFT <recipes/sft.rst>`__. SFT trains model weights on
demonstrations or on imported examples, for all task types.
``recipes/basic/`` is the record-only starting stack and stays outside the
catalog, and beta recipes join it once they publish learning results. The root
`README <../../README.md#-recipes-and-examples>`__ and `recipes/README.md
<../../recipes/README.md>`__ show the same catalog.

Open-ended discovery
--------------------

One hard problem with a measurable objective and no known answer, where the
recipe makes repeated attempts and trains on those attempts at test time.

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
     - `Guidance-TTT <recipes/guidance-ttt.rst>`__
     - `Guidance-TTT on four discovery tasks <../../recipes/tttd/examples/guidance_ttt/README.md>`__

Offline tasks with a known answer
---------------------------------

Independent tasks from a fixed set, each with a reference answer. The recipe
samples one job's responses through Reef, trains on them, and serves the next
responses from the update.

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
     - `SAO on IMOAnswerBench <../../recipes/sao/examples/imo_answerbench/README.md>`__
   * - SDFT
     - model weights
     - ``recipes/sdft/``
     - `SDFT <recipes/sdft.rst>`__
     - `SDFT on a skill stream <../../recipes/sdft/examples/skill_stream/README.md>`__
   * - SDPO
     - model weights
     - ``recipes/sdpo/``
     - `SDPO <recipes/sdpo.rst>`__
     - `SDPO on SciKnowEval Chemistry <../../recipes/sdpo/examples/sciknoweval/README.md>`__
   * - OPD
     - model weights
     - ``recipes/opd/``
     - `OPD <recipes/opd.rst>`__
     - `OPD on mathematical reasoning <../../recipes/opd/examples/math/README.md>`__
   * - GEPA
     - harness tree: rules, skills, and agent commands
     - ``recipes/gepa/``
     - `GEPA <recipes/gepa.rst>`__
     - `GEPA on AIME 2025 <../../recipes/gepa/examples/aime/README.md>`__
   * - Meta-Harness
     - harness: complete trees
     - ``recipes/meta_harness/``
     - `Meta-Harness <../../recipes/meta_harness/README.md>`__
     - Meta-Harness on Terminal-Bench:
       `example <../../recipes/meta_harness/examples/terminal_bench/README.md>`__,
       `results <../../recipes/meta_harness/RESULTS.md>`__

Online learning from usage
--------------------------

Sessions with a user, where the interaction itself is the signal, so the
recipe reads it out of the traffic it already serves.

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
   * - ACE
     - harness rules: a playbook of lessons
     - ``recipes/ace/``
     - `ACE <../../recipes/ace/README.md>`__
     - `online deployment <../../recipes/ace/examples/serve.yaml>`__
   * - ReasoningBank
     - harness skills: memory items
     - ``recipes/reasoning_bank/``
     - `ReasoningBank <../../recipes/reasoning_bank/README.md>`__
     - `online deployment <../../recipes/reasoning_bank/examples/serve.yaml>`__

How a recipe is selected
------------------------

A deployment serves exactly one recipe, named by ``recipe.implementation`` in its config.
Every scenario it creates uses that recipe. Requests never name a recipe, and
scenario snapshots do not store one. The scenario header is the only routing a
caller provides. The artifact repository is therefore deployment-owned: do not
point deployments configured with different recipes at the same repository.

.. code:: yaml

   recipe:
     implementation: recipes.sao.recipe:SAORecipe
     config:
       training_mode: auto

``recipe.implementation`` accepts the core value ``recipe``, a dotted class, or a preset.
Learning methods are imported only when selected. The ``recipes/`` tree in
this repository is a cookbook; installed method packages work the same way.
`Configuration <../reference/configuration.rst#recipe-configuration>`__
describes each spelling.

Run the recipe you chose
------------------------

Start with the `inference and feedback quickstart
<../getting-started/quickstart.rst>`__ if you have not sent traffic through
Reef yet. For a harness recipe, follow `Evolve agent prompts, rules, and skills
<evolve-your-harness.rst>`__. For a weight recipe, follow `Train model weights
from agent feedback <evolve-your-model.rst>`__. Each recipe page above provides
its own configuration and example.

When no recipe fits, write your own: `Write a recipe
<../developer-guide/write-a-recipe.rst>`__ shows the plug-in points of a
weight method and of a harness method.

Beta recipes
------------

A beta recipe joins the catalog when it publishes complete, reproducible
learning results. Its integration and smoke tests validate the wiring; they
do not establish learning performance.

- `CORAL TTT <../../recipes/beta/coral/README.md>`__
  (``recipes/beta/coral/``): test-time training, with the `coral_demo
  <../../recipes/beta/coral/examples/coral_demo/>`__ example.
- `SPADE <../../recipes/beta/spade/README.md>`__ (``recipes/beta/spade/``,
  `arXiv:2608.19197 <https://arxiv.org/abs/2608.19197>`__): a task
  generator. One policy plays an Environment Designer that writes Harbor
  tasks and a Reasoning Agent that learns in them.
- `AutoEnvScaling <../../recipes/beta/autoenvscaling/README.md>`__
  (``recipes/beta/autoenvscaling/``): a task generator. A coding agent
  rebuilds recorded coding-agent sessions as Harbor tasks.
- `Reefine <../../recipes/beta/reefine/README.md>`__
  (``recipes/beta/reefine/``): a harness method that turns a plain-language
  request into a harness update. Start it from a checkout with ``reef serve
  -c recipes/beta/reefine/examples/configs/serve.yaml``; its `example
  <../../recipes/beta/reefine/examples/README.md>`__ runs three flows.

A task generator is not a recipe of its own. The deployment selects it as
the generator of the built-in generated-task recipe, which runs the tasks
and trains on them (see `generated tasks
<../reference/configuration.rst#generated-tasks>`__ and `Task generators
<../developer-guide/task-generators.rst>`__).
