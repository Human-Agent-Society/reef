Choose a recipe for agent learning
==================================

Match a recipe to your feedback. Weight methods need the
`GPU stack <evolve-your-model.rst>`__ or a supported `Tinker objective <tinker.rst>`__.
Harness methods need a model endpoint. Most methods require a checkout;
built-in `SFT <recipes/sft.rst>`__ also ships in the wheel.

Open-ended discovery
--------------------

Use these methods for one hard problem with a measurable objective and no
known answer.

.. list-table::
   :header-rows: 1

   * - Method
     - What it changes
     - Example
   * - `TTT-Discover <recipes/tttd.rst>`__
     - Model weights from repeated attempts.
     - `Circle packing and Erdős overlap <../../recipes/tttd/examples/tttd/README.md>`__
   * - `Guidance-TTT <recipes/guidance-ttt.rst>`__
     - Guidance-model weights; the executor stays fixed.
     - `Four discovery tasks <../../recipes/tttd/examples/guidance_ttt/README.md>`__

Offline tasks with a known answer
---------------------------------

Use these methods with a fixed task set and a grader or reference answer.

.. list-table::
   :header-rows: 1

   * - Method
     - Learning signal
     - Example
   * - `SAO <recipes/sao.rst>`__
     - One scored rollout per task.
     - `IMOAnswerBench <../../recipes/sao/examples/imo_answerbench/README.md>`__
   * - `SDFT <recipes/sdft.rst>`__
     - Demonstrations supplied to the teacher.
     - `Skill stream <../../recipes/sdft/examples/skill_stream/README.md>`__
   * - `SDPO <recipes/sdpo.rst>`__
     - Correct attempts within a task group.
     - `SciKnowEval Chemistry <../../recipes/sdpo/examples/sciknoweval/README.md>`__
   * - `OPD <recipes/opd.rst>`__
     - A separate teacher scores student responses.
     - `Mathematical reasoning <../../recipes/opd/examples/math/README.md>`__
   * - `GEPA <recipes/gepa.rst>`__
     - Feedback guides harness-file revisions.
     - `AIME 2025 <../../recipes/gepa/examples/aime/README.md>`__
   * - `Meta-Harness <../../recipes/meta_harness/README.md>`__
     - Evaluation history guides complete harness trees.
     - `Terminal-Bench <../../recipes/meta_harness/examples/terminal_bench/README.md>`__

Online learning from usage
--------------------------

Use these methods with agent conversations or scored sessions.

.. list-table::
   :header-rows: 1

   * - Method
     - What it changes
     - Example
   * - `OpenClaw-RL <recipes/openclawrl.rst>`__
     - Model weights from later conversation turns.
     - `GSM8K homework stream <../../recipes/openclawrl/examples/openclawrl/README.md>`__
   * - `ACE <../../recipes/ace/README.md>`__
     - A harness playbook of lessons.
     - `Online deployment <../../recipes/ace/examples/serve.yaml>`__
   * - `ReasoningBank <../../recipes/reasoning_bank/README.md>`__
     - Harness skills containing memory items.
     - `Online deployment <../../recipes/reasoning_bank/examples/serve.yaml>`__

How a recipe is selected
------------------------

Set ``recipe.implementation``; see `recipe configuration
<../reference/configuration-recipe.rst#recipe-configuration>`__.
All scenarios use that recipe. Snapshots contain no recipe definition;
do not share artifact repositories between different recipes.

Run the recipe you chose
------------------------

Start with its linked example. For new methods, see
`Write a recipe <../developer-guide/write-a-recipe.rst>`__.

Beta recipes
------------

Beta methods lack the complete, reproducible learning results required for
the main catalog. Integration tests do not establish learning performance.

- `CORAL TTT <../../recipes/beta/coral/README.md>`__: test-time training.
- `SPADE <../../recipes/beta/spade/README.md>`__: one policy writes Harbor tasks
  as a Designer and learns from playing them.
- `AutoEnvScaling <../../recipes/beta/autoenvscaling/README.md>`__: generates
  Harbor tasks from recorded sessions or seed tasks.
- `Reefine <../../recipes/beta/reefine/README.md>`__: revises a harness from a
  plain-language request. See its `example <../../recipes/beta/reefine/examples/README.md>`__.

SPADE and AutoEnvScaling plug into the built-in generated-task recipe.
See `Generated tasks <../reference/configuration-generated-tasks.rst#generated-tasks>`__
for configuration and each method's README for results and limits.
