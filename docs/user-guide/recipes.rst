Choose a recipe for agent learning
==================================

Choose a method for your feedback, then follow its example. Weight methods need
`GPUs <evolve-your-model.rst>`__ or a supported `Tinker objective <tinker.rst>`__.
Harness methods need a model endpoint.

.. list-table::
   :header-rows: 1

   * - Goal or feedback
     - Method
   * - Optimize one problem without a known answer
     - `TTT-Discover <recipes/tttd.rst>`__ trains weights; `Guidance-TTT <recipes/guidance-ttt.rst>`__ trains a guide for a fixed executor.
   * - Train from scored tasks
     - `SAO <recipes/sao.rst>`__ uses one rollout; `SDPO <recipes/sdpo.rst>`__ uses correct attempts within a group.
   * - Learn from demonstrations or a teacher
     - `SFT <recipes/sft.rst>`__ fits target tokens; `SDFT <recipes/sdft.rst>`__ uses demonstrations; `OPD <recipes/opd.rst>`__ uses teacher scores.
   * - Learn from later conversation turns
     - `OpenClaw-RL <recipes/openclawrl.rst>`__ trains weights.
   * - Improve harness files on fixed tasks
     - `GEPA <recipes/gepa.rst>`__ revises files; `Meta-Harness <../../recipes/meta_harness/README.md>`__ revises complete trees.
   * - Improve a harness from usage
     - `ACE <../../recipes/ace/README.md>`__ maintains a playbook; `ReasoningBank <../../recipes/reasoning_bank/README.md>`__ maintains memory skills.

Set ``recipe.implementation`` in your deployment. All scenarios use that recipe;
do not share artifact repositories between recipes. Most methods need a checkout.
Built-in SFT also ships in the wheel. See `recipe configuration
<../reference/configuration-recipe.rst>`__ or `write a recipe
<../developer-guide/write-a-recipe.rst>`__.

Beta recipes
------------

Beta methods lack complete, reproducible learning results:
`CORAL TTT <../../recipes/beta/coral/README.md>`__,
`SPADE <../../recipes/beta/spade/README.md>`__,
`AutoEnvScaling <../../recipes/beta/autoenvscaling/README.md>`__,
`continual environments <../../recipes/beta/continual_envs/README.md>`__ and
`Reefine <../../recipes/beta/reefine/README.md>`__.
Their READMEs contain setup, results and limits. Integration tests do not establish
learning quality.
