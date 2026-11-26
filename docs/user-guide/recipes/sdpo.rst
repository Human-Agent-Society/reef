SDPO: learn from your own correct attempts
==========================================

Use Self-Distillation Policy Optimization
(`SDPO <https://arxiv.org/abs/2601.20802>`__) when each attempt has a score.
Successful sibling attempts become demonstrations for the others.
The student trains on its own tokens. Reef requires Slime and GPUs.

Run the example
---------------

The `SciKnowEval example
<../../../recipes/sdpo/examples/sciknoweval/README.md>`__ trains Qwen3-8B
on Chemistry. Use the Slime environment with eight GPUs: four for training
and four for serving.

.. code:: bash

   cd recipes/sdpo/examples/sciknoweval
   hf download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
   ./run.sh

Each step samples 32 questions eight times. Evaluation runs every five steps.

How it works
------------

Report the complete grid, wait for publication, then sample the next grid.
Reef drops grids that mix artifacts and responses from artifacts no longer
serving. By default, each attempt uses its first correct sibling.
Without one, its weight is 0 but it remains in the step's mean.

The default loss is Jensen-Shannon divergence over the student's top 100
tokens plus a remaining-vocabulary bucket. Use zero dropout for this
student-selected representation. The teacher moves 5% toward the policy
per step.

The report contract
-------------------

Reference one attempt, with its score and grid position.
Optional ``teacher_context`` supplies feedback; the example leaves it empty.

.. code:: json

   {
     "references": ["<inference receipt>"],
     "score": 1.0,
     "metadata": {"step": 3, "group": 12, "rollout": 5, "teacher_context": ""}
   }

Configuration
-------------

Configure the stages under ``pipeline``; stage settings under
``recipe.config`` stop startup:

.. code:: yaml

   recipe:
     implementation: recipes.sdpo.recipe:SDPORecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 32
         rollouts_per_group: 8
         max_teacher_tokens: 18944
     sample2job:
       config:
         job_samples: 256
   training:
     options:
       global-batch-size: 256
       loss-type: custom_loss
       disable-compute-advantages-and-returns: true

One job contains one grid. Match ``job_samples`` and ``global-batch-size``
to groups times rollouts. A larger ``job_samples`` fails startup.
Keep ``buffer_size`` at its default, ``job_samples``; combining grids fails
job preparation. ``use-rollout-logprobs`` is refused. Match serving and training
`chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.

Defaults under ``pipeline.record2sample.config``:

- ``groups_per_step: 32`` and ``rollouts_per_group: 8``; at least two attempts per group.
- ``max_teacher_prompt_tokens: 10240`` truncates the prompt before adding the response.
- ``max_teacher_tokens: 18432`` must cover that prompt limit and fit training's context window.
  Longer sequences do not train and count in ``unrenderable_samples``.

Defaults under ``pipeline.sample2job.config``:

- ``success_reward_threshold: 0.5`` defines a correct attempt.
- ``allow_own_success_as_demonstration: false`` excludes the attempt's own answer.
- ``remove_thinking_from_demonstration: true`` strips ``<think>`` blocks.
- ``include_environment_feedback: false`` disables reported teacher context.
- ``environment_feedback_only_without_solution: true`` restricts feedback to attempts without a correct sibling.

The Slime flags use the ``sdpo-`` prefix:

.. list-table::
   :header-rows: 1

   * - Flag suffix
     - Default
     - Meaning
   * - ``teacher``
     - self
     - ``separate`` uses ``teacher-checkpoint``.
   * - ``divergence``
     - jsd
     - ``forward``, ``reverse``, or ``jsd``; ``jsd-beta`` sets the teacher weight; default 0.5.
   * - ``top-k``
     - 100
     - Tokens per position; 0 keeps the full distribution.
   * - ``top-k-source``
     - student
     - ``student`` or ``teacher`` selects the tokens.
   * - ``top-k-distribution``
     - tail
     - ``tail`` keeps a remainder bucket; ``renormalized`` conditions on selected tokens.
   * - ``teacher-update-rate``
     - 0.05
     - Policy fraction mixed per step; 0 freezes.
   * - ``importance-sampling-level``
     - token
     - ``token`` uses per-token ratios; ``sequence`` averages across the response.
   * - ``importance-sampling-cap``
     - 2.0
     - Importance-weight cap; 0 disables correction.
   * - ``skip-response-tokens``
     - 0
     - Initial response tokens excluded from loss.

Only sampled reverse KL supports `score centering
<../../developer-guide/loss-families.rst#score-centering>`__.

Results
-------

The `100-step example run
<../../../recipes/sdpo/examples/sciknoweval/README.md>`__ raised test avg@8
from 41.2% to 74.4% at step 75, ending at 71.7%.
This measures mean accuracy, not pass@8, and does not estimate seed variance.

Related guides
--------------

- `SDFT <sdft.rst>`__: supply demonstrations directly.
- `Train model weights <../evolve-your-model.rst>`__: configure the GPU stack.
