SDFT: learn from demonstrations without forgetting
==================================================

Use Self-Distillation Fine-Tuning
(`SDFT <https://arxiv.org/abs/2601.19897>`__) to learn from demonstrations
on the student's own responses. The teacher sees a demonstration;
the student sees the original request. Reef requires Slime and GPUs.

Run the example
---------------

The `skill-stream example
<../../../recipes/sdft/examples/skill_stream/README.md>`__ trains Tool Use,
then Science Q&A. Use the Slime environment with eight GPUs: four for
training and four for serving.

.. code:: bash

   cd recipes/sdft/examples/skill_stream
   hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
   ./run.sh

It evaluates both skills every ten steps. The example freezes the teacher;
its moving-teacher runs drifted and collapsed.

How it works
------------

Generate responses, report demonstrations, then wait for publication before
generating the next job. Responses from an artifact that no longer serves
do not train. The teacher reads the demonstration in the final user message,
or a new user message after a tool result.

The default loss is full-vocabulary forward KL. The teacher starts from the
initial model and moves 1% toward the policy after each step.

The report contract
-------------------

Reference one inference record and supply ``metadata.teacher_context``.
An optional ``score`` never enters the loss.

.. code:: json

   {"references": ["<inference receipt>"], "metadata": {"teacher_context": "<demonstration>"}}

Configuration
-------------

Match ``job_samples`` to the optimizer batch size for one update per job:

.. code:: yaml

   recipe:
     implementation: recipes.sdft.recipe:SDFTRecipe
   pipeline:
     record2sample:
       config:
         max_teacher_tokens: 8192
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     config:
       global_batch_size: 32

``max_teacher_tokens`` defaults to 0, which disables the limit.
Over-limit or unrenderable teacher sequences count in ``unrenderable_samples``
and do not train. Match this limit to the trainer's context window.
Optional ``context_template`` controls the demonstration text and must contain
``{context}``. Invalid templates, unknown keys, or stage settings under
``recipe.config`` stop startup.

Set ``loss-type: custom_loss`` and
``disable-compute-advantages-and-returns: true`` in ``training.options``.
``use-rollout-logprobs`` is refused. Match serving and training
`chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.

The Slime flags use the ``sdft-`` prefix:

.. list-table::
   :header-rows: 1

   * - Flag suffix
     - Default
     - Meaning
   * - ``teacher``
     - self
     - ``separate`` uses ``teacher-checkpoint`` with the same architecture.
   * - ``divergence``
     - forward
     - ``forward``, ``reverse``, or ``jsd``; ``jsd-beta`` sets the teacher weight; default 0.5.
   * - ``top-k``
     - 0
     - Teacher-selected tokens plus the sampled token; 0 keeps the full distribution.
   * - ``teacher-update-rate``
     - 0.01
     - Policy fraction mixed per step; 0 freezes, 1 copies.
   * - ``importance-sampling-cap``
     - 2.0
     - Importance-weight cap; 0 disables correction.
   * - ``skip-response-tokens``
     - 0
     - Initial response tokens excluded from loss; paper runs used 3.

Only sampled reverse KL supports `score centering
<../../developer-guide/loss-families.rst#score-centering>`__.
Full-vocabulary scoring can need about 2 GB per 16k-token response
with a 248k vocabulary and tensor parallel size 4.
A moving teacher also keeps a host copy of actor weights per rank.

Results
-------

The `example results <../../../recipes/sdft/examples/skill_stream/README.md>`__
compare one SDFT run with one SFT control. Both learn the two skills;
SDFT retains more of the earlier skill. This does not estimate seed variance.

Related guides
--------------

- `SFT <sft.rst>`__: train directly on target tokens.
- `Train model weights <../evolve-your-model.rst>`__: configure the GPU stack.
