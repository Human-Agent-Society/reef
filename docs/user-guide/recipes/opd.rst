OPD: learn from a separate teacher on your own responses
========================================================

Use on-policy distillation (OPD) to train on the student's own responses,
scored by a frozen teacher. It requires Slime and GPUs. See the
`method description <https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

Run the example
---------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ uses
an OpenThoughts3-fine-tuned Qwen3-8B-Base student and Qwen3-32B teacher.
Use the Slime image and eight 80 GB GPUs. Follow its setup instructions
for pinned models, the chat template, and datasets. Then run:

.. code:: bash

   export OPD_MODEL_PATH=/work/models/Qwen3-8B-Base-SFT
   export OPD_TEACHER_PATH=/work/models/Qwen3-32B
   export OPD_RUN_DIR=/work/opd
   recipes/opd/examples/math/run.sh

The driver collects one job from one snapshot and waits for its committed
update. ``$OPD_RUN_DIR/results`` holds the results; the README defines
the AIME'24 evaluation.

How it works
------------

Generate responses, report them, and wait for publication before generating
the next job. Responses from an artifact that no longer serves do not train.
Teacher and student receive the same token sequence. Match serving and
training `chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.

Report contract
---------------

Reference exactly one inference receipt with empty ``teacher_context``.
Nonempty context is rejected; scores do not enter the loss.

.. code:: json

   {"references": ["<inference receipt>"], "metadata": {"teacher_context": ""}}

Configuration
-------------

Select ``recipes.opd.recipe:OPDRecipe``. Set
``pipeline.sample2job.config.job_samples`` to the optimizer batch size for
one update per job. ``pipeline.record2sample.config.max_teacher_tokens``
defaults to 0, which disables the limit. Over-limit sequences count in
``unrenderable_samples`` and do not train.

Use these distillation defaults and replace the checkpoint path:

.. code:: yaml

   training:
     options:
       loss-type: custom_loss
       disable-compute-advantages-and-returns: true
       opd-teacher: separate
       opd-teacher-checkpoint: /models/teacher
       opd-divergence: reverse
       opd-top-k: 1
       opd-teacher-update-rate: 0
       opd-importance-sampling-cap: 0
       opd-advantage-clip: 0

This uses sampled-token reverse KL with a frozen teacher and no importance
correction. ``use-rollout-logprobs`` is refused. Changing distillation flags
changes the method.

A teacher checkpoint must share the student's architecture, parallel layout,
and token IDs. It uses extra host memory and the actor's GPUs.
The teacher pass clears the student's LoRA adapter.

Optional ``opd-advantage-clip: 1.0`` bounds token advantages to ``[-1, 1]``
before weighting. The reported divergence remains unclipped. It requires
``opd-divergence: reverse``, positive ``opd-top-k``, and
``opd-top-k-distribution: renormalized``. See the
`clipping contract <../../../recipes/opd/README.md#the-teacher>`__.

Teacher engine
--------------

When only token IDs match, serve the teacher on separate GPUs.
This replaces ``opd-teacher-checkpoint``:

.. code:: yaml

   teacher:
     model-path: Qwen/Qwen3-32B
     num-gpus: 2
     port: 30001
     options:
       context-length: 32768
   training:
     options:
       opd-teacher: separate
       opd-top-k: 1
       rollout-temperature: 1.0

The engine requires teacher-selected tokens, matching token IDs, and
rollout temperature 1. Its context limit must cover every rendered sequence.
``teacher.timeout`` bounds each scoring request; ``teacher.options`` passes
native SGLang settings.

Scoring failure stops the step before training. ``perf/distill_teacher_time``
reports scoring seconds per step.
