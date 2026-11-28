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

It evaluates both skills every ten steps. The README records one SDFT run
and one SFT control; these do not estimate seed variance.

Training requirements
---------------------

Reference one inference receipt and supply a demonstration:

.. code:: json

   {"references": ["<inference receipt>"], "metadata": {"teacher_context": "<demonstration>"}}

Scores never enter the loss. Match job size to the optimizer batch size,
then wait for publication before sampling the next job.
Responses from artifacts that no longer serve do not train.

The default loss is full-vocabulary forward KL. The teacher moves 1% toward
the policy per step. The example instead sets ``sdft-teacher-update-rate: 0``;
its moving-teacher runs collapsed.

Set ``pipeline.record2sample.config.max_teacher_tokens`` to fit training's
context window and host memory. Its default, 0, disables the limit.
Over-limit or unrenderable sequences do not train.
Match serving and training chat templates. Use `SFT <sft.rst>`__ to train
directly on demonstration tokens instead.
