OPD: learn from a separate teacher on your own responses
========================================================

Use on-policy distillation (OPD) to train on the student's own responses,
scored by a frozen teacher. It requires Slime and GPUs. See the
`method description <https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

Run the example
---------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ uses
an OpenThoughts3-fine-tuned Qwen3-8B-Base student and Qwen3-32B teacher.
Use its Slime image and eight 80 GB GPUs. Follow the README's model,
chat-template, and dataset setup, then run:

.. code:: bash

   export OPD_MODEL_PATH=/work/models/Qwen3-8B-Base-SFT
   export OPD_TEACHER_PATH=/work/models/Qwen3-32B
   export OPD_RUN_DIR=/work/opd
   recipes/opd/examples/math/run.sh

The driver collects one job from one snapshot and waits for its committed
update. Results go to ``$OPD_RUN_DIR/results``; the README defines the evaluation.

Training requirements
---------------------

Reference one inference receipt with empty ``teacher_context``:

.. code:: json

   {"references": ["<inference receipt>"], "metadata": {"teacher_context": ""}}

Scores do not enter the loss; nonempty context is rejected.
Match job size to the optimizer batch size. Wait for publication before
sampling the next job. Responses from artifacts no longer serving do not train.
Match serving and training chat templates, including thinking mode.

Teacher engine
--------------

A teacher checkpoint must share the student's architecture, parallel layout,
and token IDs. Otherwise, use a separate SGLang teacher with matching token IDs,
as the example does. Set rollout temperature to 1 and fit every rendered
sequence within the teacher's context limit. Scoring failure stops training
before the step starts.

See the `teacher configuration <../../../recipes/opd/README.md#teacher-engine>`__
for engine settings and `advantage clipping
<../../../recipes/opd/README.md#the-teacher>`__ for the optional gradient change.
