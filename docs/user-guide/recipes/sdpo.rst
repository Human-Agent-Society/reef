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
The README records one 100-step run. Its avg@8 measures mean accuracy,
not pass@8, and does not estimate seed variance.

Training requirements
---------------------

Report one attempt's receipt, score, and grid position:

.. code:: json

   {
     "references": ["<inference receipt>"],
     "score": 1.0,
     "metadata": {"step": 3, "group": 12, "rollout": 5}
   }

Set the grid under ``pipeline.record2sample.config``.
Keep ``job_samples``, ``buffer_size``, and Slime's ``global-batch-size`` equal
to groups times rollouts. Each job must contain exactly one grid.
Wait for publication before sampling the next grid; mixed or stale artifacts
cannot train.

By default, scores of at least 0.5 count as correct. An attempt without a
correct sibling has weight 0 but remains in the step's mean.
The loss uses the student's top 100 tokens plus a remainder bucket;
use zero dropout. Fit ``max_teacher_tokens`` within training's context window.
See the example's ``serve.yaml`` for the complete configuration.
