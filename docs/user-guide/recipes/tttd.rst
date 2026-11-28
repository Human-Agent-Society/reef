TTT-Discover: train a model at test time
========================================

Use `TTT-Discover <https://arxiv.org/abs/2601.16175>`__ to specialize a model
while searching for a high-scoring program. Each search step evaluates a
rollout grid, trains a LoRA update, and commits the matching program archive.

Run the example
---------------

Use Linux, four NVIDIA GPUs, Docker, and the
`Reef GPU environment <../../getting-started/installation.rst>`__.
From the repository root:

.. code:: bash

   cd recipes/tttd/examples/tttd
   ./run.sh

The example downloads Qwen3-8B and runs Erdős minimum overlap with eight
groups of 64 rollouts and one optimizer step. Two GPUs train; two serve.
A successful exit confirms the requested commits.

Set ``TTTD_TASK`` to ``circle_packing_26`` or ``circle_packing_32`` for
packing problems. Each problem needs its own scenario and state directory.
See the `example README <../../../recipes/tttd/examples/tttd/README.md>`__
for program contracts, memory requirements, and restart instructions.

The report contract
-------------------

Report a finite score against each inference receipt:

.. code:: json

   {
     "references": ["<inference receipt>"],
     "score": 1.0,
     "metadata": {
       "step": 0, "group": 0, "rollout": 0,
       "groups_per_step": 8, "rollouts_per_group": 64
     }
   }

``step`` must be nonnegative. Group and rollout indices start at 0 and
must be below their dimensions. Dimensions must match the served grid,
with at least one group and two rollouts. Two-rollout groups test the
workflow only; use more to assess learning.

Configure the grid under ``pipeline.record2sample.config`` in ``serve.yaml``.
Keep ``job_samples``, ``buffer_size``, and ``global-batch-size`` equal to
groups times rollouts. Each job must contain one complete grid from one
artifact. Wait for publication before sampling the next grid.

The `Erdős results
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-erdos/README.md>`__
and `packing results
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-packing/README.md>`__
record one run per task, without seed-variance estimates.
Track the best archived score; mean rollout reward can fall independently.
