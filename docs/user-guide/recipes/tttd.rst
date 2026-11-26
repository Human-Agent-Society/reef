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
groups of 64 rollouts and one optimizer step. Two GPUs train;
two serve the policy. A successful exit confirms the requested commits.
Mismatched rollout counts, commits, artifacts, or search identity fail the run.

Choose a problem
~~~~~~~~~~~~~~~~

Set ``TTTD_TASK`` to ``erdos_min_overlap`` (default),
``circle_packing_26``, or ``circle_packing_32``:

.. code:: bash

   TTTD_TASK=circle_packing_26 ./run.sh

Each problem needs its own scenario and ``work/<task>/`` directory.
The `example README <../../../recipes/tttd/examples/tttd/README.md>`__
defines program entry points, timeouts, and rewards.

Change the search budget
~~~~~~~~~~~~~~~~~~~~~~~~

Edit the grid in ``serve.yaml``. The harness reads the same settings:

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 8
         rollouts_per_group: 64
     sample2job:
       config:
         job_samples: 512
   training:
     options:
       global-batch-size: 512

Defaults are eight groups and 64 rollouts per group. A group requires at
least two rollouts; two-rollout groups test the workflow only.
Use more rollouts to assess learning.

Configuration
-------------

Match ``job_samples`` and ``global-batch-size`` to groups times rollouts.
Each job contains one grid. Larger ``job_samples`` fails startup;
keep ``buffer_size`` at its default, ``job_samples``, to avoid combining grids.
Grid settings under ``recipe.config`` are refused.

``pipeline.record2sample.config.accept_multi_turn_policy_samples`` defaults
to false: only the last reply trains. Set true to train every reply.
The example's ``training.config`` defaults to:

- ``steps: 1``: committed search updates;
- ``max_new_tokens: 26000``: completion limit;
- ``enable_thinking: true``: Qwen3 thinking mode;
- ``num_gpus: 2``: actor GPUs, separate from serving.

Test one step at the intended grid and context length before a long run.
The example README lists per-task memory and concurrency settings.

How one step runs
-----------------

PUCT selects archived parents. Reef samples and judges sibling programs,
then waits for the complete grid. Training and publication complete before
the next grid starts. Grids that mix artifacts are discarded.

The loss uses entropic leave-one-out advantages and centered token KL
against the frozen base, without PPO clipping. Constant-reward groups are
removed; an entirely constant grid produces zero gradient.
Slime recomputes old-policy log probabilities and refuses
``use-rollout-logprobs``.

The report contract
-------------------

Report a finite ``score`` against each inference receipt:

.. code:: python

   client.report(scenario, {
       "score": reward,
       "metadata": {
           "step": step,
           "group": group,
           "rollout": rollout,
           "groups_per_step": groups_per_step,
           "rollouts_per_group": rollouts_per_group,
       },
   }, references=[receipt])

Reef validates the grid coordinates:

- ``step >= 0``;
- ``0 <= group < groups_per_step``;
- ``0 <= rollout < rollouts_per_group``;
- dimensions match the served grid, with at least one group and two rollouts.

Optional ``algorithm`` accepts ``tttd`` or ``ttt-discover`` and defaults to
``tttd``. Optional ``comparison_set`` must equal
``tttd-step-<step>-group-<group>``; Reef derives it when omitted.

Results
-------

The `Erdős results
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-erdos/README.md>`__
and `circle-packing results
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-packing/README.md>`__
record certified programs, training settings, and comparisons with the paper.
Each task has one run, so the results do not estimate seed variance.
Use the best archived score; mean rollout reward can fall while the best
solution remains unchanged.

Enable W&B tracking
-------------------

Enable `W&B tracking
<../../reference/configuration-observability.rst#experiment-tracking>`__ for
training metrics. ``tttd_step_committed`` logs ``archive_size`` and
``archive_best_reward``.

Inspect checkpoints and resume state
------------------------------------

Each task writes ``reef.log``, ``tttd-search-state.json``, and checkpoints
under ``work/<task>/``. Restart with the same task, model, grid, and sampling
settings. Incompatible identities or older state formats stop the restart.

A ``pending`` archive waits for its training commit. A ``committed`` archive
requires the restored commit number; an operator's artifact activation is
accepted at that commit. Retain the archive and its matching checkpoints.

The example saves after every step with ``save-interval: "1"`` and retains
the latest checkpoint. Retention does not change the save interval.

Add another problem
-------------------

Follow the `example's task instructions
<../../../recipes/tttd/examples/tttd/README.md>`__.
The judge must return a finite reward for valid programs and the defined
invalid reward otherwise. Start a new scenario when that contract changes.

Related guides
--------------

- `Guidance-TTT <guidance-ttt.rst>`__: train guidance with a frozen executor.
- `Train model weights <../evolve-your-model.rst>`__: configure the GPU stack.
