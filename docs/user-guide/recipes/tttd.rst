TTT-Discover: train a model at test time
========================================

Use `TTT-Discover <https://arxiv.org/abs/2601.16175>`__ to specialize a model
while searching for a high-scoring program. The harness keeps a PUCT archive,
judges generated candidates, and reports a complete rollout grid.
Reef trains and publishes a LoRA update before the next search step.

Run the example
---------------

The `example <../../../recipes/tttd/examples/tttd/README.md>`__ supports
Erdős minimum overlap and circle packing with Qwen3-8B.
Use Linux, four NVIDIA GPUs, Docker, and the
`Reef GPU environment <../../getting-started/installation.rst>`__.
From the repository root:

.. code:: bash

   git submodule update --init third_party/reef-client
   uv pip install -e ./third_party/reef-client
   uv pip install -e .
   cd recipes/tttd/examples/tttd
   ./run.sh

The first run downloads ``Qwen/Qwen3-8B`` into ``work/model``.
It runs Erdős with eight groups of 64 rollouts, thinking enabled, and one
optimizer step. Two GPUs train the actor; two serve the policy.

Reef writes its service log to ``work/erdos_min_overlap/reef.log``. The status
endpoint reports the commit number (``commit_number``) and the served artifact
(``served_artifact_id``):

.. code:: bash

   curl -s \
     -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/status

A successful exit confirms the requested training steps committed.
Mismatched rollout counts, commits, served artifacts, or search identity fail
the run.

Choose a problem
~~~~~~~~~~~~~~~~

Set ``TTTD_TASK`` to choose a Harbor task; Erdős is the default.

.. list-table::
   :header-rows: 1

   * - ``TTTD_TASK``
     - Generated entry point
     - Reward
     - Program timeout
   * - ``erdos_min_overlap``
     - ``run()``
     - Reciprocal of the verified C5 bound
     - 1,100 seconds
   * - ``circle_packing_26``
     - ``run_packing()``
     - Verified sum of 26 circle radii
     - 530 seconds
   * - ``circle_packing_32``
     - ``run_packing()``
     - Verified sum of 32 circle radii
     - 530 seconds

.. code:: bash

   TTTD_TASK=circle_packing_26 ./run.sh

Each task has its own scenario and ``work/<task>/`` directory.
Judges execute the submitted program and recompute its result.

Change the search budget
~~~~~~~~~~~~~~~~~~~~~~~~

``serve.yaml`` holds the grid and the training settings, and the harness reads
the same file. Slime's ``global-batch-size`` and ``job_samples`` in that file
must equal ``groups_per_step x rollouts_per_group``.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - ``pipeline.record2sample.config.groups_per_step``
     - 8
     - Parents selected for one step.
   * - ``pipeline.record2sample.config.rollouts_per_group``
     - 64
     - Sibling rollouts sampled from each parent.
   * - ``training.config.steps``
     - 1
     - Search and optimizer steps. Slime reads it as ``--num-rollout``.
   * - ``training.config.max_new_tokens``
     - 26000
     - Maximum completion length.
   * - ``training.config.enable_thinking``
     - true
     - Qwen3 chat-template thinking mode. The key is absent from the shipped
       ``serve.yaml``.
   * - ``training.config.num_gpus``
     - 2
     - The actor's GPUs. ``inference.num-gpus`` gives the two serving engines
       their own two GPUs.

Test the intended grid for one step before a long run. Completion length
and evaluator concurrency affect GPU memory, host memory, and CPU use.
The example README lists task-specific sequence and concurrency limits.

How one step runs
-----------------

Run one scenario per problem. Each step:

1. Select ``groups_per_step`` parents from the PUCT archive.
2. Sample ``rollouts_per_group`` programs for each parent through Reef.
3. Judge every program and report its score and grid coordinate.
4. Wait for the complete grid, then train the adapter.
5. Wait for Reef's commit and served artifact to advance, then commit the
   matching archive.

Reef discards grids that mix artifacts. It removes groups with constant
rewards before training. If all groups are constant, it keeps one group
and the step has zero gradient.

The loss uses adaptive-beta entropic leave-one-out advantages and centered
token KL against the frozen base model, without PPO clipping.
The actor recomputes old-policy log probabilities from the step's starting
weights. Slime refuses ``use-rollout-logprobs``.

The best archived score can remain unchanged while the mean rollout reward
falls. Use the archive to track the best solution.

The report contract
-------------------

Every rollout reports a finite ``score`` against the receipt of its inference
request. The report's ``metadata`` identifies one coordinate in the step grid.
Reef validates these fields when it receives the report.

.. list-table::
   :header-rows: 1

   * - ``metadata`` field
     - Required
     - Rule
   * - ``algorithm``
     - No
     - ``tttd`` or ``ttt-discover``; defaults to ``tttd``
   * - ``step``
     - Yes
     - ``>= 0``
   * - ``group``
     - Yes
     - ``0 <= group < groups_per_step``
   * - ``rollout``
     - Yes
     - ``0 <= rollout < rollouts_per_group``
   * - ``groups_per_step``
     - Yes
     - ``>= 1``; must match the served grid
   * - ``rollouts_per_group``
     - Yes
     - ``>= 2``; must match the served grid
   * - ``comparison_set``
     - No
     - ``tttd-step-<step>-group-<group>``; derived when omitted

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

Configuration
-------------

Define the grid under ``pipeline.record2sample``. Set the job and optimizer
batch sizes to the grid size for one update per grid:

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
         job_samples: 512       # the grid: groups_per_step x rollouts_per_group

.. container:: config

   ``pipeline.record2sample.config.groups_per_step`` : 8
       comparison groups in one step; at least one.

   ``pipeline.record2sample.config.rollouts_per_group`` : 64
       siblings per group; at least two, since a group of one has no relative reward.

   ``pipeline.record2sample.config.accept_multi_turn_policy_samples`` : false
       Train every reply in a reported trajectory. Otherwise, train only its
       last reply.

   ``pipeline.sample2job.config.job_samples`` : required
       Minimum samples per job. Set it to the grid size.

Each job contains one complete grid. Match Slime's ``--global-batch-size``
to ``groups_per_step x rollouts_per_group``.

- With a ``job_samples`` larger than the grid, the scenario's pipeline fails
  to build.
- Leave ``pipeline.sample2job.config.buffer_size`` at its default,
  ``job_samples``. A larger value can put two grids in one job's data, and
  ``TTTDJobData`` then raises an error.
- Reef refuses ``groups_per_step`` or ``rollouts_per_group`` under
  ``recipe.config`` and names the key's place in ``pipeline``.

Results
-------

Each run used Qwen3-8B with thinking enabled on two NVIDIA B200 GPUs: eight
groups of 64 programs per step, then one rank-32 LoRA update with an Adam
learning rate of ``4e-5``. Each task has one trajectory, so these results do
not estimate variance across seeds.

.. list-table::
   :header-rows: 1

   * - Task
     - Steps
     - Certified result
     - TTT-Discover
     - Target
   * - Erdős minimum overlap (``C5`` upper bound, lower is better)
     - 25
     - ``0.38094``
     - ``0.38093``
     - ``0.38080``
   * - Packing 26 (sum of radii, higher is better)
     - 50
     - ``2.635983``
     - ``2.635983``
     - ``2.636``
   * - Packing 32 (sum of radii, higher is better)
     - 50
     - ``2.939573``
     - ``2.939572``
     - ``2.940``

.. image:: ../../assets/tttd/packing-configurations.png
   :alt: Verified circle configurations for Packing 26 and Packing 32

The certified results come from the task judges and the best programs in the
search archives, not from the mean rollout reward. The `Erdős result details
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-erdos/README.md>`__
and the `circle-packing overview
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-packing/README.md>`__
hold the run configurations, numeric histories, saved programs, training
metrics, and W&B run IDs.

Enable W&B tracking
-------------------

Enable ``observability.wandb.enabled`` in ``serve.yaml`` and set a project.
Supply credentials through ``WANDB_API_KEY`` or your credential store.
See `Experiment tracking
<../../reference/configuration-observability.rst#experiment-tracking>`__.

The ``tttd/*`` metrics include grid size, reward statistics, and constant
reward groups removed before training. ``tttd/reward_zero_fraction`` counts
zero rewards, including invalid solutions in the bundled tasks.
These metrics describe the complete grid before filtering.

The ``tttd_step_committed`` log event reports ``archive_size`` and
``archive_best_reward``. Use these to track search progress alongside mean
rollout reward.

Inspect checkpoints and resume state
------------------------------------

Each task writes ``reef.log``, ``tttd-search-state.json``, records, artifacts,
and checkpoints under ``work/<task>/``.

The state records the task, model, recipe, grid, sampling settings, and search
identity. An incompatible identity or older state format stops the restart.

- ``pending`` means the programs were evaluated but the commit was not
  observed. The controller waits for that commit.
- ``committed`` records the commit and served artifact. After Reef restores
  the expected commit number, the controller accepts the artifact that serves.
  This permits an operator's activation change.

The example saves after each step with ``save-interval: "1"`` and retains
the latest checkpoint. Set the interval before the run; retention controls
what remains, not when checkpoints are saved. See the
`State model <../../advanced_topics/state-model.rst>`__ for recovery.

Add another problem
-------------------

1. Add a task under ``recipes/tttd/examples/tttd/harbor/`` with its instruction,
   ``task.toml``, judge, and final verifier.
2. Register it in ``run.py`` and ``run.sh``, including sequence and memory limits.
3. Adjust ``MAX_WORKERS`` in ``harness/harbor_agent.py`` for the judge's capacity.

The judge's ``grade(artifact)`` must return a finite reward for valid programs
and the task's invalid reward otherwise. Match the instruction to that
contract. Start a new scenario when the contract changes.

The example README and ``tests/recipes/tttd/`` show registration and judge
contracts. Test invalid outputs and reward calculations before a GPU run.

Related guides
--------------

- `Guidance-TTT <guidance-ttt.rst>`__: train a guidance model with a frozen
  executor on four discovery tasks.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `HTTP API reference <../../reference/http-api.rst>`__: connect your agent
  and query feedback, scenarios, and artifacts.
