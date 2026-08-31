tttd
====

TTT-Discover (`arXiv:2601.16175 <https://arxiv.org/abs/2601.16175>`__) specializes
a model to one problem while it searches for the highest-scoring solution. The
harness selects candidate programs, asks the model to improve them, evaluates
the generated programs, and adds valid candidates to a PUCT archive. Reef trains
the scenario's LoRA adapter after it receives one complete rollout grid. The next
search step uses the updated adapter.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | a finite ``score`` per rollout, addressed into a step grid |
+-------------+------------------------------------------------------------+
| Loss family | ``tttd``                                                   |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/tttd/``                                          |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, grouped                                 |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a harness that already runs a search loop        |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/tttd/examples/tttd/``                            |
+-------------+------------------------------------------------------------+

What it does
------------

Run one scenario per problem. The search archive retains the best candidate
found so far. Mean rollout reward describes the programs sampled for a training
step, so it can decrease even when the best archived solution does not change.

The cookbook example uses Qwen3-8B and includes the Erdős minimum-overlap problem
and the circle-packing problems for 26 and 32 circles.

How one step runs
-----------------

.. flow::
   :loop: use the committed adapter for the next search step

   Select parents :: PUCT chooses one parent for each group
   Generate rollouts :: each group samples sibling programs
   Evaluate and report :: the task judge returns one score per rollout
   Complete the grid :: Reef waits for every configured coordinate
   Train the adapter :: Slime and Megatron perform one optimizer step
   Publish weights :: SGLang loads the scenario's new LoRA revision
   Commit search state :: the controller records the matching PUCT archive

The harness chooses ``groups_per_step`` parent candidates and samples
``rollouts_per_group`` programs from each parent. A prompt contains the task
instruction. When a parent already has a valid program, the prompt also contains
that program, its reward, and the end of its recorded output.

Every model request passes through Reef's OpenAI-compatible chat endpoint. The
SGLang backend records the sampled token IDs, response loss mask, and rollout log
probabilities. Reef assigns the scenario's current adapter to the request and
records the adapter version with the inference.

The task judge executes the generated program and computes its reward. The
harness reports that reward with the inference receipt and the rollout's step,
group, and position. ``TTTDProcessor`` waits for every position in the grid. It
rejects a grid that contains more than one release. Groups with one
reward value across all siblings are removed after the grid reaches the barrier.
If every group is constant, the processor keeps one group so that the training
runtime receives a zero-signal batch.

Slime computes adaptive-beta entropic leave-one-out advantages within each
group. The loss uses the rollout log probabilities without PPO clipping and adds
centered token KL against the frozen base model. Megatron updates rank-32 LoRA
matrices on the attention and MLP projections. Reef publishes the adapter to
SGLang after the optimizer step succeeds.

The controller writes the updated PUCT archive to
``tttd-search-state.json`` with phase ``pending``. It waits for Reef's scenario
step to advance and for the serving runtime load ID to change. It then records
phase ``committed`` with the new scenario step and starts the next search step.

Configuration
-------------

.. config::

   groups_per_step | 8 | comparison groups in one step. Env ``REEF_TTTD_GROUPS_PER_STEP``.
   rollouts_per_group | 64 | siblings per group; at least two, since a group of one has no relative reward. Env ``REEF_TTTD_ROLLOUTS_PER_GROUP``.
   max_staleness | 0 | accepted lag between the producing and serving version. Env ``REEF_MAX_STALENESS``.

The report contract
-------------------

Every rollout reports a finite ``score`` against the receipt returned by its
inference request. The report's ``metadata`` identifies one coordinate in the
step grid. Reef validates these fields when it receives the report.

+------------------------+----------+--------------------------------------------+
| ``metadata`` field     | Required | Rule                                       |
+========================+==========+============================================+
| ``algorithm``          | no       | ``tttd`` or ``ttt-discover``; defaults to  |
|                        |          | ``tttd``                                   |
+------------------------+----------+--------------------------------------------+
| ``step``               | yes      | ``>= 0``                                   |
+------------------------+----------+--------------------------------------------+
| ``group``              | yes      | ``0 <= group < groups_per_step``           |
+------------------------+----------+--------------------------------------------+
| ``rollout``            | yes      | ``0 <= rollout < rollouts_per_group``      |
+------------------------+----------+--------------------------------------------+
| ``groups_per_step``    | yes      | ``>= 1``; must match the served grid       |
+------------------------+----------+--------------------------------------------+
| ``rollouts_per_group`` | yes      | ``>= 2``; must match the served grid       |
+------------------------+----------+--------------------------------------------+
| ``comparison_set``     | no       | must equal                                 |
|                        |          | ``tttd-step-<step>-group-<group>``;        |
|                        |          | derived when omitted                       |
+------------------------+----------+--------------------------------------------+

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

The `cookbook harness
<../../../recipes/tttd/examples/tttd/README.md>`__ keeps the ordinary PUCT
search separate from its Reef adapter. The adapter sends each inference through
the scenario endpoint, retains the returned receipt, and reports the judge's
reward against that receipt with the rollout coordinate. The search code does
not import a Reef client or construct report payloads.

Run the example
---------------

The example starts Ray, Slime, Megatron, SGLang, Reef, reef-eval, and the selected
Harbor task on one machine. It requires Linux, NVIDIA GPUs, Docker, and the Reef
training dependencies described in `Installation <../../getting-started/installation.rst>`__. Run the
following commands from the repository root:

.. code:: bash

   git submodule update --init third_party/reef-client
   python -m pip install -e ./third_party/reef-client
   python -m pip install -e . "reef-eval[harbor]"
   cd recipes/tttd/examples/tttd

``run.sh`` downloads ``Qwen/Qwen3-8B`` into ``work/model`` on its first run,
starts the stack, and runs one episode of ``harbor/erdos_min_overlap`` on the
paper grid: eight groups of 64 rollouts, one optimizer step, thinking enabled,
two GPUs.

.. code:: bash

   ./run.sh

``TTTD_TASK`` is the only user-facing environment variable. The example keeps
the grid and training settings in ``serve.yaml``. A reduced smoke can use two
groups of two rollouts, one step, and thinking disabled; that run checks the
integration path rather than the sampling budget used for the formal results.

Reef writes its service log to ``work/erdos_min_overlap/reef.log``. The public
status endpoint reports the scenario step and runtime load ID:

.. code:: bash

   curl -s \
     -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/status

The controller returns an error when the rollout count, scenario step, artifact
version, or stored search identity does not match the current run. A successful
exit confirms that Reef committed the requested number of training steps.

Choose a problem
----------------

Three Harbor tasks ship with the example:

+-----------------------+-----------------------+---------------------------------------+-----------------+
| Value                 | Generated entry point | Reward                                | Program timeout |
+=======================+=======================+=======================================+=================+
| ``erdos_min_overlap`` | ``run()``             | Reciprocal of the verified C5 bound   | 1,100 seconds   |
+-----------------------+-----------------------+---------------------------------------+-----------------+
| ``circle_packing_26`` | ``run_packing()``     | Verified sum of 26 circle radii       | 530 seconds     |
+-----------------------+-----------------------+---------------------------------------+-----------------+
| ``circle_packing_32`` | ``run_packing()``     | Verified sum of 32 circle radii       | 530 seconds     |
+-----------------------+-----------------------+---------------------------------------+-----------------+

Erdős is the default. ``TTTD_TASK`` selects another: ``run.sh`` gives each task
its own scenario and state directory, and sizes the memory limits for it,
because the packing tasks need a longer context on the same GPUs.

.. code:: bash

   TTTD_TASK=circle_packing_26 ./run.sh

The judges run each submitted program in a subprocess and recompute the returned
quantity. The circle-packing judges verify the circle count, ensure every circle
stays inside the square, check pairwise non-overlap and finite values, and then
sum the radii. They do not use the sum reported by the generated program.

Configure a trajectory
----------------------

The grid has one source in ``serve.yaml``. The harness reads the same
``groups_per_step``, ``rollouts_per_group``, ``steps``, and ``max_new_tokens``
values when it starts. Slime's ``--global-batch-size`` in that file must equal
``groups_per_step x rollouts_per_group``.

.. config::

   reef.groups_per_step | 8 | parent candidates selected for one step, in ``serve.yaml``.
   reef.rollouts_per_group | 64 | sibling rollouts sampled from each parent, in ``serve.yaml``.
   training.steps | 1 | search and optimizer steps, in ``serve.yaml``; the Slime command reads it as ``--num-rollout``.
   training.max_new_tokens | 26000 | maximum completion length, in ``serve.yaml``.
   MAX_WORKERS | 512 | concurrent rollout and evaluator calls, derived from the task in ``harness/harbor_agent.py``; packing uses ``256``.
   enable_thinking | True | Qwen3 chat-template thinking mode, in ``harness/harbor_agent.py``.
   TTTD_SEQ_LENGTH | 30000 | training and serving context length; ``run.sh`` sets it per task, ``32768`` for packing, beside ``TTTD_MAX_TOKENS_PER_GPU`` and ``TTTD_LOG_PROBS_CHUNK_SIZE``.
   num_gpus | 2 | GPUs assigned to Ray, training, and serving, in ``serve.yaml``, with ``cuda_visible_devices`` and ``--tensor-model-parallel-size`` beside it.
   TTTD_STATE_DIR | work/erdos_min_overlap | the durable state root, exported by ``run.sh`` as an absolute path because Ray workers and Git resolve a relative one from their own directories.

The reference grid contains ``8 x 64 = 512`` rollouts per optimizer step.
Training starts after all configured coordinates have arrived. Completion length and the
number of concurrent evaluator processes determine GPU memory, host memory, and
CPU use, so test the intended grid for one step before requesting a long run.

The packing configuration sets ``max_tokens_per_gpu`` to ``16384``, uses a
log-probability chunk size of ``512``, and caps evaluator concurrency at ``256``.
The Erdős configuration uses a ``30000`` token budget per GPU, a
log-probability chunk size of ``1024``, and at most ``512`` evaluator workers.

Formal circle-packing results
-----------------------------

Two formal runs completed 50 search and training steps with the same harness,
task instructions, and judges included in this example. Each step contained
eight groups of 64 programs. Each run used Qwen3-8B with thinking enabled on two
NVIDIA B200 GPUs. The optimizer used an Adam learning rate of ``4e-5``, and the
adapter used LoRA rank and alpha ``32``.

+------------+-----------------+--------------------------+----------------------+----------+--------------------+
| Task       | Completed steps | Certified sum of radii   | TTT-Discover reported| Target   | Gap to target      |
+============+=================+==========================+======================+==========+====================+
| Packing 26 | 50/50           | ``2.6359830849177777``   | ``2.635983``         | ``2.636``| ``0.000016915082`` |
+------------+-----------------+--------------------------+----------------------+----------+--------------------+
| Packing 32 | 50/50           | ``2.9395727712074926``   | ``2.939572``         | ``2.940``| ``0.000427228793`` |
+------------+-----------------+--------------------------+----------------------+----------+--------------------+

The two certified values match the values reported by TTT-Discover at six
decimal places. The targets come from the task instructions. They are not claims
about the global optima.

.. image:: ../../assets/tttd/packing-configurations.png
   :alt: Verified circle configurations for Packing 26 and Packing 32

The saved programs were executed again before the configurations were plotted.
The replay checked the circle count, finite values, whether every circle stayed
inside the square, and every pairwise distance with a tolerance of ``1e-12``.
Floating point summation changed the recorded values by less than ``1.2e-12``.

Best solution found so far
~~~~~~~~~~~~~~~~~~~~~~~~~~

TTT-Discover searches for the highest-scoring solution. The figure shows the
best-solution history for each formal run.

.. image:: ../../assets/tttd/best-solution-history.png
   :alt: Best verified circle-packing score found by search iteration

W&B metrics from the formal runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

W&B recorded 50 committed rows for each task. The static export merges resumed
run segments by the monotonic ``reef/step`` field and checks that steps 1 through
50 are present. ``rollout/rewards`` is the mean reward across one training grid.
The certified result in the table comes from the task judge and the best program
stored in the search archive.

.. image:: ../../assets/tttd/wandb-training-metrics.png
   :alt: W&B training metrics for the two formal circle-packing runs

The mean rollout reward reached its maximum at step 18 for Packing 26 and step
17 for Packing 32. The best-solution history shows that Packing 26 reached its
final certified result at iteration 13. Packing 32 reached
``2.9395727712072386`` at iteration 18 and its final result at iteration 24. The
last change was about ``2.5e-13``. Mean rollout rewards then decreased to
``1.1291`` and ``0.9264`` at step 50. Sampled-policy KL increased from about
``0.0006`` at step 20 to ``0.0491`` for Packing 26 and ``0.0430`` for Packing 32
at step 50. These metrics describe the sampled batches used for training. They
do not replace the judge's evaluation of the best archived program.

The result directory stores the final generated programs, step 20, 40, and 50
judge summaries, circle coordinates, best-solution history, W&B history,
hashes, job IDs, and scripts that regenerate the figures. See `formal 8x64 circle-packing results
<../../../recipes/tttd/examples/tttd/results/formal-8x64-v3-packing/README.md>`__ for
the files and exact provenance. Each problem currently has one formal
trajectory, so the results do not estimate variance across seeds.

Enable W&B tracking
-------------------

The example disables W&B in ``serve.yaml`` by default. Set
``observability.wandb.enabled`` to ``true`` before starting the stack, and provide a
project and an optional entity:

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team
       mode: online
       directory: work/wandb
       upload_checkpoints: false

Export ``WANDB_API_KEY`` or log in through the cluster's credential store. Do
not put the key in YAML, tags, run names, or Slime flags. Reef assigns one W&B
run identity to a scenario and resumes it after a service restart. Checkpoint
paths are logged as metadata when ``upload_checkpoints`` is false. `Experiment
tracking <../../reference/configuration.rst#experiment-tracking>`__ describes grouping,
rollback behavior, offline mode, and metric names.

When the processor receives a complete search grid, it records the following
metrics in the ``tttd`` namespace. These values describe every evaluated
rollout before constant-reward groups are removed from the training batch.

.. list-table:: TTT-Discover search metrics
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``tttd/step``
     - Zero-based search step reported by the harness.
   * - ``tttd/reward_min``, ``tttd/reward_max``, ``tttd/reward_mean``
     - Minimum, maximum, and mean reward over the complete grid.
   * - ``tttd/reward_std``
     - Population standard deviation of rewards over the complete grid.
   * - ``tttd/reward_zero_fraction``
     - Fraction of grid rewards equal to zero. In the bundled tasks, zero is
       also the configured reward for an invalid solution.
   * - ``tttd/constant_groups``, ``tttd/constant_groups_removed``
     - Number of constant-reward groups and the number removed before training.
       If every group is constant, the processor retains one group for the
       zero-gradient step.
   * - ``tttd/grid_groups``, ``tttd/grid_rollouts``
     - Size of the completed search grid.
   * - ``tttd/training_groups``, ``tttd/training_rollouts``
     - Size of the batch after constant-reward filtering.

The harness also includes ``archive_size`` and ``archive_best_reward`` in each
``tttd_step_committed`` log event. These fields show whether the archive is
still adding candidates and whether its best reward changes after the mean
grid reward stops improving.

Inspect checkpoints and resume state
------------------------------------

Each task stores its durable files under ``work/<task>/``:

.. code:: text

   work/<task>/
     reef.log
     tttd-search-state.json
     agent-record/
     artifacts.git/
     checkpoints/
     stack/
     wandb/

The search identity records the scenario, model, recipe, task-instruction hash,
grid dimensions, completion length, sampling settings, PUCT exploration
constant, and invalid reward. A restart with different identity fields stops
before it can use an incompatible archive.

A ``pending`` state contains the evaluated children for a step that the
controller has not observed as committed. On restart, the controller waits for
that transaction. A ``committed`` state records the scenario step and serving
runtime load ID that completed it. Serving runtime-load-ID strings belong to one
engine session, so a restored checkpoint can receive a new string after the
runtime starts. The controller accepts the new string after Reef restores the
expected scenario step.

The repository example creates a checkpoint every two versions and retains the
latest checkpoint under its storage policy. The formal circle-packing jobs used
an interval of one version so that each completed step could be recovered from
disk. Select the interval before the run. The retention policy controls which
completed checkpoints remain stored; it does not change the creation interval.
`Architecture <../../getting-started/architecture.rst>`__ describes the commit and recovery model.

Add another problem
-------------------

Create a sibling under ``recipes/tttd/examples/tttd/harbor/``. The task needs a
``task.toml``, an ``instruction.md``, a judge service, and a verifier that writes
the final Harbor reward. Point ``TASK_DIR`` and ``STATE_DIR`` in
``run.py`` and ``SCENARIO`` and ``SEARCH_STATE_PATH`` in
``harness/harbor_agent.py`` at it. Set the task's own sequence, evaluator, and
memory values in ``serve.yaml`` and ``harness/harbor_agent.py`` when the
existing ones do not apply.

The shared harness sends every extracted Python program to the selected judge.
The judge's ``grade(artifact)`` function must return a finite reward for a valid
program and the defined invalid reward for a failed program. Keep the program
contract in the instruction consistent with the judge, and start a new scenario
when that contract changes.

Add tests for the program contract, invalid outputs, reward calculation, and
task registration. ``tests/test_tttd_harness.py`` and
``tests/test_tttd_packing_tasks.py`` show how the shared harness hands programs
to a task-specific judge.
