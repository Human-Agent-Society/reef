Guidance-TTT: train the guide, keep the executor frozen
=======================================================

Train a small guidance model while a frozen executor writes candidate
programs. Only guidance tokens train. The loop uses
`TTT-Discover <tttd.rst>`__ with task-specific judges.

Choose a task
-------------

.. list-table::
   :header-rows: 1

   * - ``GUIDANCE_TASK``
     - Candidate
     - Metric
   * - ``polyomino_packing``
     - C++17 program
     - Packing score, higher is better
   * - ``lasso_path``
     - Python wrapper with C++17 ``CPP_CODE``
     - Inverse solve time, higher is better
   * - ``ahc058``
     - C++20 program
     - Mean raw score, higher is better
   * - ``trimul``
     - Python module with Triton kernels
     - H100 latency, lower is better

The `example README <../../../recipes/tttd/examples/guidance_ttt/README.md>`__
links each task's evaluator and test cases.

Run an example
--------------

Use Linux, Docker, and the `Reef GPU environment
<../../getting-started/installation.rst>`__. Allocate four GPUs: two train
and two serve guidance. Start from the repository root:

.. code:: shell

   uv pip install -e . -e recipes/tttd/examples/guidance_ttt
   cd recipes/tttd/examples/guidance_ttt
   export GUIDANCE_TASK=lasso_path
   python prepare.py "$GUIDANCE_TASK"

Start the selected `judge
<../../../recipes/tttd/examples/guidance_ttt/judges/README.md>`__.
Then select a separately served GPT-OSS-120B executor and run:

.. code:: shell

   export GUIDANCE_EXECUTOR=local
   export GUIDANCE_EXECUTOR_URL=http://127.0.0.1:8000/v1
   ./run.sh

Alternatively, set ``GUIDANCE_EXECUTOR=openrouter`` and provide
``OPENROUTER_API_KEY`` to use GLM-5.2. ``GUIDANCE_EXECUTOR_MAX_TOKENS=16384``
bounds smoke-run responses; otherwise the provider chooses the limit.

The default downloads Qwen3-8B and commits one update with two groups of
four rollouts. It verifies the bootstrap and best program.
The final Harbor reward is evaluation-only. This is a smoke run, not the
paper's configuration.

How one step runs
-----------------

The guide sees the problem, parent summary, and score.
The executor sees parent code and guidance. Each guidance response must end
with one nonempty ``<guidance>`` block. Malformed guidance receives zero
reward without resampling or an executor call.

The executor returns a complete program and replacement summary.
Valid programs enter the archive by raw metric; training uses a nonnegative
reward. TriMul uses ``1500 / latency_us``.

Configure a search
------------------

Edit ``serve.yaml`` or select a file with ``GUIDANCE_CONFIG``.
Keep ``pipeline.sample2job.config.job_samples`` and
``training.options.global-batch-size`` equal to groups times rollouts.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
   * - ``pipeline.record2sample.config.groups_per_step`` / ``rollouts_per_group``
     - 2 / 4
   * - ``training.config.steps`` / ``max_tokens``
     - 1 update / 8192 guidance tokens
   * - ``GUIDANCE_MAX_WORKERS``
     - 8 executor calls
   * - ``GUIDANCE_MODEL``
     - ``Qwen/Qwen3-8B``
   * - ``GUIDANCE_STATE_DIR``
     - ``work/<task>``

``run.sh`` selects GPUs ``0,1,2,3`` unless ``CUDA_VISIBLE_DEVICES`` is set.
For remote judges, set both ``GUIDANCE_JUDGE_URL`` for the host and
``GUIDANCE_JUDGE_CONTAINER_URL`` for Docker. Defaults use localhost and
``host.docker.internal`` respectively: port 8081 for Polyomino, 8082 otherwise.

For a custom bootstrap, set ``GUIDANCE_SEED`` to its source file and provide
a same-stem ``.md`` summary. Search and bootstrap use the same evaluator.

The report contract
-------------------

Reports reference guidance receipts and carry the
`TTTD grid metadata <tttd.rst#the-report-contract>`__.
Judges return training ``score``, archive ``scoreUnbounded``, and ``valid``.
Invalid AHC058 programs may earn partial reward but never enter the archive.
Rejected candidates count as PUCT visits.

Missing judges, malformed scores, and executor service errors stop the step.
They do not become zero-reward examples.

Paper results
-------------

The `paper results and solutions
<../../../recipes/tttd/examples/guidance_ttt/results/paper/README.md>`__ use
commit ``709c405``, GLM-5.2, and 30 updates of eight groups of 16 rollouts.
These are historical measurements, with task-specific final evaluations.
Search scores and final scores can differ; the baselines are not controlled
comparisons. The smoke run does not reproduce those results.

Enable experiment tracking
--------------------------

`W&B tracking <../operate.rst#track-experiments-with-w-b>`__ reports training
rewards; use the archive's raw score for the best program.

Inspect progress and resume
---------------------------

Read ``reef.log`` and ``guidance-run/`` under the state directory.

Restart with the same task, model, executor, grid, configuration, and state
directory. Reef checks ``committed-library.json`` against that identity.
Retain both the archive and training checkpoints; a program cannot restore
the optimizer. Use a new directory after changing the task or executor.

Add another task
----------------

See the `example files and checks
<../../../recipes/tttd/examples/guidance_ttt/README.md>`__ for evaluator setup
and task extension.

Related guides
--------------

- `TTT-Discover <tttd.rst>`__: objective and grid constraints.
