Guidance-TTT: train the guide, keep the executor frozen
=======================================================

Train a small guidance model while a frozen executor writes candidate
programs. Only guidance tokens train. Use an existing executor while
updating a smaller model.

Run an example
--------------

Use Linux, Docker, and the `Reef GPU environment
<../../getting-started/installation.rst>`__. Allocate four GPUs for guidance
and separately serve a GPT-OSS-120B executor. From the repository root:

.. code:: shell

   uv pip install -e . -e recipes/tttd/examples/guidance_ttt
   cd recipes/tttd/examples/guidance_ttt
   export GUIDANCE_TASK=lasso_path
   python prepare.py "$GUIDANCE_TASK"

Other tasks are ``polyomino_packing``, ``ahc058``, and ``trimul``.
Start the selected `judge
<../../../recipes/tttd/examples/guidance_ttt/judges/README.md>`__, then run:

.. code:: shell

   export GUIDANCE_EXECUTOR=local
   export GUIDANCE_EXECUTOR_URL=http://127.0.0.1:8000/v1
   ./run.sh

For GLM-5.2, set ``GUIDANCE_EXECUTOR=openrouter`` and supply
``OPENROUTER_API_KEY``. The default downloads Qwen3-8B and commits one update
with two groups of four rollouts. It verifies the bootstrap and best program;
the final Harbor reward is evaluation-only.

Training requirements
---------------------

The guide receives a parent summary; the executor receives parent code.
Guidance must end with one nonempty ``<guidance>`` block.
Malformed guidance earns zero reward without an executor call.
Missing judges, malformed scores, and executor errors stop the step.

Keep ``job_samples`` and ``global-batch-size`` equal to groups times rollouts
in ``serve.yaml``. Reports follow the
`TTTD grid contract <tttd.rst#the-report-contract>`__.
Training rewards and archive scores can differ; track the archive's raw
metric for the best program.

The `example README <../../../recipes/tttd/examples/guidance_ttt/README.md>`__
covers judge addresses, configuration, and restart requirements.
The `paper results
<../../../recipes/tttd/examples/guidance_ttt/results/paper/README.md>`__ use
a larger historical budget and task-specific evaluations. The smoke run
does not reproduce those results.
