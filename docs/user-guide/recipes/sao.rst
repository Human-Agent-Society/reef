SAO: learn from individual rollouts
===================================

Use Single-Rollout Asynchronous Optimization
(`SAO <https://arxiv.org/abs/2607.07508>`__) when each attempt has its own
score. It trains from a rollout stream without comparison groups.
Reef's implementation requires Slime and GPUs.

Run the example
---------------

The `IMOAnswerBench example
<../../../recipes/sao/examples/imo_answerbench/README.md>`__ needs two GPUs,
Docker, and the Slime training environment. It runs three problems with six
attempts each:

.. code:: bash

   cd recipes/sao/examples/imo_answerbench
   uv pip install -e .
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
   ./run.sh

The actor and critic share one GPU; SGLang uses the other.
Each scored rollout adds a training entry to the artifact history.
Batch size 1 tests the workflow; it is not a comparable learning setup.

CEO-Bench
~~~~~~~~~

Use the `CEO-Bench example <../../../recipes/sao/examples/ceobench/README.md>`__
for long conversations. It assigns weekly credit to decision turns during
a 500-day simulated startup. Its README supplies setup, rewards, and results.

How it works
------------

Report a finite ``score`` for each rollout. Training starts when
``job_samples`` eligible reports arrive. A critic supplies token-level
advantages. Its first ``num-critic-only-steps`` update only the critic,
but still commit snapshots.

The DIS ratio is exact at ``max_artifact_lag: 0``. With larger lag, it
measures drift within the job, not from the policy that generated the rollout.
Match serving and training `chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.

Configuration
-------------

Set artifact lag and job size under ``pipeline``:

.. code:: yaml

   recipe:
     implementation: recipes.sao.recipe:SAORecipe
   pipeline:
     record2sample:
       config:
         max_artifact_lag: 18
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     config:
       global_batch_size: 8

- ``max_artifact_lag`` defaults to 0 and must be nonnegative.
  Older rollouts become stale; 0 accepts only the served artifact.
- ``accept_multi_turn_policy_samples`` defaults to false under
  ``record2sample.config``. Set true to train every trajectory reply;
  otherwise only the last reply trains.
- Required ``job_samples`` counts rollouts. Match the optimizer batch size
  for one update per job. ``batch-size`` is not a valid stage or recipe key.
- The critic uses ``critic-lr``. An MoE critic freezes self-attention;
  a dense critic needs ``sao-freeze-critic-attention`` to do so.

Results
-------

The `batch-128 comparison
<../../../recipes/sao/examples/imo_answerbench/README.md>`__ records stable
SAO training and a later decline in its GRPO(+DIS) control.
It uses Qwen3-30B-A3B-Thinking-2507 without tools, unlike the paper's setting.
The control is a reported baseline, not a shipped recipe.

Related guides
--------------

- `Train model weights <../evolve-your-model.rst>`__: configure the GPU stack.
- `Inference and feedback <../../getting-started/quickstart.rst>`__: send reports.
