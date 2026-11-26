SAO: learn from individual rollouts
===================================

Use Single-Rollout Asynchronous Optimization
(`SAO <https://arxiv.org/abs/2607.07508>`__) for tasks with an independent
score per attempt. It trains model weights from a stream of rollouts,
without requiring comparison groups. Reef's implementation uses Slime and GPUs.

Run the example
---------------

Use the `IMOAnswerBench example
<../../../recipes/sao/examples/imo_answerbench/README.md>`__ with two GPUs,
Docker, and the Slime training environment. It runs three problems with six
attempts each and reports a binary answer score.

.. code:: bash

   cd recipes/sao/examples/imo_answerbench
   uv pip install -e .
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
   ./run.sh

The actor and critic share one GPU; SGLang uses the other.
This smoke configuration uses batch size 1. Each scored rollout adds a
training entry to the scenario's artifact history:

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/sao-smoke/artifacts
   # {"scenario": "sao-smoke", "artifacts": [{"operation": "training", "active": true, ...}, ...]}

Inspect ``pg_clipfrac``, ``critic/explained_variance``, and actor/critic
``grad_norm``. The ``sao/*`` metrics report producing artifacts, queue age,
and effective token rate.

CEO-Bench
~~~~~~~~~

The `CEO-Bench example <../../../recipes/sao/examples/ceobench/README.md>`__
trains during a 500-day simulated startup. When the next dashboard arrives,
the harness reports the previous week's decision turns with that week's
credit. Its README defines the reward and records the runs.

.. code:: bash

   cd recipes/sao/examples/ceobench
   hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
   export ANTHROPIC_API_KEY=...        # the simulator roles' provider
   CEOBENCH_DAYS=500 CEOBENCH_SEED=42 ./run.sh

How it works
------------

Each report supplies a finite ``score`` for one rollout. Reef starts a job
when ``job_samples`` eligible reports are available; it does not wait for
sibling attempts. The ``sao`` loss uses a critic and token-level advantages.

The critic shares the actor's GPUs and has its own ``critic-lr``. The first
``num-critic-only-steps`` fit its value head without updating the policy.
These steps still commit snapshots. An MoE critic freezes self-attention;
a dense critic does so only with ``sao-freeze-critic-attention``.

The actor recomputes old-policy log probabilities from the job's starting
weights. The DIS ratio is exact when ``max_artifact_lag`` is 0.
With a larger lag, it measures drift within the job, not from the weights
that sampled the rollout. Match training and serving
`chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.

Configuration
-------------

Set the permitted artifact lag and match the job size to the optimizer
batch size:

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

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : required
       Rollouts per job. Match ``training.config.global_batch_size`` for one
       optimizer step per job.

   ``training.config.global_batch_size``
       Rollouts per optimizer step; the paper uses 128. Reef passes it as
       Slime's ``--global-batch-size``. A value of 1 is for smoke tests, not
       a comparable learning setup.

   ``pipeline.record2sample.config.accept_multi_turn_policy_samples`` : false
       ``true`` trains every reply in the reported trajectory. ``false``
       trains its last reply, with earlier turns as context.

   ``pipeline.record2sample.config.max_artifact_lag`` : 0
       Maximum number of subsequently served artifacts. Must be nonnegative.
       Reef marks older rollouts stale. 0 accepts only the served artifact.

A ``batch-size`` key in ``recipe.config`` or in
``pipeline.sample2job.config`` is an unknown key: the recipe or the scenario's
pipeline refuses it.

Results
-------

The IMOAnswerBench example's README records a batch-128 comparison on
Qwen3-30B-A3B-Thinking-2507. SAO and a GRPO(+DIS) control trained from the
same public checkpoint on a DeepMath pool, without tools. Both were evaluated
on held-out AIME 2025, HMMT February 2025 and IMO-AnswerBench. The control is
a reported baseline, not a shipped recipe.

- SAO trained stably for 99 steps and gained 2 to 4 points on AIME and
  IMO-AnswerBench, within per-checkpoint intervals.
- GRPO(+DIS) matched it through step 40, then shortened its responses. It
  fell 5 to 13 points below SAO by step 80 and to a fraction of the base rate
  by step 140.

The README lists the numbers and the distance from the paper's
tool-integrated setting.

.. image:: ../../assets/sao/learning-curve.png
   :alt: Held-out accuracy and training dynamics of SAO and GRPO(+DIS) against optimizer step

Related guides
--------------

- `Inference and feedback quickstart <../../getting-started/quickstart.rst>`__:
  learn the request, receipt, and report workflow.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `HTTP API reference <../../reference/http-api.rst>`__: connect your agent
  and query feedback, scenarios, and artifacts.
