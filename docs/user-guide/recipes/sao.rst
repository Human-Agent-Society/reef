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
Docker, and the Slime training environment:

.. code:: bash

   cd recipes/sao/examples/imo_answerbench
   uv pip install -e .
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
   ./run.sh

It runs three problems with six attempts each. The actor and critic share
one GPU; serving uses the other. Batch size 1 checks the workflow.
Each scored rollout adds a training entry to the artifact history.

Training requirements
---------------------

Report a finite ``score`` per rollout. Match
``pipeline.sample2job.config.job_samples`` to the optimizer batch size for
one update per job.

``pipeline.record2sample.config.max_artifact_lag`` defaults to 0: only the
served artifact trains. Larger values permit older rollouts. The DIS ratio
then measures drift within the job, not from the policy that generated them.

The README records a batch-128 comparison without tools; it does not
reproduce the paper's tool setting. For long conversations, use the
`CEO-Bench example <../../../recipes/sao/examples/ceobench/README.md>`__, which
assigns weekly credit to decision turns.
