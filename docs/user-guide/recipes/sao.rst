sao
===

Single-Rollout Asynchronous Optimization (`arXiv:2607.07508
<https://arxiv.org/abs/2607.07508>`__) trains on one graded rollout at a
time. Each rollout is one training step, and the next attempt runs on the
weights it produced.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report with a finite ``score`` per rollout             |
+-------------+------------------------------------------------------------+
| Loss family | ``sao``                                                    |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/sao/``                                           |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/sao/examples/sao/``                              |
+-------------+------------------------------------------------------------+

What it does
------------

SAO has no comparison group. A rollout enters training as soon as its score
arrives without waiting for siblings and without a barrier. It can be used for a
stream of tasks where each attempt gets its own score.

.. flow::
   :loop: the next attempt runs on the updated weights

   Rollout :: one attempt at a task
   Feedback :: a score reported against the rollout's receipt
   Step* :: train on that rollout immediately
   Version :: publish the updated weights to the engine

How Reef implements it
----------------------

The processor turns every eligible ``ScoredRolloutReport`` into one
``PolicySample``. With the default ``batch_size`` of 1, each sample is its
own training step. The ``sao`` loss family runs Slime's ``policy_loss`` with
SAO's per-token primitive and a critic colocated on the actor GPUs. The
critic supplies the values, and skip-observation GAE builds the advantages
inside the training backend.

The DIS ratio compares the current policy against the log-probabilities
recorded when the rollout was generated. SAO therefore requires an inference
backend that attaches engine-native tensors.

Configuration
-------------

.. config::

   batch_size | 1 | rollouts per optimizer step. Must equal the driver's ``--global-batch-size`` because each sample is its own data-parallel unit.
   max_staleness | 0 | accepted lag between the producing and serving version.

Run the example
---------------

The `example <../../../recipes/sao/examples/sao>`__ runs three IMOAnswerBench
problems in order on a two-GPU stack. For each problem the agent makes six
attempts through Reef, extracts the ``\boxed{}`` answer, then checks it against
the gold answer for a binary reward and finally reports the result against its
receipt. The next problem is served by the weights the previous one produced.

.. code:: bash

   cd recipes/sao/examples/sao
   pip install -e . "reef-eval[harbor]"
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
   ./run.sh

``run.sh`` starts the stack that ``serve.yaml`` describes: one Megatron actor
with the critic colocated on it and one SGLang rollout engine. Each scored
rollout adds one ``training`` entry to the scenario's version chain:

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/sao-smoke/releases
   # {"scenario": "sao-smoke", "releases": [{"operation": "training", "current": true, ...}, ...]}

The runtime reports ``pg_clipfrac``, ``critic/explained_variance``, actor and
critic ``grad_norm``, and the asynchrony metrics ``sao/policy_lag_*``,
``sao/queue_age_s_*``, and ``sao/effective_token_rate``.

Results
-------

The example's README records two runs.

The comparison on Qwen3-30B-A3B trains SAO and a GRPO(+DIS) control from the
same checkpoint on the same three problems with 48 scored rollouts per arm.
The mean rewards order as the paper predicts, SAO at 0.479 above the
untrained base at 0.458 above GRPO(+DIS) at 0.417.

.. image:: ../../assets/sao/learning-curve.png
   :alt: Cumulative mean reward over the 48 scored rollouts per arm
