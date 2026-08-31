sao
===

Single-Rollout Asynchronous Optimization (`arXiv:2607.07508
<https://arxiv.org/abs/2607.07508>`__). One graded rollout is one training step.

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

SAO drops the sibling group. There is no comparison baseline and no barrier, so
a rollout enters training the moment its feedback arrives.

.. flow::

   Rollout :: one attempt at a task
   Feedback :: a report that refers to the receipt
   Step* :: train immediately on that rollout
   Version :: updated weights serve the next task

How Reef implements it
----------------------

The processor accepts every eligible ``ScoredRolloutReport`` and emits one
``PolicySample`` per report. The ``sao`` step preparer turns the batch into a
``StepSignal`` on the ``sao`` loss family, which runs Slime's ``policy_loss``
with a custom per-token primitive and a colocated critic.

The DIS ratio needs generation-time log-probabilities as its behaviour proxy, so
SAO requires an inference backend that attaches engine-native tensors.

Configuration
-------------

.. config::

   batch_size | 1 | rollouts per optimizer step. Must equal the driver's ``--global-batch-size``: each sample is its own data-parallel unit. Env ``REEF_SAO_BATCH_SIZE``.
   max_staleness | 0 | accepted lag between the producing and serving version. Env ``REEF_MAX_STALENESS``.

An ``optimization:`` section is rejected outright. Clipping bounds, critic
cadence, and GAE parameters belong to the backend, in ``training.slime_flags``.

Run the example
---------------

.. code:: bash

   cd recipes/sao/examples/sao
   pip install -e . "reef-eval[harbor]"
   ./run.sh

The example runs three IMOAnswerBench problems in order. Each drives six
independent attempts through Reef, extracts the ``\boxed{}`` answer, checks it
against gold for a binary reward, and reports the result against its receipt.

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/sao-smoke/versions
   # {"scenario": "sao-smoke", "versions": [{"kind": "training", "step": 1, ...}, ...]}

Each scored rollout adds one ``training`` entry to the chain. The runtime
surfaces ``pg_clipfrac``, ``critic/explained_variance``, actor and critic
``grad_norm``, and the async telemetry ``sao/policy_lag_*``,
``sao/queue_age_s_*``, and ``sao/effective_token_rate``.

Results
-------

`recipes/sao/examples/sao/README.md
<../../../recipes/sao/examples/sao/README.md>`__ records two runs.

The budget-limited comparison on Qwen3-30B-A3B trains SAO and a GRPO(+DIS)
control from the same checkpoint, on the same three problems, with 48 scored
rollouts per arm. Mean reward orders as the paper predicts at this budget:
SAO 0.479, base 0.458, GRPO(+DIS) 0.417. Read it as an ordering check, not a
convergence result; the README lists every protocol deviation from the paper.

.. image:: ../../assets/sao/learning-curve.png
   :alt: Cumulative mean reward over the 48 scored rollouts per arm

The committed smoke configuration has one recorded run of its own
(``results/smoke-2026-08-30/``): all 18 scored rollouts became committed
training steps with no stale drops, which is the acceptance criterion for the
smoke. Every score is 0.0 because Qwen2.5-1.5B-Instruct does not solve these
problems. The useful output is the per-step training metrics, exported from an
offline W&B run.
