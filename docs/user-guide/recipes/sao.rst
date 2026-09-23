SAO: learn from individual rollouts
===================================

Single-Rollout Asynchronous Optimization (`arXiv:2607.07508
<https://arxiv.org/abs/2607.07508>`__) samples one rollout per prompt and
grades each on its own. A scored rollout joins the next optimizer step
without waiting for siblings, and the next attempt runs on the weights that
step produced.

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
| Example     | ``recipes/sao/examples/imo_answerbench/``                  |
+-------------+------------------------------------------------------------+

What it does
------------

SAO has no comparison group. A rollout is accepted as soon as its score
arrives, without waiting for siblings and without a barrier, and ``batch_size``
accepted rollouts from as many prompts form one optimizer step. It can be used
for a stream of tasks where each attempt gets its own score.

.. flow::
   :loop: the next attempt runs on the updated weights

   Rollout :: one attempt at a task
   Feedback :: a score reported against the rollout's receipt
   Step* :: train once batch_size scored rollouts have accumulated
   Version :: publish the updated weights to the engine

How Reef implements it
----------------------

The processor turns every eligible ``ScoredRolloutReport`` into one
``TrajectoryItem``. "Single rollout" means one rollout per prompt, with no
comparison group; ``batch_size`` such rollouts, from different prompts, form
one optimizer step. The default is the paper's 128. Setting it to 1 makes
every sample its own training step, which is convenient for a smoke run but
is not the paper's estimator: the value model needs a full batch per step to
learn, and without it the single-sample advantages are noise. The ``sao``
loss family runs Slime's ``policy_loss`` with
SAO's per-token primitive and a critic colocated on the actor GPUs. The
critic supplies the values, and skip-observation GAE builds the advantages
inside the training backend.

The DIS ratio compares the current policy against the log-probabilities
recorded when the rollout was generated. SAO therefore requires an inference
backend that attaches engine-native tensors.

The value model carries the paper's cold-start mitigations: it trains at its
own, higher learning rate (``--critic-lr``) and the first
``--num-critic-only-steps`` optimizer steps fit the zero-initialized value head
before any policy update. Warmup steps still commit one training release per
step; the policy's weights first move after the warmup.

Configuration
-------------

.. config::

   batch_size | 128 | rollouts (one per prompt) per optimizer step; the paper's value. Must equal the driver's ``--global-batch-size`` because each sample is its own data-parallel unit. 1 trains on every rollout as it lands and is a smoke setting only.
   max_staleness | 0 | accepted lag between the producing and serving version.

GRPO(+DIS) control
~~~~~~~~~~~~~~~~~~

``recipes.sao.control_recipe:SAOGrpoControlRecipe`` is the paper's baseline
for the comparison: the same DIS primitive and rollout columns, with Slime's
group-relative advantages in place of the value model (loss family
``sao-grpo-dis``). The driver posts each prompt's ``--n-samples-per-prompt``
rollouts together, so a step of ``batch_size`` rollouts holds complete groups.
The example's ``serve-30b-grpo*.yaml`` stacks select it.

Run the example
---------------

The `example <../../../recipes/sao/examples/imo_answerbench>`__ runs three IMOAnswerBench
problems in order on a two-GPU stack. For each problem the agent makes six
attempts through Reef, extracts the ``\boxed{}`` answer, then checks it against
the gold answer for a binary reward and finally reports the result against its
receipt. The next problem is served by the weights the previous one produced.

.. code:: bash

   cd recipes/sao/examples/imo_answerbench
   pip install -e . "reef-eval[harbor]"
   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
   ./run.sh

``run.sh`` starts the stack that ``serve.yaml`` describes: one Megatron actor
with the critic colocated on it and one SGLang rollout engine. The smoke
config sets ``batch_size`` to 1, so each scored rollout adds one ``training``
entry to the scenario's version chain:

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     http://127.0.0.1:8900/reef/scenarios/sao-smoke/releases
   # {"scenario": "sao-smoke", "releases": [{"operation": "training", "current": true, ...}, ...]}

The runtime reports ``pg_clipfrac``, ``critic/explained_variance``, actor and
critic ``grad_norm``, and the asynchrony metrics ``sao/policy_lag_*``,
``sao/queue_age_s_*``, and ``sao/effective_token_rate``.

CEO-Bench
~~~~~~~~~

The `CEO-Bench example <../../../recipes/sao/examples/ceobench>`__ trains the
same recipe on `CEO-Bench <https://ceobench.com>`__, a 500-day simulated
startup. The harness is the benchmark's own bash agent played from the host,
its prompt, tools, and tool executor taken from the pinned checkout in the
task image and its model calls served by Reef; the two simulator roles stay
outside Reef, and the verifier reads final cash, survival days, and bankruptcy
from the run's ``world.nmdb``. The reward is weekly and online: when the
next week's dashboard appears, the finished week's decision turns (the tool
calls that changed the company) are reported with the week's credit, its
change in company value (cash plus the engine's subscription run-rate over
the weeks left) and the discounted changes of the weeks after it, scaled
against the weeks before, so the recipe trains while the episode runs. The
example's README records the reward-shaping choices and the recorded episodes.

.. code:: bash

   cd recipes/sao/examples/ceobench
   hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
   export ANTHROPIC_API_KEY=...        # the simulator roles' provider
   CEOBENCH_DAYS=500 CEOBENCH_SEED=42 ./run.sh

Results
-------

The example's README records the batch-128 comparison on
Qwen3-30B-A3B-Thinking-2507: SAO and a GRPO(+DIS) control trained from the
same public checkpoint on a DeepMath pool, without tools, and evaluated on
held-out AIME 2025, HMMT February 2025 and IMO-AnswerBench. SAO trained
stably for 99 steps and gained 2 to 4 points on AIME and IMO-AnswerBench,
within per-checkpoint intervals. GRPO(+DIS) matched it through step 40, then
shortened its responses and fell 5 to 13 points below SAO by step 80 and to
a fraction of the base rate by step 140. The README lists the numbers and the
distance from the paper's tool-integrated setting.

.. image:: ../../assets/sao/learning-curve.png
   :alt: Held-out accuracy and training dynamics of SAO and GRPO(+DIS) against optimizer step

Related guides
--------------

- `Inference and feedback quickstart <../../getting-started/quickstart.rst>`__:
  learn the request, receipt, and report workflow.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `HTTP API reference <../../reference/http-api.rst>`__: connect your agent
  and query feedback, scenarios, and releases.
