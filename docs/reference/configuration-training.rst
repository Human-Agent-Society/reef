Slime training options and jobs
===============================

For backend selection and GPU sizing, see `Evolve your model <../user-guide/evolve-your-model.rst>`__.

Slime's training options
------------------------

Set native flags under ``training.options``, without the leading ``--``.
Malformed options fail startup. Worker-specific errors fail the first job.

- Set ``global-batch-size`` explicitly. It counts samples per optimizer step.
- ``hf-checkpoint`` must match ``inference.model-path``. Explicit architecture flags override the model's Hugging Face config.
- ``actor-num-nodes`` and ``actor-num-gpus-per-node`` allocate training GPUs. ``num-gpus-per-node`` gives each node's physical GPU count.
- ``rollout-num-gpus``, ``colocate``, ``sglang-*`` and ``router-*`` configure Slime's rollout engines. Configure serving engines under ``inference.options``.
- For multi-node training, use a checkpoint directory readable by the training host. See `checkpoint cadence <../user-guide/operate.rst#set-the-checkpoint-cadence>`__.

Train without an inference engine
---------------------------------

Set ``inference.backend: none`` for supervised fine-tuning on uploaded trajectories or imported records.
Use the `trajectory upload <http-api-data-upload.rst#create-trajectory-items>`__ or import records with ``POST /reef/records``.
The trainer uses the GPUs named by ``training.options.actor-*``. Inference routes return 404.

.. code:: yaml

   recipe:
     implementation: reef.recipe.sft:SFTRecipe
   inference:
     model-path: /models/Qwen3-8B   # the weights to train; nothing serves them
     backend: none
   training:
     backend: slime
     options:
       actor-num-nodes: 1
       actor-num-gpus-per-node: 8
       global-batch-size: 32
   pipeline:
     sample2job:
       config:
         job-samples: 32

With this backend, ``reef serve`` rejects:

- ``inference.serving`` other than ``none`` or ``inference.num-gpus`` other than 0;
- ``inference.tensor-parallel-size``, ``inference.options``, ``inference.url`` or ``inference.remote``;
- ``inference.handler-factory`` or ``inference.handler-config``;
- ``teacher.model-path``, a ``generator`` section or a composite recipe.

Task jobs on Slime
------------------

Slime runs each Harbor task ``n-samples-per-prompt`` times on its rollout engines.
The verifier rewards form a group for ``advantage-estimator`` and ``policy_loss``.
Use these training and recipe options:

.. code:: yaml

   training:
     backend: slime
     options:
       colocate: true
       rollout-num-gpus-per-engine: 1
       advantage-estimator: grpo
       loss-type: policy_loss
       rollout-batch-size: 4
       n-samples-per-prompt: 4
       global-batch-size: 16
       rollout-max-response-len: 2048
   recipe:
     config:
       job-parameters:
         objective: slime_dataset
         save_interval: 10
         harbor:
           environment: modal
           concurrency: 8

- Set ``rollout-num-gpus`` for separate GPUs or ``colocate`` to share training GPUs. LoRA actors cannot use rollout engines.
- Use the ``slime`` loss family for the deployment and objective. It accepts task and dataset jobs.
- Each job runs ``ceil(tasks / rollout-batch-size)`` rollout iterations. The last batch repeats tasks to fill it. ``save_interval`` counts iterations.
- Engines accept system, user and assistant text only. Slime controls temperature, top-p, top-k and maximum response length.
- If an iteration has no trainable group, the job fails with its first rollout error. Score centering records the top ``score-centering-top-k`` log probabilities per sampled token.
- Install Harbor and the selected environment on the Ray actor's host. Set ``agent_host`` for sandboxed agents.
  The actor inherits E2B credentials, API and sandbox URLs, domain, and Modal credentials from the training service.

Reef rejects task jobs when:

- The request sets ``learning_rate_schedule`` or ``scheduling``. Set optimizer and learning-rate options in Slime.
- An option replaces task generation or scoring. This includes ``group-rm``, ``reward-key``, ``custom-generate-function-path``, ``data-source-path`` or a different ``rollout-function-path``.
- Sampling uses ``partial-rollout``, ``dynamic-sampling-filter-path`` or ``use-rollout-routing-replay``.
- ``over-sampling-batch-size`` differs from ``rollout-batch-size``, or ``rollout-top-p`` differs from 1.
- ``n-samples-per-prompt`` is 1 with an ``advantage-estimator`` that centers each reward on its group, such as ``grpo``.

Dataset jobs on Slime
---------------------

Select ``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training-mode: manual``.
Send ``POST /reef/train`` with a Slime prompt dataset:

.. code:: json

   {"text": "GRPO on the prompts", "session": "s", "artifact_id": "artifact-1",
    "parameters": {"dataset": {"path": "/data/prompts.jsonl", "epochs": 1}, "save_interval": 10}}

.. container:: config

   ``dataset.path``
       Absolute path on the Ray cluster to a Slime prompt-data JSONL file.

   ``dataset.epochs`` : 1
       Passes over the dataset. Each epoch trains ``len(dataset) // rollout-batch-size`` rollouts.

   ``save_interval``
       Rollouts between checkpoints. The default saves at job end only.

The recipe can set a default ``dataset`` in ``job-parameters``. Configure Slime's input, rollout, reward and postprocessing plugins in ``training.options``.
Reef sets ``data-source-path`` and rejects ``disable-compute-advantages-and-returns``.
