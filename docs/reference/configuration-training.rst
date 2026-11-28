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
- With ``colocate``, ``offload``, ``offload-train`` or ``advantage-estimator: ppo``, the workers keep no CPU copy of their buffers while they sleep, so Reef refuses expandable segments (``expandable_segments:True`` in ``PYTORCH_CUDA_ALLOC_CONF`` or ``PYTORCH_ALLOC_CONF``, or ``SLIME_ENABLE_EXPANDABLE_SEGMENTS=1``): their optimizer step can then write zeros into the weights. The allocator check permits ``use-critic`` alone. Native task and dataset jobs also check the critic options below.
- For multi-node training, use a checkpoint directory readable by the training host. See `checkpoint cadence <../user-guide/operate.rst#set-the-checkpoint-cadence>`__.

Train without an inference engine
---------------------------------

Set ``inference.backend: none`` for supervised fine-tuning on uploaded trajectories or imported records.
Use the `trajectory upload <http-api-data-upload.rst#create-trajectory-items>`__ or import records with ``POST /reef/records``.
The trainer uses the GPUs named by ``training.options.actor-*``. Inference routes return 404.

Use ``inference.model-path`` for the starting weights. Omit inference engine settings, teacher, generator and composite recipes.

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
- Use the ``slime`` loss family for the deployment and objective. It accepts task and dataset jobs. Both require a critic-free estimator such as ``grpo``. They refuse ``ppo`` and ``use-critic``.
- Each job runs ``ceil(tasks / rollout-batch-size)`` rollout iterations. The last batch repeats tasks to fill it. ``save_interval`` counts iterations.
- Engines accept system, user and assistant text only. Slime controls temperature, top-p, top-k and maximum response length.
- If an iteration has no trainable group, the job fails with its first rollout error. Score centering records the top ``score-centering-top-k`` log probabilities per sampled token.
- Install Harbor and the selected environment on the Ray actor's host. Set ``agent_host`` for sandboxed agents.
  The actor inherits E2B credentials, API and sandbox URLs, domain, and Modal credentials from the training service.

Set optimizer and learning-rate options in Slime. Use at least two samples per prompt with GRPO.
Leave task generation and scoring to Reef. Custom data sources and generation functions conflict with Harbor task jobs.

Dataset jobs on Slime
---------------------

Select ``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training-mode: manual``.
Send ``POST /reef/train`` with a Slime prompt dataset:

.. code:: json

   {"text": "GRPO on the prompts", "session": "s", "artifact_id": "artifact-1",
    "parameters": {"dataset": {"path": "/data/prompts.jsonl", "epochs": 1}, "save_interval": 10}}

Use an absolute ``dataset.path`` on the Ray cluster. Each epoch runs ``len(dataset) // rollout-batch-size`` rollout iterations.
``save_interval`` counts rollout iterations. Omit it to save only at job end.
Set Slime's dataset and reward plugin options under ``training.options``. Reef sets ``data-source-path`` from the request.
