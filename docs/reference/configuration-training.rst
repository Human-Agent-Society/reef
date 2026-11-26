Slime training options and jobs
===============================

How the Slime backend takes its options, how to train without an inference engine, and the jobs that Slime runs on its own rollout engines. `Settings reference <configuration-settings.rst#training>`__ lists every key of the ``training`` and ``teacher`` sections. `Train with Tinker <../user-guide/tinker.rst>`__ describes the Tinker backends, and `Evolve your model <../user-guide/evolve-your-model.rst>`__ describes how to size a stack.

Slime's training options
------------------------

``training.options`` holds Slime's native flags without the leading ``--`` (`Native backend options <configuration-inference.rst#native-backend-options>`__). The training component parses them once at startup, so a malformed option fails before the first job. A configuration error that only Slime's workers detect fails the first job.

Reef reads these options itself:

.. container:: config

   ``training.options.global-batch-size``
       Required. The samples of one optimizer step. Reef refuses a Slime deployment without it, even when ``num-steps-per-rollout`` could derive it.

   ``training.options.hf-checkpoint`` : ``inference.model-path``
       The model that Slime trains. Reef sets it; an explicit value must agree with ``inference.model-path``.

   ``training.options.actor-num-nodes``, ``training.options.actor-num-gpus-per-node``
       The training GPUs. ``training.options.num-gpus-per-node`` gives the physical size of a node.

- Reef fills Slime's architecture flags, such as layer counts and hidden sizes, from the model's Hugging Face config. An explicit flag wins.
- Slime's engine and placement options (``rollout-num-gpus``, ``colocate``, ``sglang-*``, ``router-*``) configure the rollout engines of Slime's own serving session (`Task jobs on Slime`_). They never configure the serving engines, which take ``inference.options``.
- With ``colocate``, ``offload``, ``offload-train`` or ``advantage-estimator: ppo``, Slime offloads the training workers and keeps no CPU copy of their parameter and gradient buffers. Such workers cannot use expandable segments, because their optimizer step can then write zeros into the weights. Reef refuses ``expandable_segments:True`` in ``PYTORCH_CUDA_ALLOC_CONF`` or ``PYTORCH_ALLOC_CONF``, and ``SLIME_ENABLE_EXPANDABLE_SEGMENTS=1``, for them. Two critic layouts keep expandable segments: ``use-critic`` with ``no-offload-train`` keeps the workers on the GPU, and ``use-critic`` without the options above offloads them with the CPU copy. The training service checks its own environment and ``train-env-vars`` at startup, and each worker checks its environment before it loads the model.
- With multi-node training, put the checkpoint directory on storage that the host of the training service can read.
- `Set the checkpoint cadence <../user-guide/operate.rst#set-the-checkpoint-cadence>`__ describes the ``reef-checkpoint-*`` options.

Train without an inference engine
---------------------------------

``inference.backend: none`` runs no inference engine. Use it for supervised fine-tuning on data that does not come from the served model: uploaded trajectory items (``POST /reef/namespaces/{owner}/data``) or imported records (``POST /reef/records``). The trainer takes every GPU that ``training.options.actor-*`` names, and every inference route answers 404.

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

``reef serve`` refuses the settings that configure an engine or call the model: ``inference.serving`` other than ``none``, ``inference.num-gpus`` other than 0, ``inference.tensor-parallel-size``, ``inference.options``, ``inference.url``, ``inference.remote``, ``inference.handler-factory``, ``inference.handler-config``, ``teacher.model-path``, a ``generator`` section and a composite recipe.

Task jobs on Slime
------------------

A task job trains on Harbor tasks. Slime rolls the tasks out on its own rollout engines, which Reef's serving never calls (`Training backends <../developer-guide/training-backends.rst>`__). The job runs as a dataset job (`Dataset jobs on Slime`_) whose prompts are the tasks and whose generate step is a Harbor rollout:

- ``n-samples-per-prompt`` gives the rollouts of each task, which form one prompt group.
- Each rollout runs the agent in the task's Harbor environment, and the agent's model calls reach the rollout engines.
- The rollout becomes one Slime sample with the verifier's reward. Slime computes the advantages with ``advantage-estimator`` and trains ``policy_loss``.

The Slime options and the job parameters of a task job:

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

Rules of a task job:

- **Engine GPUs.** ``rollout-num-gpus`` gives the rollout engines GPUs of their own. ``colocate`` puts them on the training GPUs. A task job whose options give the engines no GPUs fails, and a LoRA actor cannot have rollout engines.
- **Loss family.** The recipe's default objective sets the workers' loss family. A deployment that runs task jobs therefore starts with the ``slime`` family and refuses jobs of trajectories. The built-in objective of that family is ``slime_dataset``.
- **Rollout count.** The job runs ``ceil(tasks / rollout-batch-size)`` Slime rollouts, so each task runs at least once. The last batch wraps to the first tasks when the tasks do not fill it. To train the tasks again, submit another job.
- **Checkpoints.** ``save_interval`` counts Slime rollouts.
- **Model calls.** The engines answer OpenAI text chat: system, user and assistant text messages, no tools. Each call samples with Slime's ``rollout-temperature``, ``rollout-top-p`` and ``rollout-top-k``, whatever the agent asks, and ``rollout-max-response-len`` bounds each response.
- **Failed rollouts.** When no group of a Slime rollout can train, the job fails before the workers train it, with the error of the first rollout.
- **Score centering.** Slime's ``use-score-centering`` works: each sampled token carries the engines' top ``score-centering-top-k`` log probabilities.
- **Where rollouts run.** The rollouts run in a Ray actor of Slime's serving session. Its node needs the ``harbor`` package and the Harbor environment, and ``agent_host`` names that node. The actor gets ``E2B_API_KEY``, ``E2B_API_URL``, ``E2B_SANDBOX_URL``, ``E2B_DOMAIN``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` from the environment of the training service.

The training component refuses a task job, when it is submitted, in these cases:

- The loss family of the deployment or of the objective is not ``slime``.
- The job sets ``learning_rate_schedule`` or ``scheduling``. Slime's options set the learning rate and the optimizer steps.
- An option replaces a part of this loop: ``group-rm``, ``reward-key``, ``custom-generate-function-path``, ``data-source-path``, another ``rollout-function-path``, ``partial-rollout``, an ``over-sampling-batch-size`` other than ``rollout-batch-size``, ``dynamic-sampling-filter-path``, a ``rollout-top-p`` other than 1, or ``use-rollout-routing-replay``.
- ``n-samples-per-prompt`` is 1 with an ``advantage-estimator`` that centers each reward on its group, such as ``grpo``.

Dataset jobs on Slime
---------------------

A dataset job trains a prompt dataset the way Slime trains on its own: Slime rolls the prompts out on its rollout engines, scores them with its reward model, computes the advantages and trains its loss. Reef records each saved rollout as a candidate.

The deployment selects the built-in recipe ``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training-mode: manual``. Its loss family, ``slime``, refuses every other job except a task job. Each training request (``POST /reef/train``) whose ``parameters`` hold ``dataset`` is one job:

.. code:: json

   {"text": "GRPO on the prompts", "session": "s", "artifact_id": "artifact-1",
    "parameters": {"dataset": {"path": "/data/prompts.jsonl", "epochs": 1}, "save_interval": 10}}

.. container:: config

   ``dataset.path``
       An absolute path, on the Ray cluster, to a JSONL file in Slime's prompt-data format.

   ``dataset.epochs`` : 1
       The passes over the dataset. The job trains ``len(dataset) // rollout-batch-size`` rollouts in each epoch.

   ``save_interval`` : one checkpoint at the end
       The rollouts between two checkpoints.

- The recipe's ``job-parameters`` can set a default ``dataset``. Slime's own options in ``training.options`` configure the rest, for example ``input-key``, ``apply-chat-template`` and ``rollout-batch-size``.
- A rollout function, a reward model (``custom-rm-path`` or ``rm-type``) and a hook on the rollout data (``rollout-data-postprocess-path``) are Slime's plugins and run in Slime's processes.
- Reef sets ``data-source-path`` itself, so it refuses the option. It also refuses ``disable-compute-advantages-and-returns``.
