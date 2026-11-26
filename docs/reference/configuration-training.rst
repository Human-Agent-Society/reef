Slime training options and jobs
===============================

Configure Slime training, including deployments without inference and jobs that generate their own rollouts.
For backend selection and GPU sizing, see `Evolve your model <../user-guide/evolve-your-model.rst>`__.
For Tinker, see `Train with Tinker <../user-guide/tinker.rst>`__.

Slime's training options
------------------------

Put Slime's native flags under ``training.options``, without the leading ``--``.
See `Native backend options <configuration-inference.rst#native-backend-options>`__ for value syntax.
Malformed options fail at startup. Errors that only Slime's workers can detect fail the first job.

Reef reads these options itself:

.. container:: config

   ``training.options.global-batch-size``
       Required number of samples per optimizer step. Set it even when ``num-steps-per-rollout`` could derive it.

   ``training.options.hf-checkpoint`` : ``inference.model-path``
       The model that Slime trains. Reef sets it; an explicit value must agree with ``inference.model-path``.

   ``training.options.actor-num-nodes``, ``training.options.actor-num-gpus-per-node``
       The training GPUs. ``training.options.num-gpus-per-node`` gives the physical size of a node.

- Reef reads architecture flags from the model's Hugging Face config. Explicit flags take priority.
- ``rollout-num-gpus``, ``colocate``, ``sglang-*`` and ``router-*`` configure Slime's rollout engines (`Task jobs on Slime`_). Configure Reef's serving engines with ``inference.options``.
- With multi-node training, put the checkpoint directory on storage that the host of the training service can read.
- `Set the checkpoint cadence <../user-guide/operate.rst#set-the-checkpoint-cadence>`__ describes the ``reef-checkpoint-*`` options.

Train without an inference engine
---------------------------------

Set ``inference.backend: none`` for supervised fine-tuning on uploaded trajectories or imported records.
Upload trajectories with ``POST /reef/namespaces/{owner}/data`` or import records with ``POST /reef/records``.
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

A task job trains on Harbor tasks through Slime's rollout engines.
It is a `dataset job <#dataset-jobs-on-slime>`__ with tasks as prompts and Harbor as the rollout generator:

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

- **Engine GPUs.** Set ``rollout-num-gpus`` for separate GPUs, or ``colocate`` to share training GPUs. Jobs fail without rollout GPUs. LoRA actors cannot have rollout engines.
- **Loss family.** The recipe's default objective must select the ``slime`` family, which includes ``slime_dataset``. This deployment rejects trajectory jobs.
- **Rollout count.** Each job runs ``ceil(tasks / rollout-batch-size)`` Slime rollouts. The last batch repeats initial tasks to fill the batch. Submit another job to repeat training.
- **Checkpoints.** ``save_interval`` counts Slime rollouts.
- **Model calls.** Engines accept OpenAI text chat with system, user and assistant messages. Tools are unsupported. Slime's ``rollout-temperature``, ``rollout-top-p`` and ``rollout-top-k`` override agent values. ``rollout-max-response-len`` limits each response.
- **Failed rollouts.** If a Slime rollout has no trainable group, the job fails before training and reports the first rollout error.
- **Score centering.** ``use-score-centering`` records the top ``score-centering-top-k`` log probabilities for each sampled token.
- **Host.** Rollouts run in Slime's Ray actor. Its node needs ``harbor`` and the selected Harbor environment; ``agent_host`` identifies that node. The actor inherits ``E2B_API_KEY``, ``E2B_API_URL``, ``E2B_SANDBOX_URL``, ``E2B_DOMAIN``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` from the training service.

The training component rejects a submitted task job when:

- The loss family of the deployment or of the objective is not ``slime``.
- The job sets ``learning_rate_schedule`` or ``scheduling``. Slime's options set the learning rate and the optimizer steps.
- An option replaces a part of this loop: ``group-rm``, ``reward-key``, ``custom-generate-function-path``, ``data-source-path``, another ``rollout-function-path``, ``partial-rollout``, an ``over-sampling-batch-size`` other than ``rollout-batch-size``, ``dynamic-sampling-filter-path``, a ``rollout-top-p`` other than 1, or ``use-rollout-routing-replay``.
- ``n-samples-per-prompt`` is 1 with an ``advantage-estimator`` that centers each reward on its group, such as ``grpo``.

Dataset jobs on Slime
---------------------

A dataset job generates responses, scores them with Slime's reward model, computes advantages and trains the loss.
Each saved checkpoint becomes a candidate artifact.

Select ``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training-mode: manual``.
Its ``slime`` loss family accepts only dataset and task jobs.
Submit each job with ``POST /reef/train`` and a ``parameters.dataset`` object:

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
- Slime runs its plugins: the rollout function, reward model (``custom-rm-path`` or ``rm-type``), and rollout hook (``rollout-data-postprocess-path``).
- Reef sets ``data-source-path`` itself, so it refuses the option. It also refuses ``disable-compute-advantages-and-returns``.
