The ``training`` section
========================

The keys of the ``training`` and ``teacher`` sections of a weight-training deployment, the processes that such a deployment starts, and the jobs that Slime runs on its own rollout engines.

`Evolve your model <../user-guide/evolve-your-model.rst>`__ describes how to size the stack.

.. container:: config

   ``training.backend`` : slime
       ``slime``, ``tinker`` (remote LoRA training without local GPUs, `Train with Tinker <../user-guide/tinker.rst>`__), ``spindle_full`` (all the parameters of one Hugging Face revision on a Spindle deployment, `Train all the parameters on Spindle <../user-guide/tinker.rst#train-all-the-parameters-on-spindle>`__), an installed ``reef.training_backends`` entry-point name, or the ``package.module:ClassName`` of a ``TrainingDeployment`` class. The selected backend validates its own settings. Other backends do not inherit Slime's Ray or native-argument requirements

   ``training.ray-address`` : ``RAY_ADDRESS``
       the Ray cluster that the inference component and the HTTP service join. Without an address, Reef starts a local cluster and stops it on exit; Reef leaves an existing cluster running

   ``training.ray-namespace`` : reef
       the Ray namespace of the training component

   ``training.ready-timeout`` : 3600
       startup deadline, in seconds, of the services that the training backend adds, such as the inference component and the teacher engine

   ``training.max-jobs`` : 1
       training jobs that one process of the training service runs at the same time for each training component. Several jobs of one scenario run together only when the component allows it; the Tinker component does (`Training backends <../developer-guide/training-backends.rst>`__). Each running job needs its own compute

   ``training.manager`` : unset
       a ``TrainingManager`` plugin (``reef.train.manager``), as ``package.module:ClassName`` or as ``module`` and ``config``. Without it, a scenario in ``auto`` mode starts a job when ``job_samples`` untrained samples wait. With it, the manager decides (`Decide what a scenario trains <../developer-guide/training-backends.rst#decide-what-a-scenario-trains>`__)

   ``training.timeout-s`` : four hours with Slime
       bound, in seconds, of each call to the training workers (Slime: each worker call of a job, such as training and checkpoint saves; Tinker: each remote call). A positive finite number. A timeout stops the wait, not the work on the workers. The job then fails, Reef does not retry it, and the next job starts from the last saved checkpoint

   ``training.config.num_gpus``
       example-specific GPU count passed to Slime's model topology flags; does not set the Ray cluster's capacity

   ``training.config.global_batch_size``
       samples in one optimizer step, passed to Slime as ``--global-batch-size``. The examples also pass it to ``pipeline.sample2job.config.job_samples``, so one job is one optimizer step. Reef reads ``training.options.global-batch-size`` itself and refuses a Slime deployment without it, even when ``num-steps-per-rollout`` could derive it.

   ``training.config.checkpoint_dir``
       where Megatron and HF checkpoints are written

   ``training.config.megatron_checkpoint_path``
       optional pre-converted torch_dist checkpoint, to skip HF conversion on every start

   ``training.config.checkpoint_retention``
       storage-fraction bounds and the retention policy. When storage cannot hold the next checkpoint, the Slime job fails. A failed job ends, and the records of its data do not train again, because Reef consumed them when it submitted the job.

   ``training.options``
       native training flags: actor GPU layout, optimizer, sequence length, and loss settings (`Native backend options <configuration-inference.rst#native-backend-options>`__)

   ``teacher.model-path`` : unset
       the teacher engine's Hugging Face model or directory

   ``teacher.num-gpus`` : 1
       the teacher engine's GPUs, also its tensor parallel size

   ``teacher.port`` : 30001
       the teacher engine's internal HTTP port

   ``teacher.timeout`` : 300
       deadline of one teacher scoring request, in seconds

   ``teacher.options`` : {}
       native SGLang options of the teacher engine

``training.config`` holds workload variables that the file's other values reference, such as ``${training.config.global_batch_size}``. Slime fills architecture flags such as layer counts and hidden sizes from ``inference.model-path``. Do not put them in the config.

``training.colocate`` is not a key, and startup refuses it as an unknown field: Reef's serving never shares GPUs with training. Slime's own rollout engines share the training GPUs with ``training.options.colocate``.

Processes of a weight-training stack
------------------------------------

With the default Slime backend and an engine (``inference.backend: sglang`` or ``vllm``), ``reef serve`` starts:

- ``inference``: a local inference component (``python -m reef.inference.remote``) on a free loopback port. It starts the serving engines and loads each published snapshot into them. An ``inference.remote`` section replaces it with a component on another host.
- ``distill-teacher``: a frozen SGLang teacher engine, only when a ``teacher`` section is set.
- ``reef``: the HTTP service. It hosts the training component, which starts a Slime driver on Ray for each job.

Training and serving use separate processes and GPUs. The training component saves checkpoints. Reef publishes them as snapshots, and the serving engines load only published snapshots, from files. The inference component and the HTTP service join the same Ray cluster, so the engines and the training workers reserve separate GPUs. `Training backends <../developer-guide/training-backends.rst>`__ and `Serving runtimes <../developer-guide/serving-runtimes.rst>`__ describe the job and serving contracts.

An in-process integration can use ``InProcessTrainingDeployment``. It starts only the HTTP service, whose runtime factory builds the training component in that process and parses ``training.options``. The options map cannot override the runtime type, the model or the shared timeouts. Such an integration rejects Ray and training executor settings.

Reef starts only its own inference, training and HTTP services. OpenClawRL deploys its PRM and user-simulation services itself; the recipe reads ``recipe.config.prm-url`` and ``recipe.config.prm-tokenizer-path``.

Slime's training options
------------------------

Slime's own engine and placement options in ``training.options`` (``rollout-num-gpus``, ``colocate``, ``sglang-*``, ``router-*`` and the rest) reach each job's Slime driver unread. They configure the rollout engines of Slime's serving session (`Task jobs on Slime`_), never the serving engines. `Native backend options <configuration-inference.rst#native-backend-options>`__ gives the value rules.

- Native options cannot override the managed model, placement or parallelism settings. Each serving engine runs on one node. The managed path supports no pipeline or data parallel topology and no prefill/decode disaggregation.
- Reef binds ``training.options.hf-checkpoint`` to ``inference.model-path``; an explicit value must agree.
- ``training.options.actor-num-*`` set the training GPUs. The physical node size is ``training.options.num-gpus-per-node``.
- Until the first task job, each job's driver reserves the training GPUs only while it runs.
- The inference component reserves ``inference.num-gpus`` GPUs on the same Ray cluster. The inference control process takes one Ray CPU and no GPU.
- When a training worker fails, the running job fails, and the next job starts again from the latest checkpoint. A configuration error that only Slime's workers detect fails the first job.

Train without an inference engine
---------------------------------

``inference.backend: none`` runs no inference engine. Use it for supervised fine-tuning on data that does not come from the served model, such as uploaded trajectory items (``POST /reef/namespaces/{owner}/data``) or imported records (``POST /reef/records``). Only the Slime backend supports it. The trainer takes every GPU that ``training.options.actor-*`` names:

.. code:: yaml

   inference:
     model-path: /models/Qwen3-8B   # the weights to train; nothing serves them
     backend: none
   training:
     backend: slime
     options:
       actor-num-nodes: "1"
       actor-num-gpus-per-node: "8"
       global-batch-size: "32"
       # Add the recipe's optimizer, model and checkpoint options here.

Training in this mode works as follows:

- Each job trains its data on Slime's workers, which stay on their GPUs between jobs.
- Each job saves one checkpoint: the Megatron checkpoint and its HF export. Reef publishes every checkpoint as a snapshot. See `Set the checkpoint cadence <../user-guide/operate.rst#set-the-checkpoint-cadence>`__.
- A restart continues from the last saved checkpoint, as with an engine.

The deployment serves no inference. ``/v1/chat/completions``, ``/v1/responses`` and ``/v1/messages`` answer 404 for every scenario. To use a trained version, serve its published checkpoint from a separate deployment.

``reef serve`` refuses the settings that configure an engine or call the model: ``inference.num-gpus`` (other than 0), ``inference.tensor-parallel-size``, ``inference.options``, ``inference.url``, ``inference.remote``, ``inference.handler-factory``, ``inference.handler-config``, a ``teacher`` engine (``teacher.model-path``), a ``generator`` section and a composite recipe.

Task jobs on Slime
------------------

The Slime component runs task jobs on Slime's own rollout engines: the SGLang engines of Slime's serving session. Reef's serving never calls them.

- The first task job starts the session. The session holds the engines, their router and the training GPUs until the component closes. Later jobs attach to the started engines, and every job trains on the session's GPUs.
- A failed or cancelled job leaves the session running for the next job.
- ``training.options.rollout-num-gpus`` gives the engines GPUs of their own. ``training.options.colocate`` puts them on the training GPUs (all of them, unless ``rollout-num-gpus`` names fewer), and Slime moves the GPU memory between the engines and the workers.
- Slime's ``sglang-*``, ``router-*`` and ``rollout-num-gpus-per-engine`` options configure the engines.
- A task job whose options give the engines no GPUs fails. A LoRA actor cannot have rollout engines.

A task job runs as a dataset job (`Dataset jobs on Slime`_) whose prompts are Harbor tasks and whose generate step is a Harbor rollout. Slime's own options set the training method. ``n-samples-per-prompt`` gives the rollouts of each task, which form a prompt group. Each rollout runs the agent in the task's Harbor environment, and the agent's model calls reach the engines. The rollout becomes one Slime sample with the verifier's reward. Slime then computes the advantages with ``advantage-estimator`` and trains ``policy_loss``.

.. code:: yaml

   training:
     backend: slime
     options:
       colocate: true
       rollout-num-gpus-per-engine: 1
       sglang-mem-fraction-static: 0.5
       advantage-estimator: grpo
       loss-type: policy_loss
       rollout-batch-size: 4
       n-samples-per-prompt: 4
       global-batch-size: 16
       rollout-max-response-len: 2048
   recipe:
     config:
       job_parameters:
         objective: slime_dataset
         save_interval: 10
         harbor:
           environment: modal
           concurrency: 8

Rules of a task job:

- **Loss family.** The recipe's default objective sets the workers' loss family, so a deployment that runs task jobs starts with the ``slime`` family and refuses jobs of trajectories.
- **Rollout count.** The job's rollouts are ``ceil(tasks / rollout-batch-size)``, so each task runs at least once. The last batch wraps to the first tasks when the tasks do not fill it. To train the tasks again, submit another job.
- **Checkpoints.** ``save_interval`` counts Slime rollouts, as on a dataset job. A resumed job continues after its last checkpoint and skips the tasks of the rollouts that it saved. The rollouts that a stopped driver did not train run again.
- **Model calls.** The engines answer OpenAI text chat: system, user and assistant text messages, no tools. Each call samples with Slime's ``rollout-temperature``, ``rollout-top-p`` and ``rollout-top-k``, whatever the agent asks, because the trainer reads that distribution. ``rollout-max-response-len`` bounds each call's response.
- **Removed rollouts.** A rollout that fails in the infrastructure, or whose model calls do not assemble into one sample, trains nothing. Slime leaves its reward out of its group's mean and deviation, so the group's other rollouts train against each other. A group with one sample left has advantage 0. When no group of a Slime rollout can train, the job fails before the workers train it, with the first rollout's error. Slime's own metrics, such as ``rollout/raw_reward``, count a removed sample with reward 0.
- **Score centering.** Slime's ``use-score-centering`` works: each response token that the model sampled carries the engines' top ``score-centering-top-k`` log-probabilities.
- **Where rollouts run.** The rollouts run in Slime's rollout manager, a Ray actor of the serving session. Its node needs the ``harbor`` package and the Harbor environment, and reaches the job's directory and the tasks through the shared save root. ``agent_host`` names that node. The driver passes ``E2B_API_KEY``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` from its environment to the job's Ray actors. A driver that stops during a rollout leaves its Harbor rollouts running, and the next driver waits for them.

The driver refuses a task job, when it is submitted, in these cases:

- The deployment's loss family, or the objective's, is not ``slime``. The built-in objective of that family is ``slime_dataset``.
- The job sets ``learning_rate_schedule`` or ``scheduling``. Slime's options set the learning rate and the optimizer steps.
- An option replaces a part of this loop or adds data that the rollouts do not record: ``group-rm``, ``reward-key``, ``custom-generate-function-path``, ``data-source-path``, another ``rollout-function-path``, ``partial-rollout``, an ``over-sampling-batch-size`` other than ``rollout-batch-size``, ``dynamic-sampling-filter-path``, a ``rollout-top-p`` other than 1 or ``use-rollout-routing-replay``.
- ``n-samples-per-prompt`` is 1 with a group baseline: ``advantage-estimator`` ``grpo``, ``gspo``, ``cispo`` or ``reinforce_plus_plus_baseline`` with reward normalization, which would center each reward to 0.

**Log probabilities of task rollouts.** ``rollout_log_probs`` are the sampled tokens' log probabilities that a training component's own task rollouts record; serving records none. They must mean the same thing as the trainer's. Sampling runs through::

   raw logits -> penalties, logit_bias -> / temperature -> [A] -> top-k, top-p, min-p -> [B] -> sample

SGLang and Slime's trainer both read at [A] (full vocabulary, trainer with ``rollout_temperature``, no penalties), so they agree as long as a recipe uses no penalties or ``logit_bias``. vLLM ``--logprobs-mode processed_logprobs`` reads at [B], so it matches only with top-k, top-p and min-p off. Reef's vLLM client rejects a request whose effective ``top_p`` is below 1, ``top_k`` above 0 or ``min_p`` above 0. Verify with Slime's ``train_rollout_logprob_abs_diff`` on identical weights before training.

Dataset jobs on Slime
---------------------

A dataset job trains a prompt dataset the way Slime trains on its own. Slime rolls the prompts out on its serving session's engines, scores them with its reward model, computes the advantages and trains its loss. Reef records each saved rollout as a candidate.

The deployment selects the built-in recipe ``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training_mode: manual``. Its loss family, ``slime``, leaves the training signal to Slime and refuses every other job except a task job. Each training request (``POST /reef/train``) whose ``parameters`` hold ``dataset`` is one job:

.. code:: json

   {"text": "GRPO on the prompts", "session": "s", "artifact_id": "artifact-1",
    "parameters": {"dataset": {"path": "/data/prompts.jsonl", "epochs": 1}, "save_interval": 10}}

.. container:: config

   ``dataset.path``
       an absolute path, on the Ray cluster, to a JSONL file in Slime's prompt-data format

   ``dataset.epochs`` : 1
       the passes over the dataset. The job trains ``len(dataset) // rollout-batch-size`` rollouts per epoch

   ``save_interval`` : one checkpoint at the end
       rollouts between checkpoints. A checkpoint's step is the job's rollouts up to it, and its reward is the rollout's mean raw reward

A resumed job continues after its last checkpoint and skips the prompts of the rollouts that it saved. The recipe's ``job_parameters`` can set a default ``dataset``. Slime's own options in ``training.options`` configure the rest:

.. code:: yaml

   training:
     backend: slime
     options:
       colocate: true
       input-key: messages
       label-key: label
       apply-chat-template: true
       rollout-batch-size: 32
       n-samples-per-prompt: 8
       global-batch-size: 256
       rollout-max-response-len: 1024
       rollout-temperature: 1.0
       advantage-estimator: grpo
       custom-rm-path: my_package.reward.reward
       loss-type: policy_loss

A rollout function (``rollout-function-path``, ``custom-generate-function-path``), a reward model (``custom-rm-path`` or ``rm-type``) and a hook on the rollout data after the advantages (``rollout-data-postprocess-path``) are Slime's plugins and run in Slime's processes. Reef sets ``data-source-path`` itself, so the option is refused. ``disable-compute-advantages-and-returns`` is refused too.
