Generated tasks
===============

The settings of the built-in generated-task recipe: the round and training-pool keys of its stages, the rollout job parameters, the ``generator`` section of the service that writes and checks the tasks, and the ``task-import`` section.

The built-in generated-task recipe, ``reef.recipe.generated_tasks:GeneratedTaskRecipe``, generates Harbor tasks in rounds and trains on rollouts of the saved train tasks. Its stages are ``GeneratedTaskSamples`` and ``GeneratedTaskJobData``.

- The recipe names no objective. The deployment names one in ``recipe.config.job_parameters.objective``, as a dotted reference to a method that it provides.
- Each job is a task job: the training component rolls the job's tasks out on its own sampler and trains on their rollouts. ``job_parameters`` says how: ``rollout`` on Tinker, ``harbor`` on Slime, whose objective has the ``slime`` loss family (`Task jobs on Slime <configuration-training.rst#task-jobs-on-slime>`__).
- The recipe refuses a config without the objective and the rollouts.
- Without ``pipeline.record2sample.config.rounds``, no round runs, and the recipe trains only the train tasks that earlier generations saved in the scenario.

.. code:: yaml

   recipe:
     implementation: reef.recipe.generated_tasks:GeneratedTaskRecipe
     config:
       job_parameters:
         objective: my_methods.objectives:MyObjective
         rollout:
           rollouts_per_task: 4
           environment: docker
   pipeline:
     record2sample:
       config:
         jobs_per_round: 2
         generator_url: ${endpoints.generator}
         rounds: 20
         description: shell tasks with hidden state under /var and /etc
         generator:
           module: recipes.beta.spade:SpadeTaskGenerator
           config:
             generator_url: ${endpoints.generator}
             skills: [inspection, repair]
     sample2job:
       config:
         buffer_size: 2

``GeneratedTaskSamples`` paces the rounds, holds their source sessions, runs the generations, and sends their results to the generator. It trains on no report: the recipe trains on the rollouts that the training component runs, which are not records.

.. container:: config

   ``pipeline.record2sample.config.jobs_per_round`` : 1
       trained jobs after which the next round starts; 0 starts it when the previous round ends, so generation runs ahead of training until ``refill_tasks`` tasks wait

   ``pipeline.record2sample.config.source_records_per_round`` : 0
       recorded sessions that each round gives the generator, at most; 0 generates from the description only

   ``pipeline.record2sample.config.source_tag``
       the ``name=value`` tag that marks a source session: each of its model calls carries the header ``x-reef-tag-<name>: <value>``. Required when ``source_records_per_round`` is above 0.

   ``pipeline.record2sample.config.refill_tasks`` : 1
       the next round starts only while fewer saved tasks than this wait to train; a task in a running job does not wait. With ``pool_capacity`` above 0, leave it out: the pool's capacity paces the rounds.

   ``pipeline.record2sample.config.rounds`` : 0
       rounds of task generation; 0 generates nothing. A round counts when it finishes a generation or runs out of its attempts; a round of replayed sessions does not count. After a restart, only complete generations count.

   ``pipeline.record2sample.config.generator``
       the ``TaskGenerator`` class (``module``) and its ``config``. Required with rounds.

   ``pipeline.record2sample.config.description``
       what the tasks are about; every generation request carries it. Required with rounds.

   ``pipeline.record2sample.config.generator_url``
       the generator service, ``${endpoints.generator}``. Required with rounds.

   ``pipeline.record2sample.config.eval_fraction`` : 0.0
       fraction of new source groups held out before their rollouts run; between 0 and 1

   ``pipeline.record2sample.config.split_seed`` : 0
       fixed seed for the task store's train/eval split

   ``pipeline.record2sample.config.generation_attempts`` : 3
       attempts of a generation whose generator or checks fail to run; after them the round is skipped, and the next round finishes the same generation

   ``pipeline.record2sample.config.generation_deadline_seconds`` : 1800
       seconds that all attempts of one generation may take

   ``pipeline.record2sample.config.validation_concurrency`` : 1
       candidates that one generation checks at once. After a check that fails to run, no further check starts, and the attempt fails when the running checks end.

   ``pipeline.record2sample.config.pool_capacity`` : 0
       0 trains each train task once. Above 0, a task trains again after its job ends, and rounds start only while fewer tasks than this are active.

   ``pipeline.record2sample.config.review_interval`` : 0
       task jobs between two pool reviews; 0 reviews nothing

   ``pipeline.record2sample.config.review_window`` : 2
       the most recent rewards of a task that a review reads; at least 2

   ``pipeline.record2sample.config.review_band_low`` : 0
       a review retires a task whose pass rate is below this; between 0 and 1

   ``pipeline.record2sample.config.review_band_high`` : 1
       a review retires a task whose pass rate is above this; between ``review_band_low`` and 1

   ``pipeline.record2sample.config.review_min_reward_std`` : 0
       a review retires a task whose rewards have a smaller population standard deviation; 0 turns this rule off

**The generator.** Reef calls the generator class's ``from_config(config, *, scenario, environ)`` once for each scenario. A generator that calls a remote service reads its credentials through environment references in ``config``. The generator keeps its own state in the scenario's task store, so its ``config`` names no state directory; the built-in generators refuse a ``state_dir`` key. See `the task generation contract <../developer-guide/task-generators.rst>`__ for the generator interface and the round.

A source session is a report, without a training arm, whose inference records all carry ``source_tag``. The generator service accepts at most 512 KiB of records and files in one request, so keep ``source_records_per_round`` small when the sessions are long.

**Saved tasks and splits.** With rounds enabled, each valid task and its split are saved in the scenario's task store.

- A generation saves tasks and runs no rollouts. The saved train tasks run later, and the store keeps their results.
- The task files go in the task file store; see `Choose the task file store <../user-guide/storage.rst#choose-the-task-file-store>`__.
- Set ``eval_fraction`` and ``split_seed`` here, not in a generator's configuration. The defaults keep every task in train. A published task keeps its split; a change of these values applies to later tasks.
- Related source sessions keep the same split across generations. A task that joins existing train and eval source groups is rejected.
- Eval tasks are saved without rollouts or Designer rewards.
- The generator's pass-rate band does not filter the store.

**The training pool.** With ``pool_capacity`` above 0, the train tasks form a training pool. When a task's job ends, the task waits for a job again, after the tasks that already wait, so the tasks train in turn.

- A new round starts only while fewer tasks than ``pool_capacity`` are active. The pool can therefore exceed the capacity by the train tasks of one generation.
- The ``review_*`` keys need ``pool_capacity`` above 0, and ``refill_tasks`` must be left out. Imported train tasks still train once.
- When a deployment sets ``pool_capacity`` on a scenario whose train tasks trained once, those tasks join the pool at the start.
- The task store keeps each use of a task with its rewards, each review, and each retired task with its reason, so a restart continues the pool in the same order.
- The status shows the pool under ``training_pool``: ``capacity``, ``active``, ``retired``, ``reviews`` and ``last_review_retired``.

After every ``review_interval`` task jobs, a review reads the last ``review_window`` rewards of each active task. A failed task job counts too. The pass rate is the mean of those rewards. A task with fewer rewards stays. The review retires a task for one of these reasons, in this order:

- ``below_band``: the pass rate is below ``review_band_low``.
- ``above_band``: the pass rate is above ``review_band_high``.
- ``no_reward_spread``: the population standard deviation of the rewards is below ``review_min_reward_std``.

A pass rate on a bound keeps the task. A retired task does not train again, except in a job whose tasks the trainer took before the review. The review reads the rewards that training recorded, so they can be one pool cycle old. The defaults run no review, and each of their rules keeps every task. The generator still receives the first rollout of each task only.

Two limits of the training pool:

- When the trainer starts the next job before the stage reads the end of a failed task job, the tasks of the next job also get a use with no rewards, and their rewards in that job are not kept. This also holds with ``pool_capacity`` 0.
- The tasks of a job that a new training start cancelled return to the pool only when a later job ends. When that job held so many tasks that fewer than ``job_samples`` wait and the pool is full, no job and no round can start, also after a restart. Restart with a larger ``pool_capacity``.

**Task jobs.** The saved train tasks become task items that the scenario uses, ``<scenario>.generated-tasks.<task name>``, and train like every other sample. ``GeneratedTaskJobData`` makes one task job of the next job's tasks, whatever Tinker's batch size (``training.options.batch-size``) is:

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : 1
       the fewest saved tasks of one task job

   ``pipeline.sample2job.config.buffer_size`` : 4
       the most saved tasks of one task job; at least ``job_samples``, and ``job_samples`` when it is more than 4

``rollout`` configures the rollouts on Tinker's sampler, at the job's start weights. ``harbor`` configures them inside Slime's own loop, on its rollout engines (`Task jobs on Slime <configuration-training.rst#task-jobs-on-slime>`__). ``harbor`` takes the keys below except ``rollouts_per_task``: Slime's ``n-samples-per-prompt`` gives the count. These parameters say only how to run Harbor; the ``objective`` parameter says how the samples train.

.. container:: config

   ``rollout.rollouts_per_task`` : required on Tinker
       training rollouts of each task, at least 2; the samples of one task form its group

   ``rollout.agent`` : terminus-2
       the Harbor agent config, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. Its requests must fit the training component's text chat: system, user and assistant text messages, no tools.

   ``rollout.environment`` : docker
       the Harbor environment type of the rollouts, such as ``docker``, ``e2b`` or ``modal``. On ``modal``, a task with ``environment/docker-compose.yaml`` and no GPU runs on Modal's VM runtime, so that its Compose networks work

   ``rollout.agent_host`` : none
       the host at which an agent inside the sandbox reaches the component's host; none for an agent on this host

   ``rollout.concurrency`` : 2
       rollouts in flight

   ``rollout.verifier_timeout_reward`` : 0.0
       the reward of a rollout whose verifier timed out; ``null`` leaves it unscored

   ``rollout.scaffold_tolerance`` : 8
       tokens at the end of a prompt that the next prompt's history may drop, such as the empty think block of a thinking template with thinking off; the sample keeps them, and the response after them trains

**Cloud sandboxes.** Set ``rollout.environment`` to ``e2b`` for an E2B-compatible service such as AgentENV, or to ``modal`` for Modal.

- For a self-hosted E2B service, set ``E2B_API_URL``, ``E2B_SANDBOX_URL`` and ``E2B_API_KEY`` in the process that runs the rollouts. The first two select the service's API and sandbox endpoints. The Slime training component applies these settings to its workers.
- On AgentENV, also set ``REEF_E2B_DOCKERFILE_BUILD=aenv`` in that process, and put AgentENV's ``aenv`` CLI and its ``aenv-buildctl`` on ``PATH``. AgentENV's E2B template builds refuse a Dockerfile's COPY steps, so Reef then builds each task's Dockerfile with ``aenv build`` and the same ``E2B_API_URL`` and ``E2B_API_KEY``. The default, ``e2b``, uses E2B's template API.

**Sandbox lifetime.** Harbor deletes the sandbox when the rollout ends. If the rollout process is killed before cleanup, the sandbox's lifetime limits how long it remains.

- The lifetime is the sum of the task's build, agent and verifier time limits, plus one hour, and at most one day. A rollout that needs more time loses its sandbox after one day. A nonzero ``override_timeout_sec`` in ``rollout.agent`` replaces the task's agent time limit in that sum, and an ``override_setup_timeout_sec`` adds to it.
- Without an agent time limit from the task or from a nonzero ``override_timeout_sec``, the sum does not apply: a Modal sandbox lives for two hours, and an E2B sandbox for Harbor's default of one day.
- Reef asks the service to delete an E2B sandbox at the end of its lifetime. Without this request, a self-hosted service can pause the sandbox and keep it.
- These rules also apply to evaluation and generator rollouts, each with its own agent config (`Harbor task environments <../developer-guide/sandboxes.rst#harbor-task-environments>`__). Standalone ``harbor run -e e2b`` commands and the Meta-Harness Terminal-Bench ``terminus`` adapter, which is separate from the ``terminus-2`` agent of Reef's rollouts, use Harbor's default timeout behavior.

**Task job metrics.**

.. list-table::
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``rollout_rewards``
     - each task's rewards of the rollouts that ran
   * - ``rollout_samples``
     - the samples that train
   * - ``removed_rollouts``
     - the rollouts that give no sample: they fail in the infrastructure, or their model calls do not assemble into one sample
   * - ``rollout_tokens``, ``rollout_response_tokens``, ``rollout_trained_tokens``
     - Tinker only: each rollout's response tokens and the tokens of them that train, and their sums
   * - ``unrenderable_samples``
     - Tinker: samples longer than the model's context window, left out before their group forms; they are in neither ``rollout_samples`` nor the trained tokens

A sample alone in its group trains with advantage 0 on Slime and counts in ``rollout_samples``. On Tinker it does not train, and it is in neither count.

The ``generator`` section
-------------------------

Absent by default. When present, ``reef serve`` starts the generator service (``python -m reef.service.deploy.generator_process``) before the HTTP service and publishes its address as ``${endpoints.generator}``. The generator writes and checks Harbor tasks for the generated-task recipe, and runs the plays of a task generator such as SPADE. It runs under the same interpreter as the Reef service. ``execution.generator`` selects its executor, so a deployment can place it on another host.

The generator's host needs the ``harbor`` command line and the Harbor environment that ``generator.environment`` selects; the Reef service itself needs neither.

.. list-table::
   :header-rows: 1

   * - ``generator.environment``
     - Needs
   * - ``docker`` (default)
     - a Docker daemon on the host
   * - ``e2b``
     - ``E2B_API_KEY`` exported where ``reef serve`` starts, and ``reef-infra[e2b]`` in the generator's interpreter. The readiness check looks there, so ``generator.harbor`` must come from the same Python environment
   * - ``modal``
     - ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` exported where ``reef serve`` starts, and ``reef-infra[modal]`` in the generator's interpreter

With ``e2b``, a task runs only its ``environment/Dockerfile`` (no compose services), an agent inside the sandbox cannot reach a ``generator.agent-host`` on a private host, and E2B builds a template for each new task image. The E2B rules of ``rollout.environment`` (`Generated tasks`_) apply to the checks and to the rollouts: the settings of a self-hosted service, the sandbox lifetime, and the request to delete the sandbox at its end. A rollout takes the agent's time limits from ``generator.agent``. A check runs Harbor's oracle or nop agent, which sets no time limit, so only the task's time limits count for a check.

The generator runs its proposals, checks and rollouts at the same time. ``generator.concurrency`` limits only the rollouts in flight, so a proposal or a check never waits behind rollouts.

.. container:: config

   ``generator.tasks-root``
       the directory generated tasks and Harbor job files live under. Required.

   ``generator.host`` : 127.0.0.1
       bind address

   ``generator.port`` : 8910
       bind port

   ``generator.work-dir`` : ``<tasks-root>/.rollouts``
       where the rollout runner keeps trials

   ``generator.agent`` : terminus-2
       the Harbor agent each rollout runs, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. ``import_path: reef.record2sample.generator.terminus:VerifiedTerminus2`` in place of ``name: terminus-2`` also scores a rollout whose agent filled its context or lost its terminal, which Terminus 2 leaves without a reward

   ``generator.agent-host``
       an address of the host the task container can reach, for an agent that runs inside the container

   ``generator.harbor`` : ``harbor`` on PATH
       the harbor command line for the oracle check

   ``generator.environment`` : docker
       the Harbor environment of the oracle and nop checks and of the rollouts: ``docker``, ``e2b`` or ``modal``. The readiness check waits for that environment's requirements

   ``generator.concurrency`` : 2
       rollouts in flight across all rollout requests of the generator service

   ``generator.designer-url``
       a Reef service the designer calls go to instead of this deployment's

   ``generator.designer-token``
       the token for ``designer-url``

   ``generator.designer-model``
       the served model the designer asks for; the deployment's by default

   ``generator.designer-timeout-s`` : 1800
       seconds one designer call may take; ``inference.timeout-s`` must allow it too

   ``generator.designer-api``
       the API the designer asks through: ``openai`` (Chat Completions) or ``responses`` (the Responses API). The default is ``responses`` when designer calls go to this deployment and its ``inference.upstream-api`` is ``responses``, and ``openai`` otherwise

   ``generator.designer-options``
       extra fields of the designer's request, e.g. ``{"reasoning_effort": "none"}``; with ``designer-api: responses``, Responses fields such as ``{"reasoning": {"effort": "low"}}``

   ``generator.ready-timeout`` : 60
       seconds ``reef serve`` waits for the generator to answer

.. code:: yaml

   generator:
     tasks-root: ${REEF_SPADE_STATE_DIR}/tasks
   execution:
     generator: auto
   pipeline:
     record2sample:
       config:
         generator_url: ${endpoints.generator}

The generator reads the resolved deployment: the Reef address from ``reef.host`` and ``reef.port``, the token from ``reef.token``, and the served model from ``inference.model-path`` or ``inference.upstream-model``. Without ``reef serve``, ``python -m reef.service.deploy.generator_process -c serve.yaml`` runs it from the same file.

The ``task-import`` section
---------------------------

Absent by default. When present, each scenario imports the Harbor tasks of a split manifest when it opens, as its imported tasks. See `Import Harbor tasks <../user-guide/operate.rst#import-harbor-tasks>`__.

.. container:: config

   ``task-import.manifest``
       a split manifest: a JSON file with ``version``, ``seed``, ``eval_fraction``, and the task names of ``train`` and ``eval``. Required with ``tasks-root``.

   ``task-import.tasks-root``
       the directory that holds a Harbor task directory for each task name of the manifest. Required with ``manifest``.
