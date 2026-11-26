Generated tasks
===============

Use ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` to generate Harbor tasks and train on their rollouts.

This example requires a custom objective. Replace ``my_methods.objectives:MyObjective`` with your implementation:

.. code:: yaml

   generator:
     tasks-root: /srv/reef/tasks
   training:
     backend: tinker
   recipe:
     implementation: reef.recipe.generated_tasks:GeneratedTaskRecipe
     config:
       job-parameters:
         objective: my_methods.objectives:MyObjective
         rollout:
           rollouts_per_task: 4
           environment: docker
   pipeline:
     record2sample:
       config:
         rounds: 20
         jobs-per-round: 2
         description: shell tasks with hidden state under /var and /etc
         generator-url: ${endpoints.generator}
         generator:
           module: recipes.beta.spade:SpadeTaskGenerator
           config:
             generator_url: ${endpoints.generator}
             skills: [inspection, repair]
         pool-capacity: 32
         review-interval: 8
         review-band-low: 0.25

For all settings, see `Recipe <configuration-settings.rst#recipe>`__ (``GeneratedTaskSamples``),
`Generator <configuration-settings.rst#generator>`__ and `Evaluation <configuration-settings.rst#evaluation>`__ (``task-import``).

Rounds and tasks
----------------

- Every job trains tasks. Startup requires ``objective`` and the backend's rollout parameters: ``rollout`` for Tinker or ``harbor`` for `Slime <configuration-training.rst#task-jobs-on-slime>`__.
- With ``rounds: 0``, the recipe trains only tasks saved by earlier generations.
- A source session is a report whose inference records all carry ``source-tag``. Each Designer request accepts at most 512 KiB of records and files as JSON. Reduce ``source-records-per-round`` for long sessions.
- Set ``eval-fraction`` and ``split-seed`` in ``pipeline.record2sample.config``, not in the generator's ``config``.
- For a custom generator, see `the generator contract <../developer-guide/task-generators.rst#the-contract>`__.

The training pool
-----------------

Set ``pool-capacity`` above 0 to reuse generated training tasks:

- After a job ends, its tasks return to the end of the queue.
- Capacity is checked at round start. The pool can exceed it by one generation's training tasks.
- Imported train tasks still train once.
- When you enable the pool, previously trained tasks join it.
- The status shows the pool under ``training_pool``: ``capacity``, ``active``, ``retired``, ``reviews`` and ``last_review_retired``.

Every ``review-interval`` jobs, including failed jobs, Reef reviews each active task's last ``review-window`` rewards.
Tasks with fewer rewards stay active. The pass rate is the mean reward.
The first matching condition retires a task:

- ``below_band``: the pass rate is below ``review-band-low``.
- ``above_band``: the pass rate is above ``review-band-high``.
- ``no_reward_spread``: the population standard deviation of the rewards is below ``review-min-reward-std``.

Equality with a pass-rate bound does not retire a task. Recorded rewards can be one pool cycle old.

Exceptions:

- If a job starts before the stage reads the previous job's failure, the new job also counts as a use without rewards. Its rewards are discarded. This also applies with ``pool-capacity: 0``.
- When a new training start cancels a job, its tasks return to the pool only when a later job ends.

Rollout job parameters
----------------------

Set these keys under ``recipe.config.job-parameters``.
``harbor`` accepts the same keys as ``rollout``, except ``rollouts_per_task``. Both use underscores only.

.. container:: config

   ``objective``
       Required objective: a registered name or ``package.module:Objective`` reference. The recipe supplies no default.

   ``rollout.rollouts_per_task``
       Rollouts per task, at least 2. Required on Tinker. Slime uses ``training.options.n-samples-per-prompt`` instead.

   ``rollout.agent`` : terminus-2
       The Harbor agent config, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. Its requests must be text chat: system, user and assistant text messages, no tools.

   ``rollout.environment`` : docker
       The Harbor environment type of the rollouts, such as ``docker``, ``e2b`` or ``modal``.

   ``rollout.agent_host``
       Address through which a sandboxed agent reaches the rollout host. Omit for an agent running on that host.

   ``rollout.concurrency`` : 2
       Maximum concurrent rollouts. Must be at least 1.

   ``rollout.verifier_timeout_reward`` : 0.0
       The reward of a rollout whose verifier timed out. ``null`` leaves the rollout unscored.

   ``rollout.scaffold_tolerance`` : 8
       Maximum trailing prompt tokens that the next prompt's history can omit, such as a thinking template. The training sample keeps these tokens.

**Cloud sandboxes.**

- For a self-hosted E2B service, set ``E2B_API_URL``, ``E2B_SANDBOX_URL`` and ``E2B_API_KEY`` in the process that runs the rollouts. The Slime training component passes these variables and ``E2B_DOMAIN`` to its workers.
- On AgentENV, also set ``REEF_E2B_DOCKERFILE_BUILD=aenv`` in that process, and put AgentENV's ``aenv`` and ``aenv-buildctl`` on ``PATH``. Reef then builds each task's Dockerfile with ``aenv build``, which uses the same ``E2B_API_URL`` and ``E2B_API_KEY``. The Slime component does not pass ``REEF_E2B_DOCKERFILE_BUILD`` to its workers.
- Harbor deletes the sandbox when the rollout ends. When the rollout process is killed first, the sandbox stays until its `lifetime <../developer-guide/sandboxes.rst#harbor-task-environments>`__ ends.

**Task job metrics.**

.. list-table::
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``rollout_rewards``
     - each task's rewards
   * - ``rollout_samples``
     - the samples that train
   * - ``removed_rollouts``
     - the rollouts that fail in the infrastructure, or whose model calls do not assemble into one sample
   * - ``rollout_tokens``, ``rollout_response_tokens``, ``rollout_trained_tokens``
     - Tinker only: the response tokens and the trained tokens of each rollout, and their sums
   * - ``unrenderable_samples``
     - Tinker only: samples longer than the model's context window, which do not train and are not in ``rollout_samples`` or the trained tokens

On Tinker, a sample alone in its group does not train, and it is in neither ``rollout_samples`` nor ``removed_rollouts``.

The ``generator`` section
-------------------------

This section starts the generator before the HTTP service and exposes its address as ``${endpoints.generator}``.
The generator uses Reef's Python interpreter. Select its executor with ``execution.generator`` (`Executors <configuration-settings.rst#executors>`__).

The generator host needs ``harbor`` and the selected ``generator.environment``.
For Tinker, the training host also needs Harbor and the selected ``rollout.environment``.

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

With ``e2b``, a check or a rollout starts only the task's image, unless ``REEF_E2B_COMPOSE=aenv`` also starts its Compose services (`Tasks with Compose services on AgentENV <../developer-guide/sandboxes.rst#tasks-with-compose-services-on-agentenv>`__).

The generator reads its connection settings from the deployment config:

- Address: ``reef.host`` and ``reef.port``.
- Token: ``reef.token``, or the first entry of ``reef.tokens``.
- Model: ``inference.model-path`` or ``inference.upstream-model``.

To run the generator separately:

.. code:: bash

   python -m reef.deploy.generator_process -c serve.yaml
