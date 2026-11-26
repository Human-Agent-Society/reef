Generated tasks
===============

The rules of the generated-task recipe, ``reef.recipe.generated_tasks:GeneratedTaskRecipe``, for a deployment that generates Harbor tasks and trains on their rollouts.

The settings reference lists the keys of the stages (`Recipe <configuration-settings.rst#recipe>`__, stage ``GeneratedTaskSamples``), of ``generator`` (`Generator <configuration-settings.rst#generator>`__) and of ``task-import`` (`Evaluation <configuration-settings.rst#evaluation>`__).

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

``my_methods.objectives:MyObjective`` is a placeholder.

Rounds and tasks
----------------

- Every job is a task job. The training service refuses to start without ``objective`` and the rollout parameter of its backend: ``rollout`` on Tinker, ``harbor`` on Slime (`Task jobs on Slime <configuration-training.rst#task-jobs-on-slime>`__).
- With ``rounds`` 0, the recipe trains only the train tasks that earlier generations saved.
- A source session is a report whose inference records all carry ``source-tag``. The generator service accepts at most 512 KiB of JSON for the records and files of one Designer request. So keep ``source-records-per-round`` small for long sessions.
- Set ``eval-fraction`` and ``split-seed`` in ``pipeline.record2sample.config``, not in the generator's ``config``.
- Reef calls the generator's ``from_config(config, *, scenario, environ, state)`` once for each scenario (`The contract <../developer-guide/task-generators.rst#the-contract>`__).

The training pool
-----------------

With ``pool-capacity`` above 0, the train tasks form a training pool.

- When a task's job ends, the task waits for a job again, after the tasks that already wait.
- Reef checks the capacity only when a round starts, so the pool can exceed it by the train tasks of one generation.
- Imported train tasks still train once.
- When you add ``pool-capacity`` to a scenario whose train tasks trained once, they join the pool.
- The status shows the pool under ``training_pool``: ``capacity``, ``active``, ``retired``, ``reviews`` and ``last_review_retired``.

After every ``review-interval`` task jobs, a review reads the last ``review-window`` rewards of each active task. A failed task job counts too. The pass rate is the mean of those rewards. A task with fewer rewards stays. The review retires a task for the first of these reasons that applies:

- ``below_band``: the pass rate is below ``review-band-low``.
- ``above_band``: the pass rate is above ``review-band-high``.
- ``no_reward_spread``: the population standard deviation of the rewards is below ``review-min-reward-std``.

A pass rate equal to a bound keeps the task. The review reads the rewards that training recorded, so they can be one pool cycle old.

Exceptions:

- When the trainer starts a job before the stage reads the failure of the previous task job, the tasks of the new job also get a use without rewards. Their rewards in that job are not kept. This also holds with ``pool-capacity`` 0.
- When a new training start cancels a job, its tasks return to the pool only when a later job ends.

Designer replies
----------------

With ``designer-delay`` d above 0, the task jobs also train the Designer's replies, with the advantages that the generator gives them:

- The first task job of round g + d trains the replies of generation g beside its tasks. A round that published no train task passes them to the next round that did. A carrying job that fails, or that a new training start cancels, does not train them again.
- Each reply trains from the exact sample that its served call kept, so the generator must ask for it (SPADE's ``train_designer``). A reply without its sample is skipped.
- The training backend and the task jobs' objective must train delayed samples beside a task job's rollouts. Every service refuses ``designer-delay`` at its start otherwise.
- A job leaves out, with a warning in the log, the replies that the backend cannot take: replies sampled from another distribution than the job's, and replies over the number that one job takes. It also leaves out a reply that lags more than ``max-artifact-lag`` artifacts behind the served one.

See `Designer replies <../developer-guide/task-generators.rst#designer-replies>`__ for the whole flow.

Rollout job parameters
----------------------

Keys of ``recipe.config.job-parameters``. ``harbor`` takes the keys of ``rollout`` except ``rollouts_per_task``. Both accept only underscores.

.. container:: config

   ``objective``
       The method that trains the job: a registered name or a ``package.module:Objective`` reference. Required: the recipe names none.

   ``rollout.rollouts_per_task``
       The training rollouts of each task, at least 2. Required on Tinker; on Slime, ``training.options.n-samples-per-prompt`` gives the count.

   ``rollout.agent`` : terminus-2
       The Harbor agent config, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. Its requests must be text chat: system, user and assistant text messages, no tools.

   ``rollout.environment`` : docker
       The Harbor environment type of the rollouts, such as ``docker``, ``e2b`` or ``modal``.

   ``rollout.agent_host``
       The host at which an agent inside the sandbox reaches the host that runs the rollouts. Unset for an agent on that host.

   ``rollout.concurrency`` : 2
       The rollouts in flight, at least 1.

   ``rollout.verifier_timeout_reward`` : 0.0
       The reward of a rollout whose verifier timed out. ``null`` leaves the rollout unscored.

   ``rollout.scaffold_tolerance`` : 8
       The tokens at the end of a prompt that the next prompt's history may drop, such as a thinking template's scaffold. The sample keeps them.

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

With this section, ``reef serve`` starts the generator service before the HTTP service and publishes its address as ``${endpoints.generator}``. It runs under the interpreter of the Reef service. ``execution.generator`` selects its executor (`Settings reference <configuration-settings.rst#executors>`__).

The generator's host needs the ``harbor`` command and the environment that ``generator.environment`` selects. On Tinker, the training component runs the training rollouts itself, so its host needs the Harbor package and the environment of ``rollout.environment`` too.

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

The generator reads the resolved deployment: the Reef address from ``reef.host`` and ``reef.port``, the token from ``reef.token``, else from the first entry of ``reef.tokens``, and the served model from ``inference.model-path`` or ``inference.upstream-model``. Without ``reef serve``, this command runs it from the same file:

.. code:: bash

   python -m reef.deploy.generator_process -c serve.yaml
