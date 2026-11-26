Generated tasks
===============

Use ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` to generate Harbor tasks and train on their rollouts.

Start from the `SPADE deployment <../../recipes/beta/spade/examples/tinker/serve.yaml>`__.
The `settings reference <configuration-settings.rst#recipe>`__ lists the ``GeneratedTaskSamples`` stage options.

Rounds and tasks
----------------

- Every job trains tasks. Startup requires ``objective`` and the backend's rollout parameters: ``rollout`` for Tinker or ``harbor`` for `Slime <configuration-training.rst#task-jobs-on-slime>`__.
- With ``rounds: 0``, the recipe trains only tasks saved by earlier generations.
- A source session is a report whose inference records all carry ``source-tag``. Each Designer request accepts at most 512 KiB of records and files as JSON. Reduce ``source-records-per-round`` for long sessions.
- Set ``eval-fraction`` and ``split-seed`` in ``pipeline.record2sample.config``, not in the generator's ``config``.
- For a custom generator, see `the generator contract <../developer-guide/task-generators.rst#the-contract>`__.

The training pool
-----------------

With ``pool-capacity`` above zero, generated training tasks return to the queue after each job.
Previously trained tasks join when you enable the pool. Imported tasks train once.
Reef checks capacity at round start, so one generation can exceed the limit. Inspect counts under ``training_pool``.

Every ``review-interval`` jobs, including failures, Reef checks each task's last ``review-window`` rewards.
Tasks with fewer rewards stay active. Reef retires tasks in this order:

#. Mean below ``review-band-low`` (``below_band``).
#. Mean above ``review-band-high`` (``above_band``).
#. Population standard deviation below ``review-min-reward-std`` (``no_reward_spread``).

Rewards can lag one pool cycle. A job started before Reef reads the previous failure counts as a use without rewards, even with pooling disabled.
When a new training start cancels a job, its tasks return after a later job ends.

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

For self-hosted E2B, set ``E2B_API_URL``, ``E2B_SANDBOX_URL`` and ``E2B_API_KEY`` on the rollout host.
Slime forwards these and ``E2B_DOMAIN`` to workers. On AgentENV also install ``aenv`` and ``aenv-buildctl`` and set
``REEF_E2B_DOCKERFILE_BUILD=aenv`` in each rollout process. Slime does not forward that variable.
See `sandbox lifetime and cleanup <../developer-guide/sandboxes.rst#harbor-task-environments>`__.

Read ``rollout_rewards`` for task rewards and ``rollout_samples`` for trainable samples.
``removed_rollouts`` counts infrastructure and sample assembly failures.
Tinker reports per-rollout and total tokens in ``rollout_tokens``, ``rollout_response_tokens`` and ``rollout_trained_tokens``.
Overlong samples count as ``unrenderable_samples``. A singleton group contributes neither a trained sample nor a removed rollout.

The ``generator`` section
-------------------------

Reef starts the generator before HTTP and publishes its URL as ``${endpoints.generator}``.
Select its executor with ``execution.generator``. Install Harbor and the environment below in the generator's Python environment.
On Tinker, the training host also needs Harbor and its rollout environment.

- ``docker`` (default) requires a Docker daemon.
- ``e2b`` requires ``reef-infra[e2b]`` and ``E2B_API_KEY``. For multi-container tasks see `Compose on AgentENV <../developer-guide/sandboxes.rst#tasks-with-compose-services-on-agentenv>`__.
- ``modal`` requires ``reef-infra[modal]``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.

To start the generator separately with the deployment's address, model and token:

.. code:: bash

   python -m reef.deploy.generator_process -c serve.yaml
