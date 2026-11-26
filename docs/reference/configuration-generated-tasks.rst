Generated tasks
===============

Use ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` to generate Harbor tasks and train on their rollouts.
Start from the `SPADE deployment <../../recipes/beta/spade/examples/tinker/serve.yaml>`__.

Set training and evaluation splits in ``pipeline.record2sample.config``. Set ``rounds: 0`` to train on previously generated tasks.
A positive ``pool-capacity`` reuses generated training tasks across jobs. Imported tasks train once.
For review thresholds and other stage options, see the `settings reference <configuration-settings.rst#recipe>`__.

Rollout job parameters
----------------------

Under ``recipe.config.job-parameters``, set an ``objective`` and the backend's rollout configuration.

- With Tinker, set ``rollout.rollouts_per_task`` to at least two.
- With Slime, use ``harbor`` for the rollout configuration and set ``training.options.n-samples-per-prompt`` for repeats.
  See `Task jobs on Slime <configuration-training.rst#task-jobs-on-slime>`__.

Both configurations use underscore keys. The agent defaults to ``terminus-2``. Its model calls must use text chat without tools.
Set ``environment`` for the sandbox provider and ``agent_host`` if a sandboxed agent needs to reach the rollout host.

The ``generator`` section
-------------------------

Install Harbor and the sandbox provider on the generator host. With Tinker, install them on the training host too.
Docker needs a running daemon. E2B needs ``reef-infra[e2b]`` and ``E2B_API_KEY``.
Modal needs ``reef-infra[modal]``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.

Reef starts the generator before HTTP. Select its executor with ``execution.generator``.
To implement a generator, see `Task generators <../developer-guide/task-generators.rst>`__.
