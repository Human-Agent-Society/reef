Generated tasks
===============

Use ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` to generate Harbor tasks and train on their rollouts.
Start from the `SPADE deployment <../../recipes/beta/spade/examples/tinker/serve.yaml>`__.

Set training and evaluation splits in ``pipeline.record2sample.config``. Set ``rounds: 0`` to train on previously generated tasks.
A positive ``pool-capacity`` reuses generated training tasks across jobs. Imported tasks train once.
For review thresholds and other stage options, see the `settings reference <configuration-settings.rst#recipe>`__.

Designer replies
----------------

With ``designer-delay`` d above 0, the first task job of round g + d also trains generation g's Designer replies with the generator's advantages.
A round that published no train task passes them to the next round that did.
A reply trains at most once, from the exact sample that its served call kept; SPADE's ``train_designer`` asks for it.
A carrying job that fails or is cancelled does not train them again.
The last d generations of a run train only when the run continues with more rounds.
Every service refuses ``designer-delay`` at start unless the training backend and objective train delayed samples.
A job drops, with a warning, replies of another sampling distribution, replies over its limit, and replies more than ``max-artifact-lag`` artifacts behind the served one.
See `Designer replies <../developer-guide/task-generators.rst#designer-replies>`__ for the generator side.

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
