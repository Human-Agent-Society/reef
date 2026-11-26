Samples service: task generators
================================

Implement ``TaskGenerator`` to create Harbor tasks.
Select ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` and configure
``pipeline.record2sample.config.generator``.
Use `Spade <../../recipes/beta/spade>`__ as an implementation example.

How a round runs
----------------

Reef generates candidates, validates them, then publishes unique valid tasks
with a train or eval split. Train tasks become ``TaskSample`` values.
The training component runs them with the job's starting weights.
Reef delivers results after every train task has a result.

See `Generated tasks <../reference/configuration-generated-tasks.rst>`__
for pacing, pool capacity and source selection.

The contract
------------

Import ``TaskGenerator`` from ``reef.recipe.generated_tasks``:

- ``from_config(config, *, scenario, environ, state)`` builds a generator
  per scenario with durable ``GeneratorState``.
- ``async generate(request)`` returns a tuple of ``HarborTask`` values,
  possibly empty. Preserve ``source_agent_record_ids``.
  ``request.generation`` identifies the call.
- ``async validate(task)`` returns ``TaskValidationResult``.
  Empty ``errors`` means valid. Return task defects as errors;
  raise when validation cannot run.
- Optional ``async record_results(results, *, generation)`` receives one
  result per candidate, in candidate order. Make repeated delivery idempotent.

Tasks are values: write private copies for validation.
A configured generator requires scenario task storage.
See `request and result types <../../reef/recipe/generated_tasks/generator.py>`__
for complete signatures.

Results
~~~~~~~

``GeneratedTaskResult`` has four cases:

- ``RejectedTask``: validation or storage prevented publication.
- ``HeldOutTask``: eval task. Never use it as training or generation feedback.
- ``RolledOutTask``: verifier rewards; missing rewards count as 0.
- ``FailedTask``: no rollout, failed job or no rewards.
  ``infrastructure_failure`` identifies failures that prevented execution.

Verifier timeouts use ``rollout.verifier_timeout_reward`` (default 0) and
still train. Pool reuse does not repeat generator feedback.

Durable state
~~~~~~~~~~~~~

``GeneratorState`` provides atomic ``read``, ``write`` and ``remove`` document
operations, plus sorted ``keys``. Missing documents return ``None``.
``save_files(directory)`` returns a digest; ``files(digest)`` returns its
local copy. Do not modify that copy.

Writes survive restart on another host. Archival retires this state.
Use temporary directories for transient work; generators cannot publish tasks directly.

Restart behavior
----------------

Generation and validation retry within ``generation_attempts`` and
``generation_deadline_seconds``. Retries reuse saved candidates.
Exhausted rounds produce no tasks and leave source sessions for another round.

Published splits remain fixed; shared source records get the same split.
A task joining both splits is rejected.
Restart preserves generation numbering and untrained tasks, and replays
undelivered results. Delivery failures wait for another job without blocking generation.
If training finished before its result was saved, delivery reports infrastructure failure.

``GET /reef/status`` exposes progress under ``task_generation`` and ``task_store``.

Container work
--------------

Use the `generator service <../reference/configuration-generated-tasks.rst#the-generator-section>`__
for container operations and validation rollouts. These rollouts do not train.
