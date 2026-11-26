Samples service: task generators
================================

Implement ``TaskGenerator`` to create Harbor tasks for training.
Use ``reef.recipe.generated_tasks:GeneratedTaskRecipe`` and select your class
in ``pipeline.record2sample.config.generator``. No custom recipe is required.
See ``recipes/beta/spade`` for an example.

How a round runs
----------------

The samples stage runs one generation at a time on its background worker:

1. Generate candidates from the description and source sessions.
2. Save candidates and validate up to ``validation_concurrency`` tasks concurrently.
3. Publish valid, unique tasks with their train or eval split.
4. Send train tasks to the pipeline as ``TaskSample`` values.
5. Collect training rollout rewards after durable job ends.
6. Deliver results when every train task in the generation has a result.

Generation itself runs no training rollouts. The training component rolls
out tasks with the job's starting weights.
See `Generated tasks <../reference/configuration-generated-tasks.rst#generated-tasks>`__
for round pacing, pool capacity and source selection.

The contract
------------

Import ``TaskGenerator`` from ``reef.recipe.generated_tasks`` and
``TaskGenerationRequest`` from ``reef.samples.generator``.

- ``from_config(config, *, scenario, environ, state)`` builds one generator
  per scenario. ``state`` is its durable ``GeneratorState``.
- ``async generate(request)`` returns a tuple of ``HarborTask`` values,
  possibly empty. Preserve source IDs in ``source_agent_record_ids``.
  ``request.generation`` identifies the call.
- ``async validate(task)`` returns ``TaskValidationResult``.
  Empty ``errors`` means valid. Return task defects as errors;
  raise exceptions when validation could not run.
- ``async record_results(results, *, generation)`` optionally receives one
  result per candidate, in candidate order.

Tasks are values. Validation that needs files must write a private copy.
Read credentials through environment references in configuration.
A configured generator requires scenario task storage.

Generation or validation exceptions trigger retries using saved candidates,
within ``generation_attempts`` and ``generation_deadline_seconds``.
Exhausted rounds produce no tasks; the next round reuses their source sessions.

Result delivery can occur after later generations start or after restart.
Make delivery idempotent: a repeated generation must cause no additional change.
Failed delivery waits for another job without blocking later generations.

Results
~~~~~~~

``GeneratedTaskResult`` has four cases:

- ``RejectedTask(task, reason)``: validation or storage conflict prevented publication.
- ``HeldOutTask(task)``: an eval task. Never use it as training feedback
  or experience for future generation.
- ``RolledOutTask(task, rollout_rewards)``: training rollout verifier rewards.
  Missing rewards count as 0.
- ``FailedTask(task, error, infrastructure_failure)``: no rollout ran,
  the job failed or it reported no rewards. The flag identifies failures
  that prevented the task from running.

Verifier timeouts use ``rollout.verifier_timeout_reward``, default 0,
and still train. Training pools can reuse tasks, but generator feedback
contains only each task's first rollout result.

Durable state
~~~~~~~~~~~~~

Use ``GeneratorState`` for anything needed after restart:

- ``read(key)`` returns a document or ``None``.
- ``write(key, document)`` atomically replaces a document.
- ``remove(key)`` removes it; unknown keys are harmless.
- ``keys(prefix="")`` lists matching keys in sorted order.
- ``save_files(directory)`` saves a copy and returns its digest.
- ``files(digest)`` returns a local directory. Do not change its files.

Writes are durable when they return. Database documents and task-store files
remain available after restart on another machine.
Scenario archival retires this state. The generator cannot publish tasks
directly. Use temporary directories for validation and other transient work.

Restart behavior
----------------

Reef saves candidates, publications and rejections. Retries reuse candidates
instead of calling ``generate`` again.
Tasks sharing source records keep the same split. Published splits are fixed;
a task joining both splits is rejected.

Restart continues generation numbering and retains untrained task items.
Undelivered results are replayed. If a task trained before its result was
saved, Reef reports infrastructure failure to complete its generation.

``GET /reef/status`` exposes generation progress and errors under
``task_generation``, and task counts under ``task_store``.

Container work
--------------

Use the generator service for container operations or validation rollouts.
Configure it through the
`generator section <../reference/configuration-generated-tasks.rst#the-generator-section>`__.
Its rollouts are not records and do not train.
