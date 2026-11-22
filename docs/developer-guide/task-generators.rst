Samples service: task generators
================================

A task generator writes Harbor tasks for a scenario to train on. It plugs
into the built-in generated-task recipe, whose record-to-sample stage runs
in the samples service. This page defines the ``TaskGenerator`` contract and
what Reef does with the tasks.

:For: authors of a task generator
:Before you start: the `record-to-sample stage <recipe-samples-service.rst>`__, and the keys in `Generated tasks <../reference/configuration-generated-tasks.rst#generated-tasks>`__
:You will have: a generator class, its durable state, and the results that Reef sends back to it

A generator needs no recipe of its own. The deployment selects
``reef.recipe.generated_tasks:GeneratedTaskRecipe`` and names the generator
class in ``pipeline.record2sample.config.generator``. The recipe does not
depend on the generator's method, so a team can replace the generator
without a new recipe. SPADE (``recipes/beta/spade``) is one generator.

How a round runs
----------------

The recipe's stage, ``GeneratedTaskSamples``, runs task generation in
rounds, on a background worker of its own. One generation runs at a time. A
generation runs no rollouts:

1. The stage calls ``generate`` with the configured ``description`` and the
   round's source sessions.
2. It saves the candidates in the scenario's task store. Then it calls
   ``validate`` for each one, ``validation_concurrency`` at once.
3. It publishes each valid task with its train or eval split. A task
   identical to a published one is rejected as a duplicate.
4. It hands each train task to the pipeline as a ``TaskSample``
   (``made_samples``). The task trains like every other sample: the training
   component rolls it out on its own sampler, at the job's start weights.
5. When the commit of a job's end is durable, the stage reads the job's
   ``rollout_rewards`` (``job_ended``) and saves each task's result.
6. When every train task of a generation has a result, the stage calls
   ``record_results``.

`Generated tasks <../reference/configuration-generated-tasks.rst#generated-tasks>`__ lists
the keys that pace the rounds (``rounds``, ``jobs_per_round``,
``refill_tasks``, ``pool_capacity``) and that select source sessions
(``source_records_per_round``, ``source_tag``).

The contract
------------

``TaskGenerator`` is in ``reef.recipe.generated_tasks``, and
``TaskGenerationRequest`` in ``reef.record2sample.generator``. A generator
gets what it needs from its ``config`` and from these calls only.

- ``from_config(config, *, scenario, environ, state)``: class method,
  required. Reef calls it once for each scenario. ``state`` is the
  scenario's ``GeneratorState``.
- ``async generate(request) -> tuple[HarborTask, ...]``: required. Return
  zero or more tasks. Keep the request's source record IDs in each task's
  ``source_agent_record_ids``. ``request.generation`` numbers the call.
- ``async validate(task) -> TaskValidationResult``: required. Empty
  ``errors`` means that the task passed. Return task defects as errors. The
  task is a value: a check that needs a directory writes its own copy.
- ``async record_results(results, *, generation)``: optional. Receive one
  ``GeneratedTaskResult`` for each candidate of the ``generate`` call
  numbered ``generation``, in candidate order.

**Failure behavior.**

- An exception from ``generate`` or ``validate`` is a failure to run, not a
  task defect. The stage retries the generation from the saved candidates,
  up to ``generation_attempts`` times within
  ``generation_deadline_seconds``. After that the round ends without a task,
  and the next round takes the same sessions again.
- ``record_results`` can run after later generations started, and after a
  restart. A crash before Reef records the delivery sends the same results
  again. A second call for one generation must change nothing.
- A delivery that fails waits for the next job; later generations still go
  out.
- A config that names a generator is refused when the scenario has no task
  storage.

Results
~~~~~~~

Each ``GeneratedTaskResult`` is one of four cases:

- ``RejectedTask(task, reason)``: the candidate failed validation or
  conflicted with the task store. Reef never published it.
- ``HeldOutTask(task)``: the task is in the eval split and runs no rollouts.
  Do not use it as training feedback or as experience for later
  generations.
- ``RolledOutTask(task, rollout_rewards)``: a job trained the task.
  ``rollout_rewards`` holds one verifier reward for each training rollout
  that ran. A rollout without a reward counts as 0.
- ``FailedTask(task, error, infrastructure_failure)``: no rollout ran.
  ``infrastructure_failure`` is true when the task could not run at all,
  for example because the sandbox failed. Reef also sends it for a task
  whose job failed or reported no rewards.

The training component scores a rollout whose verifier ran past its timeout
with ``rollout.verifier_timeout_reward``, 0 by default, so such a rollout
trains.

Durable state
~~~~~~~~~~~~~

``state`` is the generator's ``GeneratorState``. Keep there what the
generator must read again after a restart, such as its open generations.
Keep no file of your own.

- ``read(key)`` returns the document of ``key``, or ``None``.
- ``write(key, document)`` saves the document. It replaces an earlier
  document of the key.
- ``remove(key)`` removes the document. An unknown key changes nothing.
- ``keys(prefix="")`` returns the keys that start with ``prefix``, in
  increasing order.
- ``save_files(directory)`` saves a copy of a directory and returns the
  digest of its files.
- ``files(digest)`` returns a local directory with the saved files. Do not
  change it.

Each write is durable and atomic when it returns. The documents are in the
deployment's database and the saved directories in the task file store, so a
restart on another machine reads the same state. An archive of the scenario
retires it. The generator gets no other part of the task store: it cannot
publish a task. For short-lived work, such as a task under validation, use a
temporary directory.

Restart behavior
----------------

- The stage records each publication and rejection. A retry of a generation
  uses the saved candidates and does not call ``generate`` again.
- Tasks that share a source record get the same split, and a published task
  keeps its split. A task whose sources join tasks of both splits is
  rejected.
- After a restart, the stage continues with the next generation number. The
  scenario still uses the task items that did not train, and a later job
  takes them.
- The stage sends the results that the generator did not receive. A task
  that trained before its result was saved gets an infrastructure-failure
  result, so its generation's feedback completes.

``GET /reef/status`` shows the generator, the generation in flight, the
completed generations and the last error under ``task_generation``, and the
task counts under ``task_store``.

Container work
--------------

Operations that need containers run in the generator service
(``reef.record2sample.generator``), which ``reef serve`` starts from the
``generator`` section (see `generator configuration
<../reference/configuration-generated-tasks.rst#the-generator-section>`__). A generator can
run rollouts of a task through that service, as SPADE does. Those rollouts
are not records and do not train.
