Samples service: task generators
================================

Implement ``TaskGenerator`` with ``GeneratedTaskRecipe`` to create Harbor tasks.
Select it through ``pipeline.record2sample.config.generator``.
See `Spade <../../recipes/beta/spade>`__ for an implementation and
`Generated tasks <../reference/configuration-generated-tasks.rst>`__ for configuration.

The contract
------------

Import ``TaskGenerator`` from ``reef.recipe.generated_tasks``:

- ``from_config(config, *, scenario, environ, state)`` constructs a scenario's generator.
- ``async generate(request)`` returns a tuple of ``HarborTask`` values, possibly empty.
  Preserve ``source_agent_record_ids``; ``request.generation`` identifies the call.
- ``async validate(task)`` returns ``TaskValidationResult``.
  Empty ``errors`` means valid. Return task defects there; raise when validation cannot run.
- Optional ``async record_results(results, *, generation)`` receives candidate-order results.
  Make repeated delivery idempotent, including after restart.

Tasks are values; validate private copies.
Generators require scenario task storage and cannot publish directly.
Reef publishes valid unique tasks, then training runs them with the job's starting weights.

Results and recovery
--------------------

Results are ``RejectedTask``, ``HeldOutTask``, ``RolledOutTask`` or ``FailedTask``.
Never use held-out tasks as training or generation feedback.
Rollout results contain verifier rewards; missing rewards count as zero.
Pool reuse does not repeat generator feedback.

Generation and validation retries reuse saved candidates within configured limits.
Exhausted rounds leave source sessions available.
Published train/eval splits remain fixed; cross-split tasks are rejected.
Restart replays undelivered results. Failed delivery does not block generation.

Use ``GeneratorState.read/write/remove`` for documents and ``save_files/files``
for durable file copies. Writes survive restart on another host.
Do not change returned file copies. Archival retires state.
See `generator types <../../reef/recipe/generated_tasks/generator.py>`__ for complete signatures and result fields.
