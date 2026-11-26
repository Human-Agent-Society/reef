Glossary
========

Terms used by Reef clients and method authors. Linked guides provide details.

Scenario
--------

An isolated workload: records, training state, snapshot chain and a fixed recipe. See
`Scenarios <http-api-scenarios.rst#http-api-scenarios-and-their-settings>`__.

Agent
-----

**Model + harness**. Reef updates either model weights or harness files.

Agent record
------------

A stored inference or report: ID, scenario, request type and payload.
Inference records also name the responding artifact.

Receipt
-------

A stored inference record's ID, returned in ``x-reef-agent-record-id`` or
terminal SSE ``reef.agent_record_id``. Reports use receipts in ``references``.

Report
------

Feedback about exchanges, submitted through ``POST /reef/report`` and consumed
at most once, including late reports and retries.

Feedback
--------

A report's ``score`` (number) and ``feedback`` (string or object).
The recipe defines their meaning.

Recipe
------

Configuration for pipeline stages, an objective and artifact components.
Services build runtimes and training components. Core ``recipe`` only records.
See `Choose a recipe <../user-guide/recipes.rst>`__.

Recipe reference
----------------

``recipe.implementation``: ``recipe`` for recording, ``package.module:ClassName``
for an installed class, or a preset name under ``REEF_RECIPE_CONFIG_DIR``.
There is no global recipe registry.

Preset
------

A recipe configuration file, ``<name>.yaml`` under ``REEF_RECIPE_CONFIG_DIR``.
Select it through ``recipe.implementation``, not ``reef serve -c``.

Method
------

A recipe's learning algorithm. Harness methods implement ``Method.run``
and edit the job workspace's harness tree through ``HarnessEvolutionRecipe``.

Artifact
--------

A component's immutable output, created by training, import or harness sync.
Its only public identity is opaque ``artifact_id``. Kinds:

- ``harness``: a complete executable harness.
- ``harness_plugin``: installed harness parts, such as trees, skills or config.
- ``model``: full weights or a provider model.
- ``model_adapter``: LoRA weights for a base model.

Component
---------

A scenario part, such as ``weights`` or ``harness``, declared through
``build_components``. Each selects an artifact independently; other components
keep their IDs when one changes.

Creation artifact
-----------------

The initial active artifact and training start (``operation: creation``).
Weight creation artifacts have no optimizer state. See `Starting artifacts
<../user-guide/operate.rst#starting-artifacts>`__.

Active selection
----------------

A component-to-artifact mapping used by harness routes and inference after
startup. Change it through ``served``. See `Set what a scenario serves
<../user-guide/operate.rst#set-what-a-scenario-serves>`__.

Activation rule
---------------

A rule selecting artifacts for serving: ``always``, ``evaluated`` or ``never``
(manual activation). Training can continue without activation.
See `The activation section <configuration-recipe.rst#the-activation-section>`__.

Training start
--------------

A trainer's starting artifact, initially its creation artifact and then its
latest output. It changes independently of serving through ``train-from``.
See `Train from an artifact
<../user-guide/operate.rst#train-from-an-artifact>`__.

Snapshot
--------

Stored artifact bytes and recovery metadata, located internally by
``SnapshotRef.revision``. Callers use ``artifact_id``. Activation creates no snapshot.

Snapshot chain
--------------

Repository history for publishing and recovering snapshots. Operation history
separately tracks training, imports and selection changes. See `The snapshot chain
<../advanced_topics/state-model.rst#the-snapshot-chain>`__.

Chain tip
---------

The repository publication head. New snapshots advance it; serving or activating
existing artifacts does not.

Content ID
----------

An internal checksum for reusing content, separate from artifact identity.
Two direct URI imports can share content but have different artifact IDs.

Training pipeline
-----------------

Record-to-sample selects and groups records into samples. Sample-to-job
assembles them into job data.
See `Training pipeline
<../developer-guide/write-a-recipe.rst>`__.

Reported and computed feedback
------------------------------

``ReportSamples`` reads report feedback. ``JudgedSamples`` computes feedback
from traffic without reports.

Sample
------

One ``Sample`` (``reef.core.job_data``): a ``TrajectorySample`` containing
an `ATIF document
<https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md>`__,
for training, or a ``TaskSample`` naming a `Harbor task directory
<https://www.harborframework.com/docs/tasks>`__ to run.
``job_samples`` and ``buffer_size`` count samples, not model calls.

Staleness
---------

Whether a sample can train after newer artifacts served. Record-to-sample
defines the rule; the samples service marks stale samples. See
`Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__.

Job data
--------

A ``JobData`` value containing one job's samples. Automatic jobs wait for
``job_samples``; requests select samples. Backend batch size determines optimizer steps.

Training job
------------

A data directory, starting weights and backend parameters. ``TrainingJob`` IDs
hash the scenario, backend, starting weights, job data records and parameters.
Retries submit the same job.

Training component
------------------

A ``TrainingComponent`` runs bounded jobs and saves checkpoints.
Reef submits, monitors and cancels them.
See `Training
backends <../developer-guide/training-backends.rst>`__.

Training objective
------------------

``SlimeObjective`` or ``TinkerObjective``: selects a loss family and prepares
advantages, metrics and algorithm state. Recipes choose it through ``objective``. See
`The objective <../developer-guide/recipe-training-service.rst#the-objective>`__.

Loss family
-----------

The implementation selected by ``loss_family`` to compute model-dependent
terms and tensor loss. See `Loss families
<../developer-guide/loss-families.rst>`__.

Runtime
-------

An ``InferenceRuntime`` admits inference and loads snapshots for serving,
independently of training components.

Runtime load ID
---------------

An internal ``<incarnation>:<sequence>`` token for one weight load. Reloading can
change it; restarting invalidates it. Records expose ``artifact_id`` instead.

Candidate selection
-------------------

``CandidateEvaluator`` measures a tree; ``CandidateSelector`` accepts or rejects
it under the publication policy. Activation separately decides serving.

Publication policy
------------------

``evolution.evaluation.policy``: ``compare`` requires improvement over the head;
``floor`` requires every completed run to reach the threshold; ``none`` requires
validation only. See `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__.

Harness
-------

The agent's control loop, prompts, conversations, tools, environment integration
and grading. It calls the model through inference.

Harness tree
------------

A versioned directory served by ``GET /reef/harness``. The adapter defines allowed
config, rules, prompts, skills and code. See `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__.

Adapter
-------

- **Harness:** a descriptor combining a rendered tree with an executable
  (`Harness adapters <../developer-guide/harness-adapters.rst>`__).
- **Weights:** a LoRA/PEFT adapter on a frozen base model.

Render
------

Converting a source tree and optional model binding into agent files, without
writing to disk or starting an agent. See
`Rendering <../developer-guide/harness-adapters.rst#rendering>`__.

Skill
-----

A model-readable ``SKILL.md`` under the harness tree's ``skills/`` directory.

Patch
-----

A mapping from harness source paths to new text, or ``null`` for deletion.

Workspace
---------

A harness step's directory: candidate tree in ``harness/``, learning input in
``material/``, and method files in ``state/``.

History
-------

Proposed harness trees, runs and decisions, including unpublished candidates.
Methods read it; evaluation reuses cached scores.
See `History <../user-guide/evolve-your-harness.rst#history>`__.

Task
----

An agent's assignment: instruction text, or a Harbor directory containing
instructions, an environment and scoring tests.

Run
---

One task executed by a harness in an environment. Produces a trajectory and
score: numeric, 0 for no answer, or ``failed`` for environment failure.

Rollout
-------

A policy's task run during training or evaluation. Tinker uses
``rollouts_per_task`` repeats on starting weights; Slime uses ``n-samples-per-prompt``.

Training pool
-------------

Reusable generated tasks, enabled by positive ``pipeline.record2sample.config.pool_capacity``.
Reviews retire tasks whose recent results violate limits. See `Generated
tasks <configuration-generated-tasks.rst#generated-tasks>`__.

SAO
---

Single-Rollout Asynchronous Optimization (`arXiv:2607.07508
<https://arxiv.org/abs/2607.07508>`__), the cookbook `sao
<../user-guide/recipes/sao.rst>`__ recipe. One graded rollout drives one
training step, with no comparison group.

TTT-Discover
------------

Test-time-training discover (`arXiv:2601.16175
<https://arxiv.org/abs/2601.16175>`__), the cookbook `tttd
<../user-guide/recipes/tttd.rst>`__ recipe. A model specializes to one hard
problem during the search; the search loop stays in the harness.

OpenClaw-RL
-----------

The cookbook `openclawrl <../user-guide/recipes/openclawrl.rst>`__ recipe
(`arXiv:2603.10165 <https://arxiv.org/abs/2603.10165>`__). It trains an
unmodified agent from live conversation traffic with a next-state binary
reward, without an external grader or a training report.
