Glossary
========

The vocabulary of these docs, from what a client sees to what a method
implements. Each entry is a short definition; the linked page owns the details.

Scenario
--------

One workload: its records, training state and snapshot chain, isolated from
every other scenario. ``POST /reef/scenarios``, a harness install, or the first
report or imported record with a new ``x-reef-scenario`` creates it and binds
its recipe permanently. An inference request never creates a scenario. See
`Scenarios <http-api.rst#scenarios>`__.

Agent
-----

The complete system that acts: **model + harness**. Reef improves an agent by
updating the model weights or the harness tree.

Agent record
------------

One stored exchange, either an inference call or a report, with its ID,
scenario, request type and payload. An inference record also names the
artifact that served it.

Receipt
-------

The ID of a stored inference record. Reef returns it in
``x-reef-agent-record-id`` or, for a stream, as ``reef.agent_record_id`` in the
terminal SSE metadata. Reports quote receipts in ``references``.

Report
------

A ``POST /reef/report`` request that carries feedback about one or more
exchanges. Reef consumes each report at most once, including a late or retried
report.

Feedback
--------

What you send back about a served response. A report carries it in ``score``
(a number) and ``feedback`` (a string or a structured object, such as a rubric
breakdown). Reef stores both without interpreting them; the recipe decides
what they mean.

Recipe
------

The method that a deployment runs. It binds a processor, a training
objective, an inference runtime and the components of its snapshots; a weight
recipe also binds a training component. The core ``recipe`` records without
producing updates. See `Choose a recipe <../user-guide/recipes.rst>`__.

Recipe reference
----------------

The value of ``recipe.implementation``. ``recipe`` is the core record-only
implementation. A dotted ``package.module:ClassName`` selects an installed
method class. Any other bare name resolves to a preset under
``REEF_RECIPE_CONFIG_DIR``. Reef has no global recipe registry.

Preset
------

A recipe configuration file, ``<name>.yaml`` under ``REEF_RECIPE_CONFIG_DIR``,
that ``recipe.implementation`` names. It holds the recipe's own sections. It is
not a deployment config, and ``reef serve -c`` cannot read it.

Method
------

The learning algorithm itself. A recipe binds one method to a scenario.
``HarnessEvolutionRecipe`` is one mechanism that runs many methods: each is a
``Method`` whose ``run`` edits the harness tree in a job's workspace.

Artifact
--------

One immutable output of a component, such as checkpoint weights or a harness
tree. Training, import and harness sync make artifacts. Its ``artifact_id`` is
its only public identity: treat it as an opaque string. An artifact has one of
four kinds:

- ``harness``: a whole agent harness that Reef runs as it is, such as a Harbor
  agent or an adapter's binary.
- ``harness_plugin``: a part that installs into a base harness, such as a
  harness tree that Reef evolves, a skill tree, or the configuration file of a
  ``config`` component.
- ``model``: a whole model, such as a provider's model or full weights.
- ``model_adapter``: a LoRA adapter on a base model.

Component
---------

One named part of a scenario, such as ``weights``, ``harness`` or ``skills``.
A recipe declares its components (``build_components``). The scenario selects
one artifact per component, and an update to one component keeps the other
components' artifact IDs.

Creation artifact
-----------------

The first artifact of each component, made when the scenario starts
(``operation: creation``). Reef serves it until the first activation, and it is
each trainer's first training start. A weight creation artifact holds no
optimizer state. See `Starting artifacts
<../user-guide/operate.rst#starting-artifacts>`__.

Active selection
----------------

A scenario's serving selection: a mapping from each component name to its
selected ``artifact_id``. Inference answers from the selection that completed
startup, and harness routes read the selected tree. The ``served`` setting
changes it. See `Set what a scenario serves
<../user-guide/operate.rst#set-what-a-scenario-serves>`__.

Activation rule
---------------

The deployment's rule that decides whether a new artifact serves: at once
(``always``), after an evaluation of it (``evaluated``), or when a person
activates it (``never``). Training continues from the new artifact either way.
See `The activation section <configuration.rst#the-activation-section>`__.

Training start
--------------

The artifact that a trainer's next job or evolve step starts from. It is the
creation artifact at first, and then the trainer's own newest output.
``POST /reef/scenarios/{scenario}/train-from`` sets it. The training start and
the active selection change independently. See `Train from an artifact
<../user-guide/operate.rst#train-from-an-artifact>`__.

Snapshot
--------

The repository's stored form of an artifact: its bytes and recovery metadata.
Backend interfaces locate it with ``SnapshotRef.revision``; artifact callers
use ``artifact_id``. Activation and serving publish no snapshot.

Snapshot chain
--------------

The repository history that publishes and recovers artifact snapshots. A
scenario's operation history separately records training, imports, activation,
training starts and serving changes. See `The snapshot chain
<../advanced_topics/state-model.rst#the-snapshot-chain>`__.

Chain tip
---------

The repository's publication head, which orders new snapshots. It is
independent of the active selection: activating or serving an existing
artifact does not move it.

Content ID
----------

An internal checksum of bytes or logical model state, which storage can use to
reuse content. It does not replace ``artifact_id``: two imports of the same
files create two artifacts.

Training pipeline
-----------------

The method's data side. The record-to-sample stage decides which records train
and in which groups, and turns each record set into samples. The sample-to-job
stage shapes the samples into one job's data. See `Training pipeline
<../developer-guide/write-a-recipe.rst>`__.

Reported and computed feedback
------------------------------

Two ways a signal reaches a processor. **Reported** feedback arrives
in a ``POST /reef/report``. **Computed** feedback is derived from recorded
traffic alone, with no report. The choice picks the processor engine.

Sample
------

One item of job data, a ``Sample`` (``reef.core.job_data``): a
``TrajectorySample`` that trains as it is, or a ``TaskSample``, a Harbor task
that the training component rolls out. ``job_samples`` and ``buffer_size``
count samples. A sample is not a sampling call to a model.

Staleness
---------

Whether a stored sample may still train, given the artifacts that the scenario
served after the artifact that produced it. The record-to-sample stage gives
the rule, and the samples service marks a stale sample in the store. See
`Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__.

Job data
--------

The data of one training job: a ``JobData`` value of at least
``pipeline.sample2job.config.job_samples`` samples. The training component
cuts it into optimizer steps by its own batch size.

Training job
------------

One ``TrainingJob``: its job data directory, the weights it starts from, and
the backend's parameters. The job ID is a hash of the scenario, the backend,
the start, the records of the job's data and the parameters, so a retry submits
the same job.

Training component
------------------

A ``TrainingComponent`` runs bounded training jobs with its framework's own
training loop, for example Slime or Tinker. Reef submits a job, reads its
status and can cancel it; the component saves checkpoints. See `Training
components <../developer-guide/training-components.rst>`__.

Training objective
------------------

A backend's objective (``SlimeObjective`` or ``TinkerObjective``). It declares
its loss family and prepares a job's data: advantages, metrics and algorithm
state. A recipe selects it through the job parameter ``objective``. See
`Training objective <python-api.rst#training-objective>`__.

Loss family
-----------

The backend implementation that an objective's ``loss_family`` selects. It
computes the model-dependent terms and the tensor loss. Shipped method families
include ``sao``, ``tttd`` and ``openclawrl``. See `Loss families
<../developer-guide/loss-families.rst>`__.

Runtime
-------

An ``InferenceRuntime`` executes inference requests, owns admission and loads
published snapshots for serving. It is independent of the training component:
no trainer holds a serving engine.

Runtime load ID
---------------

An engine-scoped ``<incarnation>:<sequence>`` token that names one concrete
weight load. A reload of the same artifact can produce a new one, and a restart
invalidates all of them. Runtime load IDs stay inside the engines and training;
records name the answering artifact by ``artifact_id``.

Candidate selection
-------------------

A harness trainer's decision about which candidate tree its step keeps. A
``CandidateEvaluator`` measures the candidate and a ``CandidateSelector``
selects or rejects it, under the deployment's publication policy. Selection
does not decide what serves; the activation rule does.

Publication policy
------------------

How a harness evaluation decides whether a candidate is published
(``evolution.evaluation.policy``). ``compare`` publishes a candidate that
scores better than the head tree. ``floor`` publishes one whose every completed
run reaches the floor score. ``none`` publishes every candidate that passes
validation. See `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__.

Harness
-------

The part of an agent around the model. It owns the control loop, prompts,
conversations, tools, environment integration and grading, and calls the model
through an inference runtime.

Harness tree
------------

Reef's versioned form of the files that a harness may change: config, rules,
prompt templates, skills and extension code. A tree is one directory of text
files. The adapter decides which parts a tree may hold. Reef serves the tree
over ``GET /reef/harness``. See `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__.

Adapter
-------

Two meanings. For harness evolution, the descriptor that combines a rendered
tree with an executable to make a running harness (`Harness adapters
<../developer-guide/harness-adapters.rst>`__). For weight serving, a LoRA/PEFT
adapter on a frozen base model.

Render
------

Turning a source tree, and optionally a model binding, into the files that one
agent reads. Rendering writes nothing to disk and starts no agent. See
`Rendering <../developer-guide/harness-adapters.rst#rendering>`__.

Skill
-----

A ``SKILL.md`` instruction file that the model reads, stored under ``skills/``
in a harness tree. A client pulls it with the rest of the tree. Updating one
needs no GPU.

Patch
-----

One change to a harness tree: each changed source path mapped to the file's
new text, or to ``null`` to remove the file.

Workspace
---------

The directory of one harness evolution step. It holds a copy of the head tree
(``harness/``), what the step learns from (``material/``), and the method's own
files (``state/``). The tree that the method leaves in ``harness/`` is the
step's candidate.

History
-------

Every tree that a harness scenario's steps proposed or evaluated, published or
not, with every run and every decision. It is the evaluation's score cache, and
methods read it. See `History <../user-guide/evolve-your-harness.rst#history>`__.

Task
----

What an agent is asked to do. An instruction task is text that the agent
reads. A Harbor task is a directory with an instruction, an environment and
tests that score the result.

Run
---

One run of one task by a harness tree, in one environment. It gives one
trajectory and one score: a value, 0 when the agent gave no answer, or
``failed`` when the environment could not run the task.

Rollout
-------

One run of one task by the policy model, in weight training or in a checkpoint
evaluation. It gives one trajectory and one score, as a harness run does. On
Tinker, a task job runs each task ``rollouts_per_task`` times on the job's
start weights. On Slime, ``n-samples-per-prompt`` gives the rollouts of each
task.

Training pool
-------------

The train tasks of the generated-task recipe that train again
(``pipeline.record2sample.config.pool_capacity`` above 0). A pool review
retires a task whose recent results fall outside its limits. See `Generated
tasks <configuration.rst#generated-tasks>`__.

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
