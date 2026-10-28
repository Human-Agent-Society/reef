Glossary
========

The vocabulary used throughout these docs, from what a client sees to what a
method implements.

Scenario
--------

One workload: its records, training state, and release chain, isolated from
every other scenario. The first request carrying a new ``x-reef-scenario``
creates it and permanently binds its recipe. The request may also bind a
starting release, whose content becomes the scenario's creation artifact.

Agent
-----

The complete system that acts: **model + harness**. The model supplies the
learned behavior; the harness supplies the control loop and context around it.
Reef improves an agent by updating either model weights or the harness tree.

Agent record
------------

One stored exchange, either an inference call or a report, with its id, scenario,
request type, payload, and, for inference, the release that served it.
The record store lives under ``agent_record_dir``.

Receipt
-------

The id of a stored inference record, returned in ``x-reef-agent-record-id`` or,
for a stream, as ``reef.agent_record_id`` in the terminal SSE metadata. It names
the exchange *and* the release that produced it. Reports quote receipts
in ``references``.

Report
------

A ``POST /reef/report`` request carrying feedback about one or more exchanges.
Reef consumes each report at most once, including when it arrives late or is
retried.

Feedback
--------

What you send back about a served response. A report carries it in two fields:
``score``, the numeric channel, and ``feedback``, a string or structured object
such as a rubric breakdown. Reef's core stores both without interpreting them;
the recipe decides what they mean.

Recipe
------

The method a deployment runs. It binds a processor, a training objective, an
inference runtime, and a surface; a weight recipe also binds a training
component. The core ``recipe`` records without producing updates.

Recipe reference
----------------

What ``recipe.implementation`` selects. ``recipe`` is the core record-only implementation;
a dotted ``package.module:ClassName`` selects an installed method class; any
other bare name resolves only to a YAML preset under
``REEF_RECIPE_CONFIG_DIR``. Reef has no global recipe-implementation registry.

Artifact
--------

The content of one component of a release, such as a checkpoint's weights or
a harness tree. Training makes artifacts. An artifact is of exactly one of
four kinds:

- ``harness``: a whole agent harness that Reef runs as it is, such as a Harbor
  agent or an adapter's binary.
- ``harness_plugin``: a part that installs into a base harness, such as a
  harness tree that Reef evolves, a skill tree, or the request defaults of a
  ``config`` component.
- ``model``: a whole model, such as a provider's model or full weights.
- ``model_adapter``: a LoRA adapter on a base model.

``artifact_id`` identifies an artifact. In a scenario of one component it is
the ``release_id`` of the release that made the artifact; in a scenario of
several components it is ``<release_id>:<component>``. ``content_id``
identifies the contents independently of when or why Reef published them.

Release
-------

One immutable version of a scenario's content, which binds one artifact for
each component. A release is the release that created the scenario, the
output of one training step, or, in a scenario of several components, an
activation. ``release_id`` identifies it in the scenario history and
``parent_release_id`` identifies its parent. Making a release changes nothing
else: whether it serves is a separate decision. History is appended, never
rewound.

Creation artifact
-----------------

The first artifact of each component of a scenario: the content of the
release that created the scenario. Its row in the artifact list has
``operation: creation``. Reef serves it until the first activation. It is
each trainer's first training start. A weight creation artifact holds no
optimizer state, so training from it starts with a new optimizer state.

Active release
--------------

The release that a scenario serves. It binds the active artifact of each
component. Inference answers from it, and the harness routes serve its tree.
``POST /reef/scenarios/{scenario}/activate`` makes another existing artifact
active. By default Reef activates each new artifact that passes its checks.
Every other artifact is inactive. An artifact that a training step made and
that was never active waits for activation. See `Operate a deployment
<../user-guide/operate.rst#activate-an-artifact>`__.

Chain tip
---------

The newest release on a scenario's release chain: the parent of the next
release that a training step publishes. A training step whose artifact Reef
activates moves the chain tip to its release. An inactive artifact's release,
and the release that an activation publishes in a scenario of several
components, do not move it. The chain tip and the active release can differ.
See `The release chain <../advanced_topics/state-model.rst#the-release-chain>`__.

Training start
--------------

The artifact that a trainer's next job or evolve step starts from. It is the
creation artifact until the trainer first trains. After each job or step the
trainer moves its start to its own newest output.
``POST /reef/scenarios/{scenario}/train-from`` sets it to an existing
artifact. The training start and the active release change independently. See
`Train from an artifact <../user-guide/operate.rst#train-from-an-artifact>`__.

Content ID
----------

The identity of the bytes or logical model state of a release. Equal
``content_id`` values mean equal content even when the ``release_id`` values
differ.

Release chain
-------------

A scenario's history of releases, each release carrying its parent. An
artifact of the chain can be activated when its surface and runtime can serve
it, and a trainer can start from it when the trainer can load it.

Component
---------

One named part of a release's content, such as ``weights``, ``harness``, or
``skills``. A recipe's surface declares the components its releases bind and
how each is admitted, loaded, injected, and read. A one-component release is
flat; a release with several keeps one directory per component and a manifest
naming each component's ``content_id``.

Candidate selection
-------------------

A harness trainer's decision about which candidate tree its search keeps. A
``CandidateEvaluator`` measures the candidate, a ``CandidateSelector`` makes
the select-or-reject decision, and a ``CandidateEvaluationPlugin`` implements
both. A harness recipe names it in ``evolution.selection``. Selection does
not decide what serves; the activation rule does.

Activation rule
---------------

The deployment's rule that decides whether a new artifact serves: at once
(``always``), after an evaluation of it (``evaluated``), or when a person
activates it (``never``). Training continues from the new artifact either
way. See `Configuration <configuration.rst#the-activation-section>`__.

Surface
-------

How a published artifact reaches whoever uses it: runtime loading, inference
hooks, and an optional client-pulled file tree, composed as fields on one
``Surface`` value.

Training pipeline
-----------------

The method's data side. Three stages turn records into typed job data: the
record-to-record stage decides which records train, the record-to-sample stage
turns each record set into samples, and the sample-to-job stage shapes the samples
into one job's data. Reported feedback uses valid reports with existing
inference references; computed feedback derives its signal from traffic.
Reef's ``TrainingPipeline`` drives the stages and owns job size,
consumption, and retention.

Sample
------

One item of job data, a ``Sample`` (``reef.core.job_data``): a
``TrajectorySample`` that trains as it is, or a ``TaskSample``, a Harbor task that
the training component rolls out. The record-to-sample stage makes samples;
``job_samples`` and the sample buffer count them. A sample is not a sampling
call to a model.

Training objective
------------------

The method-owned ``TrainingObjective`` declares its backend loss family and
prepares a job's complete reserved data. Its ``prepare`` method returns advantages,
metrics, and proposed algorithm state in a ``TrainingSignal`` before optimizer or
worker partitioning. It imports no training stack. A recipe selects it by
registered name or dotted class/instance path. The training component cuts
each job's data into optimizer steps. The objective declares the rows that one step
keeps together (``rollout_unit``) and rejects a job's step schedule that its
loss cannot train.

Loss family
-----------

The backend implementation selected by ``TrainingObjective.loss_family`` and
exposed through ``WeightTrainingSpec.loss_family``. It computes model-dependent
terms and the tensor loss. Shipped method families include ``sao``, ``tttd``,
and ``openclawrl``.

Harness
-------

The part of an agent around the model. It owns the control loop, prompts,
conversations, tools, environment integration, and grading, and calls the model
through an inference runtime.

Harness tree
------------

Reef's versioned representation of the mutable files in a harness: config,
rules, prompt templates, skills, extension code, and, for the ``native``
adapter, the loop's own tools, hook listeners, and control flow graph. Reef
serves it over
``GET /reef/harness``. An adapter combines the rendered tree with a harness
executable; that harness and its configured model form the running agent.

Render
------

Turning a harness tree into the files one agent uses: harness tree entries in,
paths and their text out. This process does not touch disk or start an agent.

Reef's shared engine and the adapter own it together. The engine goes first,
merging and writing by rules that are the same for every agent and rejecting a
tree that breaks them. The adapter gets the last word: its paths decide where
each file goes, and its quirks module may rewrite, drop, or refuse what the
engine produced before an agent ever reads it.

Skill
-----

A ``SKILL.md`` instruction file the model reads, stored under ``skills/`` in a
harness tree. Reef can inject one into each request or let a client pull it with
the rest of the tree. Updating one needs no GPU.

Runtime
-------

``InferenceRuntime`` executes inference requests, owns admission, and loads
published releases for serving. It loads only weights that a
training job saved and that Reef published. It is independent of the
training component: no trainer holds a serving engine.

Training component
------------------

A ``TrainingComponent`` runs bounded training jobs with its framework's own
training loop, for example Slime or Tinker. Reef submits a job, reads its
status, and can cancel it; the component saves checkpoints. A
``SavedCheckpoint`` names the directory a later job starts from and the
directory Reef publishes as a release.

Training job
------------

One ``TrainingJob``: its job data, the weights it starts from, and the backend's
parameters. The recipe's ``JobPlan`` chooses the parameters of each job. The
job ID is a hash of the scenario, the backend, the start, the records of the
job's data and the parameters, so a retry submits the same job.

Job data
--------

The data of one training job: a ``JobData`` value of samples that the
sample-to-job stage makes. A job's data holds at least
``pipeline.sample2job.config.job_samples`` samples. The training component cuts
it into optimizer batches by its own batch size.

SAO
---

Single-Rollout Asynchronous Optimization (`arXiv:2607.07508
<https://arxiv.org/abs/2607.07508>`__), the cookbook `sao <../user-guide/recipes/sao.rst>`__
recipe. One graded rollout drives one training step, with no comparison group
and no slowest-sample barrier.

TTT-Discover
------------

Test-time-training discover (`arXiv:2601.16175
<https://arxiv.org/abs/2601.16175>`__), the cookbook `tttd <../user-guide/recipes/tttd.rst>`__
recipe. A model specializes to one hard problem during the search; the search
loop stays in the harness.

OpenClaw-RL
-----------

The cookbook `openclawrl <../user-guide/recipes/openclawrl.rst>`__ recipe (`arXiv:2603.10165
<https://arxiv.org/abs/2603.10165>`__). It trains an unmodified agent from live
conversation traffic using a next-state binary reward, with no external
grader or training report. A harness-provided ``x-reef-tag-session`` value is
the supported way to correlate a conversation; it stays stable within that
conversation and is not reused for another one in the same scenario.
Transcript matching is the fallback when the tag is absent.

Runtime load ID
---------------

An engine-scoped ``<incarnation>:<sequence>`` token naming the concrete weight
load that produced a span of generated tokens. It is a serving fence, not a
publication or content identity. Reloading the same ``content_id`` can produce
a new runtime load ID, for example when an older release is activated
again. A restart invalidates runtime load IDs but not the release chain.

Staleness
---------

The number of versions that a scenario served after the runtime load that
produced a sample, bounded by the record-to-record stage's ``max_staleness``
(``pipeline.record2record.config.max_staleness``). Zero requires a sample to
train against the weights that generated it. The record-to-record stage checks
it before job data forms.

Reported and computed feedback
------------------------------

Two ways a signal reaches a processor. **Reported** feedback arrives explicitly,
in a ``POST /reef/report``. **Computed** feedback is reconstructed by the
processor from recorded traffic alone, with no report on the wire. The choice
picks the processor engine.


Method
------

The learning algorithm itself. A recipe is the object that binds one method to a
scenario; ``harness_evolve`` is one mechanism running many possible methods.

Adapter
-------

Two meanings. For harness evolution, the descriptor that combines a rendered
tree with an executable to make a running harness (`Harness adapters
<../developer-guide/harness-adapters.rst>`__). For weight
serving, a LoRA/PEFT adapter layered on a frozen base model.

Rollout
-------

One run of one task by an agent, a harness and a model, in one environment. It
goes from the task's first step to its scored end, and gives one trajectory and
one score. The score is a value, 0 when the agent gave no answer, or
``failed`` when the environment could not run the rollout. In each
step, harness evolution runs each evaluation task ``rollout_repeats`` times on
each side. A Tinker task job runs each task ``rollouts_per_task`` times on the
job's start weights.

Mutation
--------

One proposed edit to a harness tree: ``create``, ``update``, or ``remove`` on a
single root-level entry. A sequence applies as one composite proposal under one
selection result.

Preset
------

A recipe configuration file, ``<name>.yaml`` under ``REEF_RECIPE_CONFIG_DIR``,
named by ``recipe.implementation``. It carries the recipe's own sections; it is not a
deployment config and ``reef serve -c`` cannot read it.
