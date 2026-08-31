Glossary
========

The vocabulary used throughout these docs, from what a client sees to what a
method implements.

Scenario
--------

One workload: its records, training state, and version chain, isolated from
every other scenario. The first request carrying a new ``x-reef-scenario``
creates it and permanently binds its recipe. The request may also bind a
starting artifact version.

Agent record
------------

One stored exchange, either an inference call or a report, with its id, scenario,
request type, payload, and, for inference, the artifact version that served it.
The record store lives under ``agent_record_dir``.

Receipt
-------

The id of a stored inference record, returned in ``x-reef-agent-record-id`` or,
for a stream, as ``reef.agent_record_id`` in the terminal SSE metadata. It names
the exchange *and* the artifact version that produced it. Reports quote receipts
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

The method a deployment runs. It binds a processor, a step preparer, a loss
family, a runtime, and a surface. The core ``recipe`` records without
producing updates.

Recipe reference
----------------

What ``reef.recipe`` selects. ``recipe`` is the core record-only implementation;
a dotted ``package.module:ClassName`` selects an installed method class; any
other bare name resolves only to a YAML preset under
``REEF_RECIPE_CONFIG_DIR``. Reef has no global recipe-implementation registry.

Artifact
--------

The versioned thing: a model checkpoint, or a tree of text files such as a
harness or skill set. An ``ArtifactRef`` identifies one version. Between
checkpoints a live weight artifact exists only in engine memory; a durable one
stores its bytes in Git.

Version chain
-------------

A scenario's history of accepted updates, each version carrying its parent. A
durable version can be restored when its surface and runtime support rollback.

Candidate selection
-------------------

The decision whether a produced update should replace the current served
version. A ``CandidateEvaluator`` measures the candidate, a
``CandidateSelector`` makes the select-or-reject decision, and a
``CandidateEvaluationPlugin`` implements both. The docs also call this the
gate. Every recipe carries the plugin as ``candidate_evaluation``. It is
independent of checkpoint cadence and retention.

Surface
-------

How a published artifact reaches whoever uses it: runtime loading, inference
hooks, and an optional client-pulled file tree, composed as fields on one
``Surface`` value.

Processor
---------

The method's data-side component. It judges each resolved unit, consisting of
one record plus the reports referencing it, as ``TRAIN``, ``WAIT``, or
``NEVER``, and assembles
the accepted units into one typed batch. Reported and computed feedback pick
different engines.

Preparer
--------

The function that converts a reserved batch into a ``StepSignal`` containing
the loss family, advantages, and next algorithm state. It is backend neutral
and imports no training stack. A recipe names it by registered name or dotted
path.

Loss family
-----------

The tensor objective the training backend runs, declared by
``WeightTrainingSpec.loss_family``. Separate from the preparer. Bundled:
``sao``, ``opd``, ``tttd``, ``openclawrl``.

Harness
-------

Two meanings, both in use here.

1. **Your harness:** the agent program around the model, owning prompts,
   conversations, tools, environments, and grading. It runs outside Reef.
2. **The harness tree:** Reef's deliverable, a versioned artifact of config,
   rules, prompt templates, skills, and extension code, served over
   ``GET /reef/harness``.

Skill
-----

A ``SKILL.md`` instruction file the model reads, stored under ``skills/`` in a
harness tree. Reef can inject one into each request or let a client pull it with
the rest of the tree. Updating one needs no GPU.

Runtime
-------

Two meanings.

1. **The request-plane contract:** ``InferenceRuntime`` and
   ``TrainingRuntime``: the external service that executes model work. Inference
   is always required; GPU training only for weight recipes.
2. **A training backend integration:** a concrete implementation of that
   contract, such as Reef's Slime runtime.

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
conversation traffic using a next-state binary reward, with no external grader
and no session headers.

Weight version
--------------

An engine-scoped ``<incarnation>:<sequence>`` token naming the weights that
produced a span of generated tokens. Distinct from the artifact version, which
names a link in the scenario's chain: between checkpoints many weight versions
can share one artifact version, and a restart invalidates the token but not the
chain.

Staleness
---------

The accepted lag between the version that produced a sample and the version
currently serving, configured by ``max_staleness``. Zero requires a sample to
train against the weights that generated it.

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

Two meanings. For harness evolution, the descriptor that turns a rendered tree
into a running agent (`Harness adapters <../developer-guide/harness-adapters.rst>`__). For weight
serving, a LoRA/PEFT adapter layered on a frozen base model.

Episode
-------

One headless run of the agent under test against one task, in a throwaway root
holding nothing but the rendered tree. Harness evolution scores every task twice
per step, once per side.

Mutation
--------

One proposed edit to a harness tree: ``create``, ``update``, or ``remove`` on a
single root-level entry. A sequence applies as one composite proposal under one
verdict.

Preset
------

A recipe configuration file, ``<name>.yaml`` under ``REEF_RECIPE_CONFIG_DIR``,
named by ``reef.recipe``. It carries the recipe's own sections; it is not a
deployment config and ``reef serve -c`` cannot read it.
