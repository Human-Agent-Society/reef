Glossary
========

The vocabulary used throughout these docs, from what a client sees to what a
method implements.

Scenario
--------

One workload: its records, training state, and release chain, isolated from
every other scenario. The first request carrying a new ``x-reef-scenario``
creates it and permanently binds its recipe. The request may also bind a
starting release.

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

The selected content: a model checkpoint, live weights, or a harness tree.
``content_id`` identifies those contents independently of when or why Reef
published them.

Release
-------

One accepted publication decision in a scenario. ``release_id`` identifies the
link in the scenario history and ``parent_release_id`` identifies its parent.
Rollback therefore creates a new release that can point to the same
``content_id`` as an older release; history is appended, never rewound.

Content ID
----------

The identity of the bytes or logical model state selected by a release. Equal
``content_id`` values mean equal selected content even when the ``release_id``
values differ, such as after rollback or republication.

Release chain
-------------

A scenario's history of accepted updates, each release carrying its parent. A
durable release can be restored when its surface and runtime support rollback.

Candidate selection
-------------------

The decision whether a produced update should replace the current served
release. A ``CandidateEvaluator`` measures the candidate, a
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
``sao``, ``tttd``, ``openclawrl``.

Harness
-------

The part of an agent around the model. It owns the control loop, prompts,
conversations, tools, environment integration, and grading, and calls the model
through an inference runtime.

Harness tree
------------

Reef's versioned representation of the mutable files in a harness: config,
rules, prompt templates, skills, and extension code. Reef serves it over
``GET /reef/harness``. An adapter combines the rendered tree with a harness
executable; that harness and its configured model form the running agent.

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

Runtime load ID
--------------

An engine-scoped ``<incarnation>:<sequence>`` token naming the concrete weight
load that produced a span of generated tokens. It is a serving fence, not a
publication or content identity. Reloading the same ``content_id`` can produce
a new runtime load ID, while rollback can create a new ``release_id`` for old
content. A restart invalidates runtime load IDs but not the release chain.

Staleness
---------

The accepted lag between the runtime load that produced a sample and the load
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

Two meanings. For harness evolution, the descriptor that combines a rendered
tree with an executable to make a running harness (`Harness adapters
<../developer-guide/harness-adapters.rst>`__). For weight
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
