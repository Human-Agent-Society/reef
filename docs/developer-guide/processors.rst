Processors
==========

A processor is a scenario's batch builder: records in, one typed training
batch out, plus the answer to what the record store may delete. This page
explains the two engines a recipe can subclass, what each owns, and the path a
record takes to a batch.

.. page::
   :for: recipe authors choosing an engine, and anyone reading ``reef/train/processors/``
   :needs: the processor hooks from `Python API <../reference/python-api.rst#processor>`__
   :outcome: which engine a method needs, what it writes, and what it never writes

The contract
------------

The trainer drives four mutating methods on its own thread, under its lock;
none may block:

- ``ingest(record)``: one record arrives;
- ``ready()``: is a batch available;
- ``build_batch()``: produce it (the trainer validates it against ``output_schema``);
- ``acknowledge(batch_id)``: the training step consumed that batch; returns
  the consumed record ids, which the commit record persists so recovery never
  re-ingests them.

The processor also controls retention. The trainer reads
``retention_decision()`` (protected vs releasable ids) and reports deletions
back through ``compaction_applied()``.
Nothing numeric lives here. Advantages and the loss family are the step
preparer's. The read-only ``status()`` hook is empty by default; a processor
uses it only when a terminal outcome cannot become a batch and an external
runner must stop waiting (TTTD reports a complete mixed-artifact step as an
invariant failure).

``DataProcessor`` in ``base.py`` is concrete on purpose: bare, it is the
no-update default that ingests for audit and never becomes ready. Recipes
can implement their own lifecycle or reuse one of the feedback engines below.

Explicit manual training
------------------------

Pass the recipe's ``training_mode`` to ``Trainer.build``. The factory receives
it in ``ProcessorContext.training_mode``; ``with_config`` preserves it.
The processor owns ingestion, readiness, batch assembly and retention for
the chosen mode. Trainer calls the same lifecycle methods in every mode
and never substitutes a manual wrapper.

``DataProcessor`` supports ``auto`` by default. A processor implementing
both modes declares ``supported_training_modes = frozenset({"auto", "manual"})``
and uses ``self.training_mode`` in its implementation. An unsupported mode
raises ``NotImplementedError`` during initialization, before records are
ingested or a backend step runs. An unknown mode name is a ``ValueError``.

For separate implementations per mode, subclass the optional
``ModeDataProcessor`` helper and declare ``mode_processors``:

.. code:: python

   class MyProcessor(ModeDataProcessor):
       mode_processors = {
           "auto": MyAutomaticProcessor,
           "manual": MyManualProcessor,
       }

It selects the implementation at construction and delegates its complete
lifecycle, including retention, asynchronous derivation status and teardown.
An omitted mode raises ``NotImplementedError``. Each selected implementation
receives the same context and must support that mode.

Manual ingestion must include ``RequestType.TRAIN``. The processor decides
how an explicit instruction authorizes a batch: it may use the instruction
alone, or combine it with accumulated inference data. It should attach the
instruction to ``batch.request`` (``id``, ``text``, ``session``, ``release_id``)
and acknowledge the instruction receipt with the input records it consumes.
Trainer preserves batch reservation, commit and replay semantics.

Harness evolution selects an implementation based on the optional
``ManualTrainingProcessor`` engine. This engine queues instructions FIFO,
retains ordinary traffic for audit, and creates one batch per instruction.
Subclasses implement ``make_request_batch(request: AgentRecord)``; the engine
attaches ``batch.request``, assigns a stable batch id and manages request
acknowledgement. Harness requests need no inference samples, so its hook
returns an empty ``TraceBatch``. Other processors can reuse this engine or
implement their own manual lifecycle. No batch assembly method may call
models or perform training; that remains the backend's responsibility.

Dynamic configuration
---------------------

The manager binds one immutable configuration snapshot to each trainer.
``ProcessorContext.config_revision`` identifies it, and each reserved step
retains that revision until its commit. Components must not fetch newer
configuration midway through an operation.

Dynamic updates are opt-in. Declare the fields in both
``Recipe.dynamic_config_fields`` and ``DataProcessor.dynamic_config_fields``.
The recipe's ``with_runtime_config`` validates the complete proposed data
configuration and any backend-specific constraints. Implement
``prepare_reconfiguration(context)`` to return a separate, empty processor
for the requested configuration, without changing the current processor or
external state. The default raises ``NotImplementedError``.
``ModeDataProcessor`` provides construction of a replacement as an optional
helper; its subclasses still declare which fields can change dynamically.

At a step boundary, the trainer replays retained records up to its consumption
cursor into the replacement, excluding records already consumed by committed
steps. It preserves algorithm/backend state and the record cursor. Pending
records beyond that cursor are consumed normally after activation. Preparation
must preserve the requested mode and output schema; failures close the new
processor and leave the current one intact. Successful activation swaps the
processor and snapshot together under the trainer lock, then closes the old
processor. Batch IDs must remain distinct across configuration revisions;
the harness processors include ``config_revision`` in their automatic batch IDs.

The manager serializes and persists updates, while the worker owns the safe
boundary. Do not mutate a recipe, processor context, or a manager snapshot to
change live behavior. For examples, see ``test_runtime_configuration.py``.

The two feedback paths
----------------------

One question picks the engine: **does feedback arrive in a report, or must the
method compute it?**

+-----------------+---------------------------------------------+----------------------------------------------------+
|                 | reported: ``ReportedFeedbackProcessor``     | computed: ``ComputedFeedbackProcessor``            |
+=================+=============================================+====================================================+
| feedback        | reports referencing inference records       | signal mined from the traffic itself               |
| arrives as      |                                             |                                                    |
+-----------------+---------------------------------------------+----------------------------------------------------+
| ``judge`` is    | a plain method                              | an ``async def``                                   |
+-----------------+---------------------------------------------+----------------------------------------------------+
| called          | by the engine, inside its own ``ingest``    | on a private worker, after the recipe's            |
|                 |                                             | ``ingest`` dispatches                              |
+-----------------+---------------------------------------------+----------------------------------------------------+
| so it may       | only decide on data already in hand         | call models and take minutes                       |
+-----------------+---------------------------------------------+----------------------------------------------------+

Why there are two
~~~~~~~~~~~~~~~~~

Everything else, including the ``async``, follows from that one question.

A report *arrives knowing what it judges*: it names its inference records.
The decision is then a comparison on data already in hand: cheap,
synchronous, and possible the moment the last referenced record lands. What
it costs is bookkeeping about the reference: an index of reports waiting on
inferences that have not arrived, ownership of the records a report claims,
dedup for a grader that retries its POST, and a barrier for recipes whose
unit is a whole group.

A computed-feedback recipe has no report. Its signal does not exist until
later traffic completes an earlier record, and judging it calls a model.
Judgment can take seconds or minutes, which would stall serving if it ran on
the trainer's thread. It moves to a worker instead, with different
bookkeeping: which records still wait, a TTL for the ones whose completion
never comes, and absorption for judgments that land without a record to
trigger them.

Neither set follows from ``async``. Making the reported judge asynchronous
would make timing uniform while keeping every structure above, and would make
every reported recipe's readiness eventually consistent and expose it to
losing an in-flight judgment on a crash, an exposure only the computed path
carries today. The engines differ because the questions differ. What they
share (hold a pending batch, be ready while it exists or once enough units
are held, and release what it consumed) is defined once in ``base.py``. Each
engine fills in ``_ready_count``, ``_make_pending``, and ``_consume_pending``.

What a recipe writes
--------------------

**Reported feedback:** ``judge``, ``make_batch``, ``decide_group`` when it
groups, plus the class attributes ``output_schema``, ``exclusive_sources``,
``ordered_groups``.

**Computed feedback:** In ``ingest``, the correlation *is* the method. It uses
the engine's ``catch_up`` / ``dispatch`` / ``track`` / ``retire`` verbs, as
well as ``judge``, ``make_sample``, ``make_batch``, and ``expire`` for tracked
records that time out.

Neither tier contains retention, lifecycle, or recovery code.

A record's path to a batch
--------------------------

.. code:: text

   reported report ─ingest─► judge(context) ─TRAIN─► candidate ─[decide_group]─► make_batch ─► batch ─ack─► released
                               │ WAIT  parked in the waiting index; re-judged when the last referenced inference lands
                               └ NEVER terminal now; the report and the sources it owns become releasable

   computed record ─ingest─► track ──(a later record completes it)──► dispatch ─► judge (async, on the worker)
                                                                                        │
                               batch ◄─ make_batch ◄─ candidate ◄─ make_sample ◄────────┘
                                                           └ None ─► retire: terminal, releasable

Where a processor lives
-----------------------

Every file under ``reef/train/processors/`` is framework: ``base``,
``reported``, ``computed``, and ``common`` (shared report readers and sample
builders). A method that owns its judgment writes
``recipes/<name>/processor.py``, and that file should show its data flow from
top to bottom. A method that delegates to an engine backend writes none: the
backend owns its concrete processor (``reef/train/cordis_backend/processor.py``)
and wires it in its ``build``, which is why ``recipes/skillclaw/`` ships no
processor file. Machinery beyond that file's job sits beside it as
modules named by concern (``recipes/openclawrl/``: ``sessions``, ``turns``,
``prm``), never in the processor file and never in a ``utils`` grab-bag.

Each structure the engines keep answers one requirement of continual
serving; delete one and a documented failure returns. The list for the
reported engine, naming the attribute each requirement forces, is in the
module docstring of ``reported.py``; the computed engine names its per-state
structures on ``ComputedFeedbackProcessor`` itself.

Cookbook processors
-------------------

+---------------------------------------------+----------+------------------------------------------------------------------+------------------------+
| File                                        | Tier     | What its ``judge`` accepts                                       | Batch                  |
+=============================================+==========+==================================================================+========================+
| ``recipes/sao/processor.py``                | reported | a trainable, finitely scored report whose assembled sample       | ``PolicyBatch``        |
|                                             |          | passes the action-mask check; one referenced inference, or       |                        |
|                                             |          | several assembled into one multi-turn sample when                |                        |
|                                             |          | ``accept_multi_turn_policy_samples`` is set; one rollout,        |                        |
|                                             |          | one unit                                                         |                        |
+---------------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``recipes/tttd/processor.py``               | reported | a report parsing as ``TTTDGroupedRolloutReport`` on this         | ``GroupedPolicyBatch`` |
|                                             |          | scenario's grid; the step is the group, ready only when every    |                        |
|                                             |          | ``groups_per_step`` x ``rollouts_per_group`` slot is filled      |                        |
+---------------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``reef/train/cordis_backend/processor.py``  | reported | a trainable, finitely scored report referencing at least one     | ``TraceBatch``         |
|                                             |          | request, scored inside ``[min_score, max_score]``; one reference |                        |
|                                             |          | is the sample unmodified, several are one trajectory sample      |                        |
+---------------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``recipes/openclawrl/processor.py``         | computed | a main turn whose next state the PRM scores +/-1, or for which   | ``PolicyBatch``        |
|                                             |          | the teacher scored an accepted hindsight hint                    |                        |
+---------------------------------------------+----------+------------------------------------------------------------------+------------------------+
