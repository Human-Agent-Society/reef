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
and implements mode-specific methods on the same class. An unsupported mode
raises ``NotImplementedError`` during initialization, before records are
ingested or a backend step runs. An unknown mode name is a ``ValueError``.

.. code:: python

   class MyProcessor(DataProcessor):
       supported_training_modes = frozenset({"auto", "manual"})

       def ingest_auto(self, record):
           ...  # Collect inputs according to the automatic policy.

       def ingest_manual(self, record):
           ...  # Collect instructions and any required inference inputs.

       def ready_auto(self):
           ...  # Apply the recipe's batching policy.

       def ready_manual(self):
           ...  # Require an instruction and its inputs, independently of auto batching.

       def build_batch_auto(self, batch_number):
           ...  # Return the automatic training batch.

       def build_batch_manual(self, batch_number):
           ...  # Return the manual training batch with batch.request attached.

The shared ``ingest``, ``ready`` and ``build_batch`` entry points dispatch to
these methods using the instance's selected training mode. ``build_batch``
caches the selected batch until acknowledgement, so repeated reservations do
not rebuild it. There is no mapping to another processor class or instance.

The processor also implements ``acknowledge_auto`` / ``acknowledge_manual``
(return consumed receipt ids), ``retention_decision_auto`` /
``retention_decision_manual``, and ``compaction_applied_auto`` /
``compaction_applied_manual``. Missing manual hooks raise
``NotImplementedError``; declaring support never falls back to automatic
behavior. Existing automatic engines retain their ``_ready_count``,
``_make_pending`` and ``_consume_pending`` hooks through the default auto
methods. Computed-feedback subclasses implement ``ingest_auto`` for their
correlation logic and inherit automatic readiness and retention from their
engine. Processors overriding the shared entry points own their dispatch.
Background work is polled through ``derivation_pending_auto`` /
``derivation_pending_manual`` (both default to false); ``close`` remains a
shared teardown hook for all resources owned by the instance.

Manual ingestion must include ``RequestType.TRAIN``. The processor decides
how an explicit instruction authorizes a batch: it may use the instruction
alone, or combine it with accumulated inference data. It should attach the
instruction to ``batch.request`` (``id``, ``text``, ``session``, ``release_id``)
and acknowledge the instruction receipt with the input records it consumes.
Trainer preserves batch reservation, commit and replay semantics.

Harness evolution inherits the reusable ``*_manual`` methods from the optional
``ManualTrainingProcessor`` engine on the same processor instance. This engine
queues instructions FIFO,
retains ordinary traffic for audit, and creates one batch per instruction.
Subclasses implement ``make_request_batch(request: AgentRecord)``; the engine
attaches ``batch.request``, assigns a stable batch id and manages request
acknowledgement. Harness requests need no inference samples, so its hook
returns an empty ``TraceBatch``. Other processors can reuse this engine or
implement their own manual lifecycle. No batch assembly method may call
models or perform training; that remains the backend's responsibility.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/training-mode`` selects ``auto`` or
``manual`` on the existing processor through ``set_training_mode``.
The trainer serializes this operation with ingestion and reservation.
A reserved batch retains its original acknowledgement mode, so changing
mode does not interrupt a running step.

Mode-owned buffers stay on the same instance. Already ingested records are
not replayed into another mode; switching back resumes that mode's buffers.
Retention protects inputs held by either mode, and accepted TRAIN records
remain queued for manual mode even if auto is selected before ingestion.
The selector is runtime state: rebuilding a scenario uses the recipe's
configured default again.

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
|                 |                                             | ``ingest_auto`` dispatches                         |
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

**Computed feedback:** In ``ingest_auto``, the correlation *is* the method. It uses
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
