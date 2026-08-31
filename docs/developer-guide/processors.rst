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
never implement it directly; they subclass one of the two engines.

The two feedback paths
----------------------

One question picks the engine: **does feedback arrive in a report, or must the
method compute it?**

+-----------------+---------------------------------------------+----------------------------------------------------+
|                 | reported — ``ReportedFeedbackProcessor``    | computed — ``ComputedFeedbackProcessor``           |
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
builders). A method's processor is ``recipes/<name>/processor.py`` and should
show its data flow from top to bottom. Machinery beyond that file's job sits beside it as
modules named by concern (``recipes/openclawrl/``: ``sessions``, ``turns``,
``prm``), never in the processor file and never in a ``utils`` grab-bag.

Each structure the engines keep answers one requirement of continual
serving; delete one and a documented failure returns. The list, naming the
attribute each requirement forces, is in the module docstrings of
``reported.py`` and ``computed.py``.

The bundled processors
----------------------

+---------------------------------------+----------+------------------------------------------------------------------+------------------------+
| File                                  | Tier     | What its ``judge`` accepts                                       | Batch                  |
+=======================================+==========+==================================================================+========================+
| ``recipes/sao/processor.py``          | reported | a trainable, finitely scored report with exactly one referenced  | ``PolicyBatch``        |
|                                       |          | inference whose assembled sample passes the action-mask check;   |                        |
|                                       |          | one rollout, one unit                                            |                        |
+---------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``recipes/tttd/processor.py``         | reported | a report parsing as ``GroupedRolloutReport`` on this scenario's  | ``GroupedPolicyBatch`` |
|                                       |          | grid; the step is the group, ready only when every               |                        |
|                                       |          | ``groups_per_step`` × ``rollouts_per_group`` slot is filled      |                        |
+---------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``recipes/harness_evolve/processor.py``| reported | a trainable, finitely scored report with exactly one reference   | ``TraceBatch``         |
|                                       |          | and a score inside ``[min_score, max_score]``; the recorded      |                        |
|                                       |          | request is the sample, unmodified                                |                        |
+---------------------------------------+----------+------------------------------------------------------------------+------------------------+
| ``recipes/openclawrl/processor.py``   | computed | a main turn whose next state the PRM scores ±1, or for which the | ``PolicyBatch``        |
|                                       |          | teacher scored an accepted hindsight hint                        |                        |
+---------------------------------------+----------+------------------------------------------------------------------+------------------------+

See also
--------

- `Write a recipe <write-a-recipe.rst>`__: a processor in a complete recipe.
- `Loss families <loss-families.rst>`__: the numeric side, which lives in the preparer.
