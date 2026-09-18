Processors
==========

A processor is a scenario's batch builder: records in, one typed training
batch out, plus the answer to what the record store may delete. This page
explains the two feedback engines, the task-generation ABC, and the path a
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

Every concrete processor produces ``TrainingBatch.items``, an ordered tuple of
``TrainDataItem`` values: ``TrajectoryItem`` for existing trajectories or
``TaskItem`` for Harbor tasks that an algorithm will roll out. A batch may mix
both. ``make_sample`` constructs ATIF trajectory items, including captured
training tensors and feedback in ``extra.reef``. ``make_batch`` collects the
items; grouped recipes set ``group_id`` on each member. Assembly, scheduling
and group completeness remain recipe decisions. There is no separate dataset
container. See `batch values <../reference/python-api.rst#batch>`__ for formats
and algorithm support.

The processor also controls retention. The trainer reads
``retention_decision()`` (protected vs releasable ids) and reports deletions
back through ``compaction_applied()``. A batch the backend dropped as stale is
announced through ``dropped()`` before its acknowledgement, for a processor
that paces work on what actually trained.
Nothing numeric lives here. Advantages and the loss family are the step
objective's. The read-only ``status()`` hook is empty by default; a processor
uses it only when a terminal outcome cannot become a batch and an external
runner must stop waiting (TTTD reports a complete mixed-artifact step as an
invariant failure).

``DataProcessor`` in ``base.py`` is concrete on purpose: bare, it is the
no-update default that ingests for audit and never becomes ready. Recipes
can implement their own lifecycle or reuse one of the feedback engines below.

Explicit manual training
------------------------

``training_mode`` is an attribute of each ``DataProcessor``. The recipe passes
its initial value through ``Trainer.build`` and ``ProcessorContext``; it
defaults to ``auto``. Ingestion, acknowledgement, retention and compaction use
the same methods and buffers in every mode. ``GET /reef/status`` reports
``pending_instructions`` as the processor's ``buffered_requests`` plus the
instructions still unread in storage.

The shared batching cycle waits for ``batch_size`` units in ``auto``, for a
queued TRAIN instruction in ``manual``, and for either in ``hybrid``, where a
queued instruction goes first. A processor that takes instructions declares
``supported_training_modes = frozenset({"auto", "manual", "hybrid"})`` and
implements one assembly hook:

.. code:: python

   def make_training_batch(self, batch_number, request):
       if request is not None and self.training_mode == "manual":
           # Manual runs the instruction alone; harness needs no samples.
           self._pending_reports = ()
           return TrainingBatch(request.id, ())
       # Hybrid hands the instruction the units an automatic batch would take.
       return self._make_pending(batch_number)

This example extends the reported-feedback engine. ``request`` is a
``TrainingRequest`` in ``manual`` and ``hybrid`` when an instruction is queued
and ``None`` for an automatic batch. The base class attaches the instruction
to ``batch.request`` and replaces the hook's own batch id with
``<scenario>:instruction:<request id>``, the oldest instruction first, one
per step. ``_consume_pending`` releases the selected data; shared
acknowledgement also consumes the instruction. A processor that needs other
inputs for an instruction selects them in the hook or extends the shared
``ready`` predicate. Batch construction must not call models or perform
training.

Processors receive TRAIN records by including ``RequestType.TRAIN`` in
``required_request_types`` and forwarding those records to ``super().ingest``.
The base class queues them FIFO regardless of the selected mode; ``auto``
leaves the queue for a mode that takes it. Data ingestion continues normally
in manual mode, so changing to auto or hybrid can batch data already collected;
the reported-feedback engine holds at most four batches of units in manual
mode and releases the oldest beyond that with a warning, at the switch and
as reports arrive. Custom retention
implementations must preserve queued instructions and release consumed ones,
as the reported-feedback engine does.

Unsupported modes, or instruction assembly without an implementation, raise
``NotImplementedError``. An invalid mode name raises ``ValueError``.
The default automatic assembly keeps the existing ``_make_pending`` hook;
automatic processors and computed-feedback ``ingest`` implementations need no
mode-specific lifecycle methods or additional processor class.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/update`` selects ``auto``, ``manual`` or
``hybrid`` on the existing processor. The trainer serializes selection with
ingestion and reservation. A reserved batch stays unchanged and acknowledgement
consumes its actual contents, independently of subsequent mode changes.

Buffered data and queued instructions stay on the same instance. The selector
is runtime state: a scenario reload after a failed step keeps the selected
mode, and a service restart starts from the recipe's configured mode.

The two feedback paths
----------------------

``ReportedFeedbackProcessor`` consumes feedback already supplied in reports.
``ComputedFeedbackProcessor`` derives feedback from traffic, potentially using
slow model calls on its worker. Reported feedback does not have a ``judge``
hook; the computed engine retains its asynchronous ``judge``.

Reported feedback
~~~~~~~~~~~~~~~~~

Before accepting a new report, Reef checks its schema and verifies that every
reference identifies an existing inference in the same scenario. Missing,
wrong-kind, duplicate, or foreign references raise an error. Reports are not
queued waiting for future inference records. Storage is authoritative; a record
need not have reached the processor's in-memory cache at admission time.

Reports cannot opt out of training: ``metadata.training.eligible`` is rejected,
including when its value is ``true``. Valid low or negative scores are feedback,
not a reason to discard the report.

A reported-feedback recipe implements:

- ``make_sample(context) -> TrainDataItem``: assemble an ATIF trajectory or
  Harbor task directly. Use ``context.require_score()`` when the method needs
  a reward. The engine attaches the source and report ids to the returned item.
- ``make_batch(items, batch_number)``: assemble the flat tuple of selected
  training items into a batch. Consumption remains the engine's responsibility,
  including selected items that the recipe removes from training.
- ``is_training_report(report)`` when another role's reports share the
  scenario: return ``False`` for a valid report that is not this method's
  training data (SPADE's Designer reports its own score there). The engine
  releases such a report and never calls ``make_sample`` on it; its sources
  go with it only under ``exclusive_sources`` or when it references more
  than one inference, and a single referenced inference stays retained for
  a report that trains on it. The default takes every report.
- ``grouping(context)`` for grouped methods: return ``(group_key, slot)``.
  The default ``(None, None)`` makes an independent sample. A None slot uses
  the report id; repeated slots preserve the first accepted report.
- ``decide_group(key, items)`` for grouped methods: return ``INCOMPLETE``,
  ``READY``, or ``DISCARD``. The items are training data, without cache wrappers.
  The collection group can span multiple training comparison groups, as in TTTD.

The engine handles report-id deduplication, group-slot retries, reservations,
consumption, and retention. A batch is ready in automatic mode after
``batch_size`` units accumulate; a unit is one sample or one complete group.
The same batch remains reserved until acknowledged or released. Reports arriving
later for already consumed sources cannot train those sources again.

Training data errors propagate: missing required tokens/logprobs are checked by
the training backend, and unsupported trajectory assembly raises an error rather
than dropping the report. Failed sample assembly keeps its input records
protected and does not mark the report successfully processed.

Computed feedback
~~~~~~~~~~~~~~~~~

A computed-feedback recipe implements correlation in ``ingest`` using the
engine's ``catch_up``, ``dispatch``, ``track``, and ``retire`` operations.
It supplies ``async judge``, ``make_sample``, ``make_batch``, and ``expire`` for
tracked records that time out. This path derives a new signal and is unchanged
by the reported-feedback contract.

Task generation contract
------------------------

``TaskGenerationProcessor(DataProcessor, ABC)`` declares two asynchronous
methods on the processor itself:

- ``generate(request: TaskGenerationRequest) -> HarborTask`` produces one
  task specification from source records, a description and optional asset
  paths. The source records, when there are any, must have distinct ids and
  belong to one scenario, the processor's; a method whose designer writes
  from the description alone passes none. Preserve their ordered ids, or the
  designer's own inference record id, in the generated task's
  ``source_agent_record_ids``.
- ``validate(task_path: Path) -> TaskValidationResult`` checks a materialized
  candidate without modifying it. The implementation chooses the required
  structural and execution checks. Empty ``errors`` means all checks passed;
  non-empty errors reject the task. Infrastructure failures raise exceptions
  rather than reporting that the task is invalid.

Import the ABC from ``reef.train.processors`` and the request/result types
from ``reef.train.processors.task_generation``. Asset paths name generator-accessible files or
directories, such as repository snapshots or verifier fixtures; constructing
a request does not read them. Method-specific prompts and settings belong to
the processor configuration.

The ABC supplies no lifecycle: implementing the two hooks does not start a
worker or make batches ready, and the inherited lifecycle is the no-update
default. A method supplies its own, keeping the two rules every lifecycle
must keep: both hooks run outside the trainer lock, and ``ingest``,
``ready`` and ``build_batch`` never wait for them.

The first implementation is SPADE (``recipes/beta/spade/processor.py``),
which pairs the ABC with the reported-feedback engine. A private worker (the
computed engine's ``JudgingWorker``) runs one generation at a time:
``generate`` asks the designer for a task, the task is written, ``validate``
runs Harbor's oracle check on it, the task is played, and the episodes come
back as reports the reported half groups and batches; the next generation
starts after a configured number of batches was acknowledged, and a restart
carries on from the generation reports on disk. None of the container-bound
steps run in the Reef service process: ``reef.record2dataset`` is the
generator service ``reef serve`` starts beside the HTTP service from the
deployment's ``generator`` section (see `the generator section
<../reference/configuration.rst#the-generator-section>`__), and the processor
drives it over HTTP. Conversion of generated tasks into ``TaskItem`` batches
for a rollout-capable backend remains future work.

A record's path to a batch
--------------------------

.. code:: text

   report -> validate existing references -> ingest -> make_sample
       -> TrainDataItem -> optional group barrier -> make_batch -> batch -> acknowledge

   computed record -> ingest/track -> dispatch -> async judge
       -> make_sample -> candidate -> make_batch -> batch -> acknowledge

Where a processor lives
-----------------------

Shared engines live under ``reef/train/processors/``, including
``DistillProcessor``, the reported engine of the distilling recipes: it
emits the student's rollout plus the teacher sequence, the request the recipe
composes from the recorded request and the report's context (its
``teacher_request`` override) rendered with the served model's chat template. Concrete method processors
live in ``recipes/<name>/processor.py``; the harness evolution implementation
lives in ``reef/train/cordis_backend/processor.py``. Recipe-specific correlation
and model clients belong beside the method processor.

Compatibility
-------------

Reported-feedback subclasses must replace ``judge`` and ``ReportDecision`` with
``make_sample`` returning ``TrainDataItem``. Remove ``WAIT``/``NEVER`` branches: invalid
references fail admission, and training contract failures must raise. Existing
stored records replay through the new contract; legacy reports with invalid
references or eligibility flags fail explicitly and need correction before replay.

The harness recipe no longer supports ``max_score`` or filters successful
reports. Remove that setting from configuration and Python construction.
Identical report retries still return their original receipt, including after
source compaction; changed content with the same id still conflicts.
