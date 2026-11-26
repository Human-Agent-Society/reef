HTTP API: samples
=================

Inspect training samples, select job inputs and check record-to-sample processing.

Choose the samples that train
-----------------------------

Samples are assigned data items: uploaded or generated trajectories and imported or generated tasks.
``GET /reef/scenarios/{scenario}/samples`` lists them in assignment order.
``after`` defaults to 0; ``limit`` defaults to 1000, also the maximum.

.. code:: json

   {"scenario": "my-agent", "next": null, "samples": [
     {"sample_id": "trajectory-0001", "sequence": 1, "kind": "trajectory", "split": "train",
      "component": null, "group": null, "held_set": null, "train_counts": {}},
     {"sample_id": "my-agent.report-17:0", "sequence": 2, "kind": "trajectory", "split": "train",
      "component": "", "group": null, "held_set": null, "train_counts": {"": 1}}]}

- ``component`` identifies the generating trainer: ``""`` for a sole trainer;
  ``null`` for uploads or imports.
- A shared ``group`` keeps samples in one job.
- ``held_set`` identifies the assigning benchmark or trajectory set, or null.
  An item has one entry per use.
- ``train_counts`` maps trainers to committed jobs that trained the sample.
- ``next`` is the next ``after`` cursor, or null.

``GET /reef/scenarios/{scenario}/next-job`` previews automatic job inputs in use order,
with ``training_mode``. Entries contain ``sample_id``, ``kind``, ``group``,
``component`` and ``train_count``. Multiple components require ``?component=<name>``.

Select samples through `manual training requests <http-api-training.rst#manual-training>`__:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: my-agent" \
     -H "Content-Type: application/json" \
     -d '{"text": "Train these", "session": "s-1", "artifact_id": "a-1",
          "samples": ["trajectory-0001", "my-agent.report-17:0"]}'

Supply one or more distinct IDs of one kind; the body limit is 64 MiB.
Unknown or untrainable samples return 400 without queuing, including eval tasks,
unsupported inputs and another trainer's samples.
Requests can retrain samples and bypass staleness.
Samples removed from the scenario before execution are omitted.

Sample processing tasks
-----------------------

``GET /reef/scenarios/{scenario}/sample-processing`` reads tasks newest first
through the data service. The scenario must exist; reads start no work.

Query fields:

- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``.
- ``before``: positive creation-sequence cursor, excluded from the page.
- ``limit``: 1–100; default 50.
- ``changed``: boolean, default false. True hides unchanged finished tasks;
  filling a page scans at most 5,000 tasks.

The response includes ``scenario``, ``read_at``, ``retention_seconds`` and:

- ``enabled``: whether a trainer generates samples, independent of worker health.
- ``records_waiting``: unread records for the furthest-behind trainer,
  measured from its newest finished task with ``record_sequences``.
  Null means there is no trainer or retained progress is incomplete.
- ``counts``: all retained tasks, independent of filters and pagination.
  Counts and page reads are separate; their snapshots can differ.
- ``tasks`` and ``next``: the page and next ``before`` cursor, or null.
- ``hidden``: omitted runs under ``changed=true``; otherwise empty.
  Each has ``after``, ``count``, ``first_created_at`` and ``last_created_at``.
  ``after`` identifies the shown task just newer than the run, or null at the page top.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}`` reads one task.
Unknown tasks and tasks belonging to another scenario return 404.

Each task has ``id``, ``component``, ``state``, ``attempt_count``, ``created_at``,
``ended_at`` and ``available_at``. Times are Unix seconds; unavailable fields are null.

- ``started_at``: first attempt's start. ``error`` can describe a previous failed attempt.
- ``records_read``, ``samples_created``: completed-attempt counts, excluding earlier attempts.
  Reads include ignored replay records; sample counts exclude duplicate inserts.
- ``record_sequences``: completed-attempt ``[first, last]`` data sequences, or null when no records were read.
- ``changed``: new input reached training, possibly a request without new samples.
- ``background``: work remained after completion; later tasks check it.
- ``triggers``: oldest-first entries with ``name`` and first-request time ``at``.
- ``progress``: running-attempt ``{attempt, records_read, samples_created, at}``, otherwise null.

Older tasks have null counts, no ``record_sequences`` and no triggers.
Triggers are ``records``, ``data``, ``commit``, ``reload``, ``background`` or ``due``.
Only the first two supply records; others can create samples without new records.
Background polling starts after one second and doubles unchanged waits up to 300 seconds.

Set data-query ``group_key`` to ``stored_by_task`` or ``read_by_task`` to find a task's samples.
Ended tasks expire after ``storage.work_retention_s`` (default seven days).
This removes history and logs, not samples. Active tasks never expire.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` reads an attempt's log:

- ``attempt``: 1–``attempt_count``; defaults to the latest.
- ``after``: minimum and default ``-1``; returns higher line numbers.
- ``limit``: 1–1000; default 500.

The response contains ``task_id``, ``attempt``, ``attempt_count``, ``running``,
``lines`` and ``next``. ``attempt`` is null before worker assignment.
``running`` indicates whether the selected attempt is running.
Each line has ``number``, ``at``, ``level``, ``logger`` and ``message``.
``next`` is the next ``after`` cursor, or null at the current end.

Logs include stage background jobs, even after the attempt ends, but exclude
threads started outside the worker's context.
Reef writes every two seconds and at attempt completion.
Each attempt keeps 2,000 lines of at most 4,096 characters, then a truncation notice.
