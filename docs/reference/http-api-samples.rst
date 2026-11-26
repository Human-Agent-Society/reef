HTTP API: samples
=================

Inspect training samples, select job inputs and check record-to-sample processing.

Choose the samples that train
-----------------------------

Samples are assigned data items: uploaded or generated trajectories and imported or generated tasks.
``GET /reef/scenarios/{scenario}/samples`` lists them in assignment order.
``after`` defaults to 0; ``limit`` defaults to 1000, also the maximum.

The response is ``{scenario, samples, next}``. Rows include ``sample_id``,
``sequence``, ``kind``, ``split`` and:

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

The response contains ``enabled``, ``records_waiting``, retained task ``counts``,
``tasks`` and the next ``before`` cursor. Counts can change between reads.
``records_waiting`` is null when progress is unavailable.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}`` returns one task's
state, attempts, epoch-second times, error, record/sample counts and progress.
Unknown or foreign tasks return 404. Older tasks can have null counts.
Set data-query ``group_key`` to ``stored_by_task`` or ``read_by_task`` to find related samples.
Completed history/logs expire after ``storage.work_retention_s`` (default seven days);
samples and active tasks remain.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` reads an attempt's log:

- ``attempt``: 1–``attempt_count``; defaults to the latest.
- ``after``: minimum and default ``-1``; returns higher line numbers.
- ``limit``: 1–1000; default 500.

The response contains ``task_id``, ``attempt``, ``attempt_count``, ``running``,
``lines`` and ``next``. ``attempt`` is null before worker assignment.
``running`` indicates whether the selected attempt is running.
Each line has ``number``, ``at``, ``level``, ``logger`` and ``message``.
``next`` is the next ``after`` cursor, or null at the current end.

Logs are bounded to 2,000 lines per attempt and 4,096 characters per line,
followed by a truncation notice.
