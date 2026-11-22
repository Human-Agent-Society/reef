HTTP API: samples
=================

List the training samples of a scenario, read which samples the next job trains, and read the tasks of the samples service that make samples from records.

Choose the samples that train
-----------------------------

Every training sample of a scenario is a data item that it uses: an uploaded trajectory, an imported task, or a trajectory or task that a trainer's stages made. ``GET /reef/scenarios/{scenario}/samples`` lists them in the order that their uses began, at most ``limit`` (default and maximum 1000) after the sequence ``after`` (default 0):

.. code:: json

   {"scenario": "my-agent", "next": null, "samples": [
     {"sample_id": "trajectory-0001", "sequence": 1, "kind": "trajectory", "split": "train",
      "component": null, "group": null, "held_set": null, "train_counts": {}},
     {"sample_id": "my-agent.report-17:0", "sequence": 2, "kind": "trajectory", "split": "train",
      "component": "", "group": null, "held_set": null, "train_counts": {"": 1}}]}

- ``component`` is the trainer that made the sample from records (``""`` for the only trainer of a scenario), or ``null`` for an upload or an imported task.
- Samples that share a ``group`` train together.
- ``held_set`` is the name of the held benchmark or trajectory set that gives the use, or ``null``. An item that the scenario uses more than once has one entry for each use.
- ``train_counts`` holds, for each trainer, the number of committed jobs that trained the sample.
- ``next`` is the ``after`` of the next page, or ``null`` after the last one.

``GET /reef/scenarios/{scenario}/next-job`` returns the samples that the next automatic job trains, in use order, with the trainer's ``training_mode``. Each sample has ``sample_id``, ``kind``, ``group``, ``component`` and the trainer's ``train_count``. A scenario of several components needs ``?component=<name>``.

In ``manual``, the ``samples`` of ``POST /reef/train`` name the samples that the job trains (`Manual training <http-api-training.rst#manual-training>`__):

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: my-agent" \
     -H "Content-Type: application/json" \
     -d '{"text": "Train these", "session": "s-1", "artifact_id": "a-1",
          "samples": ["trajectory-0001", "report-17:0"]}'

- ``samples`` is one or more distinct sample IDs, of one kind: trajectories or tasks, not both. The request body limit of 64 MiB bounds the list.
- Reef returns 400, and queues nothing, for an ID that the store does not hold, or a sample that no trainer can train (an eval or an imported task, an upload that its recipe does not take, another trainer's sample).
- A sample that trained before trains again when the request names it. Staleness does not apply.
- A sample that leaves the scenario before the job starts is left out of the job.

Sample processing tasks
-----------------------

``GET /reef/scenarios/{scenario}/sample-processing`` reads the samples queue through the data service, newest task first. The scenario must exist. The read starts no processing. The query parameters are optional:

- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``.
- ``before``: a positive integer. It excludes creation sequences at or above it.
- ``limit``: 1 to 100, 50 by default.
- ``changed``: ``true`` or ``false`` (the default). With ``true``, the page omits finished tasks that gave the trainer nothing new. Reef reads at most 5,000 tasks to fill the page.

The response has ``scenario``, ``read_at``, ``retention_seconds`` and these fields:

- ``enabled``: the scenario has a trainer that generates samples. It does not report worker health.
- ``records_waiting``: the number of the scenario's records after the last record that a trainer read, for the trainer furthest behind. Each trainer's newest finished task with ``record_sequences`` gives that record. It is null when a trainer has no such task in the retained history.
- ``counts``: covers all retained tasks, independent of pagination and state filters. Reef reads the page and the counts separately, so tasks can change between them.
- ``tasks`` and ``next``: the page, and the next page's ``before`` value or null.
- ``hidden``: each run of omitted tasks; empty without ``changed=true``. A run has ``after``, ``count``, ``first_created_at`` and ``last_created_at``. ``after`` is the ID of the shown task just newer than the run, or null at the top of the page.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}`` returns one task. An unknown task, or a task of another scenario, returns 404.

Each task has ``id``, ``component``, ``state``, ``attempt_count``, ``created_at``, ``ended_at``, ``available_at`` and the fields below. Times are Unix seconds, and unavailable fields are null. A task from an older Reef has null counts, no ``record_sequences`` and no triggers.

- ``started_at`` is the first attempt's start. ``error`` can also describe the previous failed attempt of a queued task.
- ``records_read`` and ``samples_created`` count the completed attempt only, without earlier failed attempts. ``records_read`` includes records that replay read but the pipeline ignored. ``samples_created`` counts new inserts, without duplicate writes.
- ``record_sequences`` is ``[first, last]``: the data sequences of the first and last record of the scenario that the completed attempt read, or null when it read none.
- ``changed``: the task gave the trainer something new. This can also be a new training request, without new samples.
- ``background``: ``true`` means that work remained active when the task finished; later tasks check it.
- ``triggers`` lists why the task ran, oldest first. Each entry has ``name`` and ``at``, the time of the first request.
- ``progress`` holds the counts so far of the attempt that runs now: ``attempt``, ``records_read``, ``samples_created`` and ``at``. It is null for a task that does not run.

A trigger name is ``records``, ``data``, ``commit``, ``reload``, ``background`` or ``due``. Only ``records`` and ``data`` bring new records. The others let a stage make samples without a new record. ``background`` comes from a background job of a stage that ended, or from a fallback look while such jobs run. The fallback waits 1 second, then twice as long after each look that changed nothing, up to 300 seconds.

A sample that a task stored has the item metadata ``stored_by_task`` and ``read_by_task``. A data query with ``group_key`` set to one of them finds the samples of one task.

Control removes ended tasks after ``storage.work_retention_s`` (seven days by default). This removes task history, not stored samples. Active tasks do not expire.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` returns the log of one attempt of a task:

- The samples worker keeps each line that the attempt's work logs. This includes the stage's background jobs that the attempt submitted, also when they end after the attempt. It does not keep lines of threads that the work starts by hand.
- Reef writes the lines every 2 seconds and when the attempt ends.
- One attempt keeps at most 2,000 lines of at most 4,096 characters. The line after the last one says that the log was cut.
- The log goes with its task when the retention deletes the task.

Optional query parameters are ``attempt`` (1 to the task's ``attempt_count``; the last attempt by default), ``after`` (-1 or more, -1 by default; the page holds lines with a larger number) and ``limit`` (1 to 1000, 500 by default). The response has ``task_id``, ``attempt``, ``attempt_count``, ``running``, ``lines`` and ``next``:

- ``attempt`` is null for a task that no worker took.
- ``running`` is true while the chosen attempt runs, so more lines can come.
- Each line has ``number``, ``at``, ``level``, ``logger`` and ``message``.
- ``next`` is the next page's ``after`` value, or null at the end of the lines kept so far.
