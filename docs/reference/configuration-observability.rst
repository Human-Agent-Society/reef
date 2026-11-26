Observability settings
======================

Experiment tracking with W&B, its operational metrics, and record tracing over OpenTelemetry. Their keys take underscores only and are in the `Settings reference <configuration-settings.rst#observability>`__. An unknown key is an error at service start; ``reef serve --print-config`` does not check them.

Experiment tracking
-------------------

Install ``reef-infra[wandb]`` when the training extra does not already provide it. `Track experiments with W&B <../user-guide/operate.rst#track-experiments-with-w-b>`__ tells how to give the credential and how to read the runs.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team
       group_prefix: prod-us-east
       tags: [qwen, baseline]

- With ``mode: disabled``, Reef makes no W&B call, also when ``enabled`` is true.
- A restart resumes the same run. A change to ``project``, ``entity``, ``group_prefix`` or ``name_prefix`` starts new runs: the run id derives from them.

Operational metrics
~~~~~~~~~~~~~~~~~~~

When tracking is on, the dispatcher samples each loaded scenario every 10 seconds, inference-only scenarios included. The samples go to the scenario's run under ``operations/*``, against the Unix time in ``operations/time_seconds``. Offline mode records them locally; ``mode: disabled`` samples nothing. The names below are relative to ``operations/``.

.. list-table:: Operational metrics
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Meaning
   * - ``ingest/accepted_total``, ``ingest/duplicates_total``
     - New records appended, and identical retries acknowledged. An accepted record is not necessarily trainable.
   * - ``ingest/rejected_report_total``, ``ingest/rejected_request_total``, ``ingest/rejected_conflict_total``
     - Reports with schema or reference errors, invalid training requests, and record identities reused with other content.
   * - ``ingest/write/*``
     - Record-store appends. ``failed_total`` counts append exceptions, conflicts included.
   * - ``serve/request/*``, ``serve/retries_total``
     - `Routed requests <configuration-routing.rst>`__ that the control process serves, and their retries after a provider abort. Other requests are not measured.
   * - ``records/unread_count``, ``records/oldest_unread_age_seconds``
     - Training-visible records after the processor's read cursor, and the age of the first. They can still wait for feedback.
   * - ``records/evicted_count``, ``records/evicted_body_bytes``, ``records/data_incomplete``
     - Records that `record retention <../user-guide/storage.rst#record-retention>`__ deleted, their body bytes, and one when any was deleted.
   * - ``processor/waiting_reports``, ``processor/oldest_report_wait_seconds``
     - Reported-feedback reports that wait for their group or the sample store, and the age of the oldest.
   * - ``processor/tracked_records``, ``processor/judging_records``
     - Computed-feedback records that wait for more traffic or for a judgment.
   * - ``processor/waiting_calls``
     - Calls of the trajectory stage whose trajectory has not ended.
   * - ``processor/waiting_source_sessions``, ``processor/generation_in_flight``
     - Generated-task stage: source sessions that wait for a generation round, and one while a round runs.
   * - ``processor/ready_samples``, ``processor/job_samples``, ``processor/buffer_size``
     - Samples that the next automatic job can take now, that one automatic job needs, and that one job takes at most.
   * - ``training/reserved_job_data``, ``training/auto_enabled``
     - One while job data is reserved (its job can be running), and one for auto training.
   * - ``training/error``, ``training/failed_attempts_total``
     - One while a training error is recorded, and the failures counted. Recovery keeps the count; a dispatcher restart or a scenario deletion resets it.
   * - ``training/execution/*``
     - The trainer's job submissions and polls.

Each of ``ingest/write``, ``serve/request`` and ``training/execution`` has ``started_total``, ``completed_total``, ``failed_total``, ``active``, ``elapsed_seconds`` (the age of the oldest active call), ``duration_seconds_total`` (all ended calls) and ``last_duration_seconds`` (absent until a call ends). For the mean latency over an interval, divide the change in ``duration_seconds_total`` by the change in ``completed_total`` plus ``failed_total``.

- A completed execution call is not necessarily a commit: a poll that finds no new checkpoint commits nothing.
- The samples are not persisted history. Execution measurements reset when the trainer is rebuilt, recovery included; ``training/started_at_seconds`` shows the reset. Request and ingestion measurements reset when the scenario is rebuilt; ``started_at_seconds`` shows that reset.
- A request that fails before Reef resolves its scenario is not measured.
- Other numeric status fields of the processor appear under ``processor/``.
- When ingestion or a commit holds the trainer lock, that sample has no ``records/unread_count``, ``records/oldest_unread_age_seconds``, ``training/reserved_job_data``, ``training/auto_enabled`` or ``processor/*``.

Commit correlation
~~~~~~~~~~~~~~~~~~

Commit metrics carry ``experiment/provider``, ``experiment/project``, ``experiment/group``, ``experiment/run_id``, and ``experiment/entity`` when ``entity`` is set, to open the run of a Reef version. The run's ``reef/training_job_id`` leads back to the job.

The ``url`` of a job's ``experiment`` in ``GET /reef/scenarios/{scenario}/jobs`` is the URL that the open W&B run gives. An offline run gives none. The URL is then ``https://wandb.ai/<entity>/<project>/runs/<run_id>``, where the run is after ``wandb sync``. Without ``entity``, it is ``null``.

Record tracing
--------------

Record tracing exports every accepted record and every commit as OpenTelemetry spans over OTLP/HTTP, independent of experiment tracking. Install ``reef-infra[opentelemetry]``.

.. code:: yaml

   observability:
     tracing:
       enabled: true
       endpoint: https://cloud.langfuse.com/api/public/otel/v1/traces
       authorization: ${LANGFUSE_AUTH}

- Without ``authorization``, Reef reads ``REEF_TRACING_AUTHORIZATION``. A reference to an unset variable gives an empty value, so Reef sends no credential.
- Without ``endpoint``, the OpenTelemetry SDK reads ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``. Without ``authorization`` and ``headers``, it reads ``OTEL_EXPORTER_OTLP_HEADERS``.

The spans:

- An accepted inference record gives the root span of its own trace, ``chat <model>``. A restart or a second host gives it the same ids. It carries ``session.id``, ``reef.scenario``, ``reef.agent_record_id``, ``reef.request_type``, ``reef.artifact_id`` and the OpenTelemetry GenAI ``gen_ai.*`` attributes.
- ``reef.tags`` holds the strings of a ``metadata.tags`` list in the request body. The ``x-reef-tag-*`` headers give no span attribute.
- A feedback record gives a ``feedback`` span in the trace of the first inference that it references, with ``reef.score`` and ``reef.references``. Further references are span links.
- A commit gives an ``<operation> commit N`` span, such as ``training commit 4``. It carries the artifact ids, the job id, ``reef.component`` for one of several trainers, and the scalar metrics as ``reef.metrics.*``.
- Each record that a commit consumed gets a ``trained in commit N`` child span, linked to the commit span.

Spans have zero duration. For a streamed response, Reef reads the token usage from the SSE events. When the provider sends none, the counts are absent; for Chat Completions, request ``stream_options: {include_usage: true}``.

.. warning::

   Spans carry the exchange: the request messages and the reply in ``gen_ai.input.messages`` and ``gen_ai.output.messages``, and feedback text in ``reef.feedback``. Point tracing only at a backend that you trust with the scenario's traffic.

An export failure goes to the service log and never fails a record acceptance or a commit.
