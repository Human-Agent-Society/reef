Observability settings
======================

Use W&B for experiment metrics and OpenTelemetry for record tracing.
Keys use underscores only; see `Settings reference <configuration-settings.rst#observability>`__.
Unknown keys fail service startup. ``--print-config`` does not validate these sections.

Experiment tracking
-------------------

Install ``reef-infra[wandb]`` if the training extra does not provide it.
For credentials and run navigation, see `Track experiments with W&B <../user-guide/operate.rst#track-experiments-with-w-b>`__.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team
       group_prefix: prod-us-east
       tags: [qwen, baseline]

- ``mode: disabled`` prevents all W&B calls, even with ``enabled: true``.
- Restarts resume the same run. Changing ``project``, ``entity``, ``group_prefix`` or ``name_prefix`` changes the run ID and starts new runs.

Operational metrics
~~~~~~~~~~~~~~~~~~~

Every 10 seconds Reef samples each loaded scenario under ``operations/*``, with Unix time in ``operations/time_seconds``.
Offline mode saves samples locally. The table omits the ``operations/`` prefix.

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

For ``ingest/write``, ``serve/request`` and ``training/execution``, use ``active`` and ``elapsed_seconds`` to find outstanding calls and the age of the oldest.
``started_total``, ``completed_total`` and ``failed_total`` count calls.
Mean latency over an interval is ``Δduration_seconds_total / (Δcompleted_total + Δfailed_total)``.
``last_duration_seconds`` appears when the first call ends.

A completed poll need not produce a commit. Measurements reset on trainer or scenario rebuilds
(``training/started_at_seconds`` or ``started_at_seconds``).
Requests that fail before scenario resolution produce no measurements.
While ingestion or a commit holds the trainer lock, unread-record, reservation, automatic-training and processor gauges are absent.

Commit correlation
~~~~~~~~~~~~~~~~~~

Use ``experiment.url`` in the jobs API to open the W&B run. An offline URL works after ``wandb sync``
and requires an entity. Commit metrics identify the provider, project, group, run ID and optional entity under ``experiment/*``.
The run's ``reef/training_job_id`` identifies the job.

Record tracing
--------------

Install ``reef-infra[opentelemetry]`` to export accepted records and commits as OTLP/HTTP spans.
Tracing works independently of experiment tracking.

.. code:: yaml

   observability:
     tracing:
       enabled: true
       endpoint: https://cloud.langfuse.com/api/public/otel/v1/traces
       authorization: ${LANGFUSE_AUTH}

- Without ``authorization``, Reef reads ``REEF_TRACING_AUTHORIZATION``. A reference to an unset variable gives an empty value, so Reef sends no credential.
- Without ``endpoint``, the OpenTelemetry SDK reads ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``. Without ``authorization`` and ``headers``, it reads ``OTEL_EXPORTER_OTLP_HEADERS``.

The spans:

- An inference record starts a trace with root span ``chat <model>``. IDs remain stable across restarts and hosts. Attributes include ``session.id``, ``reef.scenario``, ``reef.agent_record_id``, ``reef.request_type``, ``reef.artifact_id`` and ``gen_ai.*``.
- ``reef.tags`` holds the strings of a ``metadata.tags`` list in the request body. The ``x-reef-tag-*`` headers give no span attribute.
- A feedback record gives a ``feedback`` span in the trace of the first inference that it references, with ``reef.score`` and ``reef.references``. Further references are span links.
- A commit gives an ``<operation> commit N`` span, such as ``training commit 4``. It carries the artifact ids, the job id, ``reef.component`` for one of several trainers, and the scalar metrics as ``reef.metrics.*``.
- Each record that a commit consumed gets a ``trained in commit N`` child span, linked to the commit span.

Spans have zero duration. Streamed responses get token counts from SSE events; counts are absent if the provider omits them.
For Chat Completions, request ``stream_options: {include_usage: true}``.

.. warning::

   Spans include request messages, replies and feedback in ``gen_ai.input.messages``, ``gen_ai.output.messages`` and ``reef.feedback``.
   Send traces only to a backend trusted with the scenario's traffic.

An export failure goes to the service log and never fails a record acceptance or a commit.
