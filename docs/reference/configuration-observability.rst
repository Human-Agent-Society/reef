Observability settings
======================

The keys of the ``observability`` section: experiment tracking with W&B (``observability.wandb``), the operational metrics that it uploads, and record tracing over OpenTelemetry (``observability.tracing``).

Experiment tracking
-------------------

Tracking is optional and off by default. It belongs to a Reef scenario, not to one training backend. Install ``reef-infra[wandb]`` when the training extra does not already provide it. `Track experiments with W&B <../user-guide/operate.rst#track-experiments-with-w-b>`__ describes how to read the runs.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team             # optional
       group_prefix: prod-us-east    # optional scenario-group namespace
       name_prefix: baseline         # optional run-name prefix
       tags: [openclawrl, qwen]
       mode: online                  # online, offline, or disabled
       directory: /var/lib/reef/wandb
       upload_checkpoints: false

Export ``WANDB_API_KEY`` before starting, or log in once with the credential store on the cluster.

.. warning::

   There is no API-key field here. Reef rejects Slime's ``--wandb-key`` flag and never writes a credential into metrics or run config. Do not put one in the YAML, in ``training.options``, in a tag, or in a run name.

``online`` sends data to the project. ``offline`` makes no network calls and writes data below ``directory`` for a later ``wandb sync``. ``disabled`` makes no calls even when ``enabled`` is true. Import, initialization, logging, summary and upload failures go to the service log and never fail a training step or its commit.

**Runs.**

- Each scenario maps to a group named after the scenario, or ``<group_prefix>/<scenario>``.
- Reef opens one run when the scenario binds and another after each train-from. The run id is deterministic, so a restart resumes the same run with ``resume=allow``.
- A train-from finishes the current run, records each component's training-start artifact ID in ``reef/train_from_artifact_ids``, and resets ``train/step`` to zero. An activation does not end a run.
- ``reef/commit_number`` increases across runs, so it joins a run back to the commit history.
- A scenario with several trainers shares the run. Each step's metrics carry the component name as a prefix (``harness/train/loss``) on the same ``train/step`` axis, with ``reef/component`` on the row. Its optimizer step rows land under ``<component>/step/*`` on their own counter. The run config lists each component's backend under ``reef.components`` and ``backend.<component>``, with ``reef.backend`` null.
- The names ``train``, ``step``, ``reef`` and ``operations`` are these prefixes, so a composite recipe refuses them as component names.

Recipe and processor code logs through the same logger without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own ``<namespace>/event`` axis. Only finite numeric values are sent.

Operational metrics
~~~~~~~~~~~~~~~~~~~

With W&B enabled, the dispatcher samples each loaded scenario, inference-only scenarios included, every 10 seconds and once during graceful shutdown. The samples go to the scenario's run under ``operations/*``, with Unix time in ``operations/time_seconds`` as their horizontal axis. They do not advance ``train/step``. Offline mode records the same samples locally; disabled tracking starts no sampling.

The following names are relative to ``operations/``:

.. list-table:: Operational measurements
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Meaning
   * - ``ingest/accepted_total``, ``ingest/duplicates_total``
     - New records appended and identical retries acknowledged by the dispatcher. Accepted records include incomplete-stream diagnostics; acceptance does not imply a trainable record or successful training.
   * - ``ingest/rejected_report_total``, ``ingest/rejected_request_total``
     - Report schema/reference violations and invalid training requests detected during dispatcher record validation.
   * - ``ingest/rejected_conflict_total``
     - Record identities reused with different content.
   * - ``ingest/write/*``
     - Record-store append calls, including identical retries that reach the store. ``failed_total`` counts append exceptions, including conflicts; validation rejections before append do not count as write failures.
   * - ``records/unread_count``
     - Training-visible records after the processor's read cursor. These may still need feedback or filtering; this is not a ready job data count.
   * - ``records/oldest_unread_age_seconds``
     - Age of the first unread record, measured from its recorded creation time; zero when none remain.
   * - ``processor/waiting_reports``
     - Reported-feedback reports that wait for their group or for the sample store. An incomplete group still counts as waiting.
   * - ``processor/oldest_report_wait_seconds``
     - Age of the oldest waiting report; zero when none remain.
   * - ``processor/tracked_records``, ``processor/judging_records``
     - Computed-feedback records waiting for more traffic or a judgment.
   * - ``processor/waiting_candidates``
     - Computed-feedback candidates that wait to become samples.
   * - ``processor/ready_samples``, ``processor/job_samples``, ``processor/buffer_size``
     - Samples that the next automatic job can take now, samples that automatic job data needs, and the most samples that one automatic job takes.
   * - ``training/reserved_job_data``
     - Zero or one. A reservation can be executing or waiting for settlement; it is not necessarily queued job data.
   * - ``training/auto_enabled``
     - One for auto training, zero for manual training. Waiting data in manual mode does not by itself indicate a stalled worker.
   * - ``training/error``, ``training/failed_attempts_total``
     - Current recorded training error (zero/one) and cumulative failures recorded by the dispatcher, including retries and worker failures. Recovery clears the current error but retains the counter. The counter resets on dispatcher restart or scenario deletion.
   * - ``training/execution/*``
     - The trainer's execution calls: a weight trainer's job submissions and polls, including checkpoint evaluations, or a harness trainer's candidate preparation.

Each operation family (``ingest/write``, ``training/execution`` and the request families) has these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``started_total``, ``completed_total``, ``failed_total``
     - calls started, completed and failed
   * - ``active``
     - calls in flight; zero or one for execution calls, which are serial
   * - ``elapsed_seconds``
     - age of the oldest active call; zero when idle
   * - ``duration_seconds_total``
     - sum of the durations of finished calls; overlapping calls can add up faster than wall time
   * - ``last_duration_seconds``
     - duration of the last finished call

To get the mean latency over an interval, divide the change in ``duration_seconds_total`` by the change in completed plus failed calls.

How to read the samples:

- A completed execution call is not necessarily a commit: a poll of a running job that saved no new checkpoint commits nothing.
- The measurements are not persisted training history. Execution measurements reset when their trainer is rebuilt, including recovery; ``training/started_at_seconds`` identifies that reset. Request and ingestion measurements reset when the scenario is rebuilt; ``operations/started_at_seconds`` identifies that reset.
- Malformed HTTP payloads and headers, and failures before Reef resolves the scenario, are outside these measurements.
- Additional finite numeric processor status fields appear under ``operations/processor/``. Sampling does not advance the processor or consume its queue. If ingestion or a commit holds the trainer lock, that sample omits the queue gauges and still reports the execution measurements.
- Samples contain no request bodies, record IDs, scenario names or error messages. Uploading them configures no alert notifications. The inference engines keep their own high-frequency monitoring.

Commit correlation
~~~~~~~~~~~~~~~~~~

Commit metrics carry ``experiment/provider``, ``experiment/project``, ``experiment/group`` and ``experiment/run_id``. Use them to open the run from a Reef version, and use the run's ``reef/training_job_id`` to go the other way.

The job listing, ``GET /reef/scenarios/{scenario}/jobs``, gives the run of each training job as ``experiment``. Its ``url`` is the URL that the open W&B run gives. An offline run gives none: the URL is then ``https://wandb.ai/<entity>/<project>/runs/<run_id>``, where the run is once it is synced, with the configured ``entity``. Without a configured ``entity``, the URL of an offline run is ``null``.

With ``upload_checkpoints: true``, Reef reads the files of each committed checkpoint from the artifact backend and uploads them as a W&B artifact. Otherwise a run names its checkpoints by their artifact IDs only.

Record tracing
--------------

Record tracing exports every accepted record and every committed training step as OpenTelemetry spans, so a tracing backend shows what an agent did in a scenario and which version it trained. It is optional, off by default, and independent of experiment tracking. Reef speaks OTLP over HTTP and names no vendor: point it at Langfuse, Arize Phoenix, Jaeger, Grafana Tempo or an OpenTelemetry Collector. Install ``reef-infra[opentelemetry]``.

.. code:: yaml

   observability:
     tracing:
       enabled: true
       endpoint: https://cloud.langfuse.com/api/public/otel/v1/traces  # full OTLP/HTTP traces URL
       authorization: ${LANGFUSE_AUTH}  # the backend credential, sent as the Authorization header
       service_name: reef              # optional resource service.name

.. container:: config

   ``endpoint``
       the complete traces URL

   ``authorization`` : ``REEF_TRACING_AUTHORIZATION``
       the credential that the backend expects in its ``Authorization`` header: ``Basic <base64 public:secret>`` for Langfuse, ``Bearer <token>`` for most others. Write it as an environment reference, or omit it and export the variable. The startup report masks it, and it never appears in a span or a log line

   ``headers``
       a mapping for a backend that expects its credential under another header name, such as ``x-api-key``. The startup report masks every header value. ``Authorization`` is rejected here, so the credential has one place

   ``service_name``
       optional resource ``service.name``

When no endpoint, credential or header is configured, the exporter reads the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_HEADERS`` environment variables instead.

Each accepted inference record becomes the root span of its own trace, named ``chat <model>``. Its trace and span ids derive from the scenario and record id, so a restart or a second Reef host produces the same ids. The span carries:

* ``session.id`` and ``reef.scenario``: the scenario.
* ``reef.agent_record_id`` and ``reef.request_type``: the receipt and record kind.
* ``gen_ai.request.model``, ``gen_ai.response.model``, ``gen_ai.response.id``, ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens`` and ``gen_ai.response.finish_reasons`` from the provider request and response, following the OpenTelemetry GenAI semantic conventions.
* ``reef.artifact_id``: the artifact that answered the request.
* ``reef.tags``: the ``x-reef-tag-*`` request tags.

Other records become these spans:

* A feedback record becomes a ``feedback`` span inside the trace of the first inference that it references, with ``reef.score`` and ``reef.references``; further references are span links.
* A training commit adds a ``training commit N`` span with the commit number as ``reef.commit_number``, component artifact IDs, job id, consumed record counts, its component as ``reef.component`` when the scenario runs several trainers, and its scalar metrics as ``reef.metrics.*``.
* Every record that the commit consumed gets one ``trained in commit N`` child span, linked back to the commit span.

Records carry one timestamp, so their spans have zero duration and start at the record's creation time.

For streamed provider responses, Reef reads token usage from the captured SSE events, including Chat Completions usage chunks, Responses terminal events and Anthropic message usage. If the provider sends no usage, token counts remain absent; for Chat Completions, request ``stream_options: {include_usage: true}`` when supported. The stored response and the forwarded stream remain unchanged.

Spans carry the exchange itself: the request messages and the reply as JSON in ``gen_ai.input.messages`` and ``gen_ai.output.messages``, and feedback text in ``reef.feedback``. The backend therefore sees the scenario's traffic; point tracing only at a backend that you trust with it. Export failures go to the service log and never fail record acceptance or a commit. The exporter batches spans in a background thread and flushes them during graceful shutdown.
