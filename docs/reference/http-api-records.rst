HTTP API: records and operation history
=======================================

Import existing inferences and reports as records, and read the records and the operations that a scenario keeps.

Record
------

Reef stores every inference exchange and every report as an ``AgentRecord`` with a record ID. The same ID with different content returns a conflict; it never overwrites. Reef tracks consumed records, so retried reports, and late reports whose references already trained, are not counted twice.

Records remain readable after training consumes them (`Record and commit history`_). Retention deletes the oldest record items only when the shared capacity budget is exceeded (20 GiB by default), and keeps durable loss totals. See `Configuration <configuration.rst>`__ and `Python API <python-api.rst>`__.

Import records for cold-start learning
--------------------------------------

``POST /reef/records`` accepts an existing inference or report without calling an inference provider. It uses the usual Bearer token and ``x-reef-scenario`` header. Submit one record per request and use a stable ``agent_record_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/records \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'x-reef-scenario: my-agent' \
     -H 'Content-Type: application/json' \
     -d '{
       "agent_record_id": "offline-0001",
       "request_type": "inference",
       "payload": {
         "model": "example-model",
         "messages": [{"role": "user", "content": "What is 2 + 2?"}],
         "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}
       }
     }'

The response is ``{"agent_record_id":"offline-0001","scenario":"my-agent", "request_type":"inference"}``. HTTP 200 acknowledges record admission; it does not mean that a training step has finished.

- The envelope has exactly the three fields shown; Reef rejects other fields. The scenario comes only from the header.
- ``request_type`` is ``inference`` or ``report``. Send a training request to ``POST /reef/train`` instead.
- An inference payload follows the selected recipe's recorded-input contract: native provider request fields with an existing ``response``, or another complete example that its processor understands.
- A report uses the same schema and reference validation as ``POST /reef/report``. Import the referenced inferences first.

Reef answers these statuses:

- 200 for an identical retry, including after capacity eviction: it returns the existing receipt.
- 400 for an invalid envelope, report schema or reference.
- 400 for an inference record with an image, when the deployment's recipes include a weight-training recipe, because training renders text only. A batch then stores nothing.
- 401 for an authentication failure.
- 409 for a known ID with different content.

Import does not assign the active artifact to externally generated examples. Training reads no token IDs or log probabilities from an imported record: the training component renders the recorded exchange with the model's chat template. To train on complete ATIF trajectories, see `Data items <http-api-data-items.rst>`__.

Imported records enter the same scenario storage and wake the same training worker as live traffic. Keep the same scenario header for later inference calls: their records continue through the same processor, with no processor replacement or end-of-dataset signal. A processor that needs report feedback needs it for imported and live records alike. Submit each example's feedback promptly so that completed jobs can drain:

.. code:: json

   {"agent_record_id": "offline-score-0001", "request_type": "report",
    "payload": {"references": ["offline-0001"], "score": 1.0}}

The selected recipe and processor own job size, buffering, retention and training policy. Uploading can overlap training. This API adds no dataset epochs, no cold-start completion barrier and no automatic algorithm switch. Use the recipe's progress and commit status to learn when training has consumed the imported examples.

Batch imports and resume
~~~~~~~~~~~~~~~~~~~~~~~~

``POST /reef/records/batch`` accepts ``{"records": [record, ...]}``, where each record uses the envelope of ``POST /reef/records``. The batch holds 1 to 1000 records within the 10 MiB body limit. All records share the request's scenario and tag headers. Put an inference before the reports that reference it, including within one batch.

Reef validates the whole batch and stores it in one database transaction. The response is ``{"records": [receipt, ...]}``, in input order. A successful import wakes the consumer once after commit. Atomicity applies to one request, not to the whole dataset or to later training.

- Malformed records or missing references return 400, and conflicting IDs return 409. Neither stores any new record of the batch.
- Identical retries are accepted, including after training or capacity eviction.
- Oversized requests return 413.

For a large local dataset, use the importer of the standalone ``reef-client`` package, which needs no Reef service or training dependencies. Save one envelope per line in an immutable UTF-8 JSONL file. Keep the original record IDs and report references: the importer generates no IDs and converts no other dataset schema.

.. code:: bash

   # Authentication is read from REEF_TOKEN when set.
   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

The importer:

- **Batches.** It reads incrementally and sends batches limited by ``--batch-size`` (default 128 records) and ``--max-batch-bytes`` (default 524288 bytes). It buffers one batch and one look-ahead record. One JSONL line must fit the byte limit; raise the limit, below 1048576, for larger records.
- **Progress.** It saves progress to ``records.jsonl.reef-import.json``, or to ``--progress PATH``: the destination, scenario, source checksum and last acknowledged byte offset and count, with no token or record payload. Each run first hashes the whole file to verify its identity, then seeks to the saved offset. The source must not change during the import. A local file lock prevents two runs on one progress file.
- **Resume.** Run the same command again after an interruption. A modified source or another destination needs a separate progress file; known IDs still deduplicate at the server.
- **Retries.** It retries transient connection failures and HTTP 408, 429, 500, 502, 503 and 504 up to ``--retries`` times (default 3), with backoff and unchanged IDs. If a reply is lost after commit, server deduplication prevents a second insert. A permanent error stops the import without advancing that batch's progress; earlier acknowledged batches stay stored.
- **Count.** The reported count includes every acknowledged source row, including retries of records already present. It does not count trained samples.

The importer supports local JSONL on platforms with POSIX file locking, including Linux and macOS. It does not convert Parquet or Hugging Face schemas, and it runs no server-side S3 import.

The batch limits do not bound processor memory. A processor that waits for reports retains inference records, so place each report close to the inferences that it references.

Record and commit history
-------------------------

All these reads require service authentication and return ``Cache-Control: no-store``. Payloads and recorded metrics can contain request content, so a platform must bind the upstream scenario to the authenticated workspace. Each response is a live read: concurrent training and retention can change later pages.

**Records.** ``GET /reef/scenarios/{scenario}/records`` reads retained record metadata, including consumed records, oldest first:

- ``after_sequence`` defaults to 0, and ``limit`` to 50 (1 to 100). ``next_after_sequence`` is null at the end.
- ``request_type`` (``inference``, ``report`` or ``train``) lists one type only; another value returns 400.
- Each row contains ``sequence``, ``agent_record_id``, ``request_type``, ``created_at``, ``references``, ``artifact_id`` and the payload's ``score`` field. ``artifact_id`` names the artifact that answered an inference call, or is null when the record names none. The row has no payload.

``GET /reef/scenarios/{scenario}/records/{record_id}`` returns that metadata and the stored ``payload``. A missing body returns 404: it may have expired, or Reef never retained it. A read never reactivates a record or changes retention.

**Operations.** ``GET /reef/scenarios/{scenario}/operations`` returns ``{scenario, operations, next_after_commit_number}``, oldest first:

- ``after_commit_number`` defaults to 0, and ``limit`` to 50 (1 to 100). ``next_after_commit_number`` is null at the end.
- ``/commits`` remains an alias, with the array named ``commits``.
- Repeat ``record_id`` to keep the operations whose ``consumed_ids`` contain any requested ID (at most 100 IDs of 1 to 256 characters). This tests exact set membership, not whether a record was eligible for training.
- Each row has ``commit_number``, ``operation``, ``operation_verified``, ``recorded_at``, ``components``, ``component``, ``activated``, ``consumed_ids`` and ``metrics``. ``components`` maps component names to the artifact IDs of the operation. ``component`` identifies the changed component when present. The row has no private storage locator, algorithm state or checkpoint recovery data.
- Operations remain readable after trace bodies expire.

The operations are:

- ``training``: a training outcome, including steps that create no artifact. ``activated`` reports whether the step selected its new artifact to serve.
- ``activate``: select an existing artifact for one component.
- ``restore``: restore the complete active selection from an earlier step.
- ``train_from``: change one trainer's starting artifact.
- ``import``: create an artifact from an external source.
- ``sync``: create a harness artifact and select it for serving and training in one operation. Its ``activated`` value is true.
- ``serve`` and ``stop_serving``: add or remove an artifact that the scenario serves beside the active one.

A ``served`` request appends ``activate``, ``serve`` and ``stop_serving`` operations, one for each artifact that changes (`Set what a scenario serves <http-api-serving.rst#set-what-a-scenario-serves>`__). Activation, restore and serving create no artifacts or repository revisions. An import or a training step can create an artifact without changing what serves. Restore appends a new operation; it does not remove or rewind history. Recovery replays serving selections and training starts independently.

Consumption does not prove that a snapshot made from the records serves. Reef assigns no learning state, explanation or evaluation result to these rows: the console that reads them owns that interpretation.
