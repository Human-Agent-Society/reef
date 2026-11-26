HTTP API: records and operation history
=======================================

Import inference exchanges and reports, then inspect retained records and operation history.

Record
------

``AgentRecord`` stores inference exchanges and reports by unique ID. Changed content conflicts; retries and late reports never train references twice. Consumed records remain readable. Retention removes oldest items above the shared capacity budget (default 20 GiB), preserving loss totals (`Record retention <../user-guide/storage.rst#record-retention>`__).

Import records for cold-start learning
--------------------------------------

``POST /reef/records`` stores an inference or report without calling a provider. Use normal authentication, ``x-reef-scenario`` and stable record IDs:

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

HTTP 200 returns ``{agent_record_id, scenario, request_type}``, acknowledging admission rather than completed training.

- Only ``agent_record_id``, ``request_type`` and ``payload`` are allowed. Scenario comes from the header.
- Types: ``inference`` or ``report``; training requests use ``POST /reef/train``.
- Inference payloads must satisfy the recipe: provider requests with ``response``, or another supported complete example.
- Reports follow `report validation <http-api-inference.rst#report>`__. Import referenced inferences first.

Reef answers these statuses:

- 200 for an identical retry, including after capacity eviction: it returns the existing receipt.
- 400 for an invalid envelope, report schema or reference.
- 400 for images on deployments with weight-training recipes; entire batches reject atomically.
- 401 for an authentication failure.
- 409 for a known ID with different content.

Imports receive no active artifact ID. Training renders exchanges with the model's chat template, ignoring imported token IDs and log probabilities. For ATIF, use `data items <http-api-data-items.rst>`__.

Imported and live records share storage and processing. Keep the same scenario header and provide required reports promptly:

.. code:: json

   {"agent_record_id": "offline-score-0001", "request_type": "report",
    "payload": {"references": ["offline-0001"], "score": 1.0}}

Recipe rules control training, buffering and retention. Uploads can overlap training; they add no epochs, completion barrier or algorithm switch. Check progress and commits for consumption.

Batch imports and resume
~~~~~~~~~~~~~~~~~~~~~~~~

``POST /reef/records/batch`` accepts ``{"records": [...]}``: 1–1000 envelopes, maximum 10 MiB, sharing scenario and tag headers. Put inferences before their reports. Each request validates and stores atomically, returning ``{"records": [receipt, ...]}`` in input order and waking the consumer after commit. Dataset-wide atomicity and training are separate.

- Malformed records or missing references return 400, and conflicting IDs return 409. Neither stores any new record of the batch.
- Identical retries are accepted, including after training or capacity eviction.
- Oversized requests return 413.

``reef-client`` imports immutable UTF-8 JSONL, one envelope per line, without service/training dependencies. It preserves IDs and references and converts no schemas.

.. code:: bash

   # Authentication is read from REEF_TOKEN when set.
   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

The importer:

- **Batch limits:** ``--batch-size`` defaults to 128; ``--max-batch-bytes`` to 524288. One batch plus one look-ahead record is buffered. Each line must fit; byte limits must remain below 1048576.
- **Progress:** ``records.jsonl.reef-import.json`` or ``--progress PATH`` stores destination, scenario, checksum, acknowledged offset and count—no token or payload. Every run verifies the whole-file checksum. A file lock prevents concurrent progress writers.
- **Resume:** rerun the command. Changed sources or destinations need separate progress files; IDs still deduplicate.
- **Retries:** connection failures and HTTP 408/429/500/502/503/504 retry with backoff and unchanged IDs, up to ``--retries`` (default 3). Permanent failures preserve earlier batches without advancing failed-batch progress.
- **Count:** acknowledged source rows, including existing records; not trained samples.

Requires local JSONL and POSIX file locking (Linux/macOS); no Parquet, Hugging Face conversion or server-side S3 import. Batch limits do not bound processor memory. Keep reports close to referenced inferences.

Record and commit history
-------------------------

Authenticated history reads use ``Cache-Control: no-store`` and can expose request content. Platforms must bind scenarios to authenticated workspaces. Concurrent training/retention can change later pages.

**Records.** ``GET /reef/scenarios/{scenario}/records`` reads retained record metadata, including consumed records, oldest first:

- ``after_sequence`` defaults to 0, and ``limit`` to 50 (1 to 100). ``next_after_sequence`` is null at the end.
- ``request_type`` (``inference``, ``report`` or ``train``) lists one type only; another value returns 400.
- Rows contain ``sequence``, ``agent_record_id``, ``request_type``, ``created_at``, ``references``, responding ``artifact_id`` (or null), and payload ``score``; no payload body.

``GET /reef/scenarios/{scenario}/records/{record_id}`` adds ``payload``. Expired or unretained bodies return 404. Reads never reactivate records or affect retention.

**Operations.** ``GET /reef/scenarios/{scenario}/operations`` returns ``{scenario, operations, next_after_commit_number}``, oldest first:

- ``after_commit_number`` defaults to 0, and ``limit`` to 50 (1 to 100). ``next_after_commit_number`` is null at the end.
- ``/commits`` remains an alias, with the array named ``commits``.
- Repeat ``record_id`` to match any exact ``consumed_ids`` member: at most 100 IDs, each 1–256 characters. This does not test training eligibility.
- Rows: ``commit_number``, ``operation``, ``operation_verified``, ``recorded_at``, ``components`` (name→artifact ID), changed ``component`` when present, ``activated``, ``consumed_ids``, ``metrics``. Private locators, algorithm state and recovery data are excluded.
- Operations remain readable after trace bodies expire.

The operations are:

- ``training``: a training outcome, including steps that create no artifact. ``activated`` reports whether the step selected its new artifact to serve.
- ``activate``: select an existing artifact for one component.
- ``restore``: restore the complete active selection from an earlier step.
- ``train_from``: change one trainer's starting artifact.
- ``import``: create an artifact from an external source.
- ``sync``: create a harness artifact and select it for serving and training in one operation. Its ``activated`` value is true.
- ``serve`` and ``stop_serving``: add or remove an artifact that the scenario serves beside the active one.

``served`` records one operation per changed artifact (`Serving <http-api-serving.rst>`__). Activation, restore and serving create no artifacts; imports and training need not change serving. Restore appends history without rewinding it. Recovery replays serving and training starts independently.

A consumed record does not establish that its resulting artifact serves. Consoles must interpret operation history; Reef adds no learning state or evaluation result.
