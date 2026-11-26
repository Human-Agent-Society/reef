State Model: Records, Artifacts, and Operations
==============================================

Use these contracts when implementing storage or artifact backends and debugging
recovery. Operators should start with `recovery
<../user-guide/operate.rst#restart-and-recovery>`__ and `backups
<../user-guide/storage.rst#back-up-a-deployment>`__.

Records
-------

Each trainer tracks its own consumed record IDs. Consumption does not delete
records. Retention removes bodies, so recovery cannot reconstruct evicted content.
See the `record interface <../../reef/storage/scenario/records.py>`__.

The snapshot chain
------------------

An artifact is an immutable component output with an opaque ``artifact_id``.
A selection maps components to artifacts. An operation records a change or
training outcome, including skipped and rejected results. Selecting an existing
artifact creates an operation, not another artifact.

Repository publication is separate from the active selection. A snapshot's
revision locates bytes; it is not a public artifact ID.
A backend must persist each artifact ID's binding with its snapshot and reject
rebinding to different files or metadata. Preserve bindings used by commits.
Publication must reject a stale parent. See `RepositoryBackend
<../../reef/artifact/repository.py>`__.

Commit ordering
---------------

The scenario store's transaction is the commit point. It appends the commit
and updates sample training counts together.
``append_commit(expected_commit_number=..., commit=...)`` must:

- Accept only the correct scenario's expected successor.
- Return the stored record for an identical retry, ignoring only ``recorded_at``.
- Reject different content at that number or a stale expected number.

A durable artifact commit follows this order:

#. Publish durable content and artifact bindings.
#. Append the commit, including the activation decision.
#. Install committed selections and synchronize the repository pointer.
#. Apply the trainer's prepared state.

Activation commits before inference loads. Pointer synchronization failure does
not undo the commit. Later commits must synchronize first. After an ambiguous
append result, read history or retry that commit before publishing again.
See `ScenarioCommitter <../../reef/control/scenario/committer.py>`__.

Recovery reconciles history with checkpoints and repairs repository pointers.
It can adopt a checkpoint ahead of history. Each trainer restores its state
and consumed IDs, then rereads retained records from the beginning, skipping
those IDs. The stored high-water mark is not a recovery cursor.

Training runs outside the transaction. Retrying a job must reuse its result or
reconcile progress; unsaved optimizer work can repeat. See `checkpoint cadence
<../user-guide/operate.rst#set-the-checkpoint-cadence>`__ for backend-specific units.

Prepare-and-commit cycles are serialized. External effects must still be
idempotent or recoverable. Expected commit numbers do not replace the deployment's
single active control writer. Multiple trainers require durable history.
