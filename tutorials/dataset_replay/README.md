# Dataset replay, then continual records

This CPU-only example shows a `DataProcessor` reading a stored dataset twice,
then consuming later records once, in the same scenario. Each batch follows
Reef's normal artifact publication and commit path. The demo backend publishes
`batch.json` descriptions; it does **not** train model weights or need a model
server.

Run from the repository root with the development environment installed:

```bash
.venv/bin/python -m tutorials.dataset_replay.demo
.venv/bin/python -m pytest tests/reef_service/test_dataset_replay_example.py -q
```

The script prints metric dictionaries describing this sequence:

```text
epoch 1: [a, b], [c]
epoch 2: [a, b], [c]
stream:  [live]
```

The three historical records are written once. The processor holds at most one
batch of trajectories and reads storage in pages no larger than `batch_size`,
including partial final batches. It never builds a list of the entire dataset.

## Wiring

`Recipe.build(scenario, records, algorithm_state=...)` already receives storage
and the latest committed state. [demo.py](demo.py) injects those into
[ReplayProcessor](processor.py), using the existing interfaces:

- `RecordStore.replay_page` reads retained records in append order. A consumed
  record remains readable until storage evicts it under capacity pressure.
- The processor's `ready` and `make_training_batch` select a batch with its
  epoch, phase and proposed next cursor. Its own cursor chooses repeated reads;
  the trainer's usual forward cursor continues to deliver TRAIN instructions.
- The backend carries `batch.next_progress` in `TrainStepResult.state["replay"]`
  alongside any model/optimizer state, and records the epoch in metrics.
  Reef persists both in the normal commit. The sample backend demonstrates this
  handoff; using another backend requires preserving it there as well.
- On restart, the recipe reconstructs the processor from that committed cursor.
  A batch prepared or acknowledged but never committed is selected again. A
  committed batch is not repeated within its pass. This does not independently
  undo external model updates: a real backend must honor Reef's existing
  training-job recovery contract too.

This processor does not add sample IDs to the trainer's permanent consumption
set: the durable `(epoch, after_sequence)` cursor is its consumption record.
Optional TRAIN instructions still use the shared acknowledgement and failure
handling. Epoch progress is fixed-size metadata; the commit log and storage's
retry metadata still grow with history.

## Choosing the dataset and continuing online

`dataset_last_sequence` selects an **already stored prefix** for cold start.
The demo obtains it from the last imported record's `get_for_audit` result.
It is recipe policy, not a dataset-end HTTP operation. Capture the range after
the desired import has finished; an empty read while uploading is not proof
that all intended records have arrived. This example does not wait for future
records up to a not-yet-written boundary.

Records with sequences above that fixed boundary are deferred until the N
passes finish, then processed once in arrival order. Records arriving during
the cold start therefore do not change the dataset between passes. The example
processes INFERENCE records directly; report correlation and reward selection
belong in a recipe that needs them.

The demo writes directly through `RecordStore` to run without an HTTP server.
In a deployment, the client JSONL importer writes through `/reef/records/batch`
and subsequent chat calls append INFERENCE records to the same scenario/store.
Their transport does not change the processor's selection logic. Do not swap
out the processor to transition from cold start to online consumption.

`epochs` here means passes over the selected dataset, with normal commits per
batch and `dataset_epoch` on each commit. `StepScheduling.epochs` instead repeats
optimizer work **inside one reserved batch**, before that batch commits; use 1
there unless that additional repetition is intentional.

Storage owns deletion throughout. Missing/evicted records are skipped and the
processor advances to surviving records; Reef's existing warnings and
`RecordLoss` metrics report the loss. N complete passes are only possible while
the inputs remain retained. Changing the range or epoch count on recovery is
rejected rather than silently restarting training.

The contract tests cover two passes over three records at batch size two, a
failed instruction, a crash before commit, restart during a pass, arrivals during
cold start, restart after the transition to streaming, bounded pages, and capacity
eviction. This is a reference for processor authors, not a large-dataset benchmark
or an SFT recipe.
