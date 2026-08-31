You are solving a Frontier-CS 2.0 open-ended optimization problem.

Create a cpp solution at `/app/solution.patch`. You can call `bash /app/submit.sh` at any time to enqueue a snapshot for the same black-box judge used by the final verifier. Submissions are asynchronous: use `bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect results. The evaluator implementation is intentionally not available in the agent workspace. Read `AGENT.md` for the shared submission workflow.

Problem id: `rocksdb_native_compaction_policy`
Language: `cpp`
Time limit: `10800s`

Original problem statement:

RocksDB Native Compaction Policy

Goal

Improve leveled compaction selection in RocksDB v10.10.1 while preserving database correctness. The workspace contains the pinned source tree at /app/rocksdb. The judge applies your patch to a clean checkout at commit 4595a5e95ae8525c42e172a054435782b3479c57, rebuilds RocksDB, and compares it with the unmodified build.

Workload

The judge runs native RocksDB workloads with changing write, point-read, scan, range-delete, snapshot, time-series, and multi-column-family phases. Options such as write-buffer size, L0 thresholds, level sizes, value sizes, and cache size vary by case. Leveled compaction is always used; universal and FIFO compaction are outside this task.

Feedback uses one fixed development case per workload family plus a smoke case. Final verification uses two fixed judge-derived seeds per family. Final seeds are not included in the agent workspace or task configuration.

Submission

Submit /app/solution.patch. After editing the checkout, run:

  bash /app/make_submission.sh
  bash /app/submit.sh

make_submission.sh rejects changes outside the editable surface instead of silently omitting them. An empty patch is a valid zero-score baseline.

Editable surface

  db/compaction/compaction_picker.cc
  db/compaction/compaction_picker.h
  db/compaction/compaction_picker_level.cc
  db/compaction/compaction_picker_level.h
  db/version_set.cc

The task covers leveled compaction selection: choosing levels and files, computing file priority, handling L0 pressure, intra-L0 decisions, marked files, tombstone-driven picks, and picker expansion. Output-file cutting is not part of the editable surface.

Correctness

Correctness is a hard gate. The candidate must build and complete every case without crash, timeout, deadlock, or background error. The harness checks point reads, range deletes, held snapshots, column families, database reopen, and a complete iterator comparison against its logical oracle.

Patches may not inspect judge identity, paths, environment variables, process state, clocks, profile names, or infrastructure details. New preprocessor directives and changes outside the five listed files are rejected. Submitted binaries and local benchmark output are ignored.

Scoring

Each case runs one isolated vanilla/candidate pair concurrently on the same deterministic operation stream. Final verification uses two seeds per workload family. The case objective is a weighted geometric mean of lower-is-better ratios:

  40% write amplification
  25% read amplification
  20% pre-drain space amplification
  15% trusted compaction output required after the policy run

The initial database load is compacted through a fixed manual path, fingerprinted, and excluded from scored counters. A candidate that changes this base state is invalid. Later writes and compactions run in fixed phase-boundary cycles so each picker decision starts from a reproducible state. Pre-drain memtables are flushed, actual table-file bytes are measured, and metadata is captured while background work is paused. After each policy run closes, an unmodified judge binary reopens the database and runs the normal vanilla policy until an additional pass produces no compaction output. It verifies the logical data before and after this residual drain. Trusted residual output is added to write amplification, and the policy plus residual drain is scored separately as 1 + output bytes divided by the larger of user-write bytes and 64 MiB, so deferred work cannot lower the measured cost. Final score uses the mean paired log improvement with a small cross-case dispersion penalty. Robust gains at or below 1.005x are treated as measurement noise and earn zero; a robust 1.017x aggregate reaches 100. Invalid or failed submissions score zero and report a strongly negative unbounded score, so they always rank below valid submissions. A positive score requires at least 40% and at least two workload families to improve by 0.5% or more, and at most one family may regress by more than 2%. Severe per-case or per-metric regressions reduce or cap the score. Extreme runtime or stall regressions are validity guards; otherwise wall-clock throughput, latency, and stall time are diagnostics, not score terms.

Feedback exposes validity, build status, aggregate gain, worst-case gain, component floor, workload breadth counts, average intra-L0 decision delta per case, case count, and a coarse score band. It does not expose per-case metrics, seeds, or final profile order.

Resources

  vCPUs: 8
  memory: 16 GiB
  storage: 32 GiB
  build timeout: 7200 seconds
  per-run timeout: 1800 seconds
