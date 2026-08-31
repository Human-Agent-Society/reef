# DuckDB E2E Query Optimization Starter

The workspace is expected to contain a pinned DuckDB checkout at:

```text
/app/duckdb
```

Modify DuckDB source code to improve end-to-end TPC-H performance while
preserving correctness. Submit a patch against the clean checkout:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

Submit early. The judge queue is asynchronous, and a full local DuckDB build in
the agent container can take long enough to exhaust the agent budget. It is
better to enqueue a small buildable-looking patch first, then keep compiling,
reviewing, or iterating while the judge scores that snapshot.

The judge applies `/app/solution.patch` to a clean DuckDB tree, builds DuckDB
with the TPC-H extension enabled, checks correctness, and scores relative
speedup against vanilla DuckDB.

The public workload family is DuckDB's built-in TPC-H q1 through q22 query set.
In normal DuckDB development workflows these queries can be inspected or run
through the TPC-H extension, including `CALL dbgen(...)` and `PRAGMA tpch(n)`.
Use this as workload context, not as a license to hard-code table names, query
numbers, or benchmark details.

The Harbor budget is 8 vCPUs, 16 GiB memory, and 32 GiB storage. DuckDB query
runs are single-threaded and use fixed judge settings: `memory_limit='6GB'`,
`max_temp_directory_size='2GB'`, an isolated temp directory, and
`preserve_insertion_order=false`. Build and query subprocesses run in a clean
environment rather than inheriting terminal or Harbor environment variables.

Only optimizer, join, filter, and narrowly related planning/execution changes
are valid. New C++ files are allowed if the build-system diff only wires those
files into existing DuckDB targets.
