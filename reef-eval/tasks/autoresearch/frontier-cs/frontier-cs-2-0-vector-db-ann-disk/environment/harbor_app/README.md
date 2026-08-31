# Vector DB Disk Skeleton

This is a starter project for the Vector DB ANN Disk task. You may use it,
modify it, or replace it entirely. The judge only requires that `/app` builds
with:

```bash
cargo build --release
PORT=<port> cargo run --release --quiet
```

and serves the required `/load` and `/search` HTTP endpoints.

`/load` receives `index_path`, `vector_path`, optional `vector_dtype`, and
optional `pq_compressed_path`/`pq_pivots_path` fields. `vector_dtype` may be
`float32`, `uint8`, or `int8`; do not infer it from the filename. The starter
also accepts legacy `graph_path` as an alias for `index_path`, and
`pq_vector_path` as an alias for `pq_compressed_path`.

Implement an online ANN service. Timed queries are evaluator inputs and may be
perturbed, resampled, or otherwise varied between runs. Do not rely on fixed
query files, query ordering, ground-truth files, baseline files, or any
judge-side sidecar data. Reading/caching `truth.bin`, `baseline.json`, or
precomputed answers from the benchmark directory is outside the task contract.

After a valid graph-based ANN implementation exists, do not spend most effort
on small parameter sweeps over search-list size, beam width, or cache limits.
Prefer real algorithmic and I/O improvements: better graph traversal,
batching/asynchronous disk reads, candidate management, vector/PQ distance
computation, and cache design are more likely to produce meaningful speedups
than repeatedly tuning a few constants.

The Harbor environment uses the Ubuntu `apt` Rust toolchain:

```text
rustc 1.75
cargo 1.75
```

Pin crate versions if newer transitive dependencies require a newer Rust
compiler.

The Harbor task provides the following resource budget:

```text
vCPUs: 8
memory: 8 GiB
query concurrency: 8
timed queries per worker: 64
```

## Attribution

This starter skeleton is adapted from KCORES/vector-db-bench, licensed under
the MIT License. See `LICENSE.KCORES` for the upstream notice.
