# Vector DB Skeleton

This is a starter project for the Vector DB ANN task. You may use it, modify it,
or replace it entirely. The judge only requires that `/app` builds with:

```bash
cargo build --release
PORT=<port> cargo run --release --quiet
```

and serves the required `/bulk_insert` and `/search` HTTP endpoints.

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
memory: 16 GiB
query concurrency: 8
timed queries per worker: 64
```

## Attribution

This starter skeleton is adapted from KCORES/vector-db-bench, licensed under
the MIT License. See `LICENSE.KCORES` for the upstream notice.
