# K-Means kernel optimization — submission workflow

You are optimizing the `kmeanslib` package at `/app/kmeanslib`. Edit the package
(rewrite the internals of `kmeans`, add Triton kernel modules under
`kmeanslib/`), then submit a patch.

## Iterate locally

Run the public self-test to check correctness and get a rough speed signal on
the two public shapes (needs a GPU in the agent container):

```bash
bash /app/public_test.sh
```

## Submit

```bash
bash /app/make_submission.sh      # writes /app/solution.patch (kmeanslib diff)
bash /app/submit.sh               # enqueues it for the black-box judge
```

Submissions are asynchronous. Submit early and keep improving; use
`bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect
results.

## Rules

- Only files under `kmeanslib/` may change.
- Do not import external optimized libraries (write the kernels yourself), and
  do not access the environment, spawn processes, or use the network.
- Keep the public `kmeans(...)` signature and return contract unchanged.
- Clustering quality is gated (inertia vs the naive baseline); do not sacrifice
  correctness for speed beyond the allowed tolerance.
