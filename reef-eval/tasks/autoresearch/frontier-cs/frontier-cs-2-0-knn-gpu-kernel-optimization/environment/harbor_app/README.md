# Brute-force k-NN kernel optimization — submission workflow

You are optimizing the `knnlib` package at `/app/knnlib`. Edit the package
(rewrite the internals of `knn`, add Triton kernel modules under `knnlib/`),
then submit a patch.

## Iterate locally

Run the public self-test to check correctness and get a rough speed signal on
the two public shapes (needs a GPU in the agent container):

```bash
bash /app/public_test.sh
```

## Submit

```bash
bash /app/make_submission.sh      # writes /app/solution.patch (knnlib diff)
bash /app/submit.sh               # enqueues it for the black-box judge
```

Submissions are asynchronous. Submit early and keep improving; use
`bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect
results.

## Rules

- Only files under `knnlib/` may change.
- Do not import external optimized libraries (write the kernels yourself), and
  do not access the environment, spawn processes, or use the network.
- Keep the public `knn(...)` signature and return contract unchanged.
- Nearest-neighbor quality is gated (recall@k vs an exact baseline); do not
  sacrifice correctness for speed beyond the allowed tolerance.
