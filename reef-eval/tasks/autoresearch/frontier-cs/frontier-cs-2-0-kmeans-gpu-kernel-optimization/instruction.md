You are solving a Frontier-CS 2.0 open-ended optimization problem.

Create a python solution at `/app/solution.patch`. You can call `bash /app/submit.sh` at any time to enqueue a snapshot for the same black-box judge used by the final verifier. Submissions are asynchronous: use `bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect results. The evaluator implementation is intentionally not available in the agent workspace. Read `AGENT.md` for the shared submission workflow.

Problem id: `kmeans_gpu_kernel_optimization`
Language: `python`
Time limit: `10800s`

Original problem statement:

# GPU K-Means Kernel Optimization

## Problem

You are given a small GPU K-Means library, `kmeanslib`, in the Harbor workspace
at `/app/kmeanslib`. Its public entry point is a **single Lloyd iteration**:

```python
kmeanslib.step(x, centroids) -> (labels, new_centroids)
```

`x` is an `(N, D)` **bfloat16** CUDA tensor of points and `centroids` is the
current `(K, D)` bfloat16 tensor. One call performs exactly one Euclidean
(squared-L2) Lloyd step: assign every point to its nearest centroid
(`labels`, an `(N,)` int64 full assignment), then recompute each centroid as the
mean of its assigned points (`new_centroids`, `(K, D)`; empty clusters keep their
previous centroid). All data is bfloat16 — treat it as the fixed working
precision. The shipped implementation is a straightforward, correct PyTorch
version (a bf16 matmul + argmin, then a scatter update).

**The judge owns the iteration loop.** It fixes the data and the initial
centroids, then calls your `step` a fixed number of times per workload, feeding
each call's `new_centroids` into the next. You do not control the data, the
initial centroids, or how many iterations run — you control only how fast a
single `step` executes.

Your goal is to make `kmeanslib.step` **as fast as possible** on the GPU while
producing the same clustering. You may rewrite the internals of the package
however you like and add new modules (including Triton kernels) under
`kmeanslib/` — in particular you may **fuse the assign and update into a single
kernel**. The public function signature and return contract above must not
change, and each result must remain a deterministic function of its inputs.

## Workload

The graded workloads are a family of held-out dense clustering problems that
vary `(N, D, K)`, spanning small/medium/large point counts and both wide-feature
(large `D`) and many-cluster (large `K`) regimes, in **bfloat16**, with a fixed
number of judge-owned `step` calls per workload and caller-supplied initial
centroids. How the point sets are generated is not disclosed — implement a
**general** exact Lloyd step (correct assignment to the given centroids + exact
cluster-mean update) and do not special-case the data.

## Iterate on a GPU

The agent workspace has no GPU, and the data generator is judge-only (it is not in
your image). To check your current code, **submit it to the judge** — it runs the
exact graded workloads on a GPU and returns your per-workload result + score. This
one command packages, submits, and waits for the result:

```bash
bash /app/public_test.sh
```

It reports, per graded workload, whether your result passes the quality gate and
your speedup, plus the geometric-mean speedup and your score (0-100) — the
identical evaluation used for your final grade. (Equivalently:
`bash /app/make_submission.sh && bash /app/submit.sh`, then poll with
`bash /app/submissions.sh`.) Submissions are asynchronous; submit early and iterate.

## Submission

The submitted artifact is a patch over the `kmeanslib` package:

```text
/app/solution.patch
```

After editing `/app/kmeanslib`, generate and submit:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

Submissions are asynchronous; submit early and keep iterating. The judge applies
your patch to a clean copy of `kmeanslib` and times it against the original
baseline on a GPU, on the same seeded data and initial centroids.

## Correctness

Correctness is a gate. After running the fixed loop of `step` calls, the judge
computes the **inertia of your returned clustering** — the sum of squared
distances of every point to `new_centroids[labels]`, i.e. using **the `labels`
and `new_centroids` your final `step` returned** — and compares it to the
baseline's, requiring your inertia to stay within a small relative tolerance.
Because the gate reads your returned `labels`, they must be the real full `(N,)`
nearest-centroid assignment to the `centroids` each `step` was given: returning
fake/empty labels, a partial or sub-sampled assignment, or centroids computed
from a subset of points all inflate this inertia and fail the gate. Each `step`
returns `labels` `(N,)` and `new_centroids` `(K, D)`. Crashes, non-finite output,
wrong shapes/dtypes, timeouts, and clustering that regresses beyond the tolerance
are penalized before speed is considered. The working precision is fixed at bfloat16 for every solution (the
inputs are bfloat16), so it is not a tuning knob — speed comes from the kernel,
not from the arithmetic precision.

## Scoring

Valid submissions are scored by speedup relative to the baseline on the same
hardware and workloads. For each workload:

```text
speedup = baseline_time / your_time
```

The objective is the geometric mean of per-workload speedups, so broad speedups
are preferred over a single large outlier. A result no faster than the baseline
earns 0; regressions earn 0. The raw geometric-mean speedup is reported in the
evaluator metrics.

## Patch Policy

The evaluator validates the patch before running it. Only Python files under the
package may change:

```text
kmeanslib/**
```

New Python modules inside `kmeanslib/` are allowed. Patches may not: modify
anything outside `kmeanslib/`; import or call an external optimized ML/kernel
library (write the kernels yourself); read or write environment variables, spawn
processes, or access the network; or otherwise tamper with the measurement
framework. The judge owns the loop, the data, and the initial centroids and
re-verifies quality from your final centroids, so faking labels, skipping the
real assign/update work, or caching results across calls does not help.

## Resource Budget

```text
GPU: single Modal GPU (H100 reference; the Triton paths also run on L40S / A100)
Agent container: CPU-only (GPU work is offloaded to Modal)
```
