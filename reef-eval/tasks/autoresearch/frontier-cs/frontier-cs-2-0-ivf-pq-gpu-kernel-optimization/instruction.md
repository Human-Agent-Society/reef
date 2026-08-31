You are solving a Frontier-CS 2.0 open-ended optimization problem.

Create a python solution at `/app/solution.patch`. You can call `bash /app/submit.sh` at any time to enqueue a snapshot for the same black-box judge used by the final verifier. Submissions are asynchronous: use `bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect results. The evaluator implementation is intentionally not available in the agent workspace. Read `AGENT.md` for the shared submission workflow.

Problem id: `ivf_pq_gpu_kernel_optimization`
Language: `python`
Time limit: `10800s`

Original problem statement:

# GPU IVF-PQ Search Kernel Optimization

## Problem

You are given a small GPU approximate-nearest-neighbour library, `ivfpqlib`, in
the Harbor workspace at `/app/ivfpqlib`. It implements **IVF-PQ** search. Its
public entry point is:

```python
ivfpqlib.ivf_pq_search(index, Q, k, *, nprobe=None) -> (vals, ids)
```

`index` is a pre-built `IvfPqIndex` (an inverted-file product-quantization index:
`nlist` coarse cells, `m` PQ sub-codebooks, and the database rows stored as `(M,
m)` uint8 PQ codes). `Q` is an `(nq, D)` float32 CUDA tensor of queries, `k` is
the number of neighbours per query, and `nprobe` is how many inverted lists to
probe. It returns `vals` `(nq, k)` float32 (approximate squared-L2 distances to
each neighbour's PQ reconstruction) and `ids` `(nq, k)` int64 (original database
row ids, `-1` padded). The shipped implementation is correct but straightforward.

Your goal is to make `ivfpqlib.ivf_pq_search` **as fast as possible** on the GPU
while returning the same neighbours. You may rewrite the internals of the package
and add new modules (including Triton kernels) under `ivfpqlib/`. The public
function signature and return contract must not change, and the result must
remain a deterministic function of the inputs.

## What is (and isn't) timed

Only `ivf_pq_search` is timed. The `index` is **built once** per workload by a
frozen builder (`ivf_pq_build`) and handed to your search unchanged; index
construction is **not** part of the score, so optimizing `ivf_pq_build` buys you
nothing. Your search must consume the index exactly as built (you may of course
reorganize its arrays inside your own call). Treat the coarse centroids, PQ
codebooks, CSR list layout, and codes as fixed inputs.

Each call receives its **own fresh copy** of that index — same contents, but a
new object with new tensors every time. Any state you attach to the index or
cache keyed on its tensor addresses is therefore gone by the next call: work
like a CSR-to-padded layout conversion cannot be hoisted into a warmup call and
amortized away. Whatever your search needs, it must produce within the call
being timed.

## Workload

The graded workloads are held-out ANN search problems over random float32
databases: `M` from tens of thousands to a few hundred thousand rows, `D` in the
64–128 range, `nlist` a few hundred lists, `m` sub-quantizers 8–16, `nprobe`
8–32, and `nq` a thousand-plus queries per call. Treat it as general IVF-PQ
search, not as specific indices to special-case.

The public self-test now runs the **exact graded workloads** — the same shapes,
thresholds, seeds, and timing the judge uses to score you — so there are no
separate hidden shapes to guess at.

## Iterate on a GPU

The agent workspace has no GPU. Use the public test to check your current code's correctness and speed on
a GPU through Modal (needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`):

```bash
bash /app/public_test.sh
```

It runs the **identical evaluation the judge uses to grade you** and reports, per
graded workload, pass/fail on the quality gate, your speedup, the geometric-mean
speedup, and your **predicted final score** (0-100). Any workload that fails its
gate makes the submission score 0.

## Submission

The submitted artifact is a patch over the `ivfpqlib` package:

```text
/app/solution.patch
```

After editing `/app/ivfpqlib`, run `bash /app/make_submission.sh` then
`bash /app/submit.sh`. Submissions are asynchronous; submit early and iterate.
The judge applies your patch to a clean copy of `ivfpqlib` and times it against
the original baseline on a GPU, searching the same seeded index and queries.

## Correctness

Correctness is a gate. On every timed iteration the judge searches the same index
with the same queries using both your code and the baseline, and requires the
**iso-result recall@k** — the fraction of your returned ids that also appear in
the baseline's top-`k` for that query — to stay at or above a high threshold.
Crashes, non-finite / wrong-shape output, timeouts, and results that disagree
with the baseline beyond the tolerance are penalized before speed is considered.

## Scoring

Valid submissions are scored by speedup relative to the baseline:

```text
speedup = baseline_time / your_time
```

The objective is the geometric mean of per-workload speedups. A result no faster
than the baseline earns 0; regressions earn 0. The raw geometric-mean speedup is
reported in the evaluator metrics.

## Patch Policy

Only Python files under `ivfpqlib/**` may change. New Python modules inside
`ivfpqlib/` are allowed. Patches may not: modify anything outside `ivfpqlib/`;
import or call an external optimized ML / ANN / kernel library (write the kernels
yourself); read/write environment variables, spawn processes, or access the
network; or tamper with the measurement framework. The timing harness regenerates
queries every iteration and re-verifies recall against the baseline each time.

## Resource Budget

```text
GPU: single Modal GPU (H100 reference; Triton paths also run on L40S / A100)
Agent container: CPU-only (GPU work is offloaded to Modal)
```
