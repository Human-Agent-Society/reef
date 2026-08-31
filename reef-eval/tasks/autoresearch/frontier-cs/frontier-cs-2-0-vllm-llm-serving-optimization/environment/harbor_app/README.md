# vLLM LLM-Serving Optimization Starter

The workspace contains a clean upstream vLLM checkout at:

```text
/app/vllm
```

Modify vLLM source code to reduce end-to-end serving latency on the agentic
SWE-bench workload while preserving the model's task-solving accuracy. Only
Python-only changes in the allowlisted scheduler/execution/serving areas are
valid (see the task statement for the exact patch policy). The model is served
on a Modal H100 with `VLLM_USE_PRECOMPILED`, so CUDA/C++ kernel changes are out
of scope.

## Submit

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

`make_submission.sh` stages your changes in `/app/vllm` and writes
`/app/solution.patch`. `submit.sh` enqueues that patch for the same black-box
judge used by the final verifier. Submissions are asynchronous — submit early,
then keep iterating. Use `bash /app/submissions.sh` and
`bash /app/wait_submission.sh <uuid>` to inspect judge results.

## Public test (local, async, real metrics)

Before (or instead of) submitting, evaluate your working tree yourself:

```bash
bash /app/public_test.sh launch        # deploys /app/vllm to a Modal H100, async
bash /app/public_test.sh status <id>   # latency + accuracy + provisional score
bash /app/public_test.sh run           # synchronous variant
```

The public test deploys your patched vLLM to a Modal H100, serves
`meta-llama/Llama-3.1-8B-Instruct`, runs the **public instance subset** (a strict
subset of the final eval set) under the same Poisson arrival workload the judge
uses, and returns:

- per-instance and mean end-to-end latency,
- an accuracy signal (patch-validity rate during iterative feedback),
- a provisional speedup and score versus the baseline (when a baseline cache is
  available in the image).

This is real serving feedback — latency and accuracy — not a build/compile flag.
Drive your loop with it: edit vLLM, run the public test, read the returned
latency/accuracy, adjust.

## What the judge measures

The judge applies `/app/solution.patch` to a clean pinned vLLM tree, builds and
serves it the same way, runs the workload, and scores **latency speedup vs the
baseline**, gated by an **accuracy guardrail**: accuracy within 5% of the
baseline does not affect the score; beyond that the score decays
inverse-proportionally with the accuracy drop. The patched server must also
reproduce the baseline's greedy (temperature 0) generations before any timing is
considered.

## Credentials

Serving the model requires `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and an
`HF_TOKEN` (gated Llama-3.1 access), provided to the workspace. Do not read,
print, or exfiltrate them, and do not reference them from patched vLLM source —
the patch policy rejects that.
