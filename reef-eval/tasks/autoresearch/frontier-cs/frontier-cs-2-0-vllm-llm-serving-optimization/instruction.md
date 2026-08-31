You are solving a Frontier-CS 2.0 open-ended optimization problem.

Create a python solution at `/app/solution.patch`. You can call `bash /app/submit.sh` at any time to enqueue a snapshot for the same black-box judge used by the final verifier. Submissions are asynchronous: use `bash /app/submissions.sh` and `bash /app/wait_submission.sh <uuid>` to inspect results. The evaluator implementation is intentionally not available in the agent workspace. Read `AGENT.md` for the shared submission workflow.

Problem id: `vllm_llm_serving_optimization`
Language: `python`
Time limit: `21600s`

Original problem statement:

# vLLM LLM-Serving Latency Optimization

## Problem

This is an experimental systems task. You are given a pinned, clean checkout of
[vLLM](https://github.com/vllm-project/vllm) in the Harbor workspace and may
modify vLLM itself. Your goal is to reduce the **end-to-end latency** of an LLM
serving system on a realistic multi-turn agentic workload while preserving the
**accuracy** (task-solving quality) of the served model.

The serving target is a deployment of
`Qwen/Qwen3-Coder-30B-A3B-Instruct` running on one NVIDIA **H100**, exposed
through vLLM's OpenAI-compatible HTTP API. The workload is an agentic
code-editing benchmark (see *Workload* below) whose requests are long,
multi-turn conversations that arrive over time as a Poisson process.

The intended optimization area is **online serving efficiency**: request
scheduling, batching, KV-cache management, prefix/prompt cache reuse,
preemption and admission control, queueing, and closely related
scheduler/execution wiring. Strong submissions improve the workload's latency
distribution without changing what the model actually generates and without
hard-coding the benchmark, dataset, queries, or judge details.

## Serving Stack (Modal + H100)

Both your local public test and the hidden judge serve the patched vLLM the
same way:

- A [Modal](https://modal.com/docs) app builds an image from **your patched
  vLLM source** and serves `Qwen/Qwen3-Coder-30B-A3B-Instruct` on one **H100**
  through the OpenAI-compatible endpoint (`<url>/v1`).
- The image is built with `VLLM_USE_PRECOMPILED=1`, which reuses vLLM's
  prebuilt CUDA kernels and rebuilds only the Python layer. **Your patch must
  therefore be Python-only** — changes that require recompiling CUDA/C++
  kernels are out of scope and rejected by the patch policy.
- The serving runtime (model, GPU, tensor-parallel size, max model length,
  dtype, and OpenAI server flags) is fixed and identical for the baseline and
  your patched build. You may not change how the server is launched; you may
  only change vLLM's internal behavior through allowlisted source files.

Running the model requires Modal and Hugging Face credentials configured in the
environment (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and an `HF_TOKEN` for model
downloads). These are provided to the workspace and the judge; do not attempt to
read, print, or exfiltrate them.

## Workload

The judge runs **two** serving workloads against each build and combines them
**50/50** (see *Scoring*):

**1. SWE-bench agentic (latency-primary).** A
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)-style SWE-bench
run: each instance is one agentic task in which the agent holds a multi-turn
conversation with the served model, issuing shell commands in a sandboxed
repository between turns. Every turn re-sends the growing conversation, so
consecutive requests for the same task share a long common prefix. Instances
arrive over time (Poisson arrivals), so many conversations are in flight at once
and compete for GPU and KV-cache capacity. The dataset is the public
`princeton-nlp/SWE-bench_Lite` set (split `test`); the agent loop, step
limit, and decoding settings (temperature `0`) are fixed.

**2. BFCL memory (agentic, correctness-primary).** The `memory` category of the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html):
each instance is a multi-turn agentic task where the model is given a key-value
memory tool suite (pre-seeded with facts from a prior conversation), asked a
question, and must issue retrieve/search tool calls across several turns and then
answer. Correctness is a deterministic word-boundary match of the final answer
against the ground truth. This gives a real, **non-zero** accuracy signal (and a
multi-step request path), so it provides a *live* accuracy guardrail and a
per-sample correctness check alongside SWE-bench. Instances arrive over time as a
Poisson process (like the SWE-bench workload), so multiple multi-step memory
conversations are in flight at once and queue.

Treat both as representative analytical serving workloads, not sets of strings
to recognize. The hidden judge may include additional non-public instances and
may vary instance order and arrival timing. Submissions should implement general
serving optimizations rather than benchmark-specific special cases.

**Request metadata.** Each request carries a stable per-conversation id in
`sampling_params.extra_args["job_id"]` (the client sends it via `vllm_xargs`);
all requests of one benchmark instance share the same value, and vanilla vLLM
ignores it. Any request metadata available on the server is fair game for your
scheduling logic, but whatever you do must be a general policy — do not key on
specific id values or otherwise special-case the benchmark (the patch policy
forbids it).

## Submission

The submitted artifact is a patch file:

```text
/app/solution.patch
```

The agent workspace contains a clean vLLM checkout at:

```text
/app/vllm
```

After modifying vLLM, generate and submit a patch:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

Submissions are asynchronous. Submit an initial small, plausible patch as soon
as it is generated, then keep iterating while the judge works. The judge applies
your patch to a clean pinned vLLM source tree, builds it on Modal, serves it,
runs the workload, and scores latency and accuracy from the judge side.
Submitted binaries, build artifacts, generated benchmark files, and local
timing logs are ignored.

## Public Test (async, latency + accuracy feedback)

You can evaluate your current working tree yourself, without going through the
judge queue, using the public test client:

```bash
# Launch an async public-test run (deploys your patched vLLM to Modal H100,
# runs the public instance subset, returns a run id):
bash /app/public_test.sh launch

# Poll for the result (latency + accuracy, not just whether it compiled):
bash /app/public_test.sh status <run_id>

# Or run synchronously:
bash /app/public_test.sh run
```

The public test reports the **same kind of feedback the judge uses**: per-instance
and aggregate end-to-end latency, an accuracy signal versus the baseline, and a
provisional score — not merely whether the build succeeded. The public instance
subset is a strict subset of the final evaluation set, so it is a fast, faithful
proxy. Use it to drive your optimization loop: change vLLM, rerun the public
test, read the returned latency/accuracy, and adjust.

## Correctness

Correctness is a hard gate, applied **before** any timing is scored. The patched
server must not change what the model generates at temperature `0`:

- **SWE-bench greedy gate.** The judge runs a fixed greedy-decoding smoke set and
  requires the patched build's outputs to match the baseline token-for-token.
- **BFCL per-sample gate.** On the BFCL slice, the patched build must not flip an
  instance from **correct** (baseline) to wrong/undecodable; a small number of
  flips is tolerated to absorb rare batch-numerics differences, but a real
  regression fails the gate.

Build failures, patch-policy violations, server start-up failures, generation
mismatches, crashes, timeouts, and out-of-memory failures all score `0` before
performance is considered.

During iterative asynchronous submissions, the judge keeps feedback focused on
the public instance subset so you can submit early and continue working while
evaluation runs. During final verification, the judge uses the broader hidden
instance set.

## Scoring

The final score blends the two workloads:

```text
final_score = 0.5 * swebench_score + 0.5 * bfcl_score
```

Each workload is scored by **latency speedup relative to the baseline** (vanilla
vLLM serving the same model on an H100 under the same workload and arrival
schedule), gated by an **accuracy guardrail**.

Latency is the end-to-end completion time per instance (first request to last
response), measured client-side. Per-instance speedups are clamped to a bounded
range and a patched instance that *fails* (errors / exits early) is counted as a
regression — so failing fast cannot inflate the score. The per-workload objective
is the **geometric mean** of those per-instance speedups:

```text
per_instance_speedup = clip(baseline_latency[i] / patched_latency[i], 1/cap, cap)
latency_speedup      = geomean(per_instance_speedup)
latency_score        = clip(100 * log2(latency_speedup), 0, 100)
```

A `1.0x` result earns `0` points and regressions also earn `0`.

Each workload's accuracy gates its latency score. For BFCL this is the
memory-retrieval accuracy (a real, non-zero signal); for SWE-bench it is the
task-solving proxy. Let

```text
rel_drop            = max(0, (baseline_accuracy - patched_accuracy) / baseline_accuracy)
accuracy_multiplier = 1.0              if rel_drop <= 0.05
accuracy_multiplier = 0.05 / rel_drop  otherwise
workload_score      = latency_score * accuracy_multiplier
```

So a fast build that meaningfully degrades memory-retrieval quality loses most of
its BFCL score, while a build that keeps accuracy within 5% of the baseline is
scored on its latency improvement. The per-workload speedups, accuracies, and
multipliers are reported in evaluator metrics.

## Patch Policy

The evaluator validates the patch before building. The policy is intentionally
strict because this task is graded by hidden benchmarks.

Allowed serving/scheduler/execution areas:

```text
vllm/v1/core/**
vllm/v1/core/sched/**
vllm/v1/core/kv_cache_utils.py
vllm/config/scheduler.py
vllm/config/cache.py
```

Conditionally allowed narrow wiring areas:

```text
vllm/v1/worker/**
vllm/v1/engine/**
vllm/v1/executor/**
vllm/v1/request.py
vllm/v1/outputs.py
vllm/v1/serial_utils.py
vllm/entrypoints/openai/protocol.py
vllm/entrypoints/openai/serving_engine.py
vllm/entrypoints/openai/serving_chat.py
vllm/entrypoints/openai/serving_completion.py
vllm/sampling_params.py
```

New Python files are allowed in these areas. The build uses `VLLM_USE_PRECOMPILED`,
so no build-system, CUDA/C++, packaging, or dependency changes are permitted.

Forbidden areas include CUDA/C++ kernels and build files (`csrc/**`, `cmake/**`,
`CMakeLists.txt`, `setup.py`, `pyproject.toml`, `requirements/**`), tests,
benchmarks, docs, examples, CI files, model definitions
(`vllm/model_executor/models/**`), tokenizer/loader internals, the workload
harness, and any timing or scoring code.

Patches may not add reads or writes of judge, Modal, Hugging Face, Frontier, or
Harbor environment variables, and may not hard-code the benchmark name, dataset
name, instance identifiers, or judge paths in scheduler/execution code. The
server is launched under a fixed configuration; patches that detect the
benchmark, sleep, short-circuit generation, or otherwise special-case the
evaluation are rejected.

## Resource Budget

The experimental Harbor budget is:

```text
agent/judge container vCPUs: 8
agent/judge container memory: 32 GiB
storage: 64 GiB
served model: Qwen/Qwen3-Coder-30B-A3B-Instruct
serving GPU: 1x NVIDIA H100 (via Modal; the operator may use H100:N for tensor parallelism)
build timeout: 7200 seconds
per-instance timeout: 1200 seconds
decoding: temperature 0, fixed max tokens
```

The judge builds and serves both baseline and patched vLLM under the same fixed
Modal configuration and the same OpenAI server flags, then runs the workload
under the same arrival schedule before measuring latency and accuracy.
