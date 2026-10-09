# AgentCL coding stream with SDPO

This example learns model weights on the pinned AgentCL coding stream through
native Reef SDPO and isolated Harbor tasks. It is a supervised weight-learning
extension, not a reproduction of AgentCL's unsupervised memory methods. A full
run requires 96 training commits and frozen 96-task repeat plus 120-task
independent evaluation. Incorrect answers and known bounded evaluation
truncations count as scores. Infrastructure faults and training truncations
block completion. Full-run coverage requires verification of the actual artifacts.

## Verified training and limits

A native Qwen2.5-7B-Instruct smoke completed two acknowledged optimizer updates
with finite gradients, complete assistant-token masks, changed weights and a
fresh checkpoint reload. Native metrics matched two uploaded W&B optimizer rows.
The saved trace viewer passed desktop/mobile interactions and exact downloads.
The SDPO smoke's frozen evaluations and supervised checkpoint byte comparison
passed. Full 96-task coverage and evaluation reliability require their own checks.

[Native training run](https://xai.wandb.io/recsys/reef-agentcl/runs/8ff664e0d864575593f5c1beeffb327a)

The three-method verification retained copied final exports and checked every
copied tensor against its recorded weight-change measurements. Node eviction
interrupted the final old-node cleanup and complete recovery-tree comparison;
those checks remain unverified. Numerical parity is also unqualified: the
provisional cross-engine log-probability tail checks failed. These smokes
establish training capability and provide no full benchmark-gain result.

Versioned export corrections preserve original dataset bytes and the 96/120
task counts. Every exported source/image must pass all 216 reference checks
before training. The restricted final-module syntax rejects known process-control
and introspection uses; it provides a bounded answer contract.

Exported-file validation, cached reference identity, persisted reports and
teardown have additional CPU regression coverage.
The recorded native runs predate that hardening and do not qualify the new
export/image bytes. Export a fresh task set and rerun all 216 reference checks
before starting another native campaign. Reference records use schema version 2;
older records are rejected. The final-module contract is version 2.

## Prerequisites and safe preflight

Use the supported Reef Python 3.12 environment with native Slime/SGLang training
dependencies, the editable `reef-client`, `reef-eval[harbor]==0.1.1`, Harbor's
Docker runtime, and a local Qwen2.5-7B-Instruct model/tokenizer. Follow the root
[development instructions](../../../../AGENTS.md). Install this example's
`pyproject.toml` editable into the **same** environment before real Harbor runs,
so `harness:HarborAgent` resolves. Do not install into a running workload.
`AGENTCL_PYTHON` can select an existing interpreter; the launcher otherwise
uses repository `.venv312`, then `.venv`. It installs nothing.

Campaign trials use the example's `DetachedDockerEnvironment` through Harbor's
custom-provider interface. Both the student and separate verifier use native
`network_mode: none`. Startup checks the container's actual network interfaces
and fails unless only loopback exists. This transport accepts only Linux,
`no-network` policies, and Dockerfile-only tasks. It rejects task Compose files,
extra Compose overlays, and allowlist additions. Harbor still owns image builds,
mounts, resource limits, artifact transfer, and cleanup. This transport needs
CPU qualification on the target runtime before native training. The separate
`verify --references` command retains Harbor's default transport.

```bash
cd recipes/sdpo/examples/agentcl
./run.sh export --data-root /path/to/sdpo-data --cache-dir /path/to/cache
./run.sh baseline --data-root /path/to/sdpo-data --run-root /path/to/full --dry-run
./run.sh train --profile smoke --data-root /path/to/sdpo-data --run-root /path/to/smoke --dry-run
./run.sh train --profile full --data-root /path/to/sdpo-data --run-root /path/to/full --dry-run
```

Export fetches `osunlp/AgentCL` at
`a01e2ca6e33fd07d9cf80e4bd69a5b3585d3400f`, checks the three source hashes,
and writes 96 training and 120 independent tasks. It never executes benchmark
code. The source card is CC-BY-NC-4.0; underlying source terms also apply.
Review redistribution/commercial use before sharing data or traces. The output
root must be empty. Keep reference solutions under the host-only `privileged/`
folder: they are never placed in student environments. Hidden tests run only
in Harbor's separate verifier. Reference-solution verification is a separate
sandbox qualification and is **not** established by a successful export.
After separate container approval, run
`AGENTCL_EXECUTION_APPROVED=1 ./run.sh verify --references --data-root /path/to/sdpo-data --run-root /path/to/reference-checks`.
This uses the privileged reference-only harness, no model calls. SDPO training
requires `reference-verification.json` with all 216 pinned references passing;
export alone does not authorize or satisfy that prerequisite.

A dry-run parses the shipped version-2 native configuration and prints the
exact model, pinned data, windows, topology, selected tasks, optimizer schedule,
outputs, and service proposal. It submits nothing and starts no service. Smoke
selects the first subtask and its matching complex task by identity, not by
matching array offsets; the two halves have different orders.

## Approved native run

Get separate approval for the exact model/runtime, GPU allocation, Docker
operations, inference and training commands. The proposal is four colocated
GPUs: a tensor-parallel-4 native Slime actor and four single-GPU SGLang engines.
Run one persistent service under an external supervisor; do not restart it
between tasks. Supply existing authentication through `REEF_TOKEN` in the host
environment. Neither script creates credentials or allocates resources.

The dry-run prints a `reef serve -c serve.yaml` proposal. Set its
`AGENTCL_MODEL_PATH`, `AGENTCL_RUN_ROOT`, `AGENTCL_STEPS`, `AGENTCL_WARMUP`,
`AGENTCL_BATCH_SIZE=4 (smoke: 2)`, `REEF_PORT`, and `AGENTCL_ROUTER_PORT` exactly as printed.
Use distinct service/runtime/output paths and free ports for each method/run.
The full profile uses 96 schedule steps/10 warmup steps. Smoke uses two schedule
steps/one warmup step: it is qualification, not optimizer-equivalent to full.

The following is an example supervisor command, **not** a resource reservation
or launch authorization. Execute it only in the approved training runtime with
its four-GPU allocation, existing credentials, and finalized free ports:

```bash
export AGENTCL_MODEL_PATH=/root/models/Qwen2.5-7B-Instruct
export AGENTCL_RUN_ROOT=/path/to/full AGENTCL_STEPS=96 AGENTCL_WARMUP=10
export AGENTCL_BATCH_SIZE=4 REEF_PORT=28902 AGENTCL_ROUTER_PORT=23002
# Run this foreground command as the external supervisor's managed workload.
reef serve -c "$PWD/serve.yaml"
```

Wait for authenticated `/healthz` readiness before using the driver from a
second process. The supervisor owns bounded shutdown and GPU-drain checks.

After approval and service readiness, use the same driver settings for every
phase and set `AGENTCL_EXECUTION_APPROVED=1` as a local execution guard:

```bash
export REEF_SERVICE_URL=http://127.0.0.1:28902 REEF_SCENARIO=agentcl-sdpo
export AGENTCL_EXECUTION_APPROVED=1
./run.sh baseline --data-root /path/to/sdpo-data --run-root /path/to/full
./run.sh train --profile full --data-root /path/to/sdpo-data --run-root /path/to/full
./run.sh evaluate --phase frozen-repeat --data-root /path/to/sdpo-data --run-root /path/to/full
./run.sh evaluate --phase independent --data-root /path/to/sdpo-data --run-root /path/to/full
./run.sh render-traces --run-root /path/to/full
./run.sh verify --run-root /path/to/full
```

For smoke use `--profile smoke` and a fresh root in **all** phases: baseline,
training, repeat, and independent are bounded to two tasks each. Start both
smoke and full from the same initial base model, never from the smoke export.

### Protocol and budgets

- Four independent complete student episodes per task (two in smoke), sampled
  from the same initial task and release in separate reset sandboxes. Each
  terminal report references every ordered receipt with one task/grid coordinate;
  the complete grid produces one native optimizer update/publication. Generic
  verifier feedback is sanitized, and the processor uses a successful sibling
  full trajectory when available. SDPO never reads dataset demonstrations.
  An inactive rollout remains in the grid with zero signal; a smoke without any
  effective signal is not successful training.
- Fenced Python execution is stateful within an episode and reset between
  episodes. Only learned weights carry across tasks; no external memory is read
  or written. Baseline evaluates both the 96-task stream and 120 held-out tasks
  before training.
- Qwen2.5-7B-Instruct, non-thinking; 8,192-token student window, 16,384-token teacher
  window, 2,048 response tokens per call, eight turns, 30-second tool timeout,
  8,000 output characters, temperature 0.7 and base seed 42. Task/attempt
  UUID5-derived sampling seeds are independent across siblings and matched
  across phases for the same task/attempt. The teacher-window limit
  includes the privileged prompt and full exact suffix; overflow is a fault.
- Student-top-100-plus-tail JSD, self-teacher EMA 0.05, importance cap 2, zero skipped response
  tokens, strict realignment/scaffolding 0. AdamW at LR 1e-5, cosine schedule,
  no weight decay, gradient clip 1. These are integration choices, not AgentCL
  paper hyperparameters. GPU capacity and live multi-turn token retention must
  be qualified before full training. Sampler and trainer rollout temperature
  both equal 0.7. The native inference proposal uses BF16, eager execution,
  and SGLang Triton attention. These backend flags need runtime qualification.
- Frozen repeat/independent phases use one attempt per task, no reports,
  demonstrations, teacher lookup, or hidden-test feedback in student context.
  The served release is checked after every episode. Verifier reward remains
  evaluation output, not training data.

## Results, resume, and verification

`run-manifest.json` pins settings and the exported manifest hash; `cursor.json`
records local intent before remote work. `episodes/`, Harbor `lab/`, ordered
turn JSONL/ATIF, `reports/`, `commits/`, `summary.json`, and `traces.html` retain
results. UUID5 identifies run/task/category/attempt/phase. The cursor advances
only when exact report **and all receipt IDs** belong to one verified,
non-pending native training commit publishing a new currently served
release. Native live releases point to the last durable checkpoint, not the
preceding live release; this checkpoint anchor is checked separately. A release-count increase alone is insufficient.

Rerun the identical command/root to reconcile retained reports, Harbor rows,
and committed consumption. Committed tasks are not replayed. Unknown inference
or report outcomes stop with an error; recover the canonical result before
continuing, rather than inventing new IDs or retrying paid inference. A service
restart is outside the driver's resume guarantee. The fresh base includes
[#728](https://github.com/Human-Agent-Society/reef/pull/728); optimizer state is
saved per update, but exact Adam/teacher/RNG equivalence after arbitrary partial
runtime restarts is not established. Do not use an HF-only export to claim
optimizer-equivalent continuation. Resume or a fresh full run must use the
original finalized schedule/settings.

The summary reports subtasks and complex tasks separately, faults/truncations,
per-task pre-update score, token/time costs, and learning curves. SDPO primary
first-pass is predeclared attempt 0 to match one-attempt baseline/evaluation;
training mean@4 (smoke mean@2) is separate, never mixed into PG/SG/GG. Gains are
explicit absolute percentage-point differences: PG = first-pass complex minus
base complex; SG = frozen complex minus first-pass complex; GG = final
independent minus base independent. SG includes a task's own prior experience.
Missing or unmatched tasks yield unavailable gains, not optimistic averages.
GPU cost needs external supervisor accounting, not a wall-time estimate.

`verify` recomputes scores/coverage/consumption and fails closed without full
qualification artifacts. The lead attaches `qualification.json` checks with
`passed` and a relative `artifact` path for real GPU training, finite gradients,
nonzero weight change, exact assistant masks/zero tool loss/teacher suffix,
reference verification, final reload, browser checks, authenticated Toolbox byte/access checks and owned cleanup/GPU drain.
Qualification check booleans plus existing files are operator declarations,
not independent validation of their contents. The final acceptance report
requires the lead to inspect actual tensors, checkpoint differences/reload,
metrics, browser results and supervisor state; `verify` alone cannot establish
those external claims.
These external claims require actual results; a smoke or CPU fixture cannot
satisfy them. The shipped recipe uses an example-owned capture processor. It calls the
original native processor and returns the unchanged batch. Private
`teacher-records/<report_id>.json` files retain exact teacher/student tokens,
masks, log probabilities, receipt order, and actual teacher messages/tools.
Teacher text is decoded from captured IDs, not a sampled completion.
Capture occurs before optimizer execution and does not establish a commit.

Use `--native-input-checks` in all smoke phases. The driver checks the first
complete training grid before sampling task two. Each nonempty episode must
preserve strict history, all assistant tokens, zero tool/context loss, and
exact teacher/student suffix identity. Valid one-turn episodes pass these checks.
Run `./run.sh qualify-inputs --run-root /path/to/run` to check all captures and
require at least one active multi-turn training episode across the campaign.
This checker makes no model calls.
The viewer labels missing capture unavailable. No teacher completion is generated.

W&B configuration is present but disabled. After **separate upload approval**,
pass `--wandb-project <approved-project> --wandb-entity <approved-entity>` to
every phase (including dry-run) and set `AGENTCL_UPLOADS_APPROVED=1`. Dry-run
prints the native service observability override for online optimizer curves;
the driver uploads aggregate evaluation scores to a separate deterministic
evaluation run. Both processes must use the approved project/entity. No trace,
demonstration or checkpoint upload is configured. Native optimizer and aggregate
evaluation logging were exercised in the smoke. Use existing approved credentials. Original JSON
records remain authoritative. Changing upload settings mid-run requires a new
run root because the manifest is immutable.
