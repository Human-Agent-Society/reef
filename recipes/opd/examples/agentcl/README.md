# AgentCL coding stream with OPD

This example trains model weights on the pinned AgentCL coding stream with
native Reef on-policy distillation (OPD). Each task produces one complete
student episode, one terminal report, and one verified native training commit.
The frozen teacher scores the student's exact recorded token IDs. Its prompt
contains no reference demonstration or terminal verifier feedback.

A full campaign has 96 ordered training tasks, a frozen 96-task repeat, and
120 independent evaluation tasks. Baseline evaluates both task sets before
training. Evaluation produces no training reports. This is a weight-learning
extension, not a reproduction of AgentCL's memory methods or paper hyperparameters.

## Implementation and qualification status

The OPD implementation comes from
[`origin/codex/opd-slime-reproduction` at `a06415d7c417dc0f180b3301e64d256aa91dcb40`](https://github.com/Human-Agent-Society/reef/commit/a06415d7c417dc0f180b3301e64d256aa91dcb40),
which supports [PR #683](https://github.com/Human-Agent-Society/reef/pull/683).
The AgentCL working tree is based on `daeaf4d42fb2f9e20445df6618afdc2a4097bdeb`;
only the OPD implementation and needed shared dependencies are integrated from
that branch pin. This is not a checkout of the whole OPD branch.
That implementation uses Han's shared distillation backend and the regular
checkpoint loader. It swaps frozen teacher weights into the actor's existing
layout for scoring. It does **not** use the separate teacher-engine path from
[PR #707](https://github.com/Human-Agent-Society/reef/pull/707).

The teacher must match the actor architecture, parameter layout, and vocabulary
shape. Student and teacher token IDs must have identical meanings. Use a
compatible, immutable checkpoint; a different-size teacher or a different
tokenizer is not supported by this proposal. The OPD processor loads no
tokenizer and never re-renders a teacher prompt.

A native Qwen2.5-7B-Instruct smoke loaded a distinct compatible frozen teacher,
completed two acknowledged optimizer updates with finite gradients, and
published changed weights. Exact assistant-token masks, receipt consumption,
frozen smoke evaluations, fresh checkpoint reload, and uploaded W&B optimizer
rows passed. The actual trace viewer passed desktop/mobile interactions and
exact downloads. The first attempt's submission-format failure is preserved;
a fresh run used clearer rejection feedback with the same parser and budgets.

[Native training run](https://xai.wandb.io/recsys/reef-agentcl/runs/3e8603d7da0fc1fa4fbc79bc19c6ad4a)

Copied final tensors matched the original recorded weight-change measurements.
The GPU node was evicted during final preservation checks, so final old-node
cleanup and complete recovery-tree byte identity remain unverified. Full
96-task coverage, benchmark gains and numerical parity remain unqualified.
A two-update smoke establishes training capability for the verified settings.

Exported-file validation, cached reference identity, persisted reports and
teardown have additional CPU regression coverage.
The recorded native runs predate that hardening and do not qualify the new
export/image bytes. Export a fresh task set and rerun all 216 reference checks
before starting another native campaign. Reference records use schema version 2;
older records are rejected. The final-module contract is version 2.

## Prerequisites and safe preflight

Use Reef's supported Python 3.12 environment, editable `reef-client`,
`reef-eval[harbor]==0.1.1`, and Harbor's Docker runtime. Native training also
requires the supported Slime/SGLang dependencies, a local
Qwen2.5-7B-Instruct student, and a compatible frozen teacher checkpoint.
Follow the root [development instructions](../../../../AGENTS.md).
Install this example's `pyproject.toml` editable into the same environment
before real Harbor runs, so `harness:HarborAgent` resolves. Do not install into
a running workload. `run.sh` installs nothing; it selects `AGENTCL_PYTHON`,
then the repository `.venv312`, then `.venv`.

Export and dry-run do not start containers, inference, training, or a service:

```bash
cd recipes/opd/examples/agentcl
./run.sh export --data-root /path/to/opd-data --cache-dir /path/to/cache
export AGENTCL_TEACHER_CHECKPOINT=/path/to/immutable-compatible-teacher
./run.sh baseline --data-root /path/to/opd-data --run-root /path/to/full --dry-run
./run.sh train --profile smoke --native-input-checks --data-root /path/to/opd-data --run-root /path/to/smoke --dry-run
./run.sh train --profile full --data-root /path/to/opd-data --run-root /path/to/full --dry-run
```

`--teacher-checkpoint` overrides `AGENTCL_TEACHER_CHECKPOINT`. A nonempty value
is required for training, campaign evaluation, and dry-run service proposals.
Export and local artifact checks need no teacher. A dry-run parses the shipped
version-2 configuration and prints the exact settings and service proposal.
It submits zero requests and creates no run artifacts. Smoke selects the first
subtask and its matching complex task by identity, not by equal array offsets.

Export fetches `osunlp/AgentCL` at
`a01e2ca6e33fd07d9cf80e4bd69a5b3585d3400f`, checks the three pinned source hashes,
and writes 96 training plus 120 independent tasks. Export executes no benchmark
code and requires an empty output root. The source card is CC-BY-NC-4.0;
underlying source terms also apply. Review those terms before sharing data or traces.

Reference solutions stay in host-only `privileged/`; student environments never
receive them. Hidden tests execute in a separate verifier. After separate
container approval, qualify all 216 exported task/verifier pairs:

```bash
AGENTCL_EXECUTION_APPROVED=1 ./run.sh verify --references \
  --data-root /path/to/opd-data --run-root /path/to/reference-checks
```

This reference-only command makes no model calls. OPD training requires a
matching `reference-verification.json` with all 216 pinned references passing.
Export alone does not satisfy this prerequisite. The driver checks the saved
qualification results and hashes against the manifest; it reads no reference
solution bytes during OPD training or evaluation. Keep that qualification
record tied to the unchanged exported tasks, tests, and verifier image.

Campaign student and verifier containers use the example's detached,
no-network Harbor provider. Startup requires only loopback interfaces.
The provider accepts Linux, Dockerfile-only tasks and `no-network` policies;
it rejects Compose files, extra overlays, and allowlists. Harbor owns builds,
mounts, resource limits, transfers, and cleanup. The reference-only command
uses Harbor's default transport. Qualify both paths in the approved runtime.

## Approved native run

Get separate approval for the exact model/runtime, GPU allocation, Docker
operations, and launch commands. The proposal uses four colocated GPUs:
a tensor-parallel-4 native Slime actor and four single-GPU SGLang engines.
The teacher shares the actor layout; there is no additional teacher engine.
Use one persistent service under an external supervisor. Do not restart it
between tasks. Supply existing authentication through `REEF_TOKEN`.
Neither script allocates resources or creates credentials.

Set the service variables exactly as printed by the dry-run. Use distinct
service, runtime, output paths, and free ports for each method and run.
Full uses 96 schedule steps and 10 warmup steps. Smoke uses two schedule
steps and one warmup step; it is not optimizer-equivalent to full.

The following is a supervisor command example, not launch authorization:

```bash
export AGENTCL_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export AGENTCL_TEACHER_CHECKPOINT=/path/to/immutable-compatible-teacher
export AGENTCL_RUN_ROOT=/path/to/full AGENTCL_STEPS=96 AGENTCL_WARMUP=10
export AGENTCL_BATCH_SIZE=1 REEF_PORT=28902 AGENTCL_ROUTER_PORT=23002
# Run only as the approved external supervisor's managed workload.
reef serve -c "$PWD/serve.yaml"
```

Wait for authenticated `/healthz` readiness. Then use a second process for
the campaign. The supervisor owns bounded shutdown and GPU-drain checks.
Use identical model, teacher, profile, and run settings for every phase:

```bash
export REEF_SERVICE_URL=http://127.0.0.1:28902 REEF_SCENARIO=agentcl-opd
export AGENTCL_EXECUTION_APPROVED=1
./run.sh baseline --data-root /path/to/opd-data --run-root /path/to/full
./run.sh train --profile full --data-root /path/to/opd-data --run-root /path/to/full
./run.sh evaluate --phase frozen-repeat --data-root /path/to/opd-data --run-root /path/to/full
./run.sh evaluate --phase independent --data-root /path/to/opd-data --run-root /path/to/full
./run.sh qualify-inputs --run-root /path/to/full
./run.sh render-traces --run-root /path/to/full
./run.sh verify --run-root /path/to/full
```

Use `--profile smoke`, a fresh run root, and `--native-input-checks` in every
smoke campaign phase. Baseline, training, repeat, and independent evaluation
then each use two tasks. Start smoke and full from the initial student, never
from the smoke export. The first-task native input check runs before task two;
campaign input qualification also requires an active multi-turn episode.

### Protocol and fixed budgets

- One attempt per task. A terminal training report retains every ordered
  episode receipt and sets `teacher_context` to `""`. The verifier score stays
  in the report as a result; it does not enter the teacher prompt or define
  a reward advantage. Incorrect completed answers remain training samples.
- Stateful fenced Python execution resets between episodes. Only learned
  weights carry across tasks; no external memory is read or written.
- Non-thinking student, 8,192-token student window, 16,384-token teacher window,
  2,048 response tokens per call, exactly eight allowed turns, temperature 0.7,
  base seed 42, 30-second tool timeout, and 8,000 output characters. Sampling
  seeds match across phases for the same task and attempt.
- Native flags: `--opd-teacher separate`, `--opd-teacher-checkpoint <checkpoint>`,
  `--opd-divergence reverse`, `--opd-top-k 1`,
  `--opd-importance-sampling-cap 0.0`, and `--opd-teacher-update-rate 0.0`.
  Zero skipped response tokens, realignment, and scaffolding preserve the full
  sampled suffix. Teacher IDs equal the complete recorded student sequence,
  including tool history; teacher-window overflow is a training fault.
- AdamW at LR 1e-5, cosine schedule, no weight decay, gradient clip 1.
  Sampler and trainer rollout temperature both equal 0.7. BF16, eager SGLang
  execution, and Triton inference attention were exercised in the native smoke.
  Other models and deployment settings require their own qualification.
- Frozen baseline, repeat, and independent phases make no reports or teacher
  calls. They read no demonstrations or terminal verifier feedback into student
  context. The served release and runtime binding are checked per episode.
  Known bounded evaluation truncations score zero; infrastructure faults and
  training truncations block completion.

## Results, resume, and verification

`run-manifest.json` pins the working-tree base, OPD implementation revision,
dataset hash, student/tokenizer, teacher checkpoint path, native OPD settings,
windows, schedule, and topology.
The checkpoint path identifies an immutable checkpoint: never replace its
contents in place. Any teacher, data, or settings change requires a fresh run
root. The dry-run prints the same specification used by the campaign.

`cursor.json` records local intent before remote work. `episodes/`, Harbor
`lab/`, turn records, `reports/`, `commits/`, and `summary.json` retain results.
The cursor advances only after one non-pending, verified native training commit
consumes the report and **all** receipt IDs, then publishes a new currently
served release. Live releases remain anchored to the last durable checkpoint.
A newer release or a larger release count alone is insufficient.

Rerun the same command and root to reconcile saved results and consumption.
Committed tasks are not replayed. Unknown inference or report outcomes stop;
recover canonical records instead of resampling or inventing new IDs.
Resume assumes the same supervised service. Exact optimizer, teacher, and RNG
continuation after arbitrary runtime restarts is not established. An HF-only
export does not establish optimizer-equivalent continuation.

Private `teacher-records/<report_id>.json` captures preserve actual native
student/teacher IDs, masks, log probabilities, receipt order, and source records.
Tool/context positions have zero loss; every assistant token stays selected.
Capture happens before optimizer execution and does not prove a commit.
OPD capture records original request messages/tools and exact token IDs without
loading a tokenizer. Decoded teacher text can be unavailable; it is never a
sampled teacher completion. `qualify-inputs` makes no model calls.

The summary separates subtask and complex-task scores, faults/truncations,
pre-update scores, and token/time costs. PG, SG, and GG are absolute
percentage-point differences; unavailable or unmatched tasks do not produce gains.
GPU cost requires external supervisor accounting.

`verify` recomputes local coverage and consumption. External qualification
artifacts still require independent review of actual tensors, gradients,
checkpoint changes, reload, sandbox isolation, browser results, and cleanup.
A CPU fixture or a two-task smoke cannot establish a qualified full campaign.

W&B is disabled. Online uploads require separate approval and
`AGENTCL_UPLOADS_APPROVED=1`; pass the same `--wandb-project` and
`--wandb-entity` in all campaign phases. The dry-run prints the corresponding
service override. Native optimizer metrics and aggregate evaluation scores use
separate runs. Trace and checkpoint uploads are not configured. This upload
option was exercised for native optimizer and aggregate evaluation metrics.
Use existing approved credentials; uploads still require approval.
