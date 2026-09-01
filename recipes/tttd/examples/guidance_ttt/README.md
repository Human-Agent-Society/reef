# Guidance-TTT on Reef

This example ports the `summary_only` Guidance-TTT loop to Reef while keeping
the execution model outside the trainable policy:

```text
PUCT archive selects G parent candidates
  -> Qwen3-8B sees each parent's canonical summary and verifier score
  -> Reef records R guidance generations and their exact SGLang tokens/logp
  -> a frozen external model sees parent code + guidance and writes a candidate
  -> FrontierCS/go-judge verifies the candidate and returns a score
  -> each score references the exact Reef guidance receipt that produced it
  -> grouped TTT-Discover (tttd) advantages update only Qwen's rank-32 LoRA adapter
  -> valid candidates become PUCT children for the next step
  -> the run controller waits for that durable commit before the next step
  -> at trial end the task's verifier scores the submitted candidate
```

The executor is therefore an ordinary OpenAI-compatible service. Its prompts,
tokens, and weights are not part of Reef training. The Reef-specific boundary
is limited to the guidance call and the score report linked to that call.

```text
harbor/                self-contained reef-eval/Harbor task definition
  polyomino_packing/
    task.toml            metadata, timeouts, resource limits
    instruction.md       the problem prompt, shown to both models
    contract.json        the task's prompt and scoring vocabulary
    environment/         container plus the route to the external judge
    solution/            verified runnable bootstrap candidate (seed archive)
    tests/               verifier: re-scores the trial's candidate, writes reward
harness/               agent harness (Guidance-TTT search + Reef adapter)
  state.py               archive records: nodes, entries, verification results
  puct.py                rank-prior PUCT scoring
  library.py             Discover-compatible archive and group accounting
  prompts.py             summary-only prompts and strict response parsing
  search.py              chat request construction and response parsing
  execution.py           frozen OpenAI-compatible executor adapters
  scorer.py              judge-protocol client: one candidate in, one score out
  contract.py            the task-shaped strings the harness takes as input
  agent.py               receipt-linked rollout and verification loop
  run_controller.py      training barrier, LoRA qualification, paired resume
  harbor_agent.py        Harbor BaseAgent (imports the harbor package)
serve.yaml             Reef + Ray + Slime/Megatron + SGLang stack config
run.py                 one reef-eval episode owning the complete trajectory
run.sh                 starts the Reef training stack, then runs run.py
pyproject.toml         makes the harness importable
```

It reuses Reef's existing `tttd` recipe, token-native SGLang capture,
TTT-Discover processor/backend preparation, Slime/Megatron optimizer,
checkpoint protocol, and serving-native LoRA publication. No
execution-model-specific behavior is added to Reef's inference engine or
training runtime.

## A task-agnostic harness

Nothing under `harness/` knows what the task is. The problem statement is the
Harbor task's `instruction.md`; the rest of the task's vocabulary — the
candidate language, the sentence constraining proposable mechanisms, the label
of the judge's raw score, the judge's problem id — is that task's
`contract.json`. Scoring is a `Scorer` callable: the harness extracts the
program from `<solution>` and hands it to `harness/scorer.py`, which speaks the
external judge's wire protocol. Adding a second discovery problem is a new
Harbor task directory, not a harness change.

Every rollout's verifier score is reported to Reef with its step-grid
coordinates and is the training signal. Harbor records the task verifier's
final trial reward in its own trial result; that evaluation-only value is not
posted to the training scenario.

## Summary-only semantics

Only `summary_only` is exposed here:

- The guidance actor receives the problem, the selected candidate's canonical
  whole-solution summary, and its verifier result. It never receives source
  code.
- The frozen executor receives the same problem, the selected candidate's full
  runnable source, and the new guidance.
- The executor returns a complete `<solution>` and a new canonical `<summary>`.
  That summary is the only implementation context shown to the guidance actor
  if this child is selected later.
- Only the Qwen guidance response mask is trainable. Executor output is used
  for verification and archive evolution, not as an RL trajectory.

The policy gets exactly one generation attempt. A response is accepted only if
it contains one non-empty terminal `<guidance>...</guidance>` block. Malformed
guidance receives reward zero, skips the executor, still counts as a PUCT visit,
and is reported against its original Reef receipt. There is no format repair,
fallback generation, or retry. Transient HTTP transport failures may be retried
by the executor client; they never cause the guidance policy to be sampled
again.

## Search and optimization

The archive preserves the Discover-compatible Guidance-TTT settings from
[`open-ttt-verl@ea47140`](https://github.com/Chonghe-Jiang/open-ttt-verl/tree/ea47140ca7aea324d89ea5585afffb03c2c01522):

- one PUCT-selected parent shared by every rollout in a comparison group;
- rank-prior PUCT with `best_child` Q and ancestor visit backpropagation;
- top two children per expansion and a top-1000 archive;
- invalid or failed candidates counted as visits but not added as executable
  search nodes;
- configurable groups and rollouts, with a complete `G × R` step barrier;
- adaptive-beta entropic leave-one-out advantages and frozen-base token KL;
- un-clipped importance sampling against captured rollout log-probabilities;
- Qwen3-8B thinking, temperature 1, top-p 1, top-k -1;
- Adam at `4e-5`, rank/alpha `32/32` LoRA, frozen base parameters, and
  serving-native adapter publication to SGLang.

The included task is FrontierCS problem 0, Polyomino Packing. Its
`solution/` seed contains a verifier-runnable C++17 parent generated for the
local GPT-OSS-120B family. Every non-format-failure execution is submitted
directly to the external FrontierCS/go-judge service; there is no approximate
local scoring fallback. The focused HTTP adapter uses only the algorithmic
judge surface, so Reef does not import FrontierCS's unrelated
model-generation or cloud-runner packages.

`harness/run_controller.py` runs a step only after the previous Reef training
transaction is durable, and fails closed unless that transaction really
happened: a positive finite grad norm, the reserved global batch consumed,
trainable LoRA parameters with a nonzero LoRA-B update, no trainable base
parameter, and a Megatron checkpoint on disk. Only then is the post-step
archive copied to `committed-library.json`, which is the sole archive a
resumed run restores.

## Results from complete Reef runs

Two full `8 × 16` searches were recovered from the Reef run artifacts and
checked against their committed archives:

| Task | Search trajectory | Valid rollouts | Evaluation check |
|---|---:|---:|---|
| Polyomino Packing | 27.8105 → 89.7965 | 3,573 / 3,840 | Deterministic 70-case FrontierCS suite |
| TriMul | 10,177.40 → 1,110.85 µs | 2,648 / 3,840 | Fixed final kernel: 1,158.46 ± 3.76 µs over three H100 repeats |

Both runs used Qwen3-14B for guidance, GLM-5.2 for execution, and 30 Reef
updates. They are single-run records, so the trajectories describe these runs
rather than variance across random seeds. For TriMul, the repeat measurement
is the stable latency result; the lower search-time value is retained to show
how the archive evolved.

The compact records, per-update trajectories, provenance hashes, and one
guidance-to-candidate case from each task are in [`results/`](results/). The
Polyomino case changes piece selection from a fixed order to a skyline-aware
decision. The TriMul case removes a global-memory round trip by reusing one
gated tile across three output blocks.

## Execution backends

Two frozen backends are built in:

| Backend | Model | Configuration |
|---|---|---|
| Local | `openai/gpt-oss-120b` | OpenAI-compatible endpoint, temperature 0, high reasoning effort, 1,200s timeout with no retries |
| OpenRouter | `z-ai/glm-5.2` | `OPENROUTER_API_KEY`, high reasoning effort, up to six transient-error retries |

`harness/harbor_agent.py` builds the local backend at
`http://127.0.0.1:8000/v1` with `high` reasoning effort; swap
`gpt_oss_120b_backend` for `openrouter_glm_5_2_backend` there to use the API
executor. The OpenRouter key is read only from the environment and
is never serialized into the library, resume state, result, or logs. The
key/account provider policy must allow a provider serving `z-ai/glm-5.2`;
request-level routing cannot override an account-level provider allowlist.

## Setup (once)

```bash
git submodule update --init third_party/reef-client
pip install -e ./third_party/reef-client
pip install -e .
```

The authoritative verifier is external: a FrontierCS checkout at the pinned
commit with its privileged go-judge started separately.

```bash
git clone https://github.com/FrontierCS/Frontier-CS.git reference/Frontier-CS
git -C reference/Frontier-CS checkout 6d597dfb60be9e592881aef051b94e30d197c436
docker compose -f reference/Frontier-CS/algorithmic/docker-compose.yml up -d --build
curl --fail http://127.0.0.1:8081/problems >/dev/null
```

The harness reaches that judge at `http://127.0.0.1:8081`; the Harbor verifier
runs inside the task container and reaches the same service through
`FRONTIERCS_JUDGE_URL` (the host gateway by default).

The local executor is one frozen SGLang server on its own GPU. An API
executor needs no third GPU:

```bash
python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --served-model-name openai/gpt-oss-120b \
  --host 127.0.0.1 --port 8000 --tp-size 1 --dtype auto \
  --trust-remote-code --context-length 32768 \
  --mem-fraction-static 0.88 --max-running-requests 4 \
  --disable-cuda-graph --reasoning-parser gpt-oss
```

## Run

```bash
./run.sh
```

The example runs the small `2 × 4`, 12,288-token qualification on two GPUs.
Groups, rollouts, sequence limits, concurrency, LoRA rank, and total steps are
deployment inputs rather than algorithm constants: they are written out twice,
as the constants at the top of `harness/harbor_agent.py` and as the matching
values in `serve.yaml`. The summary-only experiment family uses `8 × 16`; edit
both files together so the harness and the Reef/Slime stack cannot disagree.

The tested topology for the local executor is:

```text
GPU 0-1  Qwen3-8B guidance actor + Megatron LoRA trainer + SGLang serving
GPU 2    frozen GPT-OSS-120B executor (omit for the OpenRouter backend)
CPU      FrontierCS client; privileged go-judge runs as a separate service
```

To run the same loop from Reef's optional `tttd` image (whose solver
dependencies the generated programs use), build it and start `./run.sh`
inside a container with host networking and the state directory mounted:

```bash
docker build --pull -f docker/Dockerfile.reef --target tttd \
  -t reef-guidance-ttt:qwen3-8b .

docker run --rm --gpus '"device=0,1"' --network host --ipc host \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -e OPENROUTER_API_KEY \
  -v "$PWD/state:/workspace/Reef/recipes/tttd/examples/guidance_ttt/work" \
  reef-guidance-ttt:qwen3-8b \
  bash -lc 'recipes/tttd/examples/guidance_ttt/run.sh'
```

If Megatron initialization remains at zero GPU utilization in an NCCL
collective on a B200 NVLink node, retry with `-e NCCL_NVLS_ENABLE=0`; this is
a known host/driver transport interaction and does not change the algorithm.

## State and resume

All durable state lives under `work/polyomino_packing/`:

```text
work/polyomino_packing/
  guidance-run/library.json            working archive
  guidance-run/committed-library.json  archive paired with a durable step
  guidance-run/resume-state.json       next step, settings, per-step summaries
  checkpoints/megatron/                Megatron checkpoints
  checkpoints/hf/                      published HF/LoRA checkpoints
  artifacts.git/, agent-record/        Reef artifact and record stores
  lab/                                 reef-eval trial rows
  reef.log                             the stack's log
```

Re-running `./run.sh` resumes: the controller restores the committed archive,
checks that Reef's checkpoint starts at the same step, and refuses to continue
if the executor, cardinalities, sequence length, LoRA rank, or tensor-parallel
size changed. `STEPS` is the final total, not an additional count.

Reef's inference-admission controller holds requests across serving-weight
updates until the corresponding artifact head is committed, so no
recipe-specific publication barrier is needed between steps. A complete step
spanning multiple releases is reported as an explicit invariant
failure instead of leaving the run waiting for a training step that cannot
occur.

Use a fresh scenario for each discovery problem: Guidance-TTT fine-tunes on
one test problem rather than learning a general task policy.

## Tests

From the repository root:

```bash
ruff check recipes/tttd/examples/guidance_ttt tests/test_guidance_ttt.py
PYTHONPATH=. pytest -q tests/test_guidance_ttt.py tests/test_example_entrypoints.py
```

The tests cover strict parsing, summary-only code isolation, exact receipt
linkage, executor skipping on malformed guidance, dynamic cardinalities,
Discover-compatible PUCT/archive behavior, secret hygiene, the judge protocol
client, the task contract, the training barrier and its LoRA qualification
gate, the paired resume state, and the reef-eval entrypoint's task dispatch.

## Credits and license

`state.py`, `puct.py`, and `library.py`, together with the prompt/search design
used by this example, are adapted from
[`Chonghe-Jiang/open-ttt-verl@ea47140`](https://github.com/Chonghe-Jiang/open-ttt-verl/tree/ea47140ca7aea324d89ea5585afffb03c2c01522)
and were modified for Reef's receipt/report and TTTD training interfaces. The
upstream work and Reef are licensed under the Apache License 2.0; the repository
root [`LICENSE`](../../../../LICENSE) applies. The upstream NOTICE entry is:

> Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
