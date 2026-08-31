# Qwen3-8B summary-only Guidance-TTT deployment

This deployment runs Polyomino, EdgeBench VLIW, or TriMul directly on a GPU
cluster. It contains no cloud-provider launcher. The processes are deliberately
separate:

```text
Qwen/Reef node       Qwen3-8B SGLang inference + Megatron rank-32 LoRA training
execution endpoint   frozen GPT-OSS-120B or OpenRouter GLM-5.2
verifier worker      FrontierCS CPU judge, VLIW CPU Docker judge, or TriMul H100
```

Only Qwen guidance tokens are trained. The execution model writes complete
candidate programs, and the official task verifier supplies the reward. Kernel
verifiers may run remotely through the provider-neutral `POST /evaluate`
service; no hardware-provider code is part of the example.

## Prepare the Qwen/Reef node

Clone Reef, build the existing optional TTTD image, and create persistent state:

```bash
git clone https://github.com/Human-Agent-Society/reef.git
cd reef
mkdir -p state

docker build --pull -f docker/Dockerfile.reef --target tttd \
  --build-arg SLIME_IMAGE_TAG='latest@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342' \
  -t reef-guidance-ttt:qwen3-8b .
```

The node needs NVIDIA Container Toolkit, host networking, at least 256 GiB host
RAM for the two-GPU actor path, and persistent storage for Qwen3-8B,
checkpoints, and archive history. Export `HF_TOKEN` when Hugging Face
authentication is required.

The examples below use OpenRouter GLM-5.2 as the frozen execution model. Keep
the key only in the environment:

```bash
read -rsp 'OPENROUTER_API_KEY: ' OPENROUTER_API_KEY && export OPENROUTER_API_KEY
echo
```

For a local executor, start `openai/gpt-oss-120b` behind an OpenAI-compatible
endpoint and replace `--executor openrouter_glm_5_2` with:

```text
--executor gpt_oss_120b
--executor-base-url http://127.0.0.1:8000/v1
--gpt-oss-reasoning-effort high
```

GPT-OSS is deterministic by default. For the kernel settings reproduced from
the source integration, pass `--executor-temperature 0` with GLM-5.2 and set
the task-specific execution budget shown below.

## Polyomino qualification

Clone and pin FrontierCS on the host, then mount it read-only at the same path
inside the Reef container:

```bash
git clone https://github.com/FrontierCS/Frontier-CS.git reference/Frontier-CS
git -C reference/Frontier-CS checkout 6d597dfb60be9e592881aef051b94e30d197c436
docker compose -f reference/Frontier-CS/algorithmic/docker-compose.yml up -d --build
curl --fail http://127.0.0.1:8081/problems >/dev/null
```

Run one strict `2 × 4` step:

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e OPENROUTER_API_KEY \
  -v "$PWD/reference:/workspace/Reef/reference:ro" \
  -v "$PWD/state:/workspace/state" \
  reef-guidance-ttt:qwen3-8b \
  python -m examples.guidance_ttt.deployments.qwen3_8b_lora.cluster \
    --task polyomino_packing \
    --executor openrouter_glm_5_2 \
    --gpu-count 2 \
    --tensor-parallel-size 2 \
    --groups 2 \
    --rollouts 4 \
    --guidance-max-tokens 8192 \
    --sequence-length 12288 \
    --max-workers 4 \
    --judge-url http://127.0.0.1:8081 \
    --lora-rank 32 \
    --steps 1 \
    --require-nonconstant-step \
    --run-name polyomino-qualification
```

## VLIW verifier and qualification

The VLIW raw metric is official simulator cycles, lower is better. Reef uses
`1,000,000 / cycles` as its dense maximize-oriented training reward. The
official starter and judge image are pinned under
`tasks/assets/vliw_kernel_optimization/`; hidden cases remain inside the judge
image.

On a CPU worker with Docker, install Reef's Python dependencies and start the
authenticated service. The judge itself runs with no network, four CPUs, and
8 GiB memory per candidate:

```bash
python -m pip install -e .
docker pull seededge/edgebench.judge.vliw_kernel_optimization:5cdef0021634
export GUIDANCE_TTT_VERIFIER_TOKEN="$(openssl rand -hex 32)"
export VLIW_JUDGE_TOKEN="$GUIDANCE_TTT_VERIFIER_TOKEN"

PYTHONPATH=. python -m examples.guidance_ttt.verifier.service \
  --task vliw_kernel_optimization \
  --host 0.0.0.0 \
  --port 8090 \
  --concurrency 16 \
  --max-runner-timeout-s 60
```

Expose port 8090 only to the Qwen/Reef node. On that node, export the same
token and endpoint, then create a non-secret transport override:

```bash
export VLIW_JUDGE_URL=http://127.0.0.1:8090/evaluate
export VLIW_JUDGE_TOKEN='<same bearer token>'

cat >state/vliw-http-verifier.json <<'JSON'
{
  "provider": "http",
  "endpoint_env": "VLIW_JUDGE_URL",
  "api_key_env": "VLIW_JUDGE_TOKEN"
}
JSON
```

For a separate worker, replace `127.0.0.1` with its private address. A strict
one-step qualification is:

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e OPENROUTER_API_KEY \
  -e VLIW_JUDGE_URL \
  -e VLIW_JUDGE_TOKEN \
  -v "$PWD/state:/workspace/state" \
  reef-guidance-ttt:qwen3-8b \
  python -m examples.guidance_ttt.deployments.qwen3_8b_lora.cluster \
    --task vliw_kernel_optimization \
    --executor openrouter_glm_5_2 \
    --executor-temperature 0 \
    --executor-max-tokens 8192 \
    --gpu-count 2 \
    --tensor-parallel-size 2 \
    --groups 2 \
    --rollouts 4 \
    --guidance-temperature 0.9 \
    --guidance-top-p 0.95 \
    --sampling-seed 0 \
    --guidance-max-tokens 8192 \
    --sequence-length 12288 \
    --max-workers 8 \
    --verifier-config /workspace/state/vliw-http-verifier.json \
    --lora-rank 32 \
    --steps 1 \
    --require-nonconstant-step \
    --run-name vliw-qualification
```

Before starting Qwen, the runner evaluates the pinned starter through this
same official judge and creates a verified run-local seed. Training cannot
start from an unverified VLIW baseline.

## TriMul H100 verifier and qualification

TriMul requires a separate H100 because correctness and timing must use the
official GPU evaluator. The raw metric is the geometric mean of seven H100
case runtimes in microseconds, lower is better; reward is exactly
`1,500 / runtime_us`.

On an isolated H100 worker, clone the same Reef revision and create the pinned
task environment. The qualified environment is CUDA 12.8 on Ubuntu 24.04,
Python 3.13, PyTorch `>=2.7,<2.8` from the CUDA 12.8 index, and Triton 3.3.1:

```bash
python3.13 -m venv .venv-trimul
.venv-trimul/bin/python -m pip install --upgrade pip
.venv-trimul/bin/pip install ninja~=1.11 wheel~=0.45 requests~=2.32.4 \
  packaging~=25.0 numpy~=2.3 pytest PyYAML
.venv-trimul/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
  'torch>=2.7.0,<2.8.0'
.venv-trimul/bin/pip install triton==3.3.1

export GUIDANCE_TTT_VERIFIER_TOKEN="$(openssl rand -hex 32)"
export TRIMUL_JUDGE_TOKEN="$GUIDANCE_TTT_VERIFIER_TOKEN"
export TORCH_CUDA_ARCH_LIST='9.0 9.0a'

PYTHONPATH=. .venv-trimul/bin/python -m examples.guidance_ttt.verifier.service \
  --task trimul \
  --host 0.0.0.0 \
  --port 8090 \
  --concurrency 1 \
  --max-runner-timeout-s 520 \
  --evaluator-dir examples/guidance_ttt/tasks/assets/trimul_evaluator \
  --evaluation-gpu-label 'NVIDIA H100 80GB HBM3'
```

Run this worker in a disposable container or VM with outbound network disabled,
no model/API credentials, and only its bearer token. The service passes a
sanitized compiler/CUDA environment to candidate subprocesses, but the process
still needs an infrastructure sandbox because candidate Python is untrusted.

On the Qwen/Reef node:

```bash
export TRIMUL_JUDGE_URL=http://<private-h100-address>:8090/evaluate
export TRIMUL_JUDGE_TOKEN='<same bearer token>'

docker run --rm \
  --gpus '"device=0,1"' \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e OPENROUTER_API_KEY \
  -e TRIMUL_JUDGE_URL \
  -e TRIMUL_JUDGE_TOKEN \
  -v "$PWD/state:/workspace/state" \
  reef-guidance-ttt:qwen3-8b \
  python -m examples.guidance_ttt.deployments.qwen3_8b_lora.cluster \
    --task trimul \
    --executor openrouter_glm_5_2 \
    --executor-temperature 0 \
    --executor-max-tokens 16384 \
    --gpu-count 2 \
    --tensor-parallel-size 2 \
    --groups 1 \
    --rollouts 2 \
    --guidance-temperature 0.9 \
    --guidance-top-p 0.95 \
    --sampling-seed 0 \
    --guidance-max-tokens 8192 \
    --sequence-length 16384 \
    --max-workers 2 \
    --verifier-concurrency 1 \
    --lora-rank 32 \
    --steps 1 \
    --require-nonconstant-step \
    --run-name trimul-qualification
```

The default fixed TriMul seed is independently generated from the public task
statement, passes all 18 correctness cases and all seven benchmark cases, and
records an official H100 geometric-mean runtime of `10177.396849081848 us`.
The published TTT-Discover solution is retained only as an attributed reference
asset and is not the default seed.

## Large-scale adaptive/control runs

First complete the strict task qualification. Then run the adaptive experiment
and a frozen-policy control with identical task, seed, executor, sampling,
cardinality, token, and verifier settings but different run names. For the
source-matched VLIW scale, use:

```text
--groups 8 --rollouts 16 --steps 50
--training-horizon-steps 50
--guidance-temperature 0.9 --guidance-top-p 0.95 --sampling-seed 0
```

Add `--frozen-policy-control` only to the control command. It still performs
Qwen inference, execution, verification, and PUCT archive evolution, but sends
no score reports to Reef. The runner asserts after every step that the bridge
completed zero optimizer steps and the serving weight version did not change.
The adaptive run asserts a positive finite gradient, frozen base weights,
nonzero LoRA-B update, checkpoint creation, and a new served adapter version.

TriMul can use the same `8 × 16` logical batch, but each candidate performs 18
correctness and seven timing cases. For throughput, put one concurrency-1
service on each isolated H100 and use a private load balancer; never evaluate
two candidates concurrently on one H100 because contention invalidates the
timing comparison. Set `--verifier-concurrency` to the number of H100 workers
behind that endpoint.

Omit `--require-nonconstant-step` for long stochastic runs so an occasional
all-invalid or constant-reward step is recorded rather than aborting the
experiment. Keep it for the initial qualification.

## State, resume, and result audit

Each run is stored under:

```text
state/guidance-runs/<run-name>/
  library.json
  committed-library.json
  resume-state.json
  result.json
  stack.log
  checkpoints/megatron/        # adaptive runs only
```

Resume with the same run name and immutable settings plus `--resume`; `--steps`
is the new final total rather than an additional count. If a run will be
launched in shorter invocations, set the final planned
`--training-horizon-steps` on the first invocation and keep it unchanged on
resume. The runner validates
task identity, score direction, seed SHA-256, non-secret verifier settings,
sampling settings, executor configuration, cardinality, sequence length, LoRA
rank, TP, scheduler horizon, and adaptive/control mode. Adaptive checkpoint and
archive state must agree at the same step boundary. A frozen control restarts
the same base policy and resumes only its archive.

`result.json` and every step summary record task, raw metric direction, valid
rollouts, reward diversity, policy sampling-seed ranges, whether training was
reported, weight versions, gradient norm, and best raw score. Verifier tokens
and API keys are not serialized.

If Megatron initialization remains at zero GPU utilization in an NCCL
collective on a B200 NVLink node, retry the actor container with
`-e NCCL_NVLS_ENABLE=0`; this changes the transport only, not the algorithm.
