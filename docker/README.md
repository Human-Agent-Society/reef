# Docker

The Reef image bundles the full runtime (Slime, SGLang, Ray, Megatron)
so that a deployment runs entirely inside one image as plain processes. There
is no docker-compose layer: `reef serve -c <config>` reads the YAML's
`services` list and starts each declared process directly, with PID/log files
under `/tmp/reef-stack`.

## Build

```bash
docker build -f docker/Dockerfile.reef -t reef .
```

The image inherits the GPU training stack and installs Reef's exact runtime
pin from `pyproject.toml`, together with SGLang, Ray, and the training
dependencies. Having the runtime in the image does not mean its training
driver is running — only the `services` listed in your config run.

The default image pins the SGLang revision carrying Reef's adapter receiver,
so LoRA training (`--megatron-lora-rank`) works on any weight-training recipe
without a separate image. Override the revision with
`--build-arg SGLANG_COMMIT=<sha>`.

The image also replaces the base's FlashInfer with 0.7.0. Earlier releases
can compute wrong attention in SGLang's default `flashinfer` backend (upstream
FlashInfer issue #2896), so rollout log-probs drift from the trained policy.
If you serve from another environment with FlashInfer older than 0.7.0, set
`attention-backend: triton` under `inference.options`.

When Slime trains on rollout log-probs, its policy loss reports the mean
per-token gap between trainer and rollout log-probs as
`train/train_rollout_logprob_abs_diff`. This value has no fixed threshold. It
grows with the entropy of the model's predictions on the sampled tokens, which
Slime reports as `train/entropy_loss`. On RTX PRO 6000 (sm_120) with
FlashInfer 0.7.0, the `recipes/sao/examples/imo_answerbench` smoke run with
Qwen2.5-1.5B-Instruct reported 0.005 to 0.019. Each value was about 0.02 to
0.04 times `train/entropy_loss`. So this value alone does not show whether an
engine has the FlashInfer bug. Check the FlashInfer version instead. The image
build fails if the installed FlashInfer lacks the fix.

TTT-Discover's Qwen3-8B LoRA experiment uses the optional `tttd` target. It
adds only the Erdős evaluator's solver dependencies for generated programs,
alongside a qualified Slime digest:

```bash
docker build -f docker/Dockerfile.reef --target tttd \
  --build-arg SLIME_IMAGE_TAG='latest@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342' \
  -t reef-tttd:qwen3-8b .
```

## vLLM training image

`Dockerfile.reef` cannot host the vLLM training engine: vLLM 0.29 needs torch
2.13 and the `slimerl/slime` base ships torch 2.11. A deployment that pairs
`training.backend: slime` with `inference.backend: vllm` uses a second image,
`docker/Dockerfile.reef-vllm`, built on a Slime training layer that itself
starts from SGLang's torch 2.13 / CUDA 13 image. The script builds all three
stages:

```bash
docker/build-reef-vllm.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_IMAGE` | `lmsysorg/sglang:v0.5.18-cu130` | torch 2.13 / CUDA 13 base |
| `SLIME_DOCKERFILE_REF` | `2f23186...` | Slime revision whose `docker/Dockerfile` builds the training layer |
| `SLIME_COMMIT` | the `runtime` group pin in `pyproject.toml` | Slime installed in the image |
| `CUDA_ARCHS` | `90` | Transformer Engine and DeepEP architectures (`90`, `100a`, `103a`, `120f`, `;`-separated); each one adds compile time |
| `IMAGE` | `reef-vllm` | tag of the result |

The Reef stage alone, for a repository change on an existing Slime layer:

```bash
docker build -f docker/Dockerfile.reef-vllm --build-arg SLIME_IMAGE_TAG=reef-torch213 -t reef-vllm .
```

A stack on this image keeps the Slime options of any weight-training example
and changes the engine and the weight path:

```yaml
inference:
  backend: vllm
  num-gpus: "1"
  tensor-parallel-size: "1"
  options:               # vLLM engine-argument names
    max-model-len: "4096"
    gpu-memory-utilization: "0.6"
training:
  options:
    update-weight-transport: disk
    update-weight-disk-dir: /work/weights
```

The [configuration reference](https://reefinfra.ai/docs/reference/configuration/)
lists the pairing conditions (full mode, no LoRA rank, one engine per stack).

What this image is not: the Slime layer skips Slime's SGLang patches and the
Reef stage does not install the reviewed SGLang revision, so SGLang engines
and LoRA deployments stay on `Dockerfile.reef`. Expect a few
`Failed to import Triton kernels` lines at engine start: Triton 3.7 renamed
the `triton_kernels` MXFP4 MoE module that vLLM 0.29 imports, and dense
models never use it.

Verified on two 8x H200 hosts with the image this script builds (Qwen2.5-0.5B,
the SAO recipe with its critic, actor training from the first step): with one
training GPU and one engine, and with a tensor-parallel-2 trainer on two GPUs
plus one engine, three reports produced three training releases each, and the
engine's `/weight_info` version advanced after every release as it reloaded
the published weights over the disk path. The SGLang engine has not been run
on this image.

## Demo configs

| Config | Services started | GPU | Changes weights |
|---|---|---:|---:|
| `recipes/basic/local-sglang.yaml` | local SGLang + Reef | yes | no |
| `recipes/basic/external-provider.yaml` | Reef proxying to an HTTP provider | no | no |
| `recipes/<method>/examples/<example>/serve.yaml` | Ray + Slime bridge + Reef training | 2+ | yes |

Each config declares its services declaratively (`name`, `command`,
`ready` probe, `depends_on`). Commands use `${a.b.c}` interpolation against
the rest of the config, so adding or changing a service is a YAML-only edit.
The orchestrator launches Reef's HTTP child internally with the same config;
all other services are plain shell commands. See
[Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/) for the training
flow and data contract.

## Run

Inside the image (or any host with the deps installed):

```bash
export REEF_TOKEN=$(openssl rand -hex 16)
# edit the stack yaml: set model paths / provider creds
reef serve -c recipes/basic/local-sglang.yaml
```

Logs and PIDs land in `/tmp/reef-stack/`. To run just the reef HTTP service
against an already-running provider, use a config whose `services` list
contains only Reef:

```bash
reef serve -c recipes/basic/external-provider.yaml
```

## Persistence and cleanup

Agent records and exported checkpoints persist under `reef.state_dir`
(`/var/lib/reef` by default). Stop the processes (`kill` the PIDs in
`/tmp/reef-stack/`) but keep state; to also delete recorded agent records,
remove that directory.
