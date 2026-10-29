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
dependencies. Having the runtime in the image does not start it: `reef serve`
starts only the `services` that your config lists and the ones that its
training backend needs.

The default image pins the SGLang revision that Reef's serving engines run,
so LoRA training (`--megatron-lora-rank`) works on any weight-training recipe
without a separate image. Override the revision with
`--build-arg SGLANG_COMMIT=<sha>`. The build also applies
`docker/patch/sglang_qwen3_5_mrope_rope.patch`, which fixes how Qwen3.5
vision-language models rotate image tokens (sgl-project/sglang#34446). It
skips the patch when the revision already has the fix. If the revision has
no fix and the patch does not apply to it, the build fails.

The build also changes Slime's Qwen3.5 trainer so that its vision tower
computes the same image features as SGLang. It copies
`docker/patch/slime_qwen3_5_vision_numerics.py` into Slime and applies
`docker/patch/slime_qwen3_5_vl_vision_numerics.patch`. The features are
bit-identical to the engine's only when all of these are true:

* The engine uses the sdpa vision attention backend. On CUDA, SGLang uses
  another backend by default. Set `mm-attention-backend: sdpa` under
  `inference.options`. The engine log shows
  the backend in the line `Using <backend> as multimodal attention backend`.
* Each request has at most one image that misses the engine's embedding
  cache. The engine encodes those images of one request together, and the
  trainer encodes each image alone. The engine encodes per request only when it
  encodes images itself: with no `encoder` worker (EPD) and with
  `mm-enable-dp-encoder` off. The EPD encoder merges the images of up to
  `SGLANG_ENCODER_MAX_BATCH_SIZE` (default 8) requests in one call.
* SGLang is the default pin. With another `SGLANG_COMMIT`, check the features
  again.

The change was tested with transformers 5.12.1 and 5.13.0. The image
installs 5.12.1. The build fails if the image has a version outside 5.12 and
5.13, because the change then does nothing.

The build also applies `docker/patch/slime_thd_cp1_indices.patch`. Without it,
Slime's Qwen3.5 vision-language trainer fails in the first training step.
This happens when the context parallel size is 1 and a packed sequence has an
odd length.
THUDM/slime#2376 fixes the call site. This patch is a helper-side variant
with the same result.

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

## Demo configs

| Config | Services started | GPU | Changes weights |
|---|---|---:|---:|
| `recipes/basic/local-sglang.yaml` | local SGLang + Reef | yes | no |
| `recipes/basic/external-provider.yaml` | Reef proxying to an HTTP provider | no | no |
| `recipes/<method>/examples/<example>/serve.yaml` | Ray + inference component + Reef with Slime training | 2+ | yes |

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
