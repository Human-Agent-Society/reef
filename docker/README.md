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
