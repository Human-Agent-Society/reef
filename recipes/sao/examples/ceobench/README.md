# SAO on CEO-Bench

This example runs [CEO-Bench](https://ceobench.com)
([paper](https://arxiv.org/abs/2606.18543),
[code](https://github.com/zlab-princeton/ceobench-src)) through Reef and trains
the [`sao` recipe](../../../../docs/user-guide/recipes/sao.rst) on it. CEO-Bench
simulates an AI startup for 500 days from $1M in cash: 34 tools, a 19-table
database, simulated social media, and a market with hidden preferences,
competitor pressure, and delayed consequences. The primary metric is final
cash; survival days and bankruptcy are secondary. The benchmark's own
bash-agent baseline plays the game; here its agent role is served by Reef, so
every model call is recorded and attributable, while the two simulator roles
(social posts, enterprise customers) stay outside Reef, as
[Guidance-TTT](../../../tttd/examples/guidance_ttt/README.md) keeps its frozen
executor. This is step 1 of
[issue #428](https://github.com/Human-Agent-Society/reef/issues/428) with one
reward-shaping choice made for it (step 2 stays open; see below).

```text
harbor/                one CEO-Bench episode as a Harbor task
  task.toml              48h agent window, resource limits
  instruction.md         what the task is (the model never sees it: CEO-Bench owns its prompt)
  environment/
    Dockerfile           python:3.13 + uv + the pinned CEO-Bench checkout, patched and rebuilt
    reef.patch           the three hunks the checkout needs (below)
  tests/
    test.sh              runs the verifier inside the task container
    score.py             decrypts the run's world.nmdb: reward, final cash, survival days, bankrupt
harness/               agent harness (imports reef_client, not reef)
  __init__.py            lazily exports HarborAgent
  agent.py               HarborAgent: sidecar on the host, the benchmark runner in the container
  report.py              posts the verifier's score against every turn's receipt
serve.yaml             Reef + Ray + Slime/Megatron + SGLang, Qwen3.6-27B through LoRA, critic colocated
docker-compose.yaml    the stack in the reef image, host networking, six GPUs
run.py                 the loop: one episode per seed, trained between seeds
run.sh                 brings the stack up, then runs run.py through reef-eval
pyproject.toml         makes the harness importable
results/               the smoke run's manifest
```

## The harness

Harbor gives the trial a container built from `harbor/environment/Dockerfile`:
CEO-Bench at commit `d2b7b32e` with its own `uv` environment (Python 3.13,
SQLCipher reader included) and a rebuilt public bundle. `HarborAgent.run`
then does three things.

1. **A reef-client sidecar on the host.** `reef_client.serve` listens on an
   ephemeral port, replaces the `Authorization` header with the Reef token,
   stamps `x-reef-scenario`, forwards everything else to Reef unchanged, and
   keeps each `/v1/chat/completions` exchange with its
   `x-reef-agent-record-id` receipt. The benchmark's OpenAI client sees a plain
   base URL.
2. **The benchmark runner in the container.** One `exec` runs
   `saas_bench.agents.bash_agent.run_test --provider openai --base-url
   http://<host>:<port>/v1 --seed S --days D`, the paper's baseline with the
   agent role redirected. The container reaches the host by the LAN address in
   `REEF_SERVICE_URL`, which is why Reef listens on `0.0.0.0` and `run.sh`
   derives the URL from `hostname -I`. `SAAS_BENCH_*`, `OPENAI_*`,
   `ANTHROPIC_*`, and `AWS_*` variables are forwarded into that `exec` for the
   simulator roles; nothing else from the host environment is.
3. **The run directory and the receipts.** When the runner exits, the run
   directory (`world.nmdb`, `config.json`, `checkpoint.json`, `logs/`,
   `agent_workspace/`) is downloaded next to the trial's agent logs, the
   receipts go into the agent context in call order, and the sidecar stops.

Harbor then runs `tests/test.sh` in the same container. `score.py` opens the
run's `world.nmdb` with the checkout's own `load_session_db` and writes
`reward.json` with the final cash as the running sum of the `ledger` table,
survival days as the last day any daily table reached, `bankrupt` as final
cash below zero, and `reward` as final cash over the starting balance
(1.0 is break-even). A watcher thread in the harness reads Harbor's
`result.json` and posts the score to Reef (`harness/report.py`).

### The patch to CEO-Bench

`reef.patch` is applied to the pinned checkout at image build time and the
public bundle is rebuilt so the engine carries it. Five hunks:

- `agents/bash_agent/agent.py`: `SAAS_BENCH_OPENAI_CHAT_COMPLETIONS=1` pins the
  agent to `/v1/chat/completions`. The runner otherwise prefers the OpenAI
  Responses API for any OpenAI-compatible endpoint it does not recognize, and
  Reef serves chat completions and Anthropic messages.
- `server_entry.py`: `SAAS_BENCH_<FIELD>` environment variables override the
  simulator roles' provider and model (`SOCIAL_POST_LLM_PROVIDER`,
  `SOCIAL_POST_LLM_MODEL`, `ENTERPRISE_LLM_PROVIDER`, `ENTERPRISE_LLM_MODEL`)
  without editing `config.py` and rebuilding the bundle.
- `server_entry.py`: provider `none` (set for both roles) runs the engine with
  no customer LLM at all. The engine already supports that: customer, macro,
  and competitor posts come from its built-in templates, the agent's own posts
  are not judged, and enterprise negotiation is structured rather than
  generated at this commit. It is the switch for runs that should not depend
  on a paid model.
- `customer_llm.py`: token counts missing from a Responses reply count as
  zero instead of failing the cost log. SGLang's Responses endpoint fills
  `prompt_tokens` but not `input_tokens`.
- `agents/bash_agent/agent.py`: `SAAS_BENCH_MAX_COMPLETION_TOKENS` caps the
  agent's completion request (default 16384, the benchmark's value; `run.sh`
  leaves it unset). It exists for engines whose window cannot hold the
  default plus the prompt.

Everything else is the benchmark as published: default `config.py`
difficulty (competitor feedback range 0.2 to 0.5), the bash agent's prompt
and tools, and its `temperature=1.0` sampling.

### Simulator roles

By default the simulator roles keep the benchmark's settings, Haiku 4.5 for
social posts and Sonnet 4.5 for enterprise customers through the Anthropic
API, read from `ANTHROPIC_API_KEY` in the environment `run.sh` runs in.
Bedrock works the same way with `AWS_*` credentials and
`SAAS_BENCH_*_LLM_PROVIDER=bedrock`. No credential lives in the repository.

To run without any customer LLM, set both roles to `none`; the market then
speaks in the engine's template posts, which carry the same satisfaction and
virality mechanics but no generated text:

```bash
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=none SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=none
```

A local OpenAI-compatible server with a Responses endpoint can stand in for
both roles instead:

```bash
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=openai SAAS_BENCH_SOCIAL_POST_LLM_MODEL=<served name>
export SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=openai SAAS_BENCH_ENTERPRISE_LLM_MODEL=<served name>
export OPENAI_BASE_URL=http://<host>:<port>/v1 OPENAI_API_KEY=local
```

The smoke run below used one; a result meant to compare with the paper must
use the benchmark's defaults.

## Reward shaping

One episode is one terminal score over hundreds of turns, and the `sao`
recipe trains one single-reference report per step. The choice made here is
**episode-level, applied to every turn**: after the verifier scores the run,
the harness posts one report per model call, each carrying the episode's
`reward` (final cash over the starting balance) and referencing that call's
receipt. Each turn's prompt is the conversation the benchmark agent actually
sent (system prompt, the week's tool calls and outputs), so the sample is the
turn in its real context, and the critic's skip-observation GAE runs over the
turn's own tokens. Reports arrive together, so `max_staleness` in
`serve.yaml` is raised to 64: SAO's DIS calibration is what admits a rollout
whose weights have moved on, and a turn admitted past that window trains
against a policy 64 steps newer than the one that produced it.

Alternatives considered, and left to issue #428's step 2:

- **Per-period cash delta.** The simulator exposes cash daily, and the bash
  agent's context resets at every `next-week`, so a week is a natural
  conversation to score by its own cash change. It is denser but myopic: R&D
  and advertising cost cash this week and pay later, which is the delayed
  structure the benchmark is built around. The sidecar's captures carry the
  full request, so grouping turns into weeks needs no benchmark change.
- **A judged turn-level signal.** What single-stream PPO wants
  (`recipes/openclawrl/`); the simulator's own feedback (database state,
  cash) can inform the judge. Not an SAO shape.
- **One multi-reference report per week.** Reef assembles an ordered
  multi-reference report into one multi-turn sample when the recipe sets
  `accept_multi_turn_policy_samples`; SAO does not, and a thinking model's
  re-rendered history drops its earlier reasoning, which the assembly treats
  as a fork.

## Run

Prerequisites: Docker with the NVIDIA runtime, `uv`, the `reef` image
(`docker build -f docker/Dockerfile.reef -t reef .` from the repository root),
the policy model, and credentials for the simulator roles.

```bash
cd recipes/sao/examples/ceobench
hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
export ANTHROPIC_API_KEY=...
CEOBENCH_SEEDS=42,43,44 CEOBENCH_DAYS=500 ./run.sh
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (`~/models`),
`RUN_DIR` (`./work`), and `REEF_GPU_0..5` (the six devices the stack uses:
four actor GPUs, TP4 with the critic colocated, and a two-GPU rollout
engine). The policy is `Qwen3.6-27B` trained through Megatron Bridge LoRA:
the base stays frozen in the actor and in the SAO critic, the adapters and
the critic's value head train, and the rollout engine serves the published
adapter. The models tried before it are recorded below. It mints a token
into `$RUN_DIR/token`, brings the stack up with `docker compose up --wait`,
and runs `run.py` in an ephemeral `uv` environment with `reef-eval[harbor]`
and this harness. The stack stays up between runs; `docker compose down`
stops it. Per-seed rows land in `work/lab`, each trial's run directory under
the trial's `agent/ceobench/`.

`run.py` runs the seeds in order and, after each, waits for the scenario's
training releases to stop growing before the next seed, so seed N+1 is served
by what seed N taught:

```bash
curl -sS -H "Authorization: Bearer $(cat work/token)" \
  http://$(hostname -I | awk '{print $1}'):28900/reef/scenarios/ceobench-sao/releases
```

## Results

### Smoke run

`results/2026-09-13-smoke-qwen3.6-27b-lora/manifest.json` holds the lab row,
the trial result, and the run configuration. One seed, 14 simulated days,
the stack in `serve.yaml`, and a local `Qwen3-4B-Instruct-2507` on SGLang
standing in for both simulator roles, so the score is not comparable with
the paper's.

| | |
| --- | --- |
| Policy | `Qwen3.6-27B`, LoRA rank 32 on actor and critic, TP4; rollout TP2, 128k window |
| Episode | seed 42, 14 days, completed in 35 turns (10m46s), no bankruptcy |
| Final cash | $793,047 (`reward` 0.793) |
| Tokens | 501,917 in / 40,700 out; longest turn 27,673, so every turn fit the 48k trainer window |
| Reports | 35, one per turn, all accepted |
| Training | one SAO release committed per accepted report; the first critic steps brought the value loss from 10.5 to 3.3, the adapter was exported to `checkpoints/hf/<step>/adapter_model.safetensors` and loaded into the engine (`load_lora_adapter_from_distributed`) |

The agent read `docs/simulator-instructions.md`, the CLI reference, and the
SDK source, wrote `daily_scripts/week0_setup.py` (prices A/B/C at $15/$49/$149,
model tiers 1/2/3, quotas, $1,500/day development, targeted ad spend by
segment), called `next-week` with its rationale and twelve forecasts on the
first try, then spent week two querying the tables, cutting prices to
$9/$39/$99, moving ad spend to one channel, doubling development spend,
starting R&D tier 1 ($166,667), and writing a `MEMORY.md` for the next week.
Week 2 ended with 10 subscribers; the cash drop is that R&D start plus
$3,000/day of development spend, an investment the 14-day horizon cannot
repay.

Earlier attempts, same harness:

- `Qwen3-4B-Thinking-2507` (full parameters): never produced a valid
  `next-week` call, wrote a stub `next_week.py` and looped on it; stopped
  after 400 turns at day 0.
- `Qwen3-8B` (full parameters, 40k window): completed 14 days in 11 turns
  with one configuration change and no prices set (final cash $997,900);
  the per-commit Megatron checkpoint then failed with the CPU-offloaded
  optimizer.

### Policy choices on one 8-GPU node

- **Qwen3-4B-Thinking-2507** (the OpenClaw-RL example's policy): runs, but
  cannot operate the benchmark's CLI; see the smoke run above.
- **Qwen3-30B-A3B-Thinking-2507** (the SAO example's paper-scale policy):
  not trainable here with the critic. With six actor GPUs the expert weights
  force either expert tensor parallelism, which the Megatron bridge does not
  shard (`Shape mismatch loading ...experts.linear_fc1.weight0: HuggingFace
  (1536, 2048), Megatron (768, 2048)` at TP2, ETP2), or TP1, where the
  trainer's fp32 full-vocabulary logits and their gradient cap the sequence
  near 32k tokens. A rollout-only Reef stack (no training) has neither limit,
  so the untrained baseline can still use it.
- **Qwen3-8B** at TP4, full parameters, inside its native 40960-token
  window (its 128k needs YaRN, which the trainer's rotary embedding does not
  apply): runs and operates the CLI, but in a 14-day episode made one
  configuration change and never set a price. Its per-commit Megatron
  checkpoint also failed with the CPU-offloaded optimizer
  (`KeyError: 'master_param'`), which `no-save-optim` works around.
- **Qwen3.6-27B through LoRA** is the configuration shipped: frozen base
  weights in both the actor and the critic, rank-32 adapters on the
  attention and MLP projections, the critic's value head trainable. This is
  the first critic-bearing recipe to use Reef's Megatron LoRA, which until
  now applied to the actor only (`prepare_critic_args` now keeps the adapters
  for the critic). Full-parameter training of the 27B pair would need about
  216 GB for weights and gradients alone. The engine serves a 128k window;
  the trainer's is 48k, bounded by its fp32 full-vocabulary logits (248k
  entries per token), so the harness reports only turns that fit
  (`CEOBENCH_TRAIN_MAX_TOKENS` in `run.sh`). A turn that has read the
  benchmark's docs sits at 40k tokens after four calls, so early-week turns
  train and late-week ones are recorded only.

### Not yet run

- The untrained 500-day baseline with the benchmark's Anthropic simulator
  roles, at least three seeds (issue #428, acceptance criterion 1). The cost
  of one such episode sizes the seed count.
- Training runs long enough to compare against the untrained base and the
  paper's rule-based baseline.

## Open items

- **License.** `zlab-princeton/ceobench-src` carries no license file. The
  image clones a pinned commit at build time and the repository ships none of
  the benchmark's code or fixtures; confirm terms with the authors before a
  result page cites it.
- **Sandboxing.** The benchmark sandboxes the agent's shell with `bwrap` when
  present and falls back to plain execution otherwise. Here the Harbor
  container is the sandbox; the agent's shell runs unsandboxed inside it, and
  the `novamind-operation` zipapp with the database key sits at
  `/opt/ceobench/public/` inside the same container, readable by the agent.
  The benchmark's docs recommend hiding it behind a wrapper.
