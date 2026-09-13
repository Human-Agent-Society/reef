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
run.py                 one episode, trained while it is played
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
   receipts go into the agent context in call order with their week and
   token count, and the sidecar stops.

Harbor then runs `tests/test.sh` in the same container. `score.py` opens the
run's `world.nmdb` with the checkout's own `load_session_db` and writes
`reward.json` with the final cash as the running sum of the `ledger` table,
survival days as the last day any daily table reached, `bankrupt` as final
cash below zero, and `reward` as final cash over the starting balance
(1.0 is break-even). A watcher thread in the harness reads Harbor's
`result.json` for the final cash that closes the episode's last week
(`harness/report.py`).

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
- `customer_llm.py`, `simulation.py`: the two social-media functions that
  only had Bedrock and Anthropic paths (judging the agent's own post from
  each customer group's view, and a customer's reply to it) get the same
  OpenAI Responses fallback as the other simulator calls. Without it the
  engine's `next-week` failed the first time the agent posted.
- `agents/bash_agent/tools.py`, `run_test.py`: with `SAAS_BENCH_TOOL_USER`
  set and no `bwrap`, the agent's shell runs as that user through `setpriv`
  and the runner hands it the workspace. The image creates the user
  (`agent`), keeps the engine's source and host-side bundle root-only, and
  `run.sh` sets the variable.

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

The reward is online and weekly. CEO-Bench advances in weeks: the agent works
in one conversation until it calls `next-week`, the engine steps seven days
and returns the next dashboard, and the runner rebuilds the conversation from
it. Every request therefore carries the dashboard of the week it belongs to
(`=== Week N Dashboard (Day D) ===`, opening cash on the next line), and the
sidecar's captures let the harness group turns by week without touching the
benchmark. A reporter thread polls those captures while the episode runs;
when week N+1's dashboard appears, week N is over and each of its turns is
reported with

    score = (cash at the start of week N+1 - cash at the start of week N) / $1,000,000

as a single-reference report, so the `sao` recipe trains on the week's turns
while the agent is already playing week N+1 and the engine serves the
updated adapter from then on. The last week closes with the
verifier's final cash, posted by a watcher thread once Harbor writes
`result.json`. The Harbor reward itself stays the benchmark's terminal metric
(final cash over the starting balance); it is evaluation only.

Turns of one week share the week's score; the critic's skip-observation GAE
does the credit assignment inside each turn. A weekly delta is dense enough
for SAO's one-rollout-per-step cadence and lines up with the benchmark's own
decision period, at the cost of being myopic: R&D and advertising cost cash
this week and pay later. Two extensions are left open: a lag of `k` weeks
(report week N once week N+k's cash is known) and a judged turn-level signal
of the kind single-stream PPO wants (`recipes/openclawrl/`).

## Run

Prerequisites: Docker with the NVIDIA runtime, `uv`, the `reef` image
(`docker build -f docker/Dockerfile.reef -t reef .` from the repository root),
the policy model, and credentials for the simulator roles.

```bash
cd recipes/sao/examples/ceobench
hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
export ANTHROPIC_API_KEY=...
CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
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
and this harness. The episode row lands in `work/lab`, the trial's run
directory under the trial's `agent/ceobench/`.

This is test-time training: the policy adapts inside the episode it is
scored on, and the number to compare is that episode's final cash against the
same seed played by the untrained model. Replicates are independent runs from
the base model, one stack each (`docker compose down` between them, or a
fresh `RUN_DIR`): a Reef process trains one scenario for its lifetime, so a
second seed on the same stack would start from the first seed's adapter.
After the episode `run.py` waits for the scenario's training releases to
stop growing, so the adapter on disk is the one the episode ended with:

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

### Untrained baseline

`serve-baseline.yaml` serves the same model through Reef with no training
stack (SGLang TP4, the model's 262k window, the record-only recipe). An
episode on it is the untrained number a trained episode on `serve.yaml` is
compared against, same seed and simulator roles. Start it in the reef image
on four GPUs and run the episode with `CEOBENCH_TRAIN_MAX_TOKENS=0`:

```bash
docker run -d --name reef-ceobench-baseline --network host --ipc host --shm-size 32gb \
  --gpus '"device=0,1,2,3"' -v ~/models:/root/models -v "$PWD/work/baseline:/var/lib/reef" \
  -v "$(cd ../../../.. && pwd):/workspace/Reef" -e REEF_TOKEN="$(cat work/token)" \
  -e PYTHONPATH=/workspace/Reef reef \
  reef serve -c /workspace/Reef/recipes/sao/examples/ceobench/serve-baseline.yaml
```

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
  container is the outer sandbox and the agent's shell runs as an
  unprivileged user inside it (`SAAS_BENCH_TOOL_USER`), so it cannot signal
  the root-owned engine or read the engine's source and host-side bundle.
  An earlier run without this, after an engine error, saw the agent read
  the engine's source, stop the server, and start a new session. The copy of
  the `novamind-operation` zipapp in the agent's workspace still embeds the
  database key, as it does upstream; the benchmark's docs recommend hiding
  it behind a wrapper.
