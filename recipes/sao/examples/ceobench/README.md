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
serve.yaml             Reef + Ray + Slime/Megatron + SGLang, Qwen3-4B-Thinking-2507, critic colocated
docker-compose.yaml    the stack in the reef image, host networking, five GPUs
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
public bundle is rebuilt so the engine carries it. Three hunks:

- `agents/bash_agent/agent.py`: `SAAS_BENCH_OPENAI_CHAT_COMPLETIONS=1` pins the
  agent to `/v1/chat/completions`. The runner otherwise prefers the OpenAI
  Responses API for any OpenAI-compatible endpoint it does not recognize, and
  Reef serves chat completions and Anthropic messages.
- `server_entry.py`: `SAAS_BENCH_<FIELD>` environment variables override the
  simulator roles' provider and model (`SOCIAL_POST_LLM_PROVIDER`,
  `SOCIAL_POST_LLM_MODEL`, `ENTERPRISE_LLM_PROVIDER`, `ENTERPRISE_LLM_MODEL`)
  without editing `config.py` and rebuilding the bundle.
- `customer_llm.py`: token counts missing from a Responses reply count as
  zero instead of failing the cost log. SGLang's Responses endpoint fills
  `prompt_tokens` but not `input_tokens`.

Everything else is the benchmark as published: default `config.py`
difficulty (competitor feedback range 0.2 to 0.5), the bash agent's prompt
and tools, and its `temperature=1.0`, 16,384-token completion requests.

### Simulator roles

By default the simulator roles keep the benchmark's settings, Haiku 4.5 for
social posts and Sonnet 4.5 for enterprise customers through the Anthropic
API, read from `ANTHROPIC_API_KEY` in the environment `run.sh` runs in.
Bedrock works the same way with `AWS_*` credentials and
`SAAS_BENCH_*_LLM_PROVIDER=bedrock`. No credential lives in the repository.

A local OpenAI-compatible server with a Responses endpoint can stand in for
both roles:

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
hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir ~/models/Qwen3-4B-Thinking-2507
export ANTHROPIC_API_KEY=...
CEOBENCH_SEEDS=42,43,44 CEOBENCH_DAYS=500 ./run.sh
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (`~/models`),
`RUN_DIR` (`./work`), and `REEF_GPU_0..4` (the five devices the stack uses:
four actor GPUs with the critic colocated, one rollout GPU). It mints a token
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

### Smoke run (in progress)

The first end-to-end run on 2026-09-12 used `Qwen3-4B-Thinking-2507` as the
policy, 14 simulated days, seed 42, and a local `Qwen3-4B-Instruct-2507` on
SGLang standing in for both simulator roles. The wiring held: every agent
turn went through the sidecar into Reef and came back as a parsed tool call
with a receipt, and the benchmark's session, engine, and checkpoints ran
inside the task container. The policy did not: it never produced a valid
`next-week` call (13 positional arguments), wrote a stub `next_week.py` that
prints a message, and looped on it, so the simulation stayed at day 0 while
the conversation grew by roughly 6k tokens per 100-turn block toward the
131k window. The run was stopped after 400 turns before the verifier ran.
The next attempt uses a larger policy; the numbers below will be filled in
from it.

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
