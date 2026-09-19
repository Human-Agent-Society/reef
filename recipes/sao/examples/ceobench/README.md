# SAO on CEO-Bench

This example runs [CEO-Bench](https://ceobench.com)
([paper](https://arxiv.org/abs/2606.18543),
[code](https://github.com/zlab-princeton/ceobench-src)) through Reef and trains
the [`sao` recipe](../../../../docs/user-guide/recipes/sao.rst) on it. CEO-Bench
simulates an AI startup for 500 days starting from $1M in cash. The agent has
34 tools and a 19-table database and sells into a simulated market with hidden
preferences, competitor pressure, and delayed consequences. The primary metric
is final cash, with survival days and bankruptcy as secondary metrics. The
benchmark's own bash agent plays the game. `harness/` is that agent, played
from the host with its prompt, tools, and tool executor taken from the pinned
checkout and with every model call served by Reef, so each call is recorded
and attributable. The two simulator roles (social posts and enterprise
customers) stay outside Reef.

```text
harbor/                one CEO-Bench episode as a Harbor task, the world and the verifier
  task.toml              48h agent window and resource limits
  instruction.md         the task for Harbor (the model never sees it, CEO-Bench owns its prompt)
  environment/
    Dockerfile           python:3.13 + uv + the pinned CEO-Bench checkout, patched and rebuilt
    reef.patch           the changes the engine needs (described below)
    engine.py            starts the episode's session and engine and stops it for the verifier
  tests/
    test.sh              runs the verifier inside the task container
    score.py             decrypts the run's world.nmdb and writes reward, final cash, survival days, bankrupt
harness/               the benchmark's bash agent (imports reef_client, not reef)
  __init__.py            lazily exports HarborAgent
  agent.py               the benchmark's agent loop with its conversation, feedback texts, and retries
  tools.py               its six tools, run in the agent's workspace inside the task container
  harbor_agent.py        plays one trial with the agent served by Reef and its weeks credited
  report.py              the reward, from weeks and valuation to scaled scores, posting, and the pacer
serve.yaml             Reef + Ray + Slime/Megatron + SGLang, Qwen3.6-27B through LoRA, critic colocated
docker-compose.yaml    the stack in the reef image with host networking and six GPUs
run.py                 one episode, trained while it is played
run.sh                 brings the stack up and runs run.py through reef-eval
pyproject.toml         makes the harness importable
results/               the untrained baseline and the trained episode (weeks.csv, holds.csv, manifest.json)
```

## The harness

`harness/` is CEO-Bench's bash agent, split into `agent.py` and `tools.py`
the same way the benchmark's own `bash_agent/` package is. What the model is
asked, what it may call, and what its tools return all belong to the
benchmark. The harness only adds where the model calls go, when a week is
credited, and how long the game waits for the trainer.

- **`agent.py`** is the benchmark's `BashAgent` at the pinned commit and
  follows its OpenAI chat-completions path step for step. The conversation
  starts empty and is rebuilt from the system prompt, with the workspace's
  `MEMORY.md` appended, every time the week advances. One tool call runs per
  turn and any further calls in the same response are answered with
  `[Skipped - only one tool per turn ...]`. The benchmark's feedback texts
  for a response without a tool call or with arguments that are not JSON are
  kept, and so are its retry rules for API errors. Internal programming errors
  propagate to Harbor instead of being retried as model failures. The prompt and the tool
  definitions are not copied into this repository. When an episode starts
  the harness reads them from the pinned checkout in the task image, built
  by the benchmark's own classes, so they are the benchmark's byte for byte.
- **`tools.py`** is the benchmark's tool executor with `bash`, `read_file`,
  `write_file`, `edit_file`, `search_files`, and `glob_files`. The tools are
  confined to the agent's workspace and keep the same shell environment, the
  same output assembly (`[stderr]`, `[exit code: N]`, and the
  30,000-character cut), the same timeout rules, and the same file
  semantics. The harness uploads the script into the task container and runs
  each call through Harbor's `exec` as the unprivileged `agent` user. The
  benchmark sandboxes the same shell with `bwrap` where it can, and here the
  container and the user boundary serve as the sandbox.
- **`harbor_agent.py`** plays one Harbor trial. It has the task start the
  episode's engine session through `harbor/environment/engine.py`, reads the
  engine's status, dashboard, and books over its HTTP API, and serves every
  model call through a reef-client proxy on the host's loopback.
  `reef_client.serve` replaces the `Authorization` header with the Reef
  token, stamps `x-reef-scenario`, forwards the request body unchanged, and
  keeps each exchange together with its `x-reef-agent-record-id` receipt.
  The agent's OpenAI client only sees a plain base URL.
- **`report.py`** is the reward. Every captured turn is filed under the
  simulated week it was played in, since the harness knows the engine's day
  at every call. A finished week is credited as described under "Reward
  shaping" and its decision turns are posted to Reef while the episode runs.
  The last weeks close with the final cash the engine reports when the
  episode ends.

When the episode ends the run directory (`world.nmdb`, `config.json`,
`logs/`, and `agent_workspace/`) is downloaded next to the trial's agent
logs. `turns.jsonl` there lists every tool call with its output, and the
receipts go into the agent context in call order with their week, token
count, and decision. Harbor then runs `tests/test.sh` in the same container.
It stops the engine if the harness could not and scores the run. `score.py`
opens the run's `world.nmdb` with the checkout's own `load_session_db` and
writes `reward.json`. Final cash is the balance the benchmark's own
`get_cash` reads from the books, survival days is the last day any daily
table reached, `bankrupt` means final cash below zero, and `reward` is
final cash divided by the starting balance, so 1.0 is break-even.

### The patch to CEO-Bench

`reef.patch` is applied to the pinned checkout at image build time and
touches the engine only.

- `server_entry.py` reads `SAAS_BENCH_<FIELD>` variables for the simulator
  roles' provider and model. Provider `none` runs both roles on the engine's
  template posts, and `SAAS_BENCH_SIMULATOR_TIMEOUT_S` bounds one simulator
  request (300 s in `run.sh`).
- `customer_llm.py` and `simulation.py` gain an OpenAI path for the two
  social-media calls that only had Anthropic and Bedrock ones, and they
  tolerate the token counts SGLang's Responses endpoint leaves out.

Everything else is the benchmark as published, including the default
`config.py` difficulty, the bash agent's prompt and tools, and
`temperature=1.0`. The agent's tools run as the unprivileged `agent` user
the image creates (`CEOBENCH_TOOL_USER`). `CEOBENCH_BASH_TIMEOUT_S` (3600 s
in `run.sh`) widens the benchmark's 1200 s limit on one bash command, which
late-game `next-week` calls need.

### Simulator roles

The two simulator roles, the customers who post on social media and the
enterprise buyers, are LLM calls made inside the engine and outside Reef.
The episodes recorded below run both on `Qwen3-4B-Instruct-2507`, served by
SGLang on a spare GPU of the same node as an OpenAI-compatible endpoint, so
no paid API is involved.

```bash
docker run -d --name ceobench-sim --network host --gpus '"device=7"' -v ~/models:/root/models reef \
  python -m sglang.launch_server --model-path /root/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 30100 --tp 1 \
  --mem-fraction-static 0.28 --context-length 32768
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=openai SAAS_BENCH_SOCIAL_POST_LLM_MODEL=Qwen3-4B-Instruct-2507
export SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=openai SAAS_BENCH_ENTERPRISE_LLM_MODEL=Qwen3-4B-Instruct-2507
export OPENAI_BASE_URL=http://<host>:30100/v1 OPENAI_API_KEY=local
```

The benchmark's own setting is Haiku 4.5 for social posts and Sonnet 4.5
for enterprise customers through the Anthropic API with `ANTHROPIC_API_KEY`,
or through Bedrock with `AWS_*` credentials and
`SAAS_BENCH_*_LLM_PROVIDER=bedrock`. That is what `run.sh` uses when the
variables above are unset and what a result meant to compare with the paper
or the leaderboard needs. Setting both roles to `none` runs the market on
the engine's template posts with the same satisfaction and virality
mechanics but no generated text. No credential lives in the repository.

## Reward shaping

CEO-Bench advances in weeks. The agent works in one conversation until it
calls `next-week` and then rebuilds the conversation from the next
dashboard. The harness knows the engine's day at every model call, so every
turn is filed under its week. Once enough later weeks have opened, each
decision turn of week N is reported with the week's credit as its score.

    run_rate_N = the engine's MRR at the week's start (the dashboard's listed-price
                 estimate when the books cannot be read)
    V_N        = cash_N + run_rate_N x 7/30 x min(weeks left after week N, H)
    credit_N   = sum over j < K of gamma^j x (V_{N+j+1} - V_{N+j}) / $1,000,000
    score_N    = clip(credit_N, +-C) / max(median |credit| so far, F), capped at +-3

- `V_N` is the company's value at the week's start, which is cash plus its
  monthly recurring revenue over the weeks left, at most `H` = 26 of them
  (`CEOBENCH_VALUE_HORIZON_WEEKS`).
- `credit_N` sums the value changes of the next `K` = 4 weeks
  (`CEOBENCH_CREDIT_WEEKS`) discounted by `gamma` = 0.8
  (`CEOBENCH_CREDIT_DISCOUNT`), so a week that pays for acquisition is
  credited with the subscribers that arrive after it. The last weeks close
  with the engine's final cash.
- `score_N` clips the credit at `C` = 0.05 (`CEOBENCH_SCORE_CLIP`) and
  divides by the running median magnitude with a floor of `F` = 0.003
  (`CEOBENCH_SCORE_FLOOR`), so one six-figure R&D purchase cannot set the
  scale for the whole episode.
- Only decision turns are reported, meaning tool calls that change the
  company (`DECISION_CALLS` in `harness/report.py`). Turns that only read
  are recorded but not trained on, and the same goes for turns over
  `CEOBENCH_TRAIN_MAX_TOKENS` (24k, the trainer's window).

The game is paced to the trainer through `CEOBENCH_PACE_BATCH`, which is the
recipe's batch size of 8. Before each model call the harness reports the
weeks whose credit window has closed and then waits until every filled batch
has committed a release, so a week is played by a policy trained on every
week reported so far. A wait longer than `CEOBENCH_PACE_TIMEOUT_S` (20
minutes) is forgiven. `CEOBENCH_REPORTS=0` turns reporting and pacing off,
which gives the untrained baseline. The Harbor reward stays the benchmark's
final cash over the starting balance and is used for evaluation only.

## Run

You need Docker with the NVIDIA runtime, `uv`, the `reef` image (built with
`docker build -f docker/Dockerfile.reef -t reef .` from the repository
root), the policy model, and credentials for the simulator roles.

```bash
cd recipes/sao/examples/ceobench
hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
export ANTHROPIC_API_KEY=...
CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (`~/models`),
`RUN_DIR` (`./work`), and `REEF_GPU_0..5`. Those are the six devices the
stack uses, four actor GPUs at TP4 with the critic colocated and a two-GPU
rollout engine. The policy is `Qwen3.6-27B` trained through Megatron Bridge
LoRA. The base stays frozen in the actor and in the SAO critic, the adapters
and the critic's value head train, and the rollout engine serves the
published adapter. Turns longer than `CEOBENCH_TRAIN_MAX_TOKENS` (24k
tokens, the largest a step fits beside the two resident bases, with the
budget worked out in the header of `serve.yaml`) are served and recorded but
not trained on. `run.sh` mints a token into `$RUN_DIR/token`, brings the
stack up with `docker compose up --wait`, and runs `run.py` in an ephemeral
`uv` environment with `reef-eval[harbor]` and this harness. The episode row
lands in `work/lab` and the trial's run directory under the trial's
`agent/ceobench/`.

This is test-time training. The policy adapts inside the episode it is
scored on, and the number to compare is that episode's final cash against
the same seed played by the untrained model. The value model need not start
cold. `serve.yaml` points `--critic-init` at `$RUN_DIR/critic-init`, and a
copy of an earlier run's latest critic checkpoint placed there
(`checkpoints/megatron-critic/iter_N` together with its
`latest_checkpointed_iteration.txt`) is loaded with its weights and
optimizer at the first start. The critic then trains on its own for two
commits before the actor's first update (`num-critic-only-steps`). An empty
directory leaves it cold. Replicates are independent runs from the base
model with one stack each, so run `docker compose down` between them or use
a fresh `RUN_DIR`. A Reef process trains one scenario for its lifetime, and
a second seed on the same stack would start from the first seed's adapter.
After the episode `run.py` waits until the scenario's training releases
have stopped growing for fifteen quiet minutes, which is longer than one
step, so the adapter on disk is the one the episode ended with:

```bash
curl -sS -H "Authorization: Bearer $(cat work/token)" \
  http://$(hostname -I | awk '{print $1}'):28900/reef/scenarios/ceobench-sao/releases
```

## Results

Seed 42 and 500 days, which the benchmark rounds down to 71 whole weeks or
497 days, with the simulator roles as above and one episode each. Neither
number is comparable with the leaderboard, because both simulator roles are
a local `Qwen3-4B-Instruct-2507` (see "Simulator roles") and each row is one
episode at temperature 1.0. Both episodes were played by the previous form
of this harness, which ran the benchmark's own runner inside the task
container with its agent role redirected to Reef. The harness above plays
the same agent from the host and is to be re-run.

| Policy | Reward | Outcome | Final cash | `reward` |
| --- | --- | --- | ---: | ---: |
| `Qwen3.6-27B`, untrained (`CEOBENCH_REPORTS=0`) | none | bankrupt on day 255 (week 37) | -$66 | -0.00007 |
| `Qwen3.6-27B`, trained in the episode (`serve.yaml`) | the weekly credit above | completed, day 497 | $257,682 | 0.258 |

| Week | Day | Untrained cash | Subscribers | Trained cash | Subscribers | Engine MRR / month |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | $977,614 | 19 | $821,930 | 23 | $207 |
| 5 | 35 | $934,244 | 194 | $789,541 | 198 | $2,692 |
| 7 | 49 | $918,908 | 374 | $752,089 | 314 | $4,896 |
| 10 | 70 | $905,911 | 658 | $725,410 | 586 | $10,089 |
| 12 | 84 | $897,818 | 670 | $698,655 | 647 | $11,626 |
| 15 | 105 | $882,751 | 559 | $649,008 | 639 | $11,858 |
| 17 | 119 | $877,846 | 466 | $295,903 | 327 | $5,422 |
| 20 | 140 | $364,051 | 310 | $289,548 | 183 | $776 |
| 22 | 154 | $352,430 | 227 | $286,912 | 38 | $0 |
| 25 | 175 | $8,343 | 208 | $285,052 | 0 | $0 |
| 30 | 210 | $3,899 | 10 | $282,077 | 0 | $0 |
| 37 | 259 | -$66 (bankrupt on day 255) | 1 | $277,912 | 0 | $0 |
| 50 | 350 | bankrupt | | $270,177 | 0 | $0 |
| 70 | 490 | bankrupt | | $258,277 | 0 | $0 |

### Untrained baseline

The same episode with reporting off. `CEOBENCH_REPORTS=0` records the weeks
without posting them, so nothing trains and the engine serves the base
model. It gets its own scenario and run directory:

```bash
docker compose down
RUN_DIR=$PWD/work-baseline REEF_SCENARIO=ceobench-baseline CEOBENCH_REPORTS=0 \
  CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
```

`results/2026-09-13-baseline-qwen3.6-27b-seed42/` holds the manifest, the
run configuration, and `weeks.csv`. This episode was served by a standalone
SGLang engine at TP4 with the model's 262k window.

| | |
| --- | --- |
| Outcome | bankrupt on day 255 (week 37) with final cash -$66 and `reward` -0.00007 |
| Turns | 695 in 35 minutes, 11.5M input and 228k output tokens |

The agent priced low, at $15/$49/$99 and down to $4/$24/$49 by day 161. It
grew to 671 subscribers by week 11 while losing $4,000 to $7,000 a week,
bought five R&D tiers for $844,000 in all as subscribers churned, and went
bankrupt on day 255.

### Trained episode

`results/2026-09-14-trained-qwen3.6-27b-seed42/` holds the same files plus
`holds.csv` with the pacer's holds, and its `weeks.csv` also records the
engine's MRR, the values, the credit and score, and the decision counts. The
run used `serve.yaml` as shipped, with batch 8, the critic warm-started from
an earlier episode, and the pacer on.

| | |
| --- | --- |
| Outcome | completed on day 497 (week 71) with final cash $257,682 and `reward` 0.258 |
| Turns | 493 in 3h52m, of which 144 were decision turns and 125 of those were reported (19 were over the 24k window) |
| Training | 15 releases, 2 critic-only steps and then 13 actor updates with the first served from week 9. The pacer held the game for 51 minutes in all and 370 s at most |
| Rewards | one positive week (week 6, +0.006) and every other week negative, with weeks 12 to 16 at the clip |

Week 0 bought R&D for $178,070. Weeks 1 to 9 grew the base to 499
subscribers at $6,000 to $10,000 a week, or $80 to $170 per subscriber,
reaching an MRR of $8,367 a month. Each of those weeks was credited between
-0.002 and -0.023, because under a 26-week horizon a $14 subscriber is worth
about $85, which is less than it cost. From week 10 the base plateaued after
a capacity outage left 92 open issues and conversion fell from 22% to 5%.
The policy spent more, with ops at $1,200 a day, ads at $800, and a 25%
promotion, and bought a $333,000 R&D tier in week 15. Promotions above the
plan price then emptied the base by week 22, and the last 49 weeks were
cost-cutting at $595 a week.

### What the comparison shows

The trained episode ends with cash and the untrained one does not, which is
what the benchmark scores. Both episodes still take the same shape of growth
at a loss, one large purchase, and an empty company. The scheme did what it
was built to do. Only decision turns were reported, the engine's MRR was
read, scores stayed inside +-3, and the actor was updated from week 9. Yet
the credit was negative in 70 of 71 weeks. At $80 to $170 per subscriber
against $85 of horizon value no week of growth scores positive under a
26-week horizon, so the policy learns to act less. The horizon
(`CEOBENCH_VALUE_HORIZON_WEEKS`) is the lever to change first. The 24k
training window is a second limit, since weeks 11 and 19 contributed no
decision turns.
