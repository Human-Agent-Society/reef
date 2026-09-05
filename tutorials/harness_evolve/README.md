# Harness evolution quickstart

This example is the smallest full run of the harness evolution mechanism: the agent harness configuration is a composition tree of nodes, a proposal is one gated tree mutation, and a winning mutation publishes as a versioned artifact any client can pull. The proposer is the served model itself: it reads the current skill nodes and its own failing requests and proposes one skill mutation as a strict JSON object. `evaluate` grades real headless episodes on a small fixed task set by exact final answer, so a proposal only publishes when it makes previously failing tasks pass their episodes.

The pinned paper reproduction built on this mechanism lives at `recipes/skillclaw/`.
## Directory layout

```text
harness_evolve/
  deployment.yaml the minimal long-running pi deployment used by the root README;
                  run it directly with reef serve -c
  serve.yaml     the stack (reef service) plus the harness_evolve recipe
                 sections used by the self-contained ./run.sh pi demo
  serve-native.yaml
                 the ./run.sh native variant: the loop's own tools, hook
                 and graph are the seed, so the proposer can change them
  harness/       the method package serve.yaml's dotted refs name
    evolution.py the method: propose (self proposer over failures, skill
                 nodes only) and evaluate (exact last-line answer grading)
    native_evolution.py
                 the native variant: propose over skills, tools, hooks,
                 the loop graph and its agents, evaluate over the native
                 trajectory; the grader is shared
  run.py         the loop, written out: record -> report -> evolve -> pull
  run.sh         copies the recipe config out of serve.yaml, starts
                 reef serve, waits for /healthz, runs run.py
  1_evolve_your_harness.ipynb
                 the same pass cell by cell, with the service managed as a
                 subprocess from the notebook
  pyproject.toml makes harness/ an installable package
  README.md      this file
```

## Setup (once)

The client side needs only the stdlib SDK; the service side runs from the repo checkout via `run.sh`:

```bash
pip install reef-client
```

You also need an OpenAI-compatible endpoint for the model under test, and for the pi variant the pi coding agent binary on your PATH (serve.yaml's `evolution.binary`). The native variant needs no agent binary: the loop is reef's own.

## Quick start

```bash
./run.sh          # on pi (serve.yaml)
./run.sh native   # on reef's native harness (serve-native.yaml)
```

serve.yaml carries the endpoint (`upstream_url: http://127.0.0.1:8000`, no /v1 suffix) and the model (`qwen3-8b`) as literals; edit them there to point at your own. The model name appears twice, as `model.path` (the name the proposer and the evolve episodes call) and as `upstream_model` (the name served traffic is forwarded under), and run.py's `MODEL` must match; a name the endpoint does not serve fails the proposer's call, and the step records `skipped: no proposal`. The one value serve.yaml does not hold is the provider key: `export REEF_UPSTREAM_API_KEY=...` if your endpoint needs one.

## Keep a deployment running

Set `model.path` in [deployment.yaml](deployment.yaml) to your provider's model
name. From the repository root, after the source installation and with `pi` on PATH:

```bash
export REEF_UPSTREAM_URL="https://api.openai.com"  # no /v1 suffix
export REEF_UPSTREAM_API_KEY="your-openai-api-key"
reef serve -c tutorials/harness_evolve/deployment.yaml
```

Use your provider's API key, or omit it for a local endpoint without authentication.
`deployment.yaml` keeps Reef running until interrupted. It contains both the
deployment and its named recipe preset; no recipe generation or extra Python
launcher is needed. A YAML anchor shares the model name between the recipe
and the upstream runtime, so inference, proposal, and evaluation use the same model.
Defaults: port `8901`, Reef access token `reef-local`, state under
`work/deployment/`. Use `--reef.port` and `--reef.token` to override the port
and access token. The `run.sh` demo keeps its separate configuration and state.

Install a harness from this service and submit feedback as shown in the
[root README](../../README.md#harness-evolving-deployment). Adapt `deployment.yaml`'s
tasks and `harness/evolution.py`'s proposer and grader for your workload before
starting the deployment.

## Notebook

`1_evolve_your_harness.ipynb` walks the same pass cell by cell and manages the service as a subprocess, so one kernel holds the whole loop. Set the endpoint, model, and key in its first code cell; the notebook patches both serve.yaml bindings (the `reef` section's upstream values and the recipe's `model.path`) into `work/serve-notebook.yaml` and materializes the recipe config from the patched text. `run.py` stays the reference implementation of the loop; the notebook mirrors it.

Pick a model that fails at least one task and still writes the strict JSON mutation `propose` expects. A model that passes all three tasks reports no failures, so no evolve step ever runs (the DeepSeek result below); one that cannot author the JSON commits its step as `skipped: no proposal`, visible in `work/agent-record/*.commits.jsonl`. The committed outputs are a full local pass with no GPU: ollama `qwen2.5:7b` on a Mac mini scored 1.0 / 0.0 / 1.0 on the recorded pass, the self proposer updated `answer-style`, and the gate published on 1 win, 0 losses, 2 ties.

## What one run does

1. `run.sh` copies serve.yaml's recipe sections into `work/recipes/harness_evolve.yaml` and starts Reef. The recipe boots with the seed composition: one starter `answer-style` skill node. The endpoint lives only on serve.yaml's `reef` section (`upstream_url`, `upstream_api_key`, `upstream_model`) and the recipe names its model as `model.path`; Reef hands the endpoint and that name to `propose` as a model binding and renders them into each evaluation episode, so neither the method nor the published tree ever names them.
2. `run.py` sends each of the three exact-answer coding tasks once through reef inference; reef serves the reply and records the exchange. The reply is graded the same way the evolve gate grades episodes, and the score is reported against the receipt.
3. Only failures batch (`max_score: 0.0`, the SkillClaw window), and `batch_size: 1` makes every failing report one evolve step: `propose` sends the failing requests and current skills back to the same model through `models.served.chat`, which answers with one skill mutation; the candidate and current compositions each run one episode per task; the mutation publishes only on a gate win.
4. `run.py` pulls `GET /reef/harness` and prints the gate metrics and the evolved `SKILL.md` files. Point any pi at the pulled tree, with its model set to Reef, and it carries the learned skill.

## Native variant

`./run.sh native` runs the same loop on reef's own coding agent (`evolution.adapter: native`), whose tools, loop hooks, loop graph and helper agents are composition nodes. Three things differ from the pi run:

- The seed is the loop's shipped `read_file`, `write_file` and `run_bash` tools, its `loop_guard` hook and its `main` graph (`think` to `act` to `think`, `done` on a text reply), named by reference (`reef.harness.native.seed:SEED_NODES`) beside the starter skill, so the proposer sees them as nodes it may retune or replace.
- `harness/native_evolution.py`'s `propose` accepts one mutation on a skill, a `native_tool` (a schema plus a module defining `run(args, workdir) -> str`), a `native_hook` (a module defining `listen(payload, next) -> decision` at `pre_step`, `pre_execute`, `request_error` or `post_execute`) a `native_agent` (a helper with its own prompt, tools and budget that a graph's `subagent` stage calls) or the `native_graph` (the loop's stages and the edges between them, shown to the model as one worked example: the seed loop plus a `check` stage); a node's entry id is its name, so a proposal that reuses a seed tool's name updates that tool. `evaluate` grades the last `assistant/message` event of the native-jsonl trajectory with the same grader.
- No agent binary is installed: `run.sh native` writes a `reef-native` launcher for this checkout under `work/bin` and puts it on PATH, which is what an installed reef's console script provides.

The recorded pass is identical, so the two variants are comparable on the same three tasks; `run.py` prints every evolved skill, tool, hook and graph file from the pulled tree.

## Results

Measured 2026-08-17 on one B200 node, pi 0.84.2 as `evolution.binary`, vllm 0.27.0 at `http://127.0.0.1:8000`, one fresh scenario per run.

DeepSeek-V4-Flash-0731 as `deepseek-v4-flash` (tensor parallel 4, fp8 kv cache) answered every task exactly on the recorded pass (`9592`, `2880067194370816120`, `30` alone on the last line) in two runs of 3 s and 4 s each, so no report entered the `max_score: 0.0` window, nothing batched, and both runs printed `every task passed: nothing batched, no evolve step runs`. No proposal was made and no gate ran: the evolve path only opens when the recorded pass fails at least one task.

Qwen3-8B as `qwen3-8b` (tensor parallel 1) opens that window. Serve it with tool calling enabled (`--enable-auto-tool-choice --tool-call-parser hermes`); without those flags vllm rejects pi's `tool_choice: "auto"` requests with a 400, every episode scores as failed, and every gate dead ties. One run, 63 s wall clock end to end:

| pass | [sieve] | [fib] | [csv] |
|------|---------|-------|-------|
| recorded | 1.0 | 0.0 | 1.0 |
| gate: current | 1.0 | 0.0 | 1.0 |
| gate: candidate | 1.0 | 1.0 | 1.0 |

The one failing report (`[fib]`) batched and triggered one evolve step. The self proposer (qwen3-8b) proposed `create direct-answer`, a new skill beside the starter; the gate scored candidate 3.0 against current 2.0 (1 win, 0 losses, 2 ties, 0 episode failures) and published artifact `0fe30330c6ddacd400fbb905113711c3940f7071`. The pulled tree carries the proposed skill verbatim:

```text
--- pi-agent/skills/direct-answer/SKILL.md ---
# direct-answer

Enforces answers to be exact plain integers without additional text. For numerical problems requiring precise values (e.g., Fibonacci numbers, large computations), this skill ensures the response contains only the final integer result on the last line, adhering strictly to the format specified in the query.
```

### Native variant

Measured 2026-09-04 on a Mac mini, ollama `qwen2.5:7b` at `http://127.0.0.1:11434` as the model under test, `./run.sh native`, one fresh scenario per run, two runs. The recorded pass scored 1.0 / 0.0 / 1.0 both times, so the `[fib]` report batched and one evolve step ran; the whole run from launch to the gate's verdict took 93 s and 217 s. Both steps went end to end on the native loop, proposal to verdict, and both verdicts were rejections:

| run | proposal | gate: current | gate: candidate | wins / losses / ties |
|-----|----------|---------------|-----------------|----------------------|
| 1 | `create fib-solver` (native_tool) | 0.0 / 0.0 / 1.0 | 0.0 / 1.0 / 0.0 | 1 / 1 / 1 |
| 2 | `create fib-skill` (skill) | 0.0 / 0.0 / 1.0 | 0.0 / 0.0 / 0.0 | 0 / 1 / 2 |

The self proposer used both halves of its vocabulary: a tool with a memoized `fib` behind `run(args, workdir)` in run 1, a skill in run 2. The gate held both back, because a win on `[fib]` came with a loss on `[csv]` in run 1 and no win at all in run 2. Two things in the numbers are the loop's own. The seed tree scores `[sieve]` 0.0 in a gate episode while the recorded pass scored it 1.0: with tools in hand the model runs the sieve and then answers in prose, so the bare integer is not on the last line, which is what a skill mutation is there to fix. And run 1 reported `current_residue: 16`: Apple's `python3` writes its bytecode cache under `$HOME/Library/Caches/com.apple.python/`, and an episode's `HOME` is its root, so a `run_bash` call that runs that interpreter leaves those files; a Linux run reports 0, and residue is counted, never forbidden, unless `evolution.forbid_residue` is set. The first attempt at run 1 skipped its step with `no proposal` because the model wrote the kind into `id` and the name into `name`; the proposer now reads the swap off `config.name`, and the runs above are with that in place.

### Native variant with the graph node

Measured 2026-09-05 on the same Mac mini and ollama `qwen2.5:7b`, after #250 made the loop a `native_graph` node the proposer can rewrite.

The self proposer's shape for the graph mattered more than anything in the loop. With the graph described by placeholders (`"<stage>"`, `"<outcome>"`), 30 proposals over the `[fib]` failure gave 25 parsable mutations: 14 tools, all admitted, and 11 graphs, none admitted, because 9 left out the verify stage's `check` and 2 omitted the `fail` edge. Shown one worked graph instead (the seed loop plus a `check` stage with its two edges), the same 30 samples gave 29 graph proposals, 23 admitted, 6 refused for a duplicated edge or an outcome the stage does not have, and no tool or skill proposals at all: the example steers the model to the graph every time, which is the trade the tutorial makes. The refused proposals cost nothing; admission skips the step before any episode runs, and the commit names the rule.

Two newer models of the same size were probed the same way, 15 samples each with the worked example, because the served model is also the proposer and a model that cannot write the mutation format cannot evolve its own loop. The proposer gives one call 120 s and 2048 completion tokens. `qwen3:8b` returned 7 parsable proposals, 5 graphs and 2 tools, all 7 admitted; of the other 8 samples, 4 hit the 120 s limit and 4 came back inside it with no mutation in the reply. `qwen3.5:9b` returned 5, 4 graphs (3 admitted, 1 refused because its new stage was unreachable from `think`) and 1 skill; 9 samples hit the limit and 1 came back empty. Their admitted graphs were as well formed as `qwen2.5:7b`'s, but at half the parse rate the tutorial stays on `qwen2.5:7b`; a model that reasons before it answers needs that turned off or a larger budget before it fits this proposer.

Two live runs, `./run.sh native`, one fresh scenario each:

| run | recorded pass | proposal | gate: current | gate: candidate | wins / losses / ties | verdict |
|-----|---------------|----------|---------------|-----------------|----------------------|---------|
| 1 | 1.0 / 0.0 / 1.0 | `update main` (native_graph) | 1.0 / 0.0 / 0.0 | 1.0 / 0.0 / 0.0 | 0 / 0 / 3 | rejected; 645 s from launch to verdict |
| 2 | 0.0 / 0.0 / 1.0 | `update main` (native_graph): a `check` stage, `last_line_integer`, `fail` back to `think` | 0.0 / 0.0 / 0.0 | 1.0 / 0.0 / 1.0 | 2 / 0 / 1 | published; 178 s from launch to verdict |

Run 2 is the result the graph node was added for. The proposer's rewrite of `main` routes every text reply through a `check` stage that verifies the last line is a plain integer and, on `fail`, sends the model back to `think` with the message `Reply with the final answer as a plain integer alone on the last line.`; the seed graph goes straight from `think` to `done`. On the gate the seed loop scored 0 on all three tasks and the rewritten loop passed `[sieve]` and `[csv]`, 2 wins and no loss, so the graph published, and the loop the agent runs after run 2 is one its own proposal chose. Chose, not invented: the rewrite is the prompt's worked example, reproduced stage for stage and message for message. The proposer picked the graph over a tool or a skill for a format failure, and the gate confirmed the pick; whether this model can write a graph the prompt did not show is the next measurement, and the placeholder shaped prompt above says it cannot yet. `[fib]` stayed 0 on both sides; the verify stage fixes the format, not the arithmetic, and fib(90) exact is beyond this model. Run 1 is the other half of the story: a rewrite of the same node, a tie on every task, no publish, because the seed loop answered `[sieve]` in the right format that time and nothing else moved. Run 2's second failing report opened a second step, a `fib-optimize` tool, which was still on the gate when `run.py` pulled the published tree and `run.sh` stopped the service; the tutorial stops at the first publish.

A rejected proposal's content is not kept: the commit and the `rejected` history that `propose` receives carry the mutation's id and op only, so the run 1 rewrite is known to be a `main` update and nothing more. That is a gap for the proposer as much as for this table, and it is a follow up of #248.
