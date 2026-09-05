# Harness evolution quickstart

This example is the smallest full run of the harness evolution mechanism: the agent harness configuration is a composition tree of nodes, a proposal is one gated tree mutation, and a winning mutation publishes as a versioned artifact any client can pull. The proposer is the served model itself: it reads the current skill nodes and its own failing requests and proposes one skill mutation as a strict JSON object. `evaluate` grades real headless episodes on a small fixed task set by exact final answer, so a proposal only publishes when it makes previously failing tasks pass their episodes.

The pinned paper reproduction built on this mechanism lives at `recipes/skillclaw/`.
## Directory layout

```text
evolve-your-harness/
  README.md      this file: what the tutorial shows, how to run it, what it saw
  evolve-your-harness.ipynb
                 the same pass cell by cell, with the service managed as a
                 subprocess from the notebook
  run.sh         copies the recipe config out of the serve file, starts
                 reef serve, waits for /healthz, runs run.py
  run.py         the loop, written out: record -> report -> evolve -> pull
  configs/
    serve.yaml   the stack (reef service) plus the harness_evolve recipe
                 sections used by the self-contained ./run.sh pi demo
    serve-native.yaml
                 the ./run.sh native variant: the loop's own tools, hook
                 and graph are the seed, so the proposer can change them
    deployment.yaml
                 the minimal long-running pi deployment used by the root
                 README; run it directly with reef serve -c
  harness/       the method package the serve files' dotted refs name
    evolution.py the method: propose (self proposer over failures, skill
                 nodes only) and evaluate (exact last-line answer grading)
    native_evolution.py
                 the native variant: propose over skills, tools, hooks,
                 the loop graph and its agents, evaluate over the native
                 trajectory; the grader is shared
    materialize_recipe.py
                 copies a serve file's recipe sections where the recipe
                 registry reads them; run.sh calls it
    probe_proposals.py
                 samples the native proposer over one failing task and
                 counts what admission lets through, by kind
    replay.py    writes work/replay.html from a run's files: the release
                 chain, the loop graph replayed from a session, the tool
                 call ledger and the process timeline
  pyproject.toml makes harness/ an installable package
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
./run.sh self     # the same, and the model proposes the change itself through its self tools
```

Every run ends by writing `work/replay.html`: the release chain with each step's verdict and tree diff, the loop graph replayed from a session's events, the session's tool call ledger and the process timeline. Open it in a browser; `python3 run.py replay` rebuilds it from `work/` at any time.

serve.yaml carries the endpoint (`upstream_url: http://127.0.0.1:8000`, no /v1 suffix) and the model (`qwen3-8b`) as literals; edit them there to point at your own. The model name appears twice, as `model.path` (the name the proposer and the evolve episodes call) and as `upstream_model` (the name served traffic is forwarded under), and run.py's `MODEL` must match; a name the endpoint does not serve fails the proposer's call, and the step records `skipped: no proposal`. The one value serve.yaml does not hold is the provider key: `export REEF_UPSTREAM_API_KEY=...` if your endpoint needs one.

## Keep a deployment running

Set `model.path` in [deployment.yaml](configs/deployment.yaml) to your provider's model
name. From the repository root, after the source installation and with `pi` on PATH:

```bash
export REEF_UPSTREAM_URL="https://api.openai.com"  # no /v1 suffix
export REEF_UPSTREAM_API_KEY="your-openai-api-key"
reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml
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

`evolve-your-harness.ipynb` walks the same pass cell by cell and manages the service as a subprocess, so one kernel holds the whole loop. Set the endpoint, model, and key in its first code cell; the notebook patches both serve.yaml bindings (the `reef` section's upstream values and the recipe's `model.path`) into `work/serve-notebook.yaml` and materializes the recipe config from the patched text. `run.py` stays the reference implementation of the loop; the notebook mirrors it.

Pick a model that fails at least one task and still writes the strict JSON mutation `propose` expects. A model that passes all three tasks reports no failures, so no evolve step ever runs (the DeepSeek result below); one that cannot author the JSON commits its step as `skipped: no proposal`, visible in `work/agent-record/*.commits.jsonl`. The committed outputs are a full local run with no GPU, the `pi, notebook` row of the results below.

## What one run does

1. `run.sh` copies serve.yaml's recipe sections into `work/recipes/harness_evolve.yaml` and starts Reef. The recipe boots with the seed composition: one starter `answer-style` skill node. The endpoint lives only on serve.yaml's `reef` section (`upstream_url`, `upstream_api_key`, `upstream_model`) and the recipe names its model as `model.path`; Reef hands the endpoint and that name to `propose` as a model binding and renders them into each evaluation episode, so neither the method nor the published tree ever names them.
2. `run.py` sends each of the three exact-answer coding tasks once through reef inference; reef serves the reply and records the exchange. The reply is graded the same way the evolve gate grades episodes, and the score is reported against the receipt.
3. Only failures batch (`max_score: 0.0`, the SkillClaw window), and `batch_size: 1` makes every failing report one evolve step: `propose` sends the failing requests and current skills back to the same model through `models.served.chat`, which answers with one skill mutation; the candidate and current compositions each run one episode per task; the mutation publishes only on a gate win.
4. `run.py` pulls `GET /reef/harness` and prints the gate metrics and the evolved `SKILL.md` files. Point any pi at the pulled tree, with its model set to Reef, and it carries the learned skill.

## Native variant

`./run.sh native` runs the same loop on reef's own coding agent (`evolution.adapter: native`), whose tools, loop hooks, loop graph and helper agents are composition nodes. Three things differ from the pi run:

- The seed is the loop's shipped `read_file`, `write_file` and `run_bash` tools, its `loop_guard` hook and its `main` graph (`think` to `act` to `think`, `done` on a text reply), named by reference (`reef.harness.runners.native.seed:SEED_NODES`) beside the starter skill, so the proposer sees them as nodes it may retune or replace.
- `harness/native_evolution.py`'s `propose` accepts one mutation on a skill, a `native_tool` (a schema plus a module defining `run(args, workdir) -> str`), a `native_hook` (a module defining `listen(payload, next) -> decision` at `pre_step`, `pre_execute`, `request_error` or `post_execute`) a `native_agent` (a helper with its own prompt, tools and budget that a graph's `subagent` stage calls) or the `native_graph` (the loop's stages and the edges between them, shown to the model as one worked example: the seed loop plus a `check` stage); a node's entry id is its name, so a proposal that reuses a seed tool's name updates that tool. An agent alone changes nothing (only a `subagent` stage runs one), so an agent proposal carries a second mutation that routes the main graph's model text through a new `ask-<name>` subagent stage and a new `answer-<name>` model stage, whose text goes where the model's text went: the agent's reply comes back as a user message, and only a model stage writes the assistant message the grader reads. `evaluate` grades the last `assistant/message` event of the native-jsonl trajectory with the same grader.
- No agent binary is installed: `run.sh native` writes a `reef-native` launcher for this checkout under `work/bin` and puts it on PATH, which is what an installed reef's console script provides.
- The pass runs through the serve form. `run.sh native` pulls the seed tree under `work/tree` (`python3 run.py pull`), starts `reef-native serve --tree work/tree --follow head --poll-interval 5` on it, and `python3 run.py native` sends each task to that process as one turn with `reef-native turn`, grades the reply and reports the score through the wrapper's `report` command, which claims the turn's receipts from `work/captures`. When the gate publishes, the running process mounts the new release between two steps or at once while idle, one `harness/mount` line in `work/tree/native/sessions/serve.jsonl` (or in the open turn's session), with no reinstall and no restart; `run.py` then sends the first task again and prints the stages of the new session, which are the mounted graph's. `work/serve.log` is the process's stderr.

The recorded pass is identical, so the two variants are comparable on the same three tasks; `run.py` prints every evolved skill, tool, hook and graph file from the pulled tree.

## Self tools variant

`./run.sh self` is the native variant with `reef-native serve --self-tools`: the served model gets `harness_inspect`, `harness_try` and `harness_propose`, the host plane tools of the resident process. `python3 run.py self` sends one turn that tells the model it runs on a harness it can read and change and asks it to inspect the tree, propose one change through `harness_propose` that makes its answers end with the integer alone on the last line, and then answer the sieve task. The route admits the proposal into the scenario's inbox; the turn's answer is graded and reported, and the failing report opens a step that claims the proposal before it asks the method. A win publishes, the process mounts the release the model proposed (the commit's `proposal` names the proposal and the session that made it), and the first task runs again on the mounted tree. The prompt names the goal and the shape of a rules entry, not the rule's text: what the model writes is what the gate judges.

## Results

Last updated: 2026-09-05. Every row was measured on the code of the pull request in its Code column, with the tutorial files as they stood there.

The loop under measurement is one fresh scenario through `./run.sh` (pi adapter) or `./run.sh native` (native adapter): the three tasks `[sieve]`, `[fib]` and `[csv]` go through Reef once, each reply is graded 1.0 for the exact answer alone on the last line and 0.0 otherwise, only a 0.0 report batches (`max_score: 0.0`, `batch_size: 1`), and each batched report runs one evolve step whose gate runs the current and the candidate tree once per task. Scores are listed in task order; W / L / T counts the three task pairings of one gate; the model under test is also the proposer.

### Environment

| Setup | Host | Model server | Model | Agent |
|---|---|---|---|---|
| pi, B200 | one B200 node | vllm 0.27.0 | Qwen3-8B as `qwen3-8b`, tensor parallel 1; DeepSeek-V4-Flash-0731 as `deepseek-v4-flash`, tensor parallel 4, fp8 kv cache | pi 0.84.2 |
| pi, notebook | Mac mini | ollama | `qwen2.5:7b` | pi 0.84.2 |
| native | Mac mini | ollama | `qwen2.5:7b` | reef-native from the checkout |
| native + graph | Mac mini | ollama | `qwen2.5:7b` | reef-native from the checkout, after #250 |
| native, serve | Mac mini | ollama | `qwen2.5:7b` | `reef-native serve` from the checkout, #278 with #282 and #284 |
| native, self | Mac mini | ollama | `qwen2.5:7b` | `reef-native serve --self-tools` from the checkout, after #285 |

### Runs

| Setup | Model | Run | Date | Code | Recorded pass | Proposal | Gate: current | Gate: candidate | W / L / T | Verdict | Launch to verdict (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pi, B200 | deepseek-v4-flash | 1 | 2026-08-17 | #58 | 1.0 / 1.0 / 1.0 | none, nothing batched | - | - | - | no step | 3 [1] |
| pi, B200 | deepseek-v4-flash | 2 | 2026-08-17 | #58 | 1.0 / 1.0 / 1.0 | none, nothing batched | - | - | - | no step | 4 [1] |
| pi, B200 | qwen3-8b | 1 | 2026-08-17 | #58 | 1.0 / 0.0 / 1.0 | `create direct-answer` (skill) | 1.0 / 0.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1 / 0 / 2 | published `0fe30330` | 63 [2] |
| pi, notebook | qwen2.5:7b | 1 | 2026-08-31 | #70 | 1.0 / 0.0 / 1.0 | `update answer-style` (skill) | 0.0 / 0.0 / 0.0 | 1.0 / 0.0 / 0.0 | 1 / 0 / 2 | published `f98266d2` | not recorded [3] |
| native | qwen2.5:7b | 1 | 2026-09-04 | #241 | 1.0 / 0.0 / 1.0 | `create fib-solver` (native_tool) | 0.0 / 0.0 / 1.0 | 0.0 / 1.0 / 0.0 | 1 / 1 / 1 | rejected | 93 [4] |
| native | qwen2.5:7b | 2 | 2026-09-04 | #241 | 1.0 / 0.0 / 1.0 | `create fib-skill` (skill) | 0.0 / 0.0 / 1.0 | 0.0 / 0.0 / 0.0 | 0 / 1 / 2 | rejected | 217 |
| native + graph | qwen2.5:7b | 1 | 2026-09-05 | #258 | 1.0 / 0.0 / 1.0 | `update main` (native_graph) [5] | 1.0 / 0.0 / 0.0 | 1.0 / 0.0 / 0.0 | 0 / 0 / 3 | rejected | 645 |
| native + graph | qwen2.5:7b | 2 | 2026-09-05 | #258 | 0.0 / 0.0 / 1.0 | `update main` (native_graph): a `check` stage | 0.0 / 0.0 / 0.0 | 1.0 / 0.0 / 1.0 | 2 / 0 / 1 | published | 178 [6] |
| native, serve | qwen2.5:7b | 1 | 2026-09-06 | #282 + #284 | 0.0 / 0.0 / 0.0 | `update main` (native_graph): a `check` stage | 0.0 / 0.0 / 0.0 | 0.0 / 1.0 / 0.0 | 1 / 0 / 2 | published, mounted [11] | 320 [12] |
| native, self | qwen2.5:7b | 1 | 2026-09-06 | #285 | 0.0 [13] | `create answer-format`, `create answer-format-rationale` (rules), by the model through `harness_propose` | 0.0 / 0.0 / 0.0 | 0.0 / 1.0 / 1.0 | 2 / 0 / 1 | published, mounted [14] | 125 |
| native, self | qwen2.5:7b | 2 | 2026-09-06 | #287 | 0.0 [13] | `remove answer-style`, `create answer-style` (skill to rules), by the model after the route refused a kind change | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0 / 0 / 3 | rejected | 113 |

One run is one sample and no run was repeated, so the rows carry no spread. Gate episodes are stochastic: the seed tree scored `[sieve]` 1.0 on every recorded pass and 0.0 in four of the five gate episodes on the Mac, because with tools in hand the model runs the sieve and then answers in prose, so the integer is not alone on the last line.

### Proposal admission by prompt shape

Samples of the native proposer over the `[fib]` failure, measured 2026-09-05 at #258 with `harness/probe_proposals.py` against ollama at its default sampling settings; one call per sample under the proposer's 120 s and 2048 completion token budget. A sample is parsed when the reply carries a mutation and admitted when that mutation passes its node kind's admission rules, the check the evolve step runs before any episode.

| Prompt shape | Model | Samples | Parsed | Graph | Graph admitted | Tool | Skill | Unparsed |
|---|---|---|---|---|---|---|---|---|
| placeholders (`<stage>`, `<outcome>`) | qwen2.5:7b | 30 | 25 | 11 | 0 [7] | 14 | 0 | 5 |
| worked example (the seed loop plus a `check` stage) | qwen2.5:7b | 30 | 29 | 29 | 23 [8] | 0 | 0 | 1 |
| worked example | qwen3:8b | 15 | 7 | 5 | 5 | 2 | 0 | 8 [9] |
| worked example | qwen3.5:9b | 15 | 5 | 4 | 3 [10] | 0 | 1 | 10 [9] |

The worked example moved graph admission from 0 of 11 to 23 of 29 and removed every tool and skill proposal, which is the trade the tutorial makes; the two newer models write admissible graphs when they answer at all and lose more than half their samples to the budget, which is why the tutorial stays on qwen2.5:7b.

### Reading

- Both pi publishes came from one failing task and one skill proposal, on Qwen3-8B behind vllm and on qwen2.5:7b behind ollama alike.
- On the native loop without the graph node the gate held both proposals back: the tool won `[fib]` and lost `[csv]`, the skill won nothing.
- With the graph node the proposer rewrote the loop instead: a `check` stage that verifies the last line is a plain integer and, on `fail`, returns to `think` with the message `Reply with the final answer as a plain integer alone on the last line.` Run 2 published it on wins at `[sieve]` and `[csv]`; `[fib]` stayed 0.0 on both sides because fib(90) exact is beyond this model. The rewrite is the prompt's worked example reproduced stage for stage, so the proposer chose it rather than invented it; whether this model writes a graph the prompt did not show is the open measurement.

### Published artifacts

The B200 run's skill, as pulled from the published tree:

```text
--- pi-agent/skills/direct-answer/SKILL.md ---
# direct-answer

Enforces answers to be exact plain integers without additional text. For numerical problems requiring precise values (e.g., Fibonacci numbers, large computations), this skill ensures the response contains only the final integer result on the last line, adhering strictly to the format specified in the query.
```

The graph run's `main`, the stage it added and the edges that route through it:

```text
"check": {"kind": "verify", "check": "last_line_integer", "message": "Reply with the final answer as a plain integer alone on the last line."}
{"from": "think", "when": "text", "to": "check"}, {"from": "check", "when": "pass", "to": "done"}, {"from": "check", "when": "fail", "to": "think"}
```

The notebook keeps its own run's outputs cell by cell, the evolved `answer-style` skill included.

### Reproduce

```bash
cd tutorials/evolve-your-harness
# pi rows: configs/serve.yaml names the endpoint and the model. The B200 rows served
#   vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
./run.sh
# native rows: configs/serve-native.yaml, ollama on 127.0.0.1:11434
./run.sh native
# the admission table
python3 harness/probe_proposals.py --samples 30 --model qwen2.5:7b
```

The model name is set in three places, `model.path` and `upstream_model` in the serve file and `MODEL` in `run.py`; every row above had all three on the model in its Model column.

### Notes

[1] The recorded pass alone; no report batched and no step ran.
[2] End to end, including the pull of the published tree.
[3] The notebook's committed outputs hold the run cell by cell but no timing.
[4] `current_residue: 16`: on macOS a `run_bash` call that runs Apple's python3 leaves its bytecode cache under the episode's `HOME`; a Linux run reports 0, and residue is counted, never forbidden, unless `evolution.forbid_residue` is set.
[5] This run was measured before a rejected proposal's content was kept, so the rewrite is known as a `main` update and nothing more. Since then the commit record and the `rejected` history that `propose` receives carry each mutation's options, and `evolution.step_record_dir` writes every step's proposer calls, parsed proposal and gate episode logs to disk; a rerun of this table can show the rejected rewrite itself.
[6] The run's second failing report opened a second step, a `fib-optimize` tool, which was still on the gate when `run.py` pulled the published tree and `run.sh` stopped the service.
[7] 9 graphs left out the verify stage's `check` and 2 omitted the `fail` edge.
[8] 6 graphs carried a duplicated edge or an outcome the stage does not have.
[9] qwen3:8b: 4 samples over the 120 s limit and 4 replies with no mutation; qwen3.5:9b: 9 over the limit and 1 with no mutation.
[10] 1 graph had a stage unreachable from `think`.
[11] The serve form. The resident process saw the new head on its next poll; its first manifest read waited behind step 2 and timed out (`harness/mount-failed`), and the mount landed 64 s after the publish, the moment step 3 (the last batched report, no proposal) released the catalog. The same process then ran the first task again on the mounted graph: 11 stages through `check`, answer 4921, score 0.0.
[12] From the first turn's start to the step's commit: three turns (80 s), the proposer (25 s) and the paired episodes. Steps 2 and 3 followed with no proposal, 23 s and 42 s in the proposer.
[13] The self form sends one turn, not the three tasks: the model inspected its tree, proposed the two rules through `harness_propose` (admitted at the route) and answered the sieve task `\boxed{95920}`, score 0.0, reported.
[14] The step claimed the model's proposal (the commit's `proposal` names its id and session), the process mounted the release within one second of the commit, and the sieve task ran again on the 9 entry tree with the model's rules in its prompt: stages think, act, think, answer `\boxed{9657}`, score 0.0. The gate's win and the served turn's miss are both on the record. Launch to verdict is the first turn's start to the commit.
[15] `./run.sh self` as shipped. The model's first proposal updated the `answer-style` skill into a `rules` entry; the route refused it under the kind rule and said why (remove the entry and create it under the new kind); the model read the reason and proposed the remove plus create pair, admitted. Its own answer put the right number in a sentence, score 0.0. The gate tied all three tasks (both sides 0.0) and rejected; the settled proposal file carries the verdict, and the replay page shows the rejected step beside the seed.

### Known limitations

- `run.py` stops at the first publish, so a run with two failing reports records one verdict.
- A rejected proposal's content is not persisted, so a rejected row can name the mutation's id and op only.
- vllm serves Qwen3-8B for pi only with `--enable-auto-tool-choice --tool-call-parser hermes`; without them it answers pi's `tool_choice: "auto"` requests with a 400, every episode fails, and every gate ties.
- The B200 rows and the Mac rows differ in host and model server, so wall clocks compare within a setup, not across.
