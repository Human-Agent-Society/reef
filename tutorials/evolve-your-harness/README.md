# Harness evolution quickstart

This example is the smallest full run of the harness evolution mechanism: the agent harness configuration is a tree of text files, each evolve step's method changes that tree, and a change that wins the evaluation publishes as a release any client can pull. The method is the tutorial's own `RulesMethod` (`harness/plugins.py`), a short example of the method protocol: one call to the served model reads the step's failing traces and rewrites `rules.md`. The scorer grades real headless rollouts on a small fixed task set by exact final answer, so a change only publishes when it makes previously failing tasks pass their rollouts.

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
    serve.yaml   the stack (reef service) plus the harness evolution recipe
                 sections used by the self-contained ./run.sh pi demo
    serve-native.yaml
                 the ./run.sh native variant: the loop's own tools, hook
                 and graph are in the seed, so the method can change them
    deployment.yaml
                 a long-running pi deployment; run it directly with
                 reef serve -c
  seeds/
    pi/          the pi variant's first tree: a starter rules.md
    native/      the native variant's first tree: the loop's tools, its
                 loop_guard hook and main graph, and the starter skill
  harness/       the method package the serve files' dotted refs name
    plugins.py   the classes the serve files name: RulesMethod (pi),
                 NativeMethod (native) and the scorers of both variants
    rules_prompt.md
                 the prompt RulesMethod sends
    evolution.py the shared grader (exact last-line answer grading) and
                 its answer table
    native_evolution.py
                 the native variant's change: the prompt over skills,
                 tools, hooks, the loop graph and its agents, the reply
                 read into one file, and the grader over the native
                 trajectory
    native_prompt.md
                 the prompt NativeMethod sends
    materialize_recipe.py
                 copies a serve file's recipe sections where the recipe
                 registry reads them; run.sh and the notebook call it
    replay.py    writes work/replay.html from a run's files: the release
                 chain, the loop graph replayed from a session, the session
                 event log and the process timeline
  pyproject.toml makes harness/ an installable package
```

## Setup (once)

The client side needs only the stdlib SDK; the service side runs from the repo checkout via `run.sh`:

```bash
pip install reef-client
```

You also need an OpenAI-compatible endpoint for the model under test, and for the pi variant Node and npm: at startup Reef installs the pinned pi under `~/.local/share/reef-harness/pi` (set serve.yaml's `evolution.binary` to use an install of your own). The native variant needs no agent binary: the loop is reef's own.

## Quick start

```bash
./run.sh          # on pi (serve.yaml)
./run.sh native   # on reef's native harness (serve-native.yaml)
```

Every run ends by writing `work/replay.html`: the release chain with each step's result and the files its patch changed, the loop graph replayed from a session's events, the session event log and the process timeline. Open it in a browser; `python3 run.py replay` rebuilds it from `work/` at any time.

serve.yaml carries the endpoint (`inference.upstream-url: http://127.0.0.1:8000`, no /v1 suffix) and the model (`inference.upstream-model: qwen3-8b`) as literals; edit them there to point at your own. The recipe takes its model from that same `inference` section, so the method's model call, the evolve rollouts and served traffic all use the one name, and run.py's `MODEL` must match; a name the endpoint does not serve fails the method's model call, and the step is skipped with a `method_summary` that names the failure. The one value serve.yaml does not hold is the provider key: `export REEF_UPSTREAM_API_KEY=...` if your endpoint needs one.

## The method

A harness evolution method is a `Method` subclass that implements `step(context)`, which runs once per evolve step. `context.workspace` is the step's directory: `workspace.harness` holds the served tree, and the tree the method leaves there is the step's candidate. `context.services` are the services Reef runs for the method, such as model calls. `RulesMethod.step` in `harness/plugins.py` is the whole method of the pi variant:

1. `workspace.traces()` reads the step's traces, each with its task, the agent's reply, its score and the report's feedback; the method keeps the failing ones (score below 1.0). `workspace.request()` is the person's request on a step that answers one.
2. One call through `services.models.served` asks the served model to rewrite `rules.md`. The prompt is `harness/rules_prompt.md`. The call counts against the step's model-call budget and is kept in the step's record.
3. `workspace.write_tree(...)` leaves the tree with the new `rules.md`. The backend then checks the tree and runs the evaluation.
4. `workspace.write_output(...)` says what the step did: the summary is text for people, and the notes are JSON for the commit metrics.

A step that leaves the tree unchanged is skipped, and its `method_summary` metric says why: no failing trace, a failed model call, or an empty reply. `NativeMethod` follows the same steps with the prompt in `harness/native_prompt.md`, and its reply becomes one file of the native loop.

## Worker execution

The configs default to `auto`; the single service controller resolves to `uni`.
Worker placement defaults to `execution.evolution.backend: auto` with
`execution.evolution.workers: 1`: one CPU worker uses `uni`; increasing the count selects
spawned `mp` workers. Model inference still runs at the configured upstream
endpoint, so these scorers do not reserve GPUs.

The YAML only needs the worker count; `backend: auto` and resource requests
can be omitted. Local workers are not pinned or limited to one CPU core.

The worker/scorer pool is reused across evaluations and released when its
scenario closes. Each rollout still starts a separate harness process in a
fresh temporary directory. This is a fixed-size pool, not autoscaling.
`evolution.executor` is a separate rollout-isolation option (`local` or
`sandbox`), not a worker backend selector.

The demo's materializer also preserves optional top-level `execution` and
`executors` sections. For example, choose a reusable worker/resource profile:

```yaml
execution:
  evolution: cpu-pool
executors:
  cpu-pool:
    workers: 8
```

CPU requests do not enforce core limits on local workers. Ray uses CPU/GPU
requests as per-worker scheduling reservations. Set workers and resources
under `execution.evolution`.

## Keep a deployment running

Pass the model at startup using `REEF_UPSTREAM_MODEL`; no YAML edit is needed. `TIMEOUT_S` (120 s) and `MAX_TOKENS` (2048) in `harness/plugins.py` cap the method's one model call per step; raise both for a local thinking model, which answers in minutes and spends the reply budget on its reasoning first.
[deployment.yaml](configs/deployment.yaml) uses the same variable for serving and
evaluation. From the repository root, after the source installation and with Node
and npm available, replace the model ID and API key below with your provider's values:

```bash
export REEF_UPSTREAM_URL="https://api.openai.com"  # no /v1 suffix
export REEF_UPSTREAM_MODEL="REPLACE_WITH_YOUR_PROVIDER_MODEL_ID"
export REEF_UPSTREAM_API_KEY="your-openai-api-key"
reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml
```

Use the exact model ID accepted by your provider, not the placeholder above.
An unset or empty `REEF_UPSTREAM_MODEL` is reported at startup.
The deployment reads this variable in its `inference` section. The recipe
omits `model.path` and uses that runtime's model, so inference, the method's
model call and evaluation use the same model.

Use your provider's API key, or omit it for a local endpoint without authentication.
`deployment.yaml` keeps Reef running until interrupted. It contains both the
deployment and its recipe; no recipe generation or extra Python launcher is
needed.
Defaults: port `8901`, no Reef access token unless `REEF_TOKEN` is set, state
under `work/deployment/`. Use `--reef.port` and `--reef.token` to override the
port and access token. The `run.sh` demo keeps its separate configuration and state.

Pull the tree from this service with `GET /reef/harness` and report feedback
against the receipts of its model calls with `POST /reef/report`, as
[Use a published tree](../../docs/user-guide/evolve-your-harness.rst#use-a-published-tree)
describes. Adapt `deployment.yaml`'s tasks and `harness/evolution.py`'s grader
for your workload before starting the deployment.

`deployment.yaml` also sets the `activation` section's `review_kinds: [extension]`: when a win touches an `extension` file, its release stays inactive, because an evolved extension runs in pi's process with your privileges. An inactive release is served to no client until a person activates it. To activate it, find the newest row in `GET /reef/harness/releases` with `"checkpoint": true` and `"activated": false`. Send that row's `artifact_id` to `POST /reef/scenarios/{scenario}/activate` as `{"artifact_id": ...}`. The curl is under [Activate an inactive release](../../docs/user-guide/evolve-your-harness.rst#activate-an-inactive-release) in the user guide. `serve.yaml` and `serve-native.yaml` set no `activation` section, so only a kind that is always reviewed, such as native's `native_loop`, waits for a person.

## Notebook

`evolve-your-harness.ipynb` walks the same pass cell by cell and manages the service as a subprocess, so one kernel holds the whole loop. Set the endpoint, model, and key in its first code cell; the notebook loads serve.yaml, sets `inference.upstream-url` and `inference.upstream-model` to those values, writes the result to `work/serve-notebook.yaml`, and materializes the recipe config from it with `harness/materialize_recipe.py`, the step `run.sh` runs. `run.py` stays the reference implementation of the loop; the notebook mirrors it.

Pick a model that fails at least one task. A model that passes all three tasks still trains one step per report, but no step has a failing trace to learn from, so every step skips and nothing publishes (the DeepSeek result below). A step that changes nothing commits as skipped, and its `method_summary`, on the step's row in `GET /reef/harness/releases`, says why.

## What one run does

1. `run.sh` copies serve.yaml's recipe sections into `work/recipes/harness_evolve.yaml` and starts Reef. The recipe boots with the seed tree `seeds/pi`: a starter `rules.md`. The endpoint and the model live only in serve.yaml's `inference` section (`upstream-url`, `upstream-api-key`, `upstream-model`); Reef hands them to the method as the served model binding and renders them into each evaluation rollout, so neither the method nor the published tree ever names them.
2. `run.py` sends each of the three exact-answer coding tasks once through reef inference; reef serves the reply and records the exchange. The reply is graded the same way the evaluation grades rollouts, and the score is reported against the receipt.
3. Every valid scored report trains, passing ones included, and `evolution.batch_size: 1` makes each report one evolve step: `RulesMethod` sends the step's failing traces, each with its task, reply, score and feedback, and the current `rules.md` to the same model through `services.models.served`, which answers with the new rules; the candidate and current trees each run one rollout per task; the change publishes only when the candidate wins more task pairings than it loses (`evolution.evaluation.policy: compare`). A passing report still gets a step, but that step has no failing trace, so the method leaves the tree and the step skips: a passing run is not a run that batched nothing.
4. `run.py` waits for the results of the steps its reports triggered, then pulls `GET /reef/harness` and prints the release, its evaluation metrics and the files the step changed. A client that writes the pulled files where pi reads them, with pi's model bound to Reef, runs the learned rules.

## Native variant

`./run.sh native` runs the same loop on reef's own coding agent (`evolution.adapter: native`), whose tools, loop hooks, loop graph and helper agents are files of the tree. Four things differ from the pi run:

- The seed is the directory `seeds/native`: the loop's `read_file`, `write_file`, `run_bash` and `execute` tools, its `loop_guard` hook and its `main` graph (`think` to `act` to `think`, `done` on a text reply), beside the starter skill, so the method sees them as files it may retune or replace.
- `NativeMethod` asks for one change, with the prompt in `harness/native_prompt.md`: a skill, a `native_tool` (a schema plus a module defining `run(args, workdir) -> str`), a `native_hook` (a module defining `listen(payload, next) -> decision` at `pre_step`, `pre_execute`, `request_error` or `post_execute`), a `native_agent` (a helper with its own prompt, tools and budget that a graph's `subagent` stage calls) or the `native_graph` (the loop's stages and the edges between them, shown to the model as one worked example: the seed loop plus a `check` stage). The model answers in one JSON shape per kind, and the answer becomes the file the tree holds, such as `native/tools/<name>.py` with the declaration constants after the code; a change that reuses a seed tool's name rewrites that tool. An agent alone changes nothing (only a `subagent` stage runs one), so an agent change also rewrites `native/graphs/main.json` to route the main graph's model text through a new `ask-<name>` subagent stage and a new `answer-<name>` model stage, whose text goes where the model's text went: the agent's reply comes back as a user message, and only a model stage writes the assistant message the grader reads. `evaluate` grades the last `assistant/message` event of the native-jsonl trajectory with the same grader.
- No agent binary is installed: `run.sh native` writes a `reef-native` launcher for this checkout under `work/bin` and puts it on PATH, which is what an installed reef's console script provides.
- The pass runs through the serve form. `run.sh native` names the scenario and writes the model binding under `work/tree` (`python3 run.py pull`), then starts `reef-native serve --tree work/tree --scenario harness-evolve-demo --follow head --poll-interval 5`, which pulls the served source tree itself. `python3 run.py native` sends each task to that process as one turn with `reef-native turn`, grades the reply and reports the score against the receipts the turn returned. When a step publishes and activates a release, the running process mounts it between two steps or at once while idle, one `harness/mount` line in `work/tree/native/sessions/serve.jsonl` (or in the open turn's session), with no reinstall and no restart; `run.py` then sends the first task again and prints the stages of the new session, which are the mounted graph's. `work/serve.log` is the process's stderr.

The recorded pass is identical, so the two variants are comparable on the same three tasks.

## Known limitations

- `run.py` waits for the result of every step its reports triggered, then reads the active release once: a step still pending at the deadline is reported as pending and its result lands in the commit history, not in the run's output.
- vllm serves Qwen3-8B for pi only with `--enable-auto-tool-choice --tool-call-parser hermes`; without them it answers pi's `tool_choice: "auto"` requests with a 400, every rollout fails, and every evaluation ties.
