# Reefine

**Status: beta.** Reefine remains under `recipes/beta/reefine/` until
complete, reproducible results are published. It is not part of the
`reef-infra` wheel, and it is not Reef's default harness method. The tests
validate the method, the agent, the scorer and the scripted example against
stand-ins; they do not establish that a requested change works. See
[RFC #938](https://github.com/Human-Agent-Society/reef/issues/938) for this
classification.

Reefine refines a coding harness from plain-language instructions. It is a
[harness evolution](../../../docs/user-guide/evolve-your-harness.rst) recipe:
its implementation is `recipes.beta.reefine:ReefineRecipe`, a
`HarnessEvolutionRecipe` with its own method (`ReefineMethod`), scorer, seed
and evaluation defaults. It needs no training GPUs. The example configuration
refines a pi harness; [Adapters other than pi](#adapters-other-than-pi) covers
the others.

## Layout

```text
beta/reefine/
  recipe.py            ReefineRecipe: the defaults, the agent's isolation and the multimodal gateway
  method.py            ReefineMethod: answers a request, or learns one skill from failing traces
  answer.py            the text answer: the served model writes the design, the files and the review
  prompts/             the text answer's prompts and the agent's request prompt (<name>.md), and the
                       words they take for each kind of harness (env_words.yaml)
  agent.py             the agent answer: pi edits and runs the step's tree
  agent_gateway.py     the loopback gateway the agent and its trials reach models through
  agent_rules.md, agent_tools.ts, trial.py, trial_driver.mjs
                       the agent's instructions, its tools and its trial runner
  pi_extension_api.md  the summary of pi's extension API the agent reads
  harness_facts/       each adapter's facts the prompts and the review use (<adapter>.yaml)
  multimodal.py        the image, speech, embedding and decision gateway
  scorer.py            the health task's evaluator
  seed/                the first tree of a new scenario: one starter skill
  health/              the health task as a Harbor task directory, for terminus
  examples/            the example: configs/serve.yaml, the scripted demos and run.sh
```

## Start

Reefine is not installed with `reef-infra`, so run it from a Reef checkout.
Set `PYTHONPATH` to the checkout, as the example's `run.sh` does, so that the
service imports Reef and the recipe from this checkout. From the repository root, with the repository environment active, start
[the example configuration](examples/configs/serve.yaml) with an
OpenAI-compatible endpoint:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
reef serve -c recipes/beta/reefine/examples/configs/serve.yaml \
  --inference.upstream-url http://127.0.0.1:11434 \
  --inference.upstream-model gemma4:26b \
  --inference.upstream-api-key dummy \
  --recipe.config.evolution.multimodal.api_key sk-or-...
```

`--model ollama/gemma4:26b` is a short form of the three `--inference`
lines for a local Ollama model. The configuration also reads
`REEF_UPSTREAM_URL`, `REEF_UPSTREAM_MODEL` and `REEF_UPSTREAM_API_KEY` when
you do not pass the flags.

For an OpenAI Responses or Anthropic endpoint, add
`--inference.upstream-api responses` or `--inference.upstream-api anthropic`.
The last line, an OpenRouter key (or `REEF_MULTIMODAL_API_KEY`), gives the
harness image, speech, embedding and decision models (see
[Images, speech, embeddings and decisions](#images-speech-embeddings-and-decisions)).
Leave it out to go without them, or when the upstream is OpenRouter.

The service listens on `127.0.0.1:8901`. It requires no token unless
`REEF_TOKEN` is set. It keeps state under `~/.reef/reefine/`, outside the
checkout: with `sandbox: none` the agent works in a step's directory there,
and pi reads every `AGENTS.md` and `CLAUDE.md` in the directories above it
(see [The agent](#the-agent)). Move that directory aside to start fresh. For
a custom deployment, copy `examples/configs/serve.yaml` and pass your copy
with `-c`.
The [example](examples/README.md) runs a bug-fix demo, a research demo and a
measurement of which requests passed the checks.

## How it works

1. Ask. `POST /reef/train` files a plain-language instruction for one
   scenario. Create the scenario, then send the instruction:

   ```bash
   curl -sS -H "Content-Type: application/json" \
     -d '{"name": "my-harness"}' http://127.0.0.1:8901/reef/scenarios
   curl -sS -H "x-reef-scenario: my-harness" -H "Content-Type: application/json" \
     -d '{"text": "when I ask you to fix a bug, reproduce it with a failing test first", "session": "me", "release_id": ""}' \
     http://127.0.0.1:8901/reef/train
   ```

   `session` and `release_id` say where the request came from: the
   client's session and the release it runs, empty when it runs none yet.
   The answer carries the training record id and `page_path`, the
   request's page (`GET /reef/harness/requests/<id>/page`). The page
   reloads every five seconds and names the step's state until the result
   is on it, with the step's summary: the design, how to use the change and
   the review.
   `GET /reef/harness/requests/<id>/progress` gives the same state as
   JSON. The [manual training API](../../../docs/reference/http-api.rst#manual-training)
   lists the fields a request carries.
2. Step. In `training-mode: manual` the service runs one evolve step for
   each accepted instruction. Where the host can isolate it, a coding agent
   answers the instruction (see [The agent](#the-agent)): it edits the step's
   tree, runs the changed harness for real and stops, and a review call reads
   the files against the request. Otherwise the served model answers it in a
   few calls: it writes a design first (the
   request in one sentence, what triggers the behavior, what state the
   harness must know and where it comes from, what only you can provide as
   `requires` items with a `prompt` each), then the files, and a
   second call reviews them against the request. The review also says
   whether the files deliver the requested behavior or only put a
   substitute in its place, such as a rule describing it. When the review
   is partial or finds a substitute, the model writes the answer again with
   the review's findings, up to three answers in all. The step keeps the
   delivering answer with the fewest uncovered points, and its notes say
   which answer it kept when it wrote more than one (`kept_attempt`); when
   no answer delivers, it is skipped with the reason. A point the harness's
   facts say no answer there can deliver (a hard tool lockout on a harness
   whose commands cannot take a tool away) is a limit, listed apart from the
   uncovered points: it starts no retry, and the loop ends when only limits
   remain. The evaluation runs the candidate on the health task: it
   publishes when the tree still works. The health task is no test of the
   requested behavior, which only the review reads. The step's summary
   carries the design and the review either way. A review call that
   answers with no text is asked once more with room for both its
   reasoning and its reply; when it still gives none, the step records why
   and its summary says the review did not run, rather than reading like a
   step that had no review to give.
3. Result. The request page and the step's page
   (`GET /reef/harness/releases/<step>/page`) give the result: published
   as a release, inactive because it changes an extension, rejected by the
   evaluation, or skipped with the reason, followed by the step's summary.
   The request page also gives the next action: the activate route for an
   inactive release, `GET /reef/harness` for an active one.
4. Activate. A release that touches an `extension` file is inactive: Reef
   does not serve it, so it reaches no client on its own. Read its step page,
   which carries the step's summary (the design, how to use the change and
   the review) and the numbers, then activate it with the artifact id the
   page names:

   ```bash
   curl -sS -X POST -H "Content-Type: application/json" \
     -d '{"artifact_id": "<the artifact id the page names>"}' \
     http://127.0.0.1:8901/reef/scenarios/my-harness/activate
   ```

5. Use. `GET /reef/harness` serves the active release rendered for the
   deployment's adapter, and `GET /reef/harness?form=source` serves its
   source tree.
   The served files carry no endpoint and no credential: a client binds its
   agent's model to this Reef itself, as
   [Use a published tree](../../../docs/user-guide/evolve-your-harness.rst#use-a-published-tree)
   describes. The manifest's `requires` lists what the release needs over
   its whole chain, and the client checks each item before it runs the
   release. The example's [`run.py`](examples/run.py) is a worked client for pi.

## Behavior and configuration

* `training-mode: manual` runs one step for each accepted instruction on
  `POST /reef/train`. Each step also reads the failing reports in the manual
  buffer, which a client fills
  ([Choose the samples that train](../../../docs/reference/http-api.rst#choose-the-samples-that-train));
  new reports do not join it by themselves. Use `auto` to learn from failing
  reports without instructions.
* The first tree of a new scenario is the recipe's seed,
  [`seed/`](seed/): one starter skill. The agent, or the served model
  where the host cannot isolate the agent, writes skills, rules, agent
  commands, or pi extensions.
* The prompts instruct the model to integrate new slash commands into pi's native
  `/` autocomplete dropdown alongside built-in commands, with descriptions,
  using prompt templates or extension command registration. The review checks
  that integration and the path from invocation to a visible result, including
  state and an off command for modes. Headless trials cannot verify the dropdown;
  the agent records any unverified interactive checks in its design.
* The default activation rule, `always` with `review_kinds: [extension]`,
  leaves each release that changes an `extension` file inactive until a
  person activates it. On an adapter whose tree has no `extension` part,
  the list is empty. A top-level `activation` section replaces the rule.
* `evolution.evaluation.policy: floor` is the default: the evaluation runs the
  candidate alone, scored by `AnswerScorer`, and publishes it when every task
  scores at least `evolution.evaluation.floor_score` (`1.0`). The training
  start is not run. A run that gave no answer misses the floor; one that
  the environment could not run does not.
* What only you can provide (a phone number, a credential, a permission, an
  account) is a `requires` item, `{name, kind, check?, prompt?}`, whose
  `prompt` is one sentence of at most 200 characters that a client shows
  when it asks for the item; the Setup table of the step's page has a prompt
  column. A step may require at most eight distinct items, including
  those the method adds. Exceeding this limit skips the step with an
  explanation and publishes no change; split the request into smaller
  changes before retrying. On pi an `env` item's value is read at run time from
  `process.env.NAME`: the prompt says that an extension never asks
  you for it in the session, never stores it in a file of its own and never
  hardcodes it, and its review lists a value the extension asks for or
  stores itself as uncovered. On another adapter the value reaches the
  harness's environment at run time, and the same rules hold for the files
  the model writes there.

## Adapters other than pi

`evolution.adapter` runs the recipe on any bundled adapter. A request comes
through `POST /reef/train` on every adapter, the pages are the same, and
`GET /reef/harness` serves the release rendered for that adapter. Serve the
upstream in the dialect the adapter's model binding speaks:

| Adapter | Serve with |
| --- | --- |
| `claude` | `--inference.upstream-api anthropic` |
| `codex` | `--inference.upstream-api responses` |
| `opencode`, `hermes`, `dsh` | no flag (the default openai dialect) |

The agent runs on pi alone, so another adapter builds no agent and needs
no sandbox for it. On another adapter the served model answers a
request with rules, skills and commands, and with a `config/primary.json`
file where the harness's config enforces a behavior (an opencode agent with
a permission map, Claude Code permissions that pre-approve only
`WebSearch` and `WebFetch`, Codex's `web_search`). It writes no
`extension` file, since it knows pi's extension API and no other. The
prompt tells it that harness's own facts from
[`harness_facts/<adapter>.yaml`](harness_facts/): how you type a command there, which file holds the
rules, which tools the harness has (web search among them, with what it
needs: `DEEPSEEK_API_KEY` on dsh) and how the harness offers a mode. The
review judges new commands by the same facts. Its design says when the
request needs behavior these files cannot give.

On `claude`, `codex`, `hermes` and `dsh` a command cannot take a tool
away, so a mode there is guidance the model follows while every tool stays
offered: the design and the command's reply say so, the mode's state lives
in the conversation (the command's reply and the header on each reply) and
never in a file a tool writes or reads, the model declines a skill the
person's message loads while the mode is on and names the mode's off
command rather than a way around it (Claude Code's `!` line, say), and the
review lists a request for a hard restriction under limits.

On `opencode` an agent with a permission map is the mode, and the model
gets only the tools it allows: an answer whose agent has no permission map
is written again, since that agent is offered every tool. A command's
`agent:` runs only that command's turn, so the person enters the mode with
`/agents` (or with its command as a new session's first message) and
leaves it with `/agents`, choosing build, together with a leave command
whose text says every tool is back: opencode tells the model nothing when
the agent changes. The permission map limits the model's own tool calls
only; a skill the person types, a file attached with `@` and a `!`
command still run, and the review lists them under limits.

On `codex` the `web_search` key turns the hosted search on in every
session of the release, which How to use says. A `terminus` run has no
reply a person reads, so its design names the files the task leaves, the
verifier's reward and the trajectory as the visible result, and what the
release requires must hold in the Harbor task a run uses.

An answer whose form slipped is written again while attempts remain: JSON
that does not parse (a list or an object that decodes inside a broken outer
array is a fragment of it, not the answer), files every one of which was
dropped, or files the harness's validation refuses (each answer meets the
validation the step runs next). When validation refused the files, the retry
shows that answer's design and files, so what it got right stays. Each
dropped answer's reason is recorded in the step's notes
(`method_notes.dropped_attempts`) and shows in the request page's Activity
while the step runs. A review reply whose JSON a stray quote broke is asked
once more. The review reads the whole design; the step
records it cut to 4000 characters with its last paragraph, the How to use,
kept whole.

Off pi, a design that says no file this harness takes can deliver the
request is an answer, not a failure: the review runs on it, the step
records it under `method_notes.declined` with the design and what is
out of reach, and its summary says it was answered with no change. When that
review finds a point a file could still deliver, the request is written
again. On pi such a reply stays an answer with nothing to apply, its
reason under `failure`.

The step's summary carries the design with its How to use paragraph, so a
person learns the form the release takes (`$chat` on codex) rather than
one taken from the request, and the review, which lists what the harness
puts out of reach one point per line.

## The agent

The text answer writes an extension it never runs, so a model name it
guessed or a parameter a provider refuses only shows once you use the
change. The agent runs the served model as a pi coding agent instead. Its
current directory is the step's tree, `harness/` in the step's workspace,
where it may change `skills/<name>/SKILL.md`, `rules.md`,
`commands/<name>.md` and `extensions/<name>.ts`. Beside it, it writes
`output/requires.json`, `output/design.md` and `output/progress.md`, and
reads `scratch/reference/pi-extension-api.md`, this deployment's summary of
the extension API, and the step's `material/`. The step does not read back a
file the agent writes at a path that no part of pi's tree takes, a file of a
part a request may not change, or a symbolic link, and the check names each
one. It names a directory that holds no file of the harness tree, such as the
`node_modules/` of an install, once. A file of a part a request may write
must be UTF-8 text and not a link, or the check fails. When the reference omits an interface or leaves
behavior unclear, the agent may read the installed pi package's
documentation, type definitions and relevant source files. It checks the
installed version first; if source is missing locally, it may consult that
exact upstream version. The package is read-only, and delivered files must
use supported public interfaces.
A behavior that requires changes to pi itself is recorded as a limitation
in the design. The agent may also read provider documentation on the network
and has three tools of its own:

* `harness_check` runs the tree through Reef's validation, as the step
  will.
* `harness_trial` runs the changed harness for real, online, on a task the
  agent writes, and returns the session's final text, the tools it called,
  every image, speech, embedding or decision call it made with the
  provider's error when one failed, and the end of its stderr.
  Alternatively, its `script` input runs literal prompts and slash commands
  in one pi SDK session, using a fixed local OpenAI-compatible model. Steps
  can start a fresh session, force a tool attempt, and assert the actual
  outgoing tool set, system prompt, messages, and recorded fixture tool
  execution. A missing model request cannot satisfy an assertion about its
  payload. Fixture tools cannot replace built-in or candidate tools.
  Scripted trials check session behavior; an online task trial is still
  needed for behavior specific to the configured provider or model, and
  neither mode verifies an interactive dropdown.
* `harness_progress` saves short working notes in `output/progress.md`:
  confirmed API behavior and source locations, implementation status, and
  unresolved checks. Before each model call, Reef restores these notes, the
  latest check/trial observations and the remaining execution time into
  context.

While it runs, the request page's Activity lists each tool the agent calls,
each check and trial with its result, and each image or speech call with the
provider's status, so a long run shows what it is doing.

When the agent stops, its tree is the step's candidate: the head tree with
the files it changed of the parts a request may write, beside its `requires`
items and design, reviewed as a text answer is. A run that changed no file,
ran past its limit or ended on a failed model response hands back no change,
its reason under `failure`. The step record keeps the agent's session log
as `output/agent-session.jsonl`, and its check and trial observations as
`output/agent-checks.jsonl`, each identified by the candidate tree's
checksum. When the candidate is not the tree the agent left (a failed run,
or files left out), that tree stays in `output/agent-harness/`. These are
diagnostic files, not published changes or an automatic resume mechanism.
They follow the deployment's step-record retention.

The agent holds no credential. It and its trials reach models through a
loopback gateway whose address carries a random token: the served model
with the served key (always the served model, whatever a request names),
and `/v1/images`, `/v1/embeddings`, `/v1/audio/speech` and
`/v1/decisions` on the recipe's multimodal gateway (`evolution.multimodal`,
see below), and the provider's
model list (`GET /models?modality=`, fetched with its key), so the agent picks
a model that exists without holding a key. Every call spends from
`evolution.max_model_calls_per_step` and is recorded in the step's
`calls.json`. Nothing else is reachable through it, Reef's own routes
included.

Isolation (`evolution.agent.sandbox`, or `REEF_AGENT_SANDBOX`):

* `bwrap` (the default where the host can): the agent and its trials run in
  a bubblewrap jail with an empty environment and a read-only system. The
  agent may write only its tree (its working directory), `output/` and
  pi's session state; a trial runs in a new temporary directory outside
  the step's, where it may write only its own working directory and
  session state. The jail is in a network namespace pasta connects to the
  internet with no host address reachable but the gateway's port. It
  needs `bwrap` and `pasta` (the `passt` package) and a service that
  runs as a non-root user with user namespaces allowed.
* `e2b`: the agent and its trials run in an [E2B](https://e2b.dev)
  cloud sandbox, a microVM with the internet and no route to the Reef host.
  The gateway's port answers at the same loopback address inside it through
  a tunnel Reef opens from its side (a relay in the sandbox that Reef polls
  over the sandbox's public address, with a per-run secret), so it works
  from a laptop as from a server, and no other host port is reachable. The
  agent's files are copied in, refreshed for each check and trial, and copied
  back when it stops. It needs `pip install 'reef-infra[e2b]'` and an E2B
  key (`e2b_api_key`, else `E2B_API_KEY`); `e2b_template` names the
  sandbox image, else Reef builds `reef-pi-<version>-runs-v1` (the pinned
  pi and bubblewrap on Node 22) on first use. A custom template must include
  bubblewrap. The agent runs with rendered inputs read-only and only its
  workspace and declared state writable; it cannot use sudo to remove that
  restriction. Model credentials remain in Reef's gateway.

  If the command connection drops, Reef attempts to reconnect to the same
  process twice within the original time limit; it does not rerun the command.
  The tunnel retries interrupted response transfers without duplicating chunks.
  A response that cannot be completed remains a failed stream. Failed steps
  appear as `Failed` on the request and release pages, with the recorded
  error, and publish no release. After resolving the error, submit the request again.

  ```bash
  E2B_API_KEY=e2b_... reef serve -c recipes/beta/reefine/examples/configs/serve.yaml \
    --inference.upstream-url https://openrouter.ai/api \
    --inference.upstream-model z-ai/glm-5.3 \
    --recipe.config.evolution.agent.sandbox e2b
  ```

* `modal`: the same, in a [Modal](https://modal.com/docs/guide/sandboxes)
  Sandbox instead of an E2B microVM. It needs `pip install 'reef-infra[modal]'`
  and Modal's credentials (`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or the
  profile `modal setup` writes). Reef builds the image (the pinned pi,
  bubblewrap, Python 3 and coreutils on Node 22) on first use, and Modal caches
  it; `modal_image` names another, `modal_app` the Modal app the sandboxes
  belong to. A Modal sandbox lives at most a day, and the command runs to its
  end or its limit without a reconnect.
* `none`: no isolation. The agent runs with the service's user and full
  network access, fed your clients' text. Choose it only where you trust
  every client, such as your own machine. It runs in the step's directory
  under `evolution.state_dir`, and pi reads every `AGENTS.md` or
  `CLAUDE.md` in the directories above it, so keep the state directory
  outside a project that holds such files. The example configuration keeps
  it in `~/.reef/reefine/harness`.
* Left unset on a host that cannot isolate, the agent is off and the served
  model answers in text; the service logs why.

`timeout_s` (1800) bounds the whole agent run and `trial_timeout_s` (300)
each trial. Trials also respect the remaining run budget, reserving the
last 10 percent (at most 60 seconds) for saving files and finishing. The
agent receives a reminder to finish during this interval; it is still
subject to the outer hard timeout. A run past its limit hands back no change.
Set
`evolution.max_model_calls_per_step` to bound what one request may spend:
an agent run makes a model call per turn and may probe several provider
models before it settles on one.

## Images, speech, embeddings and decisions

`evolution.multimodal` names one gateway that serves these modalities behind
a single key. The agent and its trials reach it through their loopback
gateway, in the provider's own format, and hold no provider key. Reef's HTTP
service does not relay these modalities: a scenario's clients call only the
chat model routes.

```yaml
evolution:
  multimodal:
    preset: openrouter          # or openai-compatible (OrcaRouter, LiteLLM, ...)
    url: https://openrouter.ai/api   # the preset's address unless set; required for openai-compatible
    api_key: ${REEF_MULTIMODAL_API_KEY}   # or api_key_env: NAME
```

`api_key` takes the key the way `inference.upstream_api_key` takes the chat
key; the example configuration sets it to `${REEF_MULTIMODAL_API_KEY}`, and
`--recipe.config.evolution.multimodal.api_key` sets it on the command line.
Empty, the upstream key serves when the upstream is the same address, so a
deployment that chats through OpenRouter needs nothing more. Without a key,
or for a route the preset does not serve (`openai-compatible` has no
decisions), the agent's call answers 501. Recipes other than Reefine offer none.

## The health floor

The example configuration's one evaluation task is a health check:

```yaml
evaluation:
  tasks:
    - '[health] Run the shell command `echo reef-ok` with your shell tool and reply with its exact output
      as a plain word alone on the last line.'
```

Reefine's evaluator ([`scorer.py`](scorer.py)) grades the reply's last line, `reef-ok` exactly. The
floor answers one question: does the tree still work after the change? The
model binding answers, the shell tool runs, every extension loads. It says
nothing about whether the change does what was asked; the step's design and
review notes and the person judge that. Set both
`evolution.evaluation.tasks` and `evolution.evaluation.scorer` for a
workload of your own, and
`evolution.evaluation.policy: compare` to require the candidate to beat
the training start on them instead.

An adapter whose prompt is a Harbor task directory, `terminus`, cannot
take this prompt. There the recipe runs the same check as the task
directory [`health/`](health/) in its place: the agent writes the
output of `echo reef-ok` to a file in the task container, and Reefine's
evaluator reads the task verifier's reward.

## Terminus

`--recipe.config.evolution.adapter terminus` runs the recipe on
Terminal-Bench's Terminus 2. Reef does not bundle that adapter: install the
Meta-Harness Terminal-Bench campaign first (`pip install -e
recipes/meta_harness/examples/terminal_bench`), which registers it and its
`reef-terminus` runner. Prepare a model endpoint with its API key, and
Docker or a remote Harbor environment (E2B or Modal) for the task container.
With Docker, the default, prepare a working Docker installation on the service
host. On macOS, Docker must share `~/.reef/runs` with the host. Other
platforms use the temp directory; set `TMPDIR` for the service if Docker's VM
does not share it. To run the task containers remotely instead, see
[Docker or a remote environment](../../meta_harness/examples/terminal_bench/README.md#docker-or-a-remote-environment).

Terminus runs one Harbor task at a time through `reef-terminus --task`, so
there is no session to type in. Submit requests over HTTP and read their
pages:

```bash
export REEF_UPSTREAM_API_KEY=sk-or-...
reef serve -c recipes/beta/reefine/examples/configs/serve.yaml \
  --inference.upstream-url https://openrouter.ai/api \
  --inference.upstream-model openai/gpt-4o-mini \
  --recipe.config.evolution.adapter terminus
```

In another terminal, create a scenario, read its `release_id` from
`GET /reef/harness`, and send a request to `POST /reef/train`. Replace
`<release_id>` and `<what it should do>` with those values. If the service
requires authentication, set `REEF_TOKEN` to its token in this terminal:

```bash
curl -sS http://127.0.0.1:8901/reef/scenarios -H "Authorization: Bearer $REEF_TOKEN" \
  -H "Content-Type: application/json" -d '{"name": "terminus-demo"}'
curl -sS http://127.0.0.1:8901/reef/harness \
  -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: terminus-demo"
curl -sS http://127.0.0.1:8901/reef/train \
  -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: terminus-demo" \
  -H "Content-Type: application/json" \
  -d '{"text": "<what it should do>", "session": "terminal-1", "release_id": "<release_id>"}'
```

The answer carries the request's `agent_record_id`. Open
`GET /reef/harness/requests/<id>/page` for that scenario to inspect the step:
accepting the request alone does not mean it passed evaluation. The
health task should score 1; inspect the selection result and published release
on the page before using the new tree.

### Docker context and published files

With local Docker, the run uses the Docker your shell uses: it keeps the
service's `DOCKER_HOST`, `DOCKER_CONTEXT` and `DOCKER_CONFIG` (default
`~/.docker`, where colima and Docker Desktop set the current context). On
macOS its files live under `~/.reef/runs`, because colima does not
share `$TMPDIR` with its VM; on Linux, WSL and Windows they stay in the
temp directory. A run on E2B gets none of these.
`GET /reef/harness` serves the published tree;
[Harness adapters](../../../docs/developer-guide/harness-adapters.rst) shows the config that runs
it through Reef yourself, including the `x-reef-scenario` header.

## What the step records

Every step's catalog row carries the request under
`metrics.training_request`, the method's plain-text summary under
`metrics.method_summary` (why nothing changed, the design with its How to
use paragraph, and the review) and its notes under `metrics.method_notes`;
the step page (`GET /reef/harness/releases/<step>/page`) shows them:

* `design`: the plan for the request, a few sentences ending with the How
  to use paragraph.
* `review`: the review call's result, `complete` or `partial`, with
  the points of the request the files cover and the ones they leave
  uncovered, and the `limits` the harness puts out of reach. Absent when
  the review call failed, which never blocks the step; `review_failure`
  then says why.
* `refused_requires`: the `requires` items the model wrote that could
  not be honored, each with the reason. An `env` item whose check is a
  shell test is brought to the one variable it names first, so `test -n "$TOKEN"` becomes the
  variable `TOKEN` rather than a refusal.
* `undeclared_env`: the variables a written extension reads through
  `process.env` that no `requires` item names. Nothing adds them; the
  notes show them so you can set them or ask for the item.
* `failure`: why a request step produced no change: the model call
  failed (how long it took, the reply budget and the endpoint's error; a
  reply without text adds that a thinking model may have spent the budget
  on its reasoning and names `REEF_REEFINE_CALL_MAX_TOKENS`) or the reply
  held no usable file. The summary starts with it.
* `agent`: the agent's run, when the agent answered: its exit code, its
  seconds and its trials. `unread` names the files it left that the
  candidate leaves out.

Every `HarnessEvolutionRecipe` evolution setting remains available,
including another method, seeds, execution settings, and the evaluation; see
[Recipe configuration](../../../docs/reference/configuration.rst#recipe-configuration).

`REEF_REEFINE_CALL_TIMEOUT_S` and `REEF_REEFINE_CALL_MAX_TOKENS` override the
time and reply token budgets of each model call the method makes itself; the
agent's run has its own limit, `evolution.agent.timeout_s`. Defaults are 600
seconds and 65536 reply tokens for the call that answers an instruction (a
thinking model reasons for tens of thousands of tokens before it writes an
extension), 60 seconds and 4096 tokens for the short plan call before it, 120
seconds and 16384 tokens for the review after it, and 60 seconds and 8192
tokens for a skill learned from failures; the reply budgets are sized for a
thinking model, which spends part of the budget on its reasoning before the
JSON (a review budget of 8192 came back empty on one). The example's `run.sh`
sets 900 seconds and 16384 tokens for its local model.
