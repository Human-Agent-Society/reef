# Reefine: refine a pi harness from plain-language requests

Reefine turns plain-language requests into versioned harness changes: skills,
rules, agent commands, or pi extensions. This tutorial asks for a change over
HTTP, follows its step on the request page, promotes a held release, and pulls
the result. The scripted demos then run the same path end to end and run a pi
session on the new tree.

## 1. Start Reef

Use the repository environment from the [development guide](../../docs/contributing/development.rst).
The `python3` on your PATH must import `reef` and `reef_client`.
Have Node.js, npm, `rg`, and `fd` available for pi.
Start a model endpoint that supports tool calls. Keep port `8901` free.

In a terminal at the repository root:

```bash
source .venv/bin/activate
reef serve --recipe reefine \
  --inference.upstream-url http://127.0.0.1:11434 \
  --inference.upstream-model gemma4:26b \
  --inference.upstream-api-key dummy
```

This example uses a local Ollama model. Change the endpoint, model, and key
for your provider. Leave the service running throughout the tutorial.
It listens on `127.0.0.1:8901` and stores state under `.reef/reefine/`.
At startup it installs the pinned pi under `~/.local/share/reef-harness/pi`.

The local service requires no token unless `REEF_TOKEN` is set.
If you use a token, export it in both terminals and add
`-H "Authorization: Bearer $REEF_TOKEN"` to each `curl` command below.

## 2. Create a scenario

Open another terminal. Create a scenario to keep this harness's releases and
requests together, then read its first release:

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"name": "reefine-tutorial"}' \
  http://127.0.0.1:8901/reef/scenarios

curl -fsS -H 'x-reef-scenario: reefine-tutorial' \
  http://127.0.0.1:8901/reef/harness
```

The first release is the recipe's seed, one starter skill, rendered for pi.
Note its `release_id`.

## 3. Ask for a change

Send the request, with the release id from the previous step:

```bash
curl -fsS -H 'x-reef-scenario: reefine-tutorial' -H 'Content-Type: application/json' \
  -d '{"text": "When I ask you to fix a bug, reproduce it with a failing test before editing the code.", "session": "tutorial", "release_id": "<release_id>"}' \
  http://127.0.0.1:8901/reef/train
```

The answer carries the request's `agent_record_id` and its `page_path`. Open
`http://127.0.0.1:8901` followed by the `page_path` in a browser. While the
step runs, the page names its state (queued, proposing, evaluating), lists
what the proposer does, and reloads every five seconds. Local models can take
several minutes per request. For a client without a browser,
`GET /reef/harness/requests/<agent_record_id>/progress` gives the same state
as JSON.

Other requests work the same way, for example a research workflow:

```text
When I ask a research question, search for relevant papers, download and read them, then answer with citations.
```

These are example requests, not built-in Reefine modes.

## 4. Review and promote the result

When the step settles, the page shows the result, the files the change wrote,
the review, the design and its How to use section, and the next action. The
default evaluation checks that the harness can run `echo reef-ok` and return
its output. Passing this health task does not prove that your requested
behavior works: read the design and the review. Rejected or skipped requests
leave the served release unchanged.

A release that changes an extension waits as pending, because an extension
runs in pi's process with your privileges. Inspect its files and declared
requirements on the page before you serve it. Then promote it with the id the
page names:

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"release_id": "<the pending release id>"}' \
  http://127.0.0.1:8901/reef/scenarios/reefine-tutorial/promote
```

## 5. Pull the result

```bash
curl -fsS -H 'x-reef-scenario: reefine-tutorial' \
  http://127.0.0.1:8901/reef/harness
```

`files` maps each path pi reads (`pi-agent/skills/<name>/SKILL.md` for a
skill, `pi-agent/AGENTS.md` for the rules, and so on) to its text.
`requires` lists what the release needs on your machine, each item with a
`prompt` that says what to provide. The files carry no endpoint, key, or model: a client writes
them where pi reads them and binds pi's model to this Reef, as
[Use a published tree](../../docs/user-guide/evolve-your-harness.rst#use-a-published-tree)
describes. Reef ships no packaged pi client; the scripted demos below include
one.

## Scripted demos

`run.sh` runs the whole path end to end. It starts a separate deployment,
renders the served tree for pi, runs a fixed request, saves results, and stops
the service. Stop the manually started service first: both use port `8901`.

```bash
# From the repository root, with .venv activated:
cd tutorials/reefine
uv pip install -e .
export REEF_UPSTREAM_URL=http://127.0.0.1:11434  # No /v1 suffix.
export REEF_UPSTREAM_MODEL=gemma4:26b
export REEF_UPSTREAM_API_KEY=dummy
export REEF_TOKEN=reef-local                  # Matches configs/deployment.yaml.

./run.sh bugfix
./run.sh research
./run.sh measure --n 10
```

`bugfix` asks for the test-and-fix flow with a second agent's review.
`research` asks for the paper-reading workflow. The exact requests are in
[demos/bugfix.md](demos/bugfix.md) and [demos/research.md](demos/research.md).
A demo runs these steps in `run.py`:

1. Ask: it sends the request to `POST /reef/train` and prints the request's
   page link.
2. Wait: it prints each new catalog row until the step for its request
   settles.
3. Promote: it saves a pending release's step page beside the run, then
   promotes the release.
4. Install: it pulls the head's source tree
   (`GET /reef/harness?form=source`), checks each `requires` item on this
   machine, and renders the tree for pi with a model binding at this Reef
   into `work/harness`.
5. Show: it runs one headless `pi -p` session on the installed tree and
   prints the session's tool calls in order and its final answer.

**The demos promote pending extensions and run each requirement check
without asking.** They do not pause for the review described above. An unset
variable, a missing binary, or a check that does not exit 0 stops a demo with
exit code `2`. If no change is selected, the show session runs the previously
installed tree.

In the bug-fix show session, check the tool calls: the agent should run a
failing test before editing `adder.py`, then fix the code and run the test
again. A published release alone does not establish that the instruction was
followed. If the behavior is incomplete, submit a more specific request.

The [scripted deployment](configs/deployment.yaml) uses the text proposer and
stores state under `work/`. The built-in profile used above can use an agent
proposer where supported. Both use manual training and a health evaluation.
The script sets proposer limits of 900 seconds and 16,384 tokens by default.

Measurement submits up to 12 fixed skill and rule requests, one at a time.
It reports accepted requests, proposed changes, evaluated candidates, health
passes, published releases, and pending releases. It does not install those
updates or run a session after each request.

Each run saves `work/<mode>-<timestamp>.json`. A demo's session and saved step
pages are under `work/<mode>-<timestamp>/`; service logs are in `work/reef.log`.
Runs reuse `work/deployment/`. Move `work/` aside before a run to start fresh.

## Historical results

Recorded runs from September 6–8, 2026 used `gemma4:26b` on a Mac mini M4
with 32 GB of memory. They used three arithmetic tasks and `selection: always`,
which published changes regardless of evaluation results.
They do not validate the current health task.

- Bug-fix sessions reproduced, fixed, and tested the bug, but did not obtain a second agent's review.
- Research sessions answered from memory or used a local search tool, but downloaded and read no paper.
- No recorded run produced an extension.

Four measurement runs on the code from PR #315 reported:

| Run | Filed | Answered | Evaluated | Won | Published | Median request time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 2 | 0 | 2 | — |
| 2 | 10 | 3 | 3 | 1 | 3 | 248.0 |
| 3 | 10 | 8 | 8 | 1 | 8 | 379.8 |
| 4 | 10 | 10 | 10 | 0 | 10 | 427.2 |

Here, `Won` means more arithmetic-task wins than losses. Run 1 stopped after
a driver error; its counts came from catalog rows, and its median was unavailable.
Parser and request-handling code changed between runs. Runs were not repeated,
so the table does not establish a success rate or a performance comparison.
