# Reefine: refine your harness on reef-pi

Reefine turns plain-language requests into versioned harness changes: skills,
rules, agent commands, or pi extensions. This tutorial shows how to request,
review, install, and try a change in a normal `reef-pi` session.

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
Local models can take several minutes per request.

The local service requires no token unless `REEF_TOKEN` is set.
If you use a token, export it in both terminals and add
`-H "Authorization: Bearer $REEF_TOKEN"` to each `curl` command below.

## 2. Install reef-pi

Open another terminal at the repository root. Activate the same environment.
Create a scenario to keep this harness's releases and requests together:

```bash
source .venv/bin/activate
curl -fsS -H 'Content-Type: application/json' \
  -d '{"name": "reefine-tutorial"}' \
  http://127.0.0.1:8901/reef/scenarios

curl -fsS -H 'x-reef-scenario: reefine-tutorial' \
  'http://127.0.0.1:8901/reef/harness/install?adapter=pi' | bash

export PATH="$HOME/.local/bin:$PATH"
reef-pi doctor
```

The install command downloads a script from your Reef service and runs it.
It installs the harness under `~/reef-harness/reefine-tutorial` and adds the
`reef-pi` wrapper under `~/.local/bin/`. Keep the harness outside your project.
`reef-pi doctor` checks the installation and service connection.

## 3. Ask for a change

Copy the tutorial's small bug fixture into a separate workspace, then start pi:

```bash
# From the repository root:
mkdir -p "$HOME/reefine-example"
cp tutorials/reefine/demos/workspace/*.py "$HOME/reefine-example/"
cd "$HOME/reefine-example"
reef-pi
```

In the session, type:

```text
/reefine When I ask you to fix a bug, reproduce it with a failing test before editing the code.
```

Answer any clarification questions. Reef proposes and evaluates the change,
then reports the result in the session with a link to the request page.
The service performs the change; the session submits the request.

You can also submit a request from the shell:

```bash
reef-pi evolve "When I ask you to fix a bug, reproduce it with a failing test before editing the code." --wait
```

Use either entry point for the same request. Neither requires `run.sh`.

## 4. Review and install the result

In the pi session, list the versions:

```text
/versions
```

Use the version shown in the result in place of `<version>`:

```text
/versions <version>
```

Read the design, usage instructions, and review notes. The default evaluation
checks that the harness can run `echo reef-ok` and return its output.
Passing this health task does not prove that your requested behavior works.
Rejected or skipped requests leave the served version unchanged.

If you accept the change, install it:

```text
/versions <version> install
```

Confirm installation and complete any setup prompts. A version that changes
an extension waits for this approval; installation promotes it before use.
Extensions run with your privileges. Inspect declared requirements before
providing values or allowing checks. Unmet requirements prevent installation.

Load the installed version:

```text
/reload
```

Restarting `reef-pi` also loads it. For shell-based review, setup, and updates,
see the [Reefine guide](../../docs/user-guide/recipes/reefine.rst#how-it-works).

## 5. Try the changed behavior

In the reloaded session, ask:

```text
Fix the bug in adder.py.
```

Check the tool calls: the agent should run a failing test before editing
`adder.py`, then fix the code and run the test again.
A published release alone does not establish that the instruction was followed.
If the behavior is incomplete, submit a more specific `/reefine` request.

For a research workflow, try another request:

```text
/reefine When I ask a research question, search for relevant papers, download and read them, then answer with citations.
```

Review, install, and reload that version in the same way. Ask a research
question and check whether the agent actually reads sources before citing them.
These are example requests, not built-in Reefine modes.

## Optional scripted demos

`run.sh` automates experiments for this tutorial. It starts a separate deployment,
installs a harness, runs a fixed request, saves results, and stops the service.
Stop the manually started service first: both use port `8901`.

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

`bugfix` adds a second agent's review to the test-and-fix request.
`research` requests the paper-reading workflow. The exact requests are in
[demos/bugfix.md](demos/bugfix.md) and [demos/research.md](demos/research.md).

**The demos automatically promote pending extensions and run `setup --yes`.**
They do not pause for the manual review described above. Unmet requirements
or refused installation stop a demo with exit code `2`.
If no change is selected, its test session uses the previously installed harness.

The [scripted deployment](configs/deployment.yaml) uses the text proposer and
stores state under `work/`. The built-in profile used above can use an agent
proposer where supported. Both use manual training and a health evaluation.
The script sets proposer limits of 900 seconds and 16,384 tokens by default.

Measurement submits up to 12 fixed skill and rule requests, one at a time.
It reports accepted requests, proposed changes, evaluated candidates, health
passes, published releases, and pending releases. It does not install those
updates or run a session after each request.

Each run saves `work/<mode>-<timestamp>.json`. Demo workspaces and receipts
are under `work/<mode>-<timestamp>/`; service logs are in `work/reef.log`.
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
