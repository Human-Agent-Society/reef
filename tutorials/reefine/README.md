# Reefine: refine your harness on reef-pi

Reefine turns plain-language requests into versioned harness changes: skills,
rules, agent commands, or pi extensions. This tutorial runs two demos and
measures how many requests pass evaluation.

The service proposes and evaluates each change. A session after installation
shows whether the agent follows the requested behavior.

## Quick start

Use the repository environment from the [development guide](../../docs/contributing/development.rst).
The `python3` on your PATH must import `reef` and `reef_client`.
Start an OpenAI-compatible model endpoint with the model available.
Keep port `8901` free.

```bash
# From the repository root:
source .venv/bin/activate
cd tutorials/reefine
uv pip install -e .

# These are run.sh's defaults. Change them for your endpoint.
export REEF_UPSTREAM_URL=http://127.0.0.1:11434  # No /v1 suffix.
export REEF_UPSTREAM_MODEL=gemma4:26b
export REEF_UPSTREAM_API_KEY=dummy              # Use your endpoint's key.

./run.sh bugfix
```

`run.sh` starts Reef with [configs/deployment.yaml](configs/deployment.yaml),
installs the harness, runs the demo, and stops the service on exit.
The deployment uses token `reef-local`. If you set `REEF_TOKEN`, keep it equal
to the deployment token.

The service installs the pinned pi version under `~/.local/share/reef-harness/pi`
on first start. The harness goes into `work/harness/`. The wrapper goes into
`~/.reef/installs/`, and `~/.local/bin/reef-pi` links to it.

Local models can take several minutes per request. The script defaults to
`REEF_PROPOSER_TIMEOUT_S=900` and `REEF_PROPOSER_MAX_TOKENS=16384`.
Set these variables before running the script to override them.

## Run the demos

| Command | Requested behavior | Session after installation |
| --- | --- | --- |
| `./run.sh bugfix` | Reproduce a bug with a failing test, fix it, run tests, then request a second agent's review. | Fix `adder.py` in a copy of [demos/workspace/](demos/workspace/). |
| `./run.sh research` | Search for papers, download and read them, then answer with citations. | Explain the comparison-sorting lower bound with a source. |

The exact requests are in [demos/bugfix.md](demos/bugfix.md) and
[demos/research.md](demos/research.md). Each demo:

1. Submits the request with `reef-pi evolve`.
2. Waits for the proposal and evaluation result.
3. Prints the changes and timing.
4. Promotes a pending release, if present.
5. Runs `reef-pi setup --yes` for declared requirements.
6. Installs a selected or promoted release.
7. Runs a session and prints its tool calls and final answer.

If no release is selected, the session uses the previously installed harness.
An unmet setup requirement or refused installation stops the demo with exit code `2`.

**The demos automatically promote extensions and run setup checks.**
Extensions run with your privileges. For manual use, read the version page
before promotion and inspect the declared requirements before setup.

## Understand the result

The deployment uses manual training: each accepted instruction runs one step.
Failed session reports do not trigger additional steps.

The proposer designs the change, writes entries, and reviews them against the request.
The service checks the entries, then evaluates the candidate with `selection: floor`.
The health task must run `echo reef-ok` through the shell tool and return its output.

- `1 / 0` means the health task passed; `0 / 1` means it failed.
- `selected` means the release was published.
- `pending` means the release needs promotion because it changes a `code_extension`.
- `rejected` means evaluation rejected the candidate.
- `skipped` includes the reason no change was evaluated.

Passing the health task does not prove that the requested behavior works.
Read the design and review notes with `reef-pi page <version>` or `/versions <version>`.
Check the session's tool calls against the request.

Declared `requires` items can be incomplete or incorrect. Review the version
page for unmet requirements and undeclared variables. Generated extensions
can use `fetch` and system commands. Additional npm dependencies are not installed.

## Measure requests

```bash
./run.sh measure          # 10 requests by default.
./run.sh measure --n 12   # All 12 requests in the fixed list.
```

Measurement submits skill and rule requests one at a time.
It does not promote or install their releases, or run a session after each request.

| Count | Meaning |
| --- | --- |
| Filed | Requests accepted by the service. |
| Answered | Results containing a proposed change. |
| Admitted | Candidates evaluated. |
| Won | Candidates that passed the health task. |
| Published | Releases published. |
| Pending | Releases awaiting promotion. |

These counts measure proposal and evaluation outcomes, not compliance with each request.

## Inspect saved results

| Path | Contents |
| --- | --- |
| `work/reef.log` | Service log. |
| `work/<mode>-<timestamp>.json` | Run results and catalog rows. |
| `work/<mode>-<timestamp>/` | Demo workspace, session receipts, and any saved pending-release page. |
| `work/deployment/steps/` | Proposer replies, parsed changes, and evaluation records. |

Runs reuse the state in `work/deployment/`. Queued requests from an interrupted
run can finish before a new request. To start a fresh chain, move `work/` aside
before running a demo.

## Historical results

The recorded runs from September 6–8, 2026 used `gemma4:26b` on a Mac mini M4
with 32 GB of memory. They used three arithmetic tasks and `selection: always`.
Publication therefore did not require an evaluation win. These results do not
validate the current health task.

- Bug-fix sessions reproduced, fixed, and tested the bug, but did not obtain a second agent's review.
- The first research session answered from memory. A later session used a local search tool, but downloaded and read no paper.
- No recorded run produced an extension.

The four measurement runs on the code from PR #315 reported:

| Run | Filed | Answered | Admitted | Won | Published | Median request time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 2 | 0 | 2 | — |
| 2 | 10 | 3 | 3 | 1 | 3 | 248.0 |
| 3 | 10 | 8 | 8 | 1 | 8 | 379.8 |
| 4 | 10 | 10 | 10 | 0 | 10 | 427.2 |

Here, `Won` means more arithmetic-task wins than losses. Run 1 stopped after
a driver error; its counts came from catalog rows, and its median was unavailable.
Parser and request-handling code changed between runs. Runs were not repeated,
so the table does not establish a success rate or a performance comparison.

## Use the built-in recipe

Reefine also ships in `reef-infra`. To start its built-in service profile:

```bash
reef serve --recipe reefine --model ollama/gemma4:26b
```

The profile listens on `127.0.0.1:8901` and stores state under `.reef/reefine/`.
Set `REEF_TOKEN` to require a Bearer token; otherwise, this loopback service
has no authentication. This profile differs from the tutorial deployment.
See the [Reefine guide](../../docs/user-guide/recipes/reefine.rst) for configuration.
