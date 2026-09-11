# Harness requests on reef-pi

A person asks their coding agent for a capability in plain words, from the shell (`reef-pi harness "..."`) or from inside a pi session (`/reef-harness ...`), and the ask posts a training instruction to reef (`POST /reef/train`) with the installed release and the session it came from as provenance. Nothing in the session writes the change: the agent side only asks.

The service writes it. The deployment runs in `training_mode: manual`, so one evolve step runs for each accepted instruction: it hands the request to the recipe's proposer, the served model, which reads the current tree, the request and the pi extension API reference and answers with the change the request names: a skill, a rules entry, an agent command or a pi extension. Admission screens it, the gate runs it against the current tree on the recipe's tasks, and the catalog row carries the request (`metrics.training_request`), the mutations and the verdict, with one page per step that says why the version exists and what it changed.

The person promotes what runs as code. A release that touches a `code_extension` waits as pending until a person reads its page and promotes it; a release whose `requires` items (a permission, a variable, a service) are not checked off with `reef-pi setup` is never installed. The demos here script that path end to end on this machine and record what the model did, working or not; this is RFC #310's stage 5, and `./run.sh measure` counts the requests that won the gate, the first of the two measurements its stage 6 names before the recipe promotion (the held out shapes are not here).

## Directory layout

```text
harness-requests/
  README.md          this file: what the tutorial shows, how to run it, what it saw
  run.sh             starts reef serve on configs/deployment.yaml, waits for /healthz,
                     installs the served tree under work/harness, runs run.py <mode>,
                     stops the service; usage: ./run.sh bugfix | research | measure
  run.py             the driver: ask -> step -> promote if pending -> setup if required
                     -> install -> show; and the measurement
  configs/
    deployment.yaml  the pi deployment of tutorials/evolve-your-harness with requests,
                     version_check and review_kinds: [code_extension], training_mode
                     manual, selection: always, the same three tasks, and every path
                     under tutorials/harness-requests/work/
  demos/
    bugfix.md        the bug fix flow request and the workspace fixture
    research.md      the research loop request
    workspace/       a tiny Python project with one failing test (sum_to stops one short)
  pyproject.toml     makes the directory installable; the method package stays in
                     tutorials/evolve-your-harness/harness
  work/              the runs: reef.log, harness/ (the installed tree), captures/,
                     <mode>-<timestamp>.json and <mode>-<timestamp>/ (the page, the show
                     session's receipts and workspace); not committed
```

## Quick start

```bash
cd tutorials/harness-requests
pip install -e .          # reef-client for run.py; reef itself runs from the checkout
export REEF_UPSTREAM_URL=http://127.0.0.1:11434   # an OpenAI compatible endpoint, no /v1 suffix
export REEF_UPSTREAM_MODEL=gemma4:26b             # a model that endpoint serves
export REEF_UPSTREAM_API_KEY=dummy                # the endpoint's key; anything for a local ollama
./run.sh bugfix
```

The three variables are `run.sh`'s defaults, so with ollama on this machine `./run.sh bugfix` alone runs. `run.sh` also sets `REEF_PROPOSER_TIMEOUT_S=900` and `REEF_PROPOSER_MAX_TOKENS=16384`: the method package gives one proposer call 120 s and 4096 reply tokens for a request (60 s and 2048 for a failure step), a local model of this size needs minutes, and a thinking model spends the reply budget on its reasoning first, so the smaller budget came back empty. `run.sh` refuses when a reef already answers on `127.0.0.1:8901`, the port `deployment.yaml` uses. The `python3` on your PATH must import reef (the checkout's environment) and have the pinned pi under `~/.local/share/reef-harness/pi`, which the service installs at its first start. The install step points `~/.local/bin/reef-pi` at `work/harness/reef-pi`, as every install through the install route does.

## What each demo does

### Bug fix flow

The request, from [demos/bugfix.md](demos/bugfix.md): when I ask you to fix a bug: reproduce it first with a failing test, fix it, run the tests, then have a second agent review the diff before you tell me it is done. A working answer names the four steps in order as a rules entry or a skill, or adds a `/fix-bug` command, or writes an extension that runs a second session over the diff. `run.py bugfix` posts the request with `reef-pi harness`, polls the catalog until the row that carries the request under `metrics.training_request` settles, prints the verdict, the mutations and the proposer's seconds, promotes a pending release after printing its page URL (the demo is scripted; a person reads the page first), runs `reef-pi setup --yes` when the head names `requires` (an unmet item stops the demo, exit 2), installs the head through the install route, then runs the show session: `reef-pi -p "fix the bug in adder.py"` in a copy of `demos/workspace/`, and prints the session's tool calls in order and its final answer. Under the change the session should run the tests and see the failure, edit `adder.py`, run the tests again, and review the diff before it says it is done.

### Research loop

The request, from [demos/research.md](demos/research.md): when I ask a research question, first search for the relevant papers, download and read them, then answer with citations. A working answer says to search and read before answering and to cite what was read, or writes an extension that fetches papers into the workspace first; an extension that needs a search service names it under `requires`. The steps are the bug fix flow's; the show session is `reef-pi -p "what is the best known lower bound for sorting by comparisons, with a source"` in an empty directory, and it should search, fetch at least one source, and answer with a citation.

## The measurement

```bash
./run.sh measure          # --n 10 by default, up to the fixed list's length
```

`run.py measure` posts the requests of its fixed list one after another (skills and rules, no extension: "answer in one sentence when the question is arithmetic", "always show the command you ran before you show its output", and so on), each once the step of the one before it settled, since manual mode runs one step per accepted instruction and no failure driven step between them, and prints one row per request (request, kind proposed, verdict, W / L / T, seconds) and the totals: filed, answered (a mutation came back), admitted (the gate ran), won (more wins than losses in the recorded verdict), published, pending. Under `selection: always` a publish says nothing about the gate, so won and published are counted apart; "requests that won the gate" is the won column.

## Environment

| Setup | Host | Model server | Model | Agent |
|---|---|---|---|---|
| Mac mini M4, 32 GB | one machine, service and agent | ollama at `127.0.0.1:11434` | `gemma4:26b` (26B with 4B active, 18.6 GB) by default; `qwen3.8:27b` (dense, 17.7 GB q4) is the slower alternative | pi 0.84.2 through `reef-pi` from the checkout, `configs/deployment.yaml` |

Rows 1 to 4 of the Runs table and rows 1 to 3 of the measurement table ran on the request store implementation of the earlier stage 5 head, commit 7e3982bb, and the working states before it that the notes describe, where the ask filed a request with `POST /reef/harness/requests` and reported a session's receipts so a step ran for it. Those rows stay as they were measured. The rows after them ran on the code of this pull request as it stands: the ask is a training instruction on `POST /reef/train` with the deployment in `training_mode: manual`, and no session runs before the ask. Rows dated 2026-09-08 ran on main after the stack merged, at commit 00d40ef2, with a fresh `work/`.

## Runs

Every row was measured on the code of the pull request in its Code column, with the tutorial files as they stood there; the model is the one the row names. One run is one sample and no run was repeated, so the rows carry no spread. Dates are this machine's local clock (KST) at the run's start. The gate still runs and records its verdict; under `selection: always` the demo publishes on any verdict, and a `code_extension` still waits for a promote. Proposer is the step's proposer call as the catalog row records it. Ask to install counts from the `reef-pi harness` call to the end of the install, or to the verdict when nothing new installed; on 7e3982bb the driver ran a ready session before the ask (note 10), and the rows measured there count from the start of that session.

| Demo | Model | Run | Date | Code | Proposal | Verdict | W / L / T | Pending | Promoted | Requires | Show session | Proposer (s) | Ask to install (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix | `gemma4:26b` | 1 | 2026-09-06 | #315 | none [1] | skipped: no proposal | - / - / - | no | - | nothing | bash ls -R, read adder.py, read test_adder.py, bash pytest, edit adder.py, bash pytest [2] | 183.6 | 196.7 |
| bugfix | `gemma4:26b` | 2 | 2026-09-07 | #315 | none [3] | skipped: no proposal | - / - / - | no | - | nothing | bash find, read adder.py, write test_adder.py, bash python3 test_adder.py, edit adder.py, bash python3 test_adder.py [2] | 165.5 | 176.0 |
| bugfix | `gemma4:26b` | 3 | 2026-09-07 | #315 | create bug-fix-protocol (rules) | selected [4] | 0 / 0 / 3 | no | - | nothing | bash find, read adder.py, write test_adder.py, bash python3 test_adder.py, edit adder.py, bash python3 test_adder.py [5] | 157.7 | 554.3 |
| research | `gemma4:26b` | 1 | 2026-09-07 | #315 | update answer-style (skill) [6] | selected | 0 / 0 / 3 | no | - | nothing | bash (a comment, no command) [7] | 194.7 | 417.3 |
| bugfix | `gemma4:26b` | 4 | 2026-09-07 | #315 | create bug-fix-protocol (rules) | selected [17] | 0 / 0 / 3 | no | - | nothing | bash find, read adder.py, write test_adder.py, bash python3 test_adder.py, edit adder.py, bash python3 test_adder.py | 207.6 | 428.2 |
| bugfix | `gemma4:26b` | 5 | 2026-09-08 | #315 | create bug-fix-workflow (rules) | selected [19] | 0 / 0 / 3 | no | - | nothing | bash find, read adder.py, write test_adder.py, bash python3 test_adder.py, edit adder.py, bash python3 test_adder.py | 188.2 | 425.2 |
| research | `gemma4:26b` | 2 | 2026-09-08 | #315 | create research-workflow (rules) | selected [20] | 0 / 0 / 3 | no | - | nothing | read smart-search SKILL.md, bash opencli list, bash opencli gemini -h, bash opencli gemini ask | 183.1 | 379.6 |
<!-- rows -->

### Measurement runs

Every row is one `./run.sh measure` run on the code of the pull request in its Code column; Parser names the entry parser the service ran, since the parser changed between runs. Requests is how many the run filed; Answered how many rows carry a mutation (the parser took the reply and admission let it through; a refused reply and a step whose instruction failed carry none); Admitted how many the gate ran; Won how many had more wins than losses; Published how many released; Skipped how many steps took nothing from the reply. Median is the seconds from the ask to the verdict over the run's requests.

| Run | Model | Date | Code | Parser | Requests | Answered | Admitted | Won | Published | Skipped | Median (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 [14] | `gemma4:26b` | 2026-09-07 | #315 | before the null id fix | 2 | 2 | 2 | 0 | 2 | 0 | - |
| 2 [15] | `gemma4:26b` | 2026-09-07 | #315 | before the null id fix | 10 | 3 | 3 | 1 | 3 | 7 | 248.0 |
| 3 [16] | `gemma4:26b` | 2026-09-07 | #315 | null id fix | 10 | 8 | 8 | 1 | 8 | 2 | 379.8 |
| 4 [18] | `gemma4:26b` | 2026-09-07 | #315 | current, training record path | 10 | 10 | 10 | 0 | 10 | 0 | 427.2 |
<!-- measure rows -->

## Reading

- A row's Proposal column names the mutations the step recorded (`op id (kind)`); a `skipped` verdict with no mutation means the model answered nothing the parser took (`skipped: no proposal`) or admission refused what it took (the verdict names the refusal), and the step record under `work/deployment/steps/` holds the reply and the parsed mutations either way.
- W / L / T is the gate on the recipe's three tasks: a workflow change is expected to tie all three, so the column shows whether the change moved the tasks, not whether it works.
- The Show session column is the tool sequence of the session after the install, from the receipts the wrapper spooled; whether the change works is read there.

## Reproduce

```bash
cd tutorials/harness-requests
# the demos: ollama on 127.0.0.1:11434, the model in the row's Model column
REEF_UPSTREAM_MODEL=gemma4:26b ./run.sh bugfix
REEF_UPSTREAM_MODEL=gemma4:26b ./run.sh research
# the measurement
REEF_UPSTREAM_MODEL=gemma4:26b ./run.sh measure --n 10
```

Every run leaves `work/<mode>-<timestamp>.json` with the catalog rows it read and the result it printed, so a README row can be checked against the record's catalog rows. `work/` keeps the commit log across runs: a second run on the same `work/deployment/` continues the same chain, so a rerun of a fresh chain removes `work/` first.

## Notes

1. Run 1's proposer reply was empty: the step record (`steps/harness-requests-demo/1/proposer.json` under the run's deployment directory) shows a 4096 token reply budget and no text (the record has no finish reason field). The model's benchmark reply opens with its reasoning, so the budget going to the reasoning is the likely cause, not a recorded one. `run.sh` sets `REEF_PROPOSER_MAX_TOKENS=16384` since; run 1 ran with the method package's default.
2. The show sessions of runs 1 and 2 ran on the seed tree, since no release published: each found the file, ran a test that failed (run 2 wrote its own instead of reading `test_adder.py`), edited `adder.py`, ran the test again, and answered done without a second review, which is what the request asks to add.
3. Run 2's reply, under the larger budget, was one `rules` entry for the request, with its kind under the key `kind`; the parser read `name`, as the prompt's schema line says, and took nothing. The parser accepts `kind` as well since, and the prompt says which key the kind goes under.
4. Run 3's reply was a `rules` entry that names the four steps of the request in order. The gate ran both trees on the three tasks and both passed every task (three ties, as a workflow change is expected to); `selection: always` published it as release `0df7ee42`, and the install put it on the tree the show session ran on. The W / L / T column is counted from the per task scores the catalog row records, since `selection: always` records no counts of its own; for runs 3 and the research run the driver of the day printed no count, and the column was counted from the rows in their records afterwards.
5. Run 3's show session, on the tree with the rules entry, wrote a failing test first, ran it, fixed `adder.py`, ran the test again and reported those steps in order; it did not have a second agent review the diff. Nothing in the tree gives it one: that step needs an `agent_command` or a `code_extension`, and no run has produced either yet.
6. The research run's reply rewrote the seed's `answer-style` skill into a three step protocol (search for the papers, download and read them, answer with citations): a text change, not the `code_extension` with a search tool the request names. The gate tied all three tasks and `selection: always` published it as release `02717d57`.
7. The research show session, on that tree, made one tool call: a `bash` call whose command was a comment saying no command was needed and it would search its own knowledge, then answered from memory with a textbook citation (the decision tree bound, Cormen et al.). It had `bash`, and searched nothing and downloaded nothing: the row shows what a skill can and cannot do for a request that names a tool, and whether a local model writes the extension form is the question RFC #310 lists under its risks.
8. A first measurement run, under the parser as it stood after note 3, was stopped during its second request: its first reply was a `rules` entry with its `text` beside the id instead of under `config`, a third shape of the same answer, and the parser took nothing again. The parser reads the config fields from beside the id as well since; the measurement table above ran with that parser.
9. A request is a training record on the service, and manual mode runs the accepted instructions oldest first: a run stopped while its step is in flight leaves its instruction queued, the next run's first step goes to it, and `run.py` prints that row as an earlier request's and waits for the row whose `training_request.text` is its own. On 7e3982bb the request lived in a store under `proposals_dir` (the default `.reef/proposals` under the service's directory, outside `work/`, until `configs/deployment.yaml` put it under `work/`), a second measurement run's first step went to a stopped run's request, and the driver of the day reported one more session so the next step ran; the store is gone, `proposals_dir` is the proposal inbox alone, and the driver has no such branch. The measurement table above ran on a fresh `work/deployment/`, with the demo chain kept beside it.
10. The ask form: `reef-pi harness "<request>"` posts the text to `POST /reef/train` with the installed release id from the sidecar and, as provenance, the id of the oldest spooled session or a fresh one, prints `training request <id> accepted`, and the deployment's `training_mode: manual` runs one step for it, with no receipts and no report. On 7e3982bb the ask filed a request with `POST /reef/harness/requests` and reported the receipt of a `reef-pi -p "Reply with the single word ready."` session the driver ran first, so the step ran with that batch; a request with no receipt behind it waited for a batch no session sends, and the driver stopped there. The new path needs no session before the ask, so the driver runs none.
11. The install script checks `python3 -c 'import reef_client.serve, reef.harness.client.wrapper'` and writes that interpreter's absolute path into `reef-pi`, and the wrapper reads the token back from `models.json`, so a shell that runs `reef-pi` later needs neither the venv on its PATH nor `REEF_TOKEN`. `run.py` puts its own interpreter first on the PATH and the checkout on `PYTHONPATH` when it runs the script, so an editable reef that points elsewhere does not get in the way.
12. The show session's tool calls come from the receipts the wrapper's proxy captured and spooled at exit, moved into the run directory as the run's own record; what stays in the spool is for `reef-pi report` to claim, and `reef-pi harness` names its oldest session as a request's provenance and nothing more. pi's own session file lands under its default directory, outside the install root.
13. The measurement never promotes or installs: skills and rules publish at once, and the installed tree stays the one `run.sh` installed; the measurement runs no session, so that tree is not used while the chain moves on the service. Each request is posted once the step of the one before it settled, and manual mode runs no failure driven step between them, so the rows are one step per request.
14. Measurement run 1 stopped at its second request of ten: its second step had a candidate episode that timed out (the fib task, a loss for the candidate, so the tally reads one win, one loss, one tie), the driver of the day compared that episode's missing score as a number and crashed, and the record of the run was never written, so the row is read from the catalog rows of the chain. Both requests it filed were answered and published; the median is absent because the second request's ask to verdict clock died with the driver.
15. Measurement run 2 ran all ten requests on the same chain. Seven steps took nothing from the reply: six replies were a `rules` entry with `"id": null`, copied from the tree listing, where a rules entry shows a null id because it has no name of its own, and one had no id key and its text beside the kind. The three answered requests were two `rules` entries and one skill; one won the gate outright, the other two lost one task each and published under `selection: always`. The parser gives a `rules` entry with a null id one from its text since, and the prompt says every entry needs an id; the next row ran on that parser.
16. Measurement run 3 ran the same ten requests on the same chain with the null id fix: every reply came back as a mutation, eight ran the gate and published, one of them with a win on one task. The two steps that took nothing were admission refusals, not parser refusals: the reply reused an id the tree already carried (the seed skill's name for a `rules` entry, and the id a rules entry of run 2 already had for the same request). The proposer sees the tree's entries as kind and body, without their ids, so it cannot tell that a rules id is taken; a `rules` entry that reuses another kind's name takes an id from its text since, and a reused rules id still stops at admission. Those two rows carry no mutation, so the Answered column counts them as not answered.
17. The first run on the training record path (this pull request's code): `reef-pi harness` was accepted as a training instruction with no session before it, manual mode ran the step at once, and the proposer wrote a `rules` entry with run 3's id and kind and a shorter text; the gate tied every task and `selection: always` published it; the show session, on the installed tree, reproduced the bug with a test, fixed it and ran the test again, three of the four steps the request names, as in run 3.
18. Measurement run 4 is the first on the training record path (this pull request's code): the ten requests were posted one after another with no session before any of them, manual mode ran one step per request, every reply came back as a mutation the parser took (seven `rules` entries and three skills) and every step published under `selection: always`; none won the gate outright, nine tied all three tasks and one lost one task. The parser fixes of notes 3, 8, 15 and 16 are all in this code. Run 3's two admission refusals did not recur: `brevity` because this chain is fresh, `answer-style` because this reply updated the seed skill instead of naming it for a rules entry, a shape the parser now renames anyway.
19. Bug fix run 5 and research run 2 are the first runs on main after the stack merged (commit 00d40ef2, a fresh `work/`): each ask was accepted with no session before it, manual mode ran one step per ask, both replies were a `rules` entry the parser took at once, both gates tied on the three tasks and published, and nothing waited for a promote or named `requires`. The bug fix show session wrote and ran a failing test, fixed `adder.py` and ran the test again; it did not have a second agent review the diff, as in note 5.
20. The research show session read the machine's own `smart-search` skill, listed the opencli tools and asked gemini through opencli for the bound, then answered with the decision tree argument and a textbook citation (CLRS, chapter 8): a search through a tool this time, unlike note 7, though no paper was downloaded or read.

## Known limitations

- `selection: always` is what makes the demos publish: the gate scores the candidate on the three arithmetic tasks, which a workflow change does not move, so every pairing ties and the score comparison would reject. The gate still runs and its verdict is recorded; RFC #308's acceptance tasks are the real fix, and the recipe promotion in RFC #310's stage 6 waits for them.
- A local model of this size may not write a working pi extension from the API reference in one step; the rows record what it wrote and what the show session did, working or not.
- `requires` items are the model's word: the proposer names what its extension needs, `reef-pi setup --yes` runs the checks it wrote without a person reading them first (the scripted demo), and nothing verifies that the list is complete or right.
- An extension cannot import an npm package: an extension the proposer writes has `fetch` and system commands, and a dependency mechanism is a later RFC beside `requires`. No run here has produced an extension yet.
- The proposer sees the tree's entries as kind and body and not their ids, so a reply that reuses the id of an existing `rules` entry is refused at admission as an existing entry and the step skips; the measurement's third run shows two.
- `training_mode: manual` takes instructions only: the deployment learns nothing from a failed session's report between requests, which is what keeps one step per request in the measurement. The other tutorial's deployment runs in `hybrid` for a person who wants both.
- Wall clocks are one machine's: the proposer and the six gate episodes share one model server, so seconds compare within a model, not across.
