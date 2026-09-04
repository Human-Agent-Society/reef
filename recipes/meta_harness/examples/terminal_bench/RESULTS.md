# Reproduction result

Reef's Meta-Harness selects identically to upstream's. That claim is about
mechanism, it is proven exactly by replay, and it does not depend on either
live run.

This document is careful to separate it from a claim it does not make: that
the two arms, run end to end, arrive at the same frontier. They cannot, for
reasons that are architectural rather than accidental, and a run in which they
happened to agree would be coincidence rather than a result.

## What is claimed

| | Claim | How it is shown | Status |
| --- | --- | --- | --- |
| Selection semantics | Given the same scores, both implementations decide the same way | `replay.py` drives upstream's own `update_frontier` | **exact**, deterministic |
| Configuration | Both arms run the same executor, model, tasks, trials and iterations | this document plus `terminal_bench.yaml` | matched, deviations listed |
| Execution | The Reef arm runs the whole search end to end and produces well-formed results | run B below | demonstrated |
| Outcome parity | Both arms reach the same frontier | -- | **not claimed; not achievable** |

## Settings, matched between arms

| | Both arms |
| --- | --- |
| Proposer model | `gpt-5.6-sol` |
| Executor agent | vanilla Terminus 2 (`baseline_terminus2` upstream, empty seed tree in Reef) |
| Target model | `openai/gpt-5.6-luna` |
| Sandbox | e2b |
| Tasks | upstream's 30-task hard subset, identical lists |
| Scale | 2 trials x 5 iterations |
| Temperature | unset in both. `gpt-5.6-luna` rejects function tools unless temperature is 1; upstream's `--ak temperature=0.7` was removed, which fixes the request and matches the arms at once |

Known differences, each deliberate:

| | Upstream | Reef | Why |
| --- | --- | --- | --- |
| Proposer invocation | Codex CLI subprocess | API `ModelBinding` | Same model, different transport. A fidelity gap against "proposer tool: Codex CLI". |
| Candidate space | Python agent files (`Terminus2` subclasses) | declarative nodes (`rules`, `skill`, `agent_command`) | Architectural. This is why outcome parity is unreachable. |
| Episodes per iteration | 60 (candidate only) | 120 (candidate paired with incumbent) | Cordis pairs to cancel drift; the frontier then discards the incumbent measurement. See below. |
| Executor | -- | `local`, never `sandbox` | terminus is `self_isolating`: Harbor's container is the boundary and Docker does not nest in bubblewrap. |

## The claim: the two implementations select identically

`replay.py` imports upstream's `update_frontier` and drives it alongside
`MetaHarnessSelector`, comparing the decisions rather than a description of
them. Across 200 random score sequences, and across each arm's real measured
scores, there are no disagreements.

Run it against an upstream checkout -- upstream's module-scope imports have to
resolve, so the interpreter needs its dependencies:

    METAHARNESS_DIR=/path/to/terminal_bench_2 \
      uv run --extra dev --with python-dotenv python -m pytest \
      tests/reef_service/test_meta_harness_replay.py

One detail decides the whole comparison: the baseline must be seeded into the
frontier first. Without that, 110 of 200 sequences disagree, because upstream
starts from a seeded high-water mark and a selector that starts from nothing
gives the first candidate a free win.

## Why outcome parity is not the claim

Three reasons, in descending order of how fundamental they are.

**The arms search different spaces.** Upstream mutates Python agent files; its
candidates are `Terminus2` subclasses. Reef mutates declarative nodes. Neither
can propose the other's candidate, so the sequences of things being scored
have no correspondence.

**The proposer is stochastic.** Two runs of the same arm propose different
candidates from the same history.

**The measurement cannot resolve the difference anyway.** See the floor below:
two measurements of the *same* agent land 0.10 apart.

## The measurement floor

Reef's seed is the empty tree, which the terminus adapter renders as stock
Terminus 2 -- upstream's `baseline_terminus2`. The same agent, measured
independently by each arm on the same 30 tasks and 2 trials:

| | mean | passes |
| --- | --- | --- |
| Reef seed (run B) | 0.3833 | 23/60 |
| Upstream baseline | 0.2833 | 17/60 |

At 60 episodes the pooled standard error is 0.086, so the gap is **1.16
sigma**. Task by task, 21 of 30 agree exactly, and 8 of the 9 that differ are
one episode moving on a two-trial task, in both directions -- the signature of
noise, not of a systematic difference.

### Open question: Reef measures the seed higher than upstream, every time

Four independent measurements of the same agent:

| | run 1 | run 2 | combined |
| --- | --- | --- | --- |
| Reef seed | 0.3833 | 0.3667 | 45/120 = **0.375** |
| Upstream baseline | 0.2833 | 0.2833 | 34/120 = **0.283** |

Reef is higher in every pairing. Combined, the gap is 0.092 at about 1.5
sigma -- not significant, but consistently signed in a way noise usually is
not. On a single pairing this was called noise here; four measurements make
that call premature.

Ruled out as causes: Reef's empty seed tree renders `terminus/config.json` as
`{}` with no instruction paths and no agent kwargs, so it is stock Terminus 2
with Harbor's defaults, and upstream no longer passes a temperature either.
Upstream's retry can only raise its score, not lower it.

What remains untested is that the arms reach Harbor differently: Reef drives
`Trial` directly through reef-eval, upstream goes through `harbor run` as a
Job. Job-level and trial-level defaults are not known to be identical. That is
the next thing to check, and until it is checked neither "noise" nor
"divergence" is the honest label.

### The same configuration, run twice

Upstream's baseline was measured twice, same agent, same 30 tasks, same two
trials. Both runs returned **0.2833**. Task by task they do not agree:

```
configure-git-webserver           1.0 -> 0.5
fix-ocaml-gc                      1.0 -> 0.5
llm-inference-batching-scheduler  1.0 -> 0.5
fix-code-vulnerability            0.5 -> 1.0
path-tracing                      0.0 -> 0.5
write-compressor                  0.0 -> 0.5
```

Six of thirty tasks flipped, one episode each, three down and three up. The
identical mean is those flips cancelling exactly -- coincidence, not
stability. **Ten percent of episodes change outcome between identical runs.**

Read the matching seed measurements above with that in mind: a mean that
reproduces to four decimal places can sit on top of a fifth of the task set
disagreeing.

The consequence is larger than the table. **At 30 tasks x 2 trials the search
selects on margins smaller than its own measurement error.** Separating a real
5-point improvement would need roughly 8 trials per task, four times the
episodes. This is the honest limit of the reproduction as scoped: it shows the
implementations decide alike, not that either decides well.

### Reading the seed partition

Splitting the seed's per-task results is descriptive, not a ranking of which
tasks are worth running:

| | Tasks (upstream baseline) | Can a better candidate show it? |
| --- | --- | --- |
| Seed always passes | 6 | No -- only regression is visible |
| Seed always fails | 19 | **Yes -- this is headroom** |
| Seed sometimes passes | 5 | Yes -- and seed variance lives here |

A task the seed always fails is where a candidate has room to win, so do not
prefilter a task set down to the seed-variable tasks. `select_panel.py`
partitions; it does not recommend.

## What the live runs show

### Run B (Reef, current)

Seed 0.3833. Every candidate so far scores below it and is rejected:

| Iteration | Candidate | Seed | Decision | Wall clock | Spend |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.350 | 0.3833 | reject | 5208s | ~$4.0 |
| 2 | 0.100 | 0.3833 | reject | 2755s | -- |
| 3 | 0.050 | 0.3833 | reject | 2572s | -- |
| 4 | 0.000 | 0.3833 | reject | **74s** | ~$0.002 |
| 5 | 0.000 | 0.3833 | reject | **123s** | ~$0.002 |

**Only the seed and iteration 1 are measurements.** Iterations 4 and 5 claim
120 episodes each in 74 and 123 seconds, which is impossible for tasks whose
median timeout is 1800s, and they spent about $0.002 where a real episode
costs roughly $0.02. They did not reach the model. Iterations 2 and 3 took
real time but their cost is far below iteration 1's, so they are doubtful and
are not reported as results either.

`episode_failures` read 0 for iterations 4 and 5 and did not catch this. That
metric counts an episode that produced *no score*; an adapter that exits
before reaching the model still yields a reward of zero, which is a score. The
driver now also counts episodes that produced a score without costing
anything, since cost is the ground truth for whether an episode called the
model, and stops the run on the combined figure.

What run B establishes is that the pipeline executes end to end and that the
selector rejects every candidate it should. It does not establish how the
proposed candidates perform, because four of five iterations did not measure
them.

### Diagnosing the fast-failing episodes

Ruled out by running them again, each a real episode on `password-recovery`
against `gpt-5.6-luna` on e2b:

| Suspect | Test | Result |
| --- | --- | --- |
| The seed composition | empty tree, one episode | exit 0, reward, 193s |
| The candidate composition | iteration 5's `rules` node, one episode | exit 0, reward 1.0, 151s |
| The adapter binary resolving the wrong worktree | `reef-terminus` from a neutral cwd | resolves `/Users/dobby/reef-pr4/reef`, retry helpers present |

So it was neither the candidates nor the tooling. The remaining explanation is
the environment: iterations 4 and 5 ran while the shared e2b account was at
100/100, and a sandbox that cannot start makes an episode fail in seconds. That
is circumstantial -- the timing fits and nothing else survives -- and it is not
proven, because the run recorded no cause.

**Why nothing caught it.** Cordis marks an episode `None` only when
`run_episode` raises; the scorer must return a float, so an episode that ran
and produced no reward becomes a zero. The runner does exit 1 in that case and
Cordis records a `FailureObservation` with stage `exit` -- the signal existed
the whole time, in `candidate_failures`/`current_failures`, and the driver
dropped it by logging only scalars. It now records the observation count and a
sample of causes per iteration, and stops on them. A recurrence will name
itself.

Reproducing a single episode needs the runtime venv, which has `reef-eval`:

    OPENAI_API_KEY=... E2B_API_KEY=... REEF_TERMINUS_ENVIRONMENT=e2b \
    REEF_REAL_TERMINUS_TASK=terminal-bench/password-recovery \
    REEF_REAL_TERMINUS_MODEL=openai/gpt-5.6-luna \
    PYTHONPATH=$PWD /path/to/runtime-venv/bin/python -m pytest \
    tests/smoke/test_real_terminus.py

### Upstream

Baseline 0.2833, measured cleanly. Its first candidate iteration was lost to a
full sandbox account (below) and the arm was stopped. Nothing in the claim
depends on it; it is worth rerunning for a cost and throughput comparison when
the shared account is quiet.

### Run A is not a result

An earlier Reef run reported seed 0.2167 rising to a 0.350 frontier. It was
measured while 37% of its episodes never started, and those score 0:

| | seed | episodes that never ran |
| --- | --- | --- |
| Run A | 0.2167 (13/60) | 37% |
| Run B | 0.3833 (23/60) | 9% |

Dividing passes by the episodes that actually ran gives ~0.34 and ~0.42, about
one standard error apart. The seed is near **0.40**; both figures understate
it and run A badly. So run A's "13-point improvement" was not one -- iteration
2's candidate at 0.350 was selected because the incumbent had been depressed
by an outage, and the five tasks credited to it as "unlocked" were tasks the
seed failed on infrastructure. Two of them, `fix-ocaml-gc` and
`sparql-university`, pass cleanly for the seed in both run B and upstream's
baseline.

Run A's value is as the case that produced the failure accounting, the outage
guard and the retry. These corrections are estimates: the failure share is
counted across each iteration's full 120 episodes, so a specific zero cannot
be attributed to a specific episode.

## Operating on a shared, oversubscribed account

The e2b account is shared with other teams and this project's slice is 32 of
its 100 sandboxes. The slice is a convention, not a quota: other teams have
been observed holding 68-71, so 32 + their usage exceeds the cap and a job can
find no capacity at all.

What that broke, and what now guards it:

| | |
| --- | --- |
| An episode that cannot get a sandbox scores 0 and enters the mean | `run.py` records `episodes` and `episode_failures` per iteration and stops when most episodes produce no score |
| Upstream averaged 3 completed trials against 57 `RateLimitException`s and logged "no improvement" at 0.033 | `check_job_health`/`OutageDetected` in the fork, at the `harbor_run` chokepoint |
| A brief spike starves an episode | the terminus runner retries on rate-limit and quota errors with jittered exponential backoff; upstream passes `--max-retries` with `--retry-include RateLimitException` |
| Retry does not survive a sustained outage | it fired and exhausted all three attempts against a full account. Retry is necessary, not sufficient; the guards are what stop a bad number being recorded |
| Upstream reported cost but never stopped | `charge_job`/`SpendCapReached`, ported from Reef's `ObservedCostLedger` |

**Reaping is not a remedy.** The quota is not ours to reclaim. Killing
sandboxes on age alone here destroyed 97, most of them other teams' running
work. `reap_sandboxes.py` filters on `environment_name` against the shipped
task list before it considers age, and no flag disables that half.

## Reef pays twice for every iteration after the first

Cordis evaluates in pairs: for each task and repeat it runs both
`candidate_files` and `current_files`, so an iteration costs 120 episodes, not
60. The interleaving is deliberate -- drift lands on both sides of a pair.

The Meta-Harness frontier does not want that measurement. An incumbent keeps
the score it was admitted on:

```python
incumbent_scores = incumbent.scores or current_scores
```

Once `incumbent.scores` is set, the fresh `current_scores` is discarded. The
live runs show it exactly: each iteration's `incumbent_mean` is the previous
`candidate_mean` to the digit. Iteration 1 is real work -- `served.scores`
starts empty, so that pairing is what gives the seed its score. From iteration
2 on, 60 target-model episodes per iteration are executed, billed and thrown
away: 240 of 600 across a five-iteration run.

Selections are unaffected, which is why the arms still agree. Skipping the
incumbent leg when `incumbent.scores` is set would halve the cost of every
run, at the price of making Cordis's pairing a no-op for this method. That
belongs to whoever owns the selector.

## The task set

`harbor dataset download 'terminal-bench@2.0'` returns all 89 official tasks
in a few seconds; `tasks-89.txt` records them with the command that
regenerates it. Upstream's 30-task hard subset is a strict subset, so **59
tasks have never been evaluated in either arm**. Both entry points take
`--tasks-file`, which reads commas or newlines and ignores `#` comments.

Every task declares an `agent.timeout_sec` cap. These bound the tail rather
than predict a runtime:

| Group | Tasks | Median cap | Longest |
| --- | --- | --- | --- |
| All official | 89 | 900s | 12000s |
| Upstream's hard subset | 30 | 1800s | 7200s |
| The other 59 | 59 | 900s | 12000s |

The hard subset is the slower half by median, so a sample drawn from all 89
runs faster per task than the runs above, not slower. One task dominates the
tail: `build-pov-ray` at 12000s against a 900s median, enough to set an
iteration's wall clock by itself at concurrency 16. No task requests a GPU, so
e2b covers the whole set.

Each phase blocks on its slowest task, which is what sets the wall clock: a
60-episode baseline took an hour because 58 trials finished quickly and one
ran 51 minutes.

## Defects this found

Each produced a plausible result rather than an error, which is why running it
mattered:

| Defect | How it presented |
| --- | --- |
| Proposer prompt named the node vocabulary but not each kind's config shape | first proposal rejected for a missing `text` |
| `reef-terminus` not installed | a "successful" 7.5s iteration with both sides scoring 0.0 |
| `score_episode(result)` vs Cordis's `(task, result)` | TypeError mid-evaluation |
| Spend guard never called `record_trial` | \$0.00 recorded against a \$3 cap |
| Codex blocks on stdin after finishing | session hung to timeout with its files already written |
| `terminal-bench` pinned to rewritten history | both upstream baselines crashed in 9s |
| KIRA's native function calling unsupported on the target model | evaluation raised `BadRequestError`; settled the KIRA-vs-terminus2 question |
| Harbor's local Docker default collects no verifier reward | every trial failed until the environment was set to e2b |
| `--ak temperature=0.7` with function tools on `gpt-5.6-luna` | both baselines scored 0.0 from `BadRequestError`, reading as a failed agent |
| A full sandbox account | an iteration scored 0.017, then 0.033, reading as collapsing candidates |
| Driver logged only means and timings | an outage and a weak candidate were indistinguishable for two hours |
| Replay tests skipped on a missing checkout but failed on missing upstream deps | five red tests in the file carrying the reproduction claim |
