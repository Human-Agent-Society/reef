# Handoff: Terminal-Bench 2 reproduction (#42, PR 4)

Branch `simon/tb2-reproduction`, 66 commits ahead of main. No GitHub PR opened
yet. Read `RESULTS.md` first -- it is the reviewer-facing report and is kept
honest about what is and is not shown.

## The claim, and the one thing that is settled

**Reef's Meta-Harness selects identically to upstream's, given the same
scores.** Proven exactly, deterministically, and independently of any live
run, by `replay.py` driving upstream's own `update_frontier` alongside
`MetaHarnessSelector`. 200 random score sequences plus both arms' real scores,
no disagreements.

    METAHARNESS_DIR=/path/to/terminal_bench_2 \
      uv run --extra dev --with python-dotenv python -m pytest \
      tests/reef_service/test_meta_harness_replay.py

Note the `--with python-dotenv`: upstream imports its own deps at module
scope, and without them the tests skip (they used to fail, which read as a
mismatch).

**Outcome parity is not claimed and is not achievable.** The arms mutate
different things -- upstream rewrites Python agent files, Reef mutates
declarative nodes (`rules`, `skill`, `agent_command`) -- so they can never
propose the same candidate.

## Where things stand

| | Result |
| --- | --- |
| Reef arm (run C, `/tmp/tb2-reef5`) | **finished**. Seed 0.3667, five candidates, none selected. 531 episodes, $14.24 |
| Upstream arm (`/tmp/upstream5.log`) | **stopped** by the outage guard in iteration 3. Frontier 0.350, $45.39 |
| Sandbox reaper (`/tmp/reaper2.log`) | running, `--older-than 9000 --watch 600` |

Both arms are finished. Upstream stopped on the outage guard before it
reached its spend cap, so the cap question is moot.

**A guard I added is wrong, and this is the second thing to fix.** Both arms
were halted by a candidate that slowed episodes until they stopped completing
-- Reef's iteration 5, upstream's `stalled_terminal_watchdog` with 32
`TimeoutException`. That is a candidate to reject, not a reason to stop the
search, and the guard's message misdiagnoses it as an environment problem. The
signal to separate them exists and is unused: under Cordis's pairing an outage
takes down the incumbent's episodes too, while a destructive candidate leaves
them healthy. Halt only when both sides fail; otherwise reject and continue.

## The finding that matters most for the Reef numbers

`--episode-timeout-s` defaulted to **1800s**, covering the whole adapter
invocation (build + agent + verifier), while 18 of the hard subset's 30 tasks
declare an agent timeout of 1800s or more, up to 7200s. Upstream allows each
job 28800s. So Reef cut long tasks short and scored them zero -- about a sixth
of every iteration.

Fixed to 14400s, but **the finished Reef run used the old value**, so its
figures are lower bounds.

**This is not just noise, and it is the open question worth chasing.** The
truncation penalises exactly one class of candidate: those that make the agent
more thorough. Upstream's `evidence_gated_completion` was such a candidate --
a verification rule that ran 139 minutes, cost $37.59, and scored 32%.
Upstream let it finish and scored it. Reef would have truncated it. Reef's own
iteration 5 was the same shape: a candidate that slowed episodes, 65 of them
cut off, scored 0.15, refused by the guard.

So Reef's search was systematically biased against thoroughness candidates,
which is a large fraction of what the proposer generates. **A rerun with the
14400s timeout could plausibly change which candidates are selected**, not
just their absolute scores. That is the first thing to redo.

## The measurement cannot adjudicate the search

Measured directly, not estimated:

- Upstream's baseline was run twice under identical configuration. Both
  returned **0.2833**, and **6 of 30 tasks flipped** between them -- three
  down, three up, cancelling exactly. Roughly 10% of episodes change outcome
  between identical runs.
- Upstream's one selection was +6.7%, which is 4 episodes out of 60 -- smaller
  than that run-to-run variance.

At 30 tasks x 2 trials the search selects on margins smaller than its own
error. Separating a real 5-point improvement needs roughly 8 trials per task.

## Open questions

1. **Reef measures the seed ~0.09 higher than upstream, every time**
   (0.3833 / 0.3667 against 0.2833 / 0.2833). ~1.5 sigma, consistently signed.
   Ruled out: seed composition (renders `{}`, no kwargs, no instruction
   paths), temperature (unset both sides), timeout multipliers (`JobConfig`
   and `TrialConfig` defaults identical), failed-episode handling (both score
   zero), upstream's retry (can only raise). Untested: nothing specific left.
   Needs a third measurement each.
2. **Reef evaluates twice what upstream does.** Cordis pairs candidate with
   incumbent (120 episodes/iteration); the frontier then discards the
   incumbent half. 240 of 600 episodes bought and thrown away. Halving it
   makes Cordis's pairing a no-op for this method -- a call for whoever owns
   the selector (#214).
3. **Harbor leaks e2b sandboxes** and swallows the failed delete. Not fixable
   from here; `reap_sandboxes.py` bounds it but cannot eliminate it, because
   the age threshold must exceed the longest live episode (~2h). 32 concurrent
   episodes held ~54 sandboxes.

## Operating notes -- read before running anything

- **The e2b account is shared with other teams.** This project's slice is 32
  of the account's 100. Reaping on age alone once killed 97 sandboxes, most of
  them other teams' running work. `reap_sandboxes.py` filters on
  `environment_name` against `tasks-89.txt` and there is no flag to disable
  that. Do not bypass it.
- Run the reaper alongside any job: `--older-than <longest task timeout +
  margin> --watch 600 --yes`. The default 14400s is right for the full 89 and
  too loose for the hard 30 (use 9000).
- `run.py` refuses to start without headroom in the 32 share and prints the
  reaper command.
- Launch scripts in `/tmp` end in `exec`. **Never `source` them** to read a
  variable -- doing so launches a second run against the same output
  directory. Read keys with `sed` instead.
- Long runs need `nohup ... & disown`; the Bash tool's background mode is
  still killed at 10 minutes.
- Always push with an explicit refspec:
  `git push origin simon/tb2-reproduction:refs/heads/simon/tb2-reproduction`.

## What to trust in the artefacts

| Run | Verdict |
| --- | --- |
| Run A (`/tmp/tb2-scaled2`) | **discard.** 37% of episodes never ran; its "13-point improvement" was the outage |
| Run B (`/tmp/tb2-reef3`) | **partial.** Only the seed and iteration 1 measured anything |
| Run C (`/tmp/tb2-reef5`) | **usable, lower bound.** Iterations 1-4 truncated, iteration 5 refused by the guard |
| Upstream (`jobs/upstream-matched`) | **usable.** Baseline and 2 iterations clean |

## Suggested next steps

1. Decide upstream's cap, then let it stop or finish.
2. Rerun the Reef arm with `--episode-timeout-s 14400` and compare which
   candidates get selected, not just the scores. This is the highest-value
   experiment left.
3. If search quality matters, raise trials per task to ~8 and draw tasks from
   `tasks-89.txt` rather than the hard subset -- 59 of the 89 have never been
   evaluated in either arm.
4. Open the PR for this branch, and decide whether the upstream fork becomes a
   pinned submodule instead of `upstream/meta_harness.patch`.
5. **The OpenAI key the user pasted is still unrotated.** It is in the session
   transcript and in `/tmp/run_scaled2.sh`.
