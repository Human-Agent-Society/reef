# Reproduction result

Both arms ran live on Terminal-Bench 2 with matched settings, and the two
implementations select identically. The live *outcomes* differed, and the
reason is measurement noise rather than a difference in the search.

## Settings, matched between arms

| | Both arms |
| --- | --- |
| Proposer tool | Codex CLI |
| Proposer model | `gpt-5.6-sol` |
| Executor agent | vanilla Terminus 2 (`baseline_terminus2` upstream, empty seed tree in Reef) |
| Target model | `openai/gpt-5.6-luna` |
| Sandbox | e2b |
| Task | `password-recovery`, 1 trial, 1 iteration |

## What each arm did

| | Baseline | Candidate | Decision |
| --- | --- | --- | --- |
| Upstream | 0.0 | 1.0 (`default_temperature_guard`) | **select** — `1.0 > 0.0` |
| Reef | 1.0 | 1.0 | **reject** — `1.0 > 1.0` is false |

Upstream: proposer 9m42s, evaluation 3m18s, \$0.01.
Reef: 167s end to end, \$0.0204 across two paired episodes.

## The arms did not disagree

Each applied the same rule correctly to the inputs it observed. What differed
was the baseline measurement of the *same agent on the same task*: upstream's
`terminus2-baseline` scored 0.0, Reef's seed scored 1.0, and a standalone
probe of the same configuration also scored 1.0. `password-recovery` is close
to a coin flip for this target model at one trial.

Replaying each arm's actual measured scores through both implementations
confirms it — neither case produces a disagreement:

```
upstream live (baseline 0.0, candidate 1.0): disagreements=none
reef live     (seed     1.0, candidate 1.0): disagreements=none
```

Together with the 200 random sequences in `replay.py`, that is the matching
claim. It is deterministic and it does not depend on either arm's sampling.

## What this does not show

A single task at one trial cannot support a claim about the search. With this
target model passing roughly a third of tasks, agreement between two live arms
would be as likely to be luck as evidence, and so would disagreement. This run
demonstrates that both pipelines execute an identically-configured experiment
end to end and that their selection rules agree on real data. It is not a
measured comparison of search quality, and the published tree is a mechanism
demonstration rather than a measured improvement.

## What the seed partition does and does not tell you

A scaled run (30 tasks x 2 trials x 5 iterations) measured the seed on
upstream's hard subset. `select_panel.py` recovers the per-task result:

| | Tasks | Can a better candidate show it here? |
| --- | --- | --- |
| Seed always passes | 4 | No — only regression is visible |
| Seed always fails | 21 | **Yes — this is headroom** |
| Seed sometimes passes | 5 | Yes — and seed variance lives here |

Iteration 2 settled which of those columns is right. The candidate scored
0.350 against the incumbent's 0.233, and the gain did not come from the
five variable tasks:

```
unlocked (seed always failed, candidate passes)
  mcmc-sampling-stan     [0,0] -> [1,1]
  sparql-university      [0,0] -> [1,1]
  fix-ocaml-gc           [0,0] -> [0,1]
  password-recovery      [0,0] -> [1,0]
  polyglot-rust-c        [0,0] -> [1,0]
regressed (seed always passed)
  fix-code-vulnerability [1,1] -> [0,1]
```

Seven of the net eight episodes came from tasks the seed never solved. So a
task the seed always fails is not a constant added to both sides: it is the
main thing a candidate can win. Only the four always-pass tasks are
signal-free in the upward direction, and they still register regressions —
`fix-code-vulnerability` did.

Two consequences for the next run. Do **not** prefilter the task set down to
the seed-variable tasks; that would have discarded the five tasks carrying
most of iteration 2's improvement. And read `partition()` as a description of
the seed, not as a prediction about candidates — its always-fail bucket is
where the headroom is.

What stays true is that the panel is small for judging *marginal* changes:
iteration 1 selected on 0.233 against 0.217, one episode, which is noise.
Iteration 2's margin is a different matter — five distinct tasks unlocked,
two of them on both trials.

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
