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

## The full 89-task set is available

Earlier runs used upstream's 30-task hard subset because the official set was
not enumerated here. It is now: `harbor dataset download 'terminal-bench@2.0'`
returns all 89 in a few seconds, and `tasks-89.txt` records them. Upstream's
hard subset is a strict subset, so 59 tasks have never been evaluated in
either arm.

Both entry points now take `--tasks-file`, which reads commas or newlines and
ignores `#` comments, so the shipped list can be passed directly.

## Wall clock for a larger run

Every task declares an `agent.timeout_sec` cap. These are caps, not expected
durations -- most episodes finish far inside them -- but they bound the tail,
which is what a run has to be planned around:

| Group | Tasks | Median cap | Longest |
| --- | --- | --- | --- |
| All official | 89 | 900s | 12000s |
| Upstream's hard subset | 30 | 1800s | 7200s |
| The other 59 | 59 | 900s | 12000s |

The hard subset is the slower half by median, so a sample drawn from all 89
runs faster per task than the runs recorded above, not slower.

One task dominates the tail: `build-pov-ray` at 12000s, roughly 3.3 hours,
against a 900s median. At concurrency 16 a single task like that can set an
iteration's wall clock on its own. A sample that includes it should either
raise concurrency or accept the tail deliberately.

No task in the official set requests a GPU, so e2b is sufficient throughout.

## Reef pays twice for every iteration after the first

Cordis evaluates in pairs. For each task and repeat it builds one pairing of
`candidate_files` and `current_files` and runs both, so the incumbent is
re-measured alongside the candidate on every iteration -- 120 episodes here,
not 60. The interleaving is deliberate: drift lands on both sides of a pair
instead of one whole side.

Meta-Harness's frontier does not want that measurement. The incumbent keeps
the score it was admitted on, so the selector reads

```python
incumbent_scores = incumbent.scores or current_scores
```

and once `incumbent.scores` is set, the freshly measured `current_scores` is
discarded. The live run shows it exactly: each iteration's `incumbent_mean` is
the previous iteration's `candidate_mean` to the digit.

| Iteration | candidate | incumbent | = previous candidate |
| --- | --- | --- | --- |
| 1 | 0.2333 | 0.2167 | seeds the frontier, used |
| 2 | 0.3500 | 0.2333 | yes -- measurement discarded |
| 3 | 0.3000 | 0.3500 | yes -- measurement discarded |

Iteration 1 is real work: `served.scores` starts empty, so that pairing is
what gives the seed its 0.2167. From iteration 2 on, 60 target-model episodes
per iteration are executed, billed, and thrown away. Across a 5-iteration run
that is 240 of 600 episodes, about 40% of the target spend, against upstream
which evaluates only the new candidate.

**This does not affect the matching claim.** Both implementations decide on
the same stored high-water score, which is why the arms agree. It is a cost
divergence, not a behavioural one.

It is worth a decision rather than a quiet fix, because the two designs
disagree on purpose: pairing buys drift cancellation, and a frontier that
never refreshes its incumbent cannot spend it. Skipping the incumbent leg when
`incumbent.scores` is already set would halve the cost of every run, at the
price of making Cordis's pairing a no-op for this method. That belongs to
whoever owns the selector, not to a reproduction run.

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
