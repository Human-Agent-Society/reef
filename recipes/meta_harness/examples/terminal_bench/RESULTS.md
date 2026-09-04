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
