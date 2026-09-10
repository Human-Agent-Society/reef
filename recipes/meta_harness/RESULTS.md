# Terminal-Bench reproduction results

The comparison starts from vanilla Terminus 2 and runs a baseline plus four
full-history iterations. Each measurement covers the same 30 tasks with two
repeats: 60 trials. A candidate replaces the current choice only when its mean
score is strictly higher; a tie keeps the current choice.

| Step | Reef score | Reef selection | Upstream score | Upstream selection |
| --- | ---: | --- | ---: | --- |
| Baseline | 20/60 | Start with baseline | 24/60 | Start with baseline |
| Iteration 1 | 23/60 | Select iteration 1 | 22/60 | Keep baseline |
| Iteration 2 | 20/60 | Keep iteration 1 | 20/60 | Keep baseline |
| Iteration 3 | 20/60 | Keep iteration 1 | 21/60 | Keep baseline |
| Iteration 4 | 23/60 | Tie: keep iteration 1 | 21/60 | Keep baseline |

**Both selectors made identical decisions when replaying the same completed
score histories.** All 600 recorded scores were replayed through Reef's
selector and upstream's own `update_frontier`. All eight candidate decisions
agreed, including Reef's tie. This verifies the overall selection rule given
identical observations; independent proposals and scores can differ.

The [selected Reef harness](results/reef_harness.py) is based on the iteration 1
output, with terminology simplified in its comments and completion prompt.
The measurements below used the original output, whose SHA-256 was
`abe8e8b703bd31baaf9ec063fd44596c890f6faebc8804867c98989241d04b89`.
The version with revised wording has not been benchmarked again.

Evaluating the chosen harnesses again, with two fresh repeats on the same tasks,
gave **22/60 (36.67%) for Reef's iteration 1** and **21/60 (35.00%) for upstream's
baseline**. These results did not feed back into search and are not a held-out
task evaluation. Two infrastructure losses were replaced, one per arm; no
admissible outcome was repeated. Reef includes one terminal-loss zero under
the shared scoring policy, with its raw invalid/null verifier reward retained
in the internal run records.

## Configuration and scope

- Upstream: `stanford-iris-lab/meta-harness@44b9942127847f7421db70d8c7e48407f09a3c70`.
- Target: `gpt-5.6-luna`; proposer: `gpt-5.6-sol`, Responses API, `xhigh` effort.
- Tasks: 30-task hard subset at revision `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`.
- Runtime: Python 3.12.14, Harbor 0.20.0, LiteLLM 1.99.0, OpenAI 2.54.0, E2B 2.46.4.

This is a method reproduction on 30 tasks, with shared API, sandbox, and verifier
adaptations. It is not the paper's full benchmark or an unmodified upstream
performance reference. The local campaign measured the baseline once and only
new candidates thereafter; the reusable recipe uses Reef's paired evaluator.
The campaign scripts, raw histories, and audit records remain internal.

These measurements used the local experiment runner, before the shared
Terminus adapter supported Python extensions. They validate the search method;
they are not benchmark measurements of the updated adapter. A focused contract
test now loads the checked-in harness as a `code_extension` through the
shared recipe, episode lifecycle, Terminus runner, and publication path, with
process launch and the remote trial replaced by test doubles.
