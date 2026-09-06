# Terminal-Bench reproduction results

This experiment checks whether Reef reproduces Meta-Harness selection and
execution correctly. It compares the full-history method against an adapted
upstream implementation, starting from vanilla Terminus 2. Independent agent
executions can produce different scores; performance equality is not the
correctness criterion.

## Baseline and four evaluated iterations

Each score is successful trials out of 60: the same 30 tasks with two repeats.
Selection compares a candidate against the highest mean score already observed.
A tie keeps the existing harness.

| Step | Reef score | Reef selection | Upstream score | Upstream selection |
| --- | ---: | --- | ---: | --- |
| Baseline | 20/60 | Start with baseline | 24/60 | Start with baseline |
| Iteration 1 | 23/60 | Replace baseline with iteration 1 | 22/60 | Keep baseline |
| Iteration 2 | 20/60 | Keep iteration 1 | 20/60 | Keep baseline |
| Iteration 3 | 20/60 | Keep iteration 1 | 21/60 | Keep baseline |
| Iteration 4 | 23/60 | Tie: keep iteration 1 | 21/60 | Keep baseline |

**Both selectors made identical decisions when replaying the same completed
score histories.** All 600 recorded scores—five measurements per arm—were
replayed through Reef's selector and upstream's own `update_frontier`.
All eight candidate promotion decisions agreed, including Reef's actual tie.
This establishes the same overall selection rule given identical observations;
it does not assert identical proposals or per-task frontier serialization.

The [result artifact](results/scores.json) contains the recorded score vectors
and decisions. [reef_harness.py](results/reef_harness.py) is the exact Reef
implementation chosen by the search.

## Separate evaluation of the chosen harnesses

After search, each chosen harness was evaluated again on the same 30 tasks
with two fresh repeats. Reef's iteration 1 scored **22/60 (36.67%)**;
upstream's baseline scored **21/60 (35.00%)**. The harnesses were fixed for
this check, and its results were not fed into either search. These are fresh
executions on the same tasks, not held-out-task results or proof of statistical
equivalence.

Two trials lost infrastructure completion evidence and were replaced, one per
arm; no admissible outcome was repeated. Both arms count verifier outcomes
returned after agent timeout. Reef also has one completed, billed terminal-loss
benchmark zero under the shared scoring policy, with its raw invalid/null
verifier reward preserved. This policy does not establish the cause of the loss.

## Configuration

- Upstream: `stanford-iris-lab/meta-harness@44b9942127847f7421db70d8c7e48407f09a3c70`.
- Target: `gpt-5.6-luna`; proposer: `gpt-5.6-sol`, Responses API, `xhigh` effort.
- Tasks: [fixed hard subset](tasks-hard30.txt), revision
  `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`, two repeats.
- Runtime: Python 3.12.14, Harbor 0.20.0, LiteLLM 1.99.0, OpenAI 2.54.0,
  E2B 2.46.4. Each runner has 4 GiB; original task phase budgets are retained.
- Shared adaptations: API proposer transport, E2B execution, verifier cleanup
  compatibility, explicit terminal-failure scoring and protected completion.

The baseline source is byte-identical to upstream. Six core upstream functions,
including scoring and frontier selection, retain their original ASTs. The
upstream evolution loop differs only by three verified guards for runtime
identity, fresh outputs and baseline health. Its surrounding transport and
measurement setup are documented adaptations. This is a 30-task reproduction
of the method, not the paper's full benchmark or a stock upstream golden score.
