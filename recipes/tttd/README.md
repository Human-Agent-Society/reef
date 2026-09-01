# tttd

Reproduction of [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175) as a Reef weight-training recipe. TTT-Discover makes repeated attempts at one hard problem at test time: rollouts arrive as a fixed grid of sibling attempts, the step trains on the whole group, and the next grid runs on the weights it produced. The package holds the method; the search loop that drives it lives in [examples/tttd](examples/tttd/README.md).

- Paper: [arXiv:2601.16175](https://arxiv.org/abs/2601.16175)
- Pins: the repo's `third_party/slime` submodule for training; the completed run used `Qwen/Qwen3-8B` with thinking, rank-32 LoRA, the paper's Adam learning rate, and the Erdos minimum-overlap task
- Claim scope: a result-level reproduction of the paper's Qwen3-8B Erdos experiment at half its rollout budget (eight groups of 32 against the paper's 64 per group). 50 steps in about 22.6 hours on one four-B200 node reached a best certified C5 upper bound of 0.380916 against 0.380932 in the paper's Table 2.

## Layout

```text
tttd/
  recipe.py        TTTDRecipe: training spec, loss family "tttd", grid configuration
  processor.py     reported feedback, grouped: the step is the full grid of sibling attempts
  preparer.py      builds the grouped policy batch for the driver
  report.py        TTTDGroupedRolloutReport, the declared report schema
  slime/           the training-plane objective: the grouped entropic loss
  examples/tttd/   the runnable search: Harbor tasks, PUCT archive, serve.yaml, results
  examples/guidance_ttt/  the summary-only guidance variant on the same recipe
```

## Where the rest is documented

[The tttd recipe page](../../docs/user-guide/recipes/tttd.rst) covers the runtime sequence, a reduced smoke, and recovery behavior, and the [example README](examples/tttd/README.md) records implementation details, paper fidelity, and the completed reproduction.
