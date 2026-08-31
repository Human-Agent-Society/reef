# SWE-bench Verified

[SWE-bench Verified](https://github.com/SWE-bench/SWE-bench): 500
real-world software-engineering tasks (fix a reported issue in a real
repository), in the Harbor task format published by
[harbor-datasets](https://github.com/laude-institute/harbor-datasets).
A resolved issue is reward 1.0.

It is here because [AgentStream](https://arxiv.org/abs/2608.00155)
builds its task streams from six benchmarks, and this is the hardest of
them with a published Harbor version. The two the paper measures as
hardest, HLE and BrowseComp-Plus, have no Harbor version yet.

The upstream dataset repository carries no license, so these tasks are
never committed here. [`fetch.py`](fetch.py) fetches them onto your
machine from the exact commit the Harbor registry pins as v1.0; a blob
filter keeps subset fetches small:

```bash
reef-eval fetch swebench-verified --limit 50   # or task names, or all 500
reef-eval stream swebench-verified --agent claude-code --model anthropic/claude-opus-5
```

The isolated baseline for `metrics.transfer` is a plain
`reef-eval run swebench-verified/<task> --agent <a>` over the same tasks.
