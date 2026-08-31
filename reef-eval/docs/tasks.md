# Benchmarks

Everything reef-eval ships, in one table. Each benchmark is a directory of stock
Harbor tasks, so every task runs two ways:

```bash
reef-eval run cl-bench/bsm-s01 --agent oracle                        # through reef-eval
harbor trial start -p tasks/continual-learning/cl-bench/bsm-s01 # stock Harbor
```

`reef-eval list` shows what is runnable where you are.

| Benchmark | Regime | Tasks | Run |
|---|---|---|---|
| [first-party](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/autoresearch/first-party) | autoresearch | 6 | `reef-eval run autoresearch/first-party --agent oracle` |
| [EdgeBench](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/autoresearch/edgebench) | autoresearch | 51 | `reef-eval run edgebench/<task> --budget <h>` |
| [FrontierCS](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/autoresearch/frontier-cs) | autoresearch | 208 | `reef-eval run frontier-cs/<task> --agent <a>` |
| [terminal-bench](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/continual-learning/terminal-bench) | stream | 89 | `reef-eval stream terminal-bench --agent <a>` |
| [CL-Bench](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/continual-learning/cl-bench) | stream | 301 | `reef-eval stream cl-bench --agent <a>` |
| [SWE-bench Verified](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/continual-learning/swebench-verified) | stream | 500 | `reef-eval fetch swebench-verified --limit 50` first |

The first-party tasks each teach one hard part of autoresearch, and each
benchmark states its upstream, license, and oracle scores in
[the catalog](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks),
next to the tasks themselves.

To write your own, start from
[`tasks/_template`](https://github.com/Human-Agent-Society/reef/tree/main/reef-eval/tasks/_template)
and follow [Authoring tasks](authoring-tasks.md).
