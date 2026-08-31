# terminal-bench

[terminal-bench 2.0](https://github.com/laude-institute/terminal-bench-2)
(Apache-2.0): 89 pass/fail terminal tasks (build a project, fix a repo,
drive a CLI), each a stock Harbor task with its own container and
verifier. A pass is reward 1.0, so the stream metrics work on it
directly.

Only version 2.0 is supported. The pin in [`fetch.py`](fetch.py) is the
exact commit the Harbor registry publishes as v2.0; terminal-bench 1.x
predates the Harbor task format.

All 89 tasks are committed here and run out of the box; `fetch.py`
re-syncs them from the pin if you need to regenerate.

```bash
reef-eval stream terminal-bench --agent claude-code --model anthropic/claude-opus-5
```

That streams every task in name order with the agent's state directory
(`$REEF_EVAL_STATE_DIR`) carried between tasks. For a custom order, a subset,
or repeats, list the task folders yourself:

```bash
reef-eval stream terminal-bench/chess-best-move terminal-bench/build-pmars \
  terminal-bench/chess-best-move --agent claude-code --model anthropic/claude-opus-5
```

The isolated baseline for `metrics.transfer` is a plain run over the same
tasks: `reef-eval run terminal-bench/<task> --agent <a>`.
