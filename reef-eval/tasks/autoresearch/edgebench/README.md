# EdgeBench

51 released tasks from [ByteDance-Seed/EdgeBench](https://github.com/ByteDance-Seed/EdgeBench):
real-world environments where agents iterate for 2-12 hours against
executable feedback. Continuous scores; the official metric is score vs
interaction budget.

```bash
cd tasks/autoresearch/edgebench
python fetch.py                        # list all 51 tasks
python fetch.py ann_vector_search_qps  # convert one into this folder
python fetch.py --all                  # convert all
```

Converted folders are stock Harbor tasks. The local [`convert.py`](convert.py)
script is tested against unmodified published specs; it maps EdgeBench's two
containers onto Harbor's separate verifier and turns `submit_paths` into
declared artifacts.

Run one (the budget is a run parameter, not a task property):

```python
await lab.run(
    "tasks/autoresearch/edgebench/ann_vector_search_qps",
    agent={
        "name": "claude-code",
        "model_name": ...,
        "override_timeout_sec": 2 * 3600,
    },
    tags={"budget": 2},
)
# metrics.scaling(lab.df("episode")) reproduces their budget curves;
# rescale params ride in each task.toml's [metadata].
```

All 51 tasks are committed in this folder; the underlying specs are
[CC-BY-4.0](https://huggingface.co/datasets/ByteDance-Seed/EdgeBench),
© ByteDance-Seed. The local `fetch.py` and `convert.py` scripts regenerate
them from the published specs.

Note: tasks reference EdgeBench's prebuilt work/judge images; pull access
follows their docs (`sforge pull`).

**Fidelity vs upstream.** EdgeBench's own harness runs its judge as a live
server during the session, and reef-eval's first-party tasks now use the same
shape (a judge sidecar scoring limited submissions, peak counted). This
conversion, however, still maps their judge onto a run-once separate
verifier: converted tasks give the agent only the work environment's own
feedback, and keeping the best submission in place is the agent's job.
Regenerating the conversion onto the judge protocol is on the
[Reef issue tracker](https://github.com/Human-Agent-Society/reef/issues).
