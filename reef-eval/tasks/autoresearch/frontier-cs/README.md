# FrontierCS

Open-ended CS problems from
[FrontierCS/Frontier-CS](https://github.com/FrontierCS/Frontier-CS) (MIT):
the optimal solution is unknown, but quality is objectively scored;
continuous partial credit, expert-curated
([paper](https://arxiv.org/abs/2512.15699)). Two tracks are supported,
both generated verbatim by their official Harbor adapters:

- **1.0 algorithmic track**: 188 open-ended competitive-programming
  problems (many NP-hard variants), C++17, per-problem testlib checkers.
  Their adapter is judge-native: an HTTP judge sidecar grades submissions
  during the session (`bash /app/submit.sh`), and the final score is the
  best of the last solution and the best graded submission. All 188
  committed.
- **2.0 track**: 20 open research problems (Erdős constructions,
  BBOPlace, GPU kernels …) with in-task `submit.sh` self-scoring; all 20
  committed, including four GPU kernel tasks whose judge sidecars
  demonstrate the compose-overlay wiring from
  [authoring tasks](../../../docs/authoring-tasks.md).

## The algorithmic track needs the upstream problem data

Its judge grades against test data that lives in the upstream repository,
which is 2.6 GB for all 188 problems and so is not committed here. The
generated `docker-compose.yaml` mounts it through
`$FRONTIER_CS_ALGORITHMIC_PATH`, and Compose reads an unset variable as an
empty string, so without it the judge starts, fails to find anything, and
returns 0. reef-eval checks for the variable and refuses the run instead.

Fetch the data for the problems you mean to run, and they are ready:

```bash
cd tasks/autoresearch/frontier-cs
python fetch.py --problems 1 17          # judge data for problems 1 and 17
```

There is nothing to export. Compose reads a `.env` beside each task's
compose file, so `fetch.py` writes one there pointing at what it just
downloaded. A task whose data you have not fetched is refused before the
containers start, rather than running to a score of 0.

The 2.0 track is self-contained and needs none of this.

Run any task:

```bash
reef-eval run frontier-cs/frontier-cs-2-0-erdos-demo --agent oracle
reef-eval run frontier-cs/frontier-cs-algorithm-1 --agent claude-code --model anthropic/claude-opus-5
```

Regenerate any task from upstream (ids pick the track: numeric =
algorithmic, named = 2.0):

```bash
cd tasks/autoresearch/frontier-cs
python fetch.py                      # list problem ids
python fetch.py 1 17                 # algorithmic track
python fetch.py erdos_unit_distance  # 2.0 track
```

The algorithmic judge runs from a published image
(`yanagiorigami/frontier-cs-harbor-judge`), pulled on first use.
