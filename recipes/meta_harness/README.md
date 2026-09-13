# Meta-Harness

Meta-Harness is full-history search over a fixed model's harness. This method
represents every candidate as a complete Reef composition, so the same search
works with any Reef adapter and any episode scorer. It has no Harbor,
Terminal-Bench, Terminus, or coding-agent dependency.

The implementation is a specialization of Reef's existing harness-evolution
recipe. Reef still owns node validation, adapter rendering, model binding,
episode execution, timeouts, residue policy, finite-score checks, selection
settlement, artifact publication, and scenario commits. Meta-Harness adds only
the population-aware proposal and selection policies.

## Configuration

```yaml
implementation: recipes.meta_harness.recipe:MetaHarnessRecipe

model:
  path: model-under-test

evolution:
  adapter: pi
  evaluate: my_harness.scoring:score_episode
  tasks:
    - First validation task
    - Second validation task
  seed:
    - id: rules
      name: rules
      config:
        text: Work carefully and verify the result.
  models:
    proposer:
      url: https://api.openai.com
      model: gpt-5
      api_key_env: OPENAI_API_KEY
  meta_harness:
    archive: ${REEF_WORK}/meta-harness
    mode: full_history
    max_candidates: 20
    max_target_episodes: 200
    max_nodes: 32
```

`evaluate` has the standard Reef episode-scorer signature:

```python
def score_episode(task: str, result: EpisodeResult) -> float:
    ...
```

The built-in proposal policy also has the standard Reef proposer signature:

```python
def propose(nodes, samples, models, *, manifest=None, rejected=()):
    ...
```

It calls `evolution.models.proposer`, and refuses to run when
that name is not configured. The model sees all retained candidate
compositions and scores, the parent each came from, the current trace batch, the
validation task names, and the adapter's Reef node vocabulary. It returns one
parent id and one complete composition. In `full_history` mode the parent may
be any retained candidate; `incumbent_only` is the greedy control.

`components` can narrow what the proposer may change. Other nodes from the
selected parent must remain byte-for-byte equivalent in the proposed
composition:

```yaml
  meta_harness:
    archive: ${REEF_WORK}/meta-harness
    components: [rules, skill]
```

With no `components` setting, every node kind the selected adapter exposes is
eligible. The adapter's own render/finalization checks remain authoritative;
for example, an adapter may still reject an otherwise valid Reef node kind.

### Terminus 2 code evolution

Use Reef's `terminus` adapter and its `code_extension` node to evolve Python
behavior. It accepts one self-contained module defining `Agent(Terminus2)`.
Start from a no-op subclass so the proposer sees the adapter contract:

```yaml
evolution:
  adapter: terminus
  executor: sandbox
  sandbox:
    egress_hosts: [api.e2b.dev, api.openai.com]
    env_from: [REEF_TERMINUS_ENVIRONMENT, E2B_API_KEY]
  seed:
    - id: agent
      name: code_extension
      config:
        name: agent
        code: |
          from harbor.agents.terminus_2 import Terminus2
          class Agent(Terminus2):
              pass
  meta_harness:
    archive: ./meta-harness
    components: [code_extension]
```

Combine this with the tasks, scorer, model, and proposer settings above. Export
`REEF_TERMINUS_ENVIRONMENT=e2b` and `E2B_API_KEY` in the deployment. Install
`reef-infra[terminus]` and `harbor[e2b]` on Linux with Python 3.12+ and bubblewrap;
place the runtime and any local task directories under `/usr` or `/opt`, which
the existing sandbox mounts read-only. Registry task ids also work.

Reef's sandbox isolates the Python runner; Harbor uses E2B for the terminal
task. Rendering never imports candidate code, and local execution refuses
extensions. Rules, skills, model binding, timeouts, trajectories, scoring, and
publication all use Reef's existing components. `egress_hosts` currently enables
network access; it does not enforce a hostname firewall. Only explicitly named
deployment variables are forwarded. A tree without an extension can still run
stock Terminus 2 locally with Docker.

## Population and commits

Every unique valid candidate is retained, including non-winners, and can be a
future parent. Selection is a strict improvement in mean validation score over
the score the incumbent was admitted on, which reproduces upstream's frontier:
it keeps the highest recorded score. Reef's paired gate
still measures both sides, so this method spends two evaluations per iteration
where upstream spends one; budgets expressed in episodes are not directly
comparable to upstream's iteration counts. Failed episodes count as zero; a
non-finite score is rejected by Reef before settlement.

The [Terminal-Bench example](examples/terminal_bench/README.md) runs the shared
recipe on all 89 tasks in the pinned dataset. The [results](RESULTS.md) compare a
matched search on the 30-task subset with upstream and include the selected
harness. The experiment measured the baseline once and each new candidate
once per measurement; its local campaign tooling is separate from this
reusable recipe.

The evaluation suite must stay fixed so historical scores remain comparable.
Task promotion, periodic rechecks, and review-only publication are therefore
rejected by this method. Selection and automatic publication commit together.

The complete population, parents, scores, served id, proposal attempts, and
budget counters live under `meta_harness_population` in Reef's algorithm
state. Proposal and selection changes are staged until the scenario commit is
durable. Evaluation, settlement, activation, publication, or commit failures
cannot advance the committed population.

The file under `archive` is only a human-readable post-commit mirror. Scenario
names are SHA-256 encoded into filenames so they cannot escape the directory.
On restart Reef ignores the file as input and rewrites a stale mirror from the
committed algorithm state.

`max_target_episodes` counts both sides of Reef's paired gate: candidate and
incumbent, multiplied by `episode_repeats`. A proposal is not started unless
the complete next gate fits. Zero disables a budget; Reef's shared
`max_steps`, `max_model_calls_per_step`, executor, timeout, and residue options
remain available as usual.
