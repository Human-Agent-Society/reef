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

It calls `evolution.models.proposer`, falling back to the served model when
that name is not configured. The model sees all retained candidate
compositions and scores, their lineage, the current trace batch, the
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

## Population and commits

Every unique valid candidate is retained, including non-winners, and can be a
future parent. Selection is a strict improvement in mean validation score over
the committed incumbent. Failed episodes count as zero; a non-finite score is
rejected by Reef before settlement.

The complete population, lineage, scores, served id, proposal attempts, and
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

