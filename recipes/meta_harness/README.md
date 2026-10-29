# Meta-Harness

Meta-Harness is full-history search over a fixed model's harness. This method
represents every candidate as a complete Reef harness tree, so the same search
works with any Reef adapter and any rollout scorer. It has no Harbor,
Terminal-Bench, Terminus, or coding-agent dependency.

The implementation is a specialization of Reef's harness evolution recipe, in
the standard method layout:

```text
meta_harness/
  recipe.py       MetaHarnessRecipe: the evolution.meta_harness section, the proposer and the selection
  proposer.py     MetaHarnessProposer: one complete tree per step from the retained population
  selection.py    MetaHarnessSelection: a strict mean-score improvement over the committed incumbent
  population.py   every candidate tree, its scores and the budgets, committed as method state
  examples/terminal_bench/
                  the runnable Terminal-Bench search
```

Reef still owns tree validation, adapter rendering, model binding, rollout
execution, timeouts, residue policy, finite-score checks, publication, and
scenario commits. Meta-Harness adds only the population-aware proposer and
selection.

## Configuration

```yaml
recipe:
  implementation: recipes.meta_harness.recipe:MetaHarnessRecipe
  config:
    evolution:
      adapter: pi
      seed: seed                 # the first tree, a directory
      evaluate: my_harness.scoring:RolloutScore
      tasks:
        - First validation task
        - Second validation task
      models:
        proposer:
          url: https://api.openai.com
          model: gpt-5
          api_key_env: OPENAI_API_KEY
      meta_harness:
        mode: full_history
        max_candidates: 20
        max_target_rollouts: 200
        max_files: 32

inference:
  upstream-model: model-under-test
```

`seed` is a directory of tree files, such as `rules.md` with the starting
rules. `evaluate` names a standard Reef rollout scorer class:

```python
class RolloutScore(RolloutScorer):
    def score(self, task: str, result: RolloutResult) -> float:
        ...
```

The `meta_harness` section takes these keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `full_history` | `full_history` lets the proposer branch from any retained candidate; `incumbent_only`, the greedy control, from the served one only |
| `model` | `proposer` | the `evolution.models` binding that proposes |
| `components` | every kind the adapter renders | the tree kinds the proposer may change |
| `max_candidates` | `0` (no limit) | new candidates to evaluate |
| `max_target_rollouts` | `0` (no limit) | evaluation rollouts to spend |
| `max_files` | `32` | files in one proposed tree |

The proposer is a standard Reef `Proposer`: `propose(context)` reads the
population from the method state of the latest commit. It calls the binding
`meta_harness.model` names and refuses to run when `evolution.models` does
not declare it: the harness under test must not propose its own
improvements. The model sees every retained candidate's files and scores
(the served candidate's alone in `incumbent_only` mode), the parent each
came from, the traces of the current step, the validation task names, and a guide
to the files the adapter takes. It returns one parent id and one complete
tree as `files`, path to text. Reef validates the tree and states it as a
patch against the served tree. A tree the population already
holds is recorded as a duplicate and the step proposes nothing.

`components` can narrow what the proposer may change. Files of other kinds
must stay byte-for-byte equal to the selected parent's:

```yaml
      meta_harness:
        components: [rules, skill]
```

With no `components` setting, every kind the selected adapter renders is
eligible. The adapter's own validation remains authoritative; for example,
an adapter may still refuse an otherwise valid file.

### Terminus 2 code evolution

Use Reef's `terminus` adapter and its `code_extension` kind to evolve Python
behavior. It accepts one self-contained module defining `Agent(Terminus2)`,
at `extensions/<name>.py` in the tree. Start from a no-op subclass so the
proposer sees the adapter contract: a seed directory holding
`extensions/agent.py` with

```python
from harbor.agents.terminus_2 import Terminus2


class Agent(Terminus2):
    pass
```

and this configuration:

```yaml
recipe:
  config:
    evolution:
      adapter: terminus
      seed: seed
      executor: sandbox
      sandbox:
        egress_hosts: [api.e2b.dev, api.openai.com]
        env_from: [REEF_TERMINUS_ENVIRONMENT, E2B_API_KEY]
      meta_harness:
        components: [code_extension]
```

Combine this with the tasks, scorer, model, and proposer settings above. Export
`REEF_TERMINUS_ENVIRONMENT=e2b` and `E2B_API_KEY` in the deployment. Install
`harbor[e2b]` on Linux with Python 3.12+ and bubblewrap;
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
future parent; a candidate's id is its tree's checksum. Selection is a strict
improvement in mean validation score over the score the incumbent was admitted
on, which reproduces upstream's frontier: it keeps the highest recorded score.
Reef's paired evaluation still measures both sides, so this method spends two
evaluations per iteration where upstream spends one; budgets expressed in
rollouts are not directly comparable to upstream's iteration counts. Failed
rollouts count as zero; a non-finite score is rejected by Reef before
selection.

The [Terminal-Bench example](examples/terminal_bench/README.md) runs the shared
recipe on the full pinned dataset. The [results](RESULTS.md) compare a matched
search with upstream and include the selected harness. The experiment measured
the baseline once and each new candidate once per measurement; its local
campaign tooling is separate from this reusable recipe.

The evaluation suite must stay fixed so historical scores remain comparable.
Task promotion, periodic rechecks, and review-only publication are therefore
rejected by this method, and so is an `evolution.selection` setting: the
method owns the selection. Selection and automatic publication commit
together.

The complete population, parents, scores, served id, proposal attempts, and
budget counters live under `population` in the method state Reef commits with
each step. A step reads the population from the latest commit and folds in
the evaluation of the candidate the previous step proposed, so evaluation,
publication, or commit failures cannot advance the committed population.

`max_target_rollouts` counts both sides of Reef's paired evaluation: candidate
and incumbent, multiplied by `rollout_repeats`. A proposal is not started
unless the complete next evaluation fits. Zero disables a budget; Reef's shared
`max_steps`, `max_model_calls_per_step`, executor, timeout, and residue options
remain available as usual.
