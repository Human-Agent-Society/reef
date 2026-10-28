# Meta-Harness

Meta-Harness is full-history search over a fixed model's harness. This method
represents every candidate as a complete Reef harness tree, so the same search
works with any Reef adapter and any episode scorer. It has no Harbor,
Terminal-Bench, Terminus, or coding-agent dependency.

The method is a Reef harness evolution method (`Method`) in the standard
layout:

```text
meta_harness/
  method.py            MetaHarness: one complete tree per step from the whole search history
  system_prompt.md     the proposer model's system prompt
  proposal_prompt.md   the instructions in front of the search state
  examples/terminal_bench/
                       the runnable Terminal-Bench search
  results/             the selected harness of the reported comparison
```

Reef owns tree validation, adapter rendering, model binding, episode
execution, timeouts, residue policy, finite-score checks, the score cache,
selection, publication, and scenario commits. Meta-Harness adds the proposal:
what the proposer model sees and how its reply becomes a candidate.

## Configuration

A deployment names the method in `evolution.method` of Reef's harness
evolution recipe, and selects with the `compare` policy on means:

```yaml
recipe:
  implementation: reef.recipe.harness:HarnessEvolutionRecipe
  config:
    evolution:
      adapter: pi
      seed: seed                 # the first tree, a directory
      method:
        module: recipes.meta_harness:MetaHarness
        config:
          mode: full_history
          max_candidates: 20
          max_target_episodes: 42
      models:
        proposer:
          url: https://api.openai.com
          model: gpt-5
          api_key_env: OPENAI_API_KEY
      evaluation:
        tasks:
          - First validation task
          - Second validation task
        scorer: my_harness.scoring:EpisodeScore
        policy: compare
        statistic: mean

inference:
  upstream-model: model-under-test
```

`seed` is a directory of tree files, such as `rules.md` with the starting
rules. `evaluation.scorer` names a standard Reef episode scorer class:

```python
class EpisodeScore(EpisodeScorer):
    def score(self, task: str, result: EpisodeResult) -> float:
        ...
```

`evolution.method.config` takes these keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `full_history` | `full_history` lets the proposer branch from any candidate; `incumbent_only`, the greedy control, from the served one only |
| `model` | `proposer` | the `evolution.models` binding that proposes; `served` is refused |
| `components` | every kind the adapter renders | the tree kinds a proposal may change |
| `max_candidates` | `0` (no limit) | the candidates the method proposes |
| `max_target_episodes` | `0` (no limit) | the evaluation episodes after which the method proposes nothing |
| `max_files` | `32` | the files of one proposed tree |
| `max_traces` | `16` | the traces whose record files one prompt shows |
| `trace_chars` | `4000` | the characters one prompt shows of one trace's record files |

The method refuses to run when `evolution.models` does not declare `model`:
the harness under test must not propose its own improvements.

## One step

The population is the scenario's history. A step shows the proposer model
these items, as data:

- The candidates of the search: the served tree, the seed, and each tree the
  method proposed; in `incumbent_only` mode, the served tree alone. Each has its
  files, the parent it branched from, the proposer's hypothesis and changes,
  its outcome (served, selected, rejected, or undecided), and its scores on
  the evaluation tasks.
- Excerpts of the episode records that the history keeps, and of the step's
  own traces. The step's traces come first. Then the candidates take turns,
  the served one first and the others newest first, each with its lowest
  scores first. A long record keeps its beginning and its end.
- The latest replies that proposed no candidate, and why.
- The person's instruction, on a step that answers a training request.
- A guide to the files the adapter takes.

The reply names one parent and returns one complete tree as `files`, path to
text. The step refuses the reply when the parent is not a candidate it
showed, when the tree changes a file of a kind outside `components` (such
files must stay byte for byte equal to the parent's), when the tree holds
more than `max_files` files, or when the tree fails the adapter's
validation. It also refuses a tree that the history already holds. A refused
reply leaves the head unchanged, so the step skips and Reef evaluates
nothing. The adapter's validation remains authoritative: an adapter may
still refuse an otherwise valid file.

A proposed tree becomes the step's candidate. The history keeps it with the
proposer's hypothesis, changes, and parent, so the next step shows it.

## Selection and budgets

Select with `policy: compare` and `statistic: mean`. The candidate is
published when its mean score is strictly higher than the head's; an episode
that could not run counts as zero. The score cache keeps the first scores of each tree,
so the head's mean is the one it was admitted on, and the head does not run
again. The exception is an episode that could not run: it runs again in the
next evaluation. This reproduces upstream's frontier: it keeps the highest recorded
mean, and a tie keeps the current choice. The seed runs once with the first
candidate, and then each candidate runs once.

The budgets count the history: `max_candidates` counts the trees the method
proposed, and `max_target_episodes` counts the evaluation episodes that ran,
the episodes of a client's proposal included. A spent budget ends the search
before the model call; the last evaluation may pass `max_target_episodes`.
Zero turns a budget off. Reef's own `max_steps`, `max_model_calls_per_step`,
executor, timeout, and residue options apply as usual.

Keep the evaluation suite fixed, so that the scores of all candidates stay
comparable: do not combine this method with task collection or rechecks.

## State and commits

The history directory under `evolution.state_dir` holds every candidate, its
episodes and records, and each step's decision. The method's own state,
committed with each step, is `state/attempts.jsonl`: one line per reply that
proposed no candidate.

The history keeps an attempt whose step never committed, such as an
evaluation or a commit that failed. A retry of that step does not count the
attempt's tree, does not show it, and does not refuse it as a known tree. If
the proposer returns it again, the cache scores it; only the episodes that
could not run are run again.

## Terminus 2 code evolution

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
      method:
        module: recipes.meta_harness:MetaHarness
        config:
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

## Example and results

The [Terminal-Bench example](examples/terminal_bench/README.md) runs the
method on the full pinned dataset. The [results](RESULTS.md) compare a matched
search with upstream and include the selected harness.
