# ace

ACE ([Agentic Context Engineering](https://arxiv.org/abs/2510.04618), Zhang et al.) as a Reef harness method.
ACE keeps a playbook: a list of short lessons, called bullets, that the agent reads in its context. After a
session, a Reflector reads the session and finds what went right or wrong. A Curator turns that analysis into
new bullets. Code, not a model, merges the new bullets into the playbook. The playbook grows by small additions,
so it keeps detail instead of collapsing into a short summary.

ACE needs no correct answer. The Reflector judges a session from what happened in it, such as errors, failed
commands and retries. It uses a score or feedback only when the trace has one.

- Paper: [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- Upstream code, for reference only: [ace-agent/ace](https://github.com/ace-agent/ace). The method does not
  depend on it, and its prompts are written for this method.

## How ACE learns online

ACE is a harness method: Reef runs the loop, and ACE decides how the harness tree changes in one step. The
served agent is ACE's Generator. It reads the playbook in `rules.md`, and it does not change during a session.
The learning happens in Reef, after the session, in a training step:

```text
pi session: ACE's Generator, which reads the playbook in rules.md
  │  model calls through Reef, then POST /reef/report (score, feedback, receipts)
  ▼
one trace in Reef's records
  │  a step starts when job_samples traces wait (default 1)
  ▼
training step: ACE.step(context)
  │  per run the Reflector and the Curator, then the merge and the refine; a new rules.md
  ▼
check, evaluation policy, new release, activation rule
  │  reef-client installs the active release when the next session starts
  ▼
next pi session: reads the new playbook
```

| Question | Answer |
| --- | --- |
| When does ACE learn? | When `pipeline.sample2job.config.job_samples` traces wait. With the default of 1, each report starts one step. |
| What does one step read? | The traces of the step's material (`workspace.traces()`) and the playbook in `state/playbook.json`. |
| What does one step change? | `state/playbook.json`, and one marked block of `rules.md`. Nothing else in the tree. |
| When does the agent use the change? | The example uses `evaluation.policy: none` and the default activation rule `always`, so every valid new playbook serves at once. The client installs the active release when a session starts, so the next session reads it. |

The two sides split the work as follows:

| Reef does | ACE does |
| --- | --- |
| Records the agent's model calls and the reports, and makes them traces. | Selects the traces to learn from. |
| Starts a step when `job_samples` traces wait, with a copy of the head tree, the traces and the method's `state/`. | Calls the Reflector and the Curator for each trace. |
| Gives the model bindings (`served` or an `evolution.models` entry) and records each call in the step. | Changes the playbook by fixed rules, and renders it into `rules.md`. |
| Checks the new tree, applies the evaluation policy, publishes a release, and commits `state/` with the step. | Writes a summary and notes about the step. |

## Code layout

| File | Responsibility | Upstream counterpart |
| --- | --- | --- |
| [method.py](method.py) | `ACESettings`, and `ACE.step`: the order of one step, from the state and the traces to the new tree. | `ace/ace.py` (`_train_single_sample`) |
| [roles.py](roles.py) | The Reflector and the Curator: what each model reads, one call each, and the check of each reply. | `ace/core/reflector.py`, `ace/core/curator.py` |
| [playbook.py](playbook.py) | `Bullet`, `Addition` and `Playbook`: the counts, the merge, the refine rules, and the block in `rules.md`. No model call. | `playbook_utils.py` |
| [prompts/](prompts/) | The Reflector's and the Curator's instructions. | `ace/prompts/` |
| [examples/](examples/) | `serve.yaml`, an online deployment on pi, and `seed/`, its first tree. | |

## One step

`ACE.step` has four stages. Each stage reads the output of the stage before it.

### 1. Select the runs

A run is a trace with a score or a reply. A trace with neither is a model call that ends on a tool call, in the
middle of a session. The trace of the call that ends the session carries the whole session, so the step skips
the middle one. The step reads the runs in order, at most `max_traces_per_step` of them.

### 2. Reflect, curate and merge, per run

For each run, in order:

1. **The Reflector reads the playbook and the trace text.** The trace text starts with the score, the feedback
   and the failure, each when the trace has one. Then it has the conversation without the system prompt: one
   line per message and one line per tool call. A trace of a train-task run has no trajectory, so it shows the
   task and the reply instead. A text longer than 30,000 characters keeps its start and its end.
2. **The tags change the counts.** Each `helpful` or `harmful` tag adds 1 to that count of its bullet. A
   `neutral` tag, or a tag for an id that is not in the playbook, has no effect.
3. **The Curator reads the sections, the playbook, the reflection and the trace's task.** It returns operations.
   An `ADD` of text to a configured section is an addition. Every other operation is `ignored`.
4. **Code merges each addition.** An addition is `refused` when its text has the shape of a credential or of an
   instruction override, or holds a marker line of the block. It is a `duplicate` when a bullet has the same
   text, compared without case and extra spaces. Otherwise it is `added` as a new bullet with the next id.

The next run reads the playbook that the previous run left. A reply without the asked JSON skips the rest of
its run: a failed Reflector reply changes nothing, and a failed Curator reply adds no bullet, but the tags of
its reflection still count.

ACE learns from traces only. A step that answers a training request learns from the traces that come with the
request; the request's text reaches neither the Reflector nor the Curator.

### 3. Refine the playbook

Once per step, after the last run:

1. The step removes each bullet whose `harmful` count exceeds its `helpful` count by `prune_margin` or more.
2. When more than `max_bullets` bullets remain, it removes the bullets with the lowest `helpful` minus
   `harmful` count. Of equal counts, the oldest bullet goes first.

### 4. Render and write

The step renders the playbook into `rules.md` and checks the new tree with `services.check`. When the tree
passes the check, the step writes `state/playbook.json` and the tree. When it fails, the step changes neither,
so the state and `rules.md` always agree. A step that leaves the tree unchanged publishes nothing.

## State and the rules block

`state/playbook.json` holds the playbook: `next_number`, the number of the next bullet's id, and the bullets,
each `{id, section, content, helpful, harmful, step}`. `step` is the step that added the bullet. A removed
bullet's id is never given again.

Each step renders the playbook into one block of `rules.md`, between two marker lines:

```markdown
<!-- ace playbook: begin -->
## Playbook

Lessons from earlier sessions: use the ones that fit your task. The counts say how often a review found a lesson helpful or harmful.

### Strategies and insights
- [ace-00001] helpful=2 harmful=0 :: Read the failing test before you edit the code.
<!-- ace playbook: end -->
```

The method keeps the text of `rules.md` outside the block. You can edit that text. Do not edit the block: the
state holds the playbook, so the next step replaces the block.

The step record is under `<evolution.state_dir>/<scenario>/harness/steps/<step>/`. It holds the new tree in
`harness/`, the playbook in `state/playbook.json`, each model call in `calls.json`, and the summary and notes in
`output/`. The notes count the runs that the step read and skipped, the tags, the outcome of each operation
(`added`, `duplicates`, `refused`, `ignored`), and the bullets removed by `prune_margin` (`pruned`) and by
`max_bullets` (`capped`).

## Configuration

The keys go under `evolution.method.config`:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `served` | The model of the Reflector and the Curator: `served` (the model under test, as in the paper) or a name under `evolution.models`. |
| `sections` | `Strategies and insights`, `Code and commands`, `Common mistakes to avoid`, `Others` | The sections of the playbook. The Curator puts each new bullet in one of them. |
| `max_bullets` | `100` | The most bullets that the playbook keeps. |
| `prune_margin` | `2` | Remove a bullet when its harmful count exceeds its helpful count by this many or more. |
| `max_traces_per_step` | `0` | The most runs that one step reads; 0 reads every run. Each run that the step reads costs two model calls. |

## Deployment

### Online, on pi

[examples/serve.yaml](examples/serve.yaml) evolves the rules of pi from live sessions. Its seed,
[examples/seed/rules.md](examples/seed/rules.md), holds base rules that the agent reads from the first
release; the method adds its block below them. Run it from the repository root:

```bash
export REEF_UPSTREAM_URL=http://127.0.0.1:8000 REEF_UPSTREAM_MODEL=qwen3-8b
reef serve -c recipes/ace/examples/serve.yaml
```

Start pi through this Reef as
[Use a published tree](../../docs/user-guide/evolve-your-harness.rst#use-a-published-tree) describes. After each
session, post one report with `POST /reef/report`: a `score`, optional `feedback`, and the receipts of the
session's model calls in `references`. The score does not have to be a correct answer; a plain 1 for done and 0
for not done is enough. The step records go under `recipes/ace/work/harness/`.

To learn from recorded traffic without reports, name the inference stage in the deployment file. Then each
recorded model call is one trace, without a score:

```yaml
pipeline:
  record2sample:
    module: reef.record2sample.traces:InferenceSamples
```

To keep a basic check on each new playbook, use `evaluation.policy: floor` with a scorer and a health task: Reef
publishes the playbook only when the agent still passes the task.

### On a fixed task set

Use `evaluation.policy: compare` with a scorer and evaluation tasks. Reef then publishes a new playbook only when
it scores better than the served one on the same tasks. A rejected step keeps its playbook changes in `state/`,
so the next candidate carries them with that step's changes. With
`reef.sample2job.traces:TrainTaskJobData`, each step runs uploaded train tasks on the head, and ACE learns from
those runs.

## Differences from the paper and the upstream code

- **Bullets used.** In the paper, the Generator names the bullets that it used, and the Reflector tags those. A
  served agent does not name them, so the Reflector reads the whole playbook and tags the bullets that apply.
- **One reflection per run.** Upstream reflects on a wrong answer for up to three rounds
  (`max_num_rounds`) and generates the answer again after each round. Reef does not run the served agent again,
  so the Reflector answers once.
- **A Curator call per run.** Upstream can curate every `curator_frequency` samples. This method curates after
  each reflection.
- **De-duplication.** The method drops only exact duplicates, compared without case and extra spaces. The
  paper's de-duplication by semantic embeddings (upstream `BulletpointAnalyzer`, off by default) is out of
  scope.
- **Refinement by counts.** Neither the paper nor the upstream code removes bullets by their counts: upstream
  gives the Curator a token budget in its prompt. This method removes bullets by `prune_margin` and keeps at
  most `max_bullets`.
- **Unknown sections.** Upstream puts a bullet for an unknown section in `OTHERS`. This method ignores the
  operation.
- **Offline epochs.** The paper revisits the training samples for several epochs and keeps the playbook with
  the best validation score. In Reef, the evaluation policy decides which playbook is served.
- **Screens.** The prompts fence trace text as data, and the merge refuses a bullet with the shape of a
  credential or of an instruction override, or with a marker line of the block. The paper has none of these.
- **No labels, weak signal.** The paper notes that without labels or reliable feedback, the playbook can learn
  wrong lessons. The harmful counts and an evaluation policy limit that risk; they do not remove it.
