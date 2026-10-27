# ace

ACE ([Agentic Context Engineering](https://arxiv.org/abs/2510.04618), Zhang et al.) as a Reef harness method.
ACE keeps a playbook: a list of short lessons, called bullets, that the agent reads in its context. After each
task, a Reflector reads the trajectory and finds what went right or wrong. A Curator then turns that analysis
into new bullets. Code, not a model, merges the new bullets into the playbook. The playbook grows by small
additions, so it keeps detail instead of collapsing into a short summary.

ACE needs no correct answer. The Reflector judges a trajectory from what happened in it, such as errors, failed
commands and retries, and it uses a score or feedback only when the trace has one. In Reef, this method is a
small acceptance case for the method protocol: it learns from traces, keeps what it learns as method state and
one block of `rules.md`, and runs no evaluation episodes.

- Paper: [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
- Upstream code, for reference only: [ace-agent/ace](https://github.com/ace-agent/ace). The method does not
  depend on it, and its prompts are written for this method.

## Layout

```text
ace/
  method.py              ACE (the Method), ACESettings, the playbook, and the merge
  prompts/reflector.md   the Reflector's instructions
  prompts/curator.md     the Curator's instructions
  examples/serve.yaml    an online deployment on pi
  examples/seed/         the deployment's first tree: a rules.md with base rules
```

## How Reef runs ACE

| ACE | Reef |
| --- | --- |
| Generator | The served agent. It reads the playbook in `rules.md`, which every session reads. |
| Reflector | One model call per run that the step reads. It returns an error analysis, a key insight, and a tag (`helpful`, `harmful` or `neutral`) for each bullet that applies to the run. |
| Curator | One model call per run that the step reads. It returns `ADD` operations, each one new bullet in one section. |
| Merge, grow and refine | Code in `method.py`. It counts the tags, adds the bullets, and removes duplicates, harmful bullets and the bullets past the cap. |

`state/playbook.json` holds the playbook: `next_number`, the number of the next bullet's id, and the bullets,
each `{id, section, content, helpful, harmful, step}`. Each step renders the playbook into one block of
`rules.md`, between two marker lines:

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

### One step

A step reads runs: the traces of its material that have a score or a reply. A trace with neither is a model
call that ends on a tool call, in the middle of a run; the trace of the call that ends the run carries the
whole run, so the step skips the middle one. For each run, in order, and at most `max_traces_per_step` runs:

1. The Reflector reads the playbook and the trace. The trace shows the messages, the tool calls and their
   results without the system prompt, and the score, the feedback and the failure when the trace has them.
   The trace of a rollout (an episode of an offline step) has no trajectory, so it shows its task and its
   reply instead. A trace longer than 30,000 characters keeps its start and its end.
2. Each tag adds 1 to the `helpful` or the `harmful` count of its bullet. A tag for an id that is not in the
   playbook has no effect.
3. The Curator reads the sections, the playbook, the reflection and the trace's task, and returns `ADD`
   operations.
4. Each `ADD` becomes a new bullet with the next id, with these exceptions:
   - an operation that is not an `ADD`, names a section that is not configured, or has no text is ignored;
   - a text with the shape of a credential or of an instruction override is refused, and so is a text that
     holds a marker line of the block (`<!-- ace playbook: begin -->` or `<!-- ace playbook: end -->`), which
     would end the block early when the next step reads `rules.md`;
   - a text that a bullet already has, compared without case and extra spaces, is a duplicate and adds nothing.

Then the step removes each bullet whose `harmful` count exceeds its `helpful` count by `prune_margin` or more.
When more than `max_bullets` bullets remain, it removes the bullets with the lowest `helpful` minus `harmful`
count; of equal counts, the oldest bullet goes first. The step checks the new tree with `tools.check`, and then
writes `state/playbook.json` and `rules.md`. When the new tree fails the check, the step changes neither.

A Reflector reply that does not hold the asked JSON skips its trace. A Curator reply that does not hold it adds
no bullet for its trace, but the reflection's tags still count. The step's notes name each skipped trace with
the reason. The notes also count the tags, the operations (`added`, `duplicates`, `refused`, `ignored`), and the
bullets removed by pruning and by the cap. A step that changes nothing in the tree publishes nothing.

ACE learns from traces only. A step that answers a training request learns from the traces that come with the
request; the request's text reaches neither the Reflector nor the Curator.

## Configuration

The keys go under `evolution.method.config`:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `served` | The model of the Reflector and the Curator: `served` (the model under test, as in the paper) or a name under `evolution.models`. |
| `sections` | `Strategies and insights`, `Code and commands`, `Common mistakes to avoid`, `Others` | The sections of the playbook. The Curator puts each new bullet in one of them. |
| `max_bullets` | `100` | The most bullets that the playbook keeps. |
| `prune_margin` | `2` | Remove a bullet when its harmful count exceeds its helpful count by this many or more. |
| `max_traces_per_step` | `0` | The most runs (traces with a score or a reply) that one step reads; 0 reads every run. Each run that the step reads costs two model calls. |

## Deployment

### Online, on pi

[examples/serve.yaml](examples/serve.yaml) evolves the rules of pi from live sessions, with
`evaluation.policy: none`: Reef runs no episode and publishes every new playbook that passes validation. Its
seed, [examples/seed/rules.md](examples/seed/rules.md), holds base rules that the agent reads from the first
release; the method adds its block below them. Run it from the repository root:

```bash
export REEF_UPSTREAM_URL=http://127.0.0.1:8000 REEF_UPSTREAM_MODEL=qwen3-8b
reef serve -c recipes/ace/examples/serve.yaml
```

Point pi at this Reef as the client section of
[Evolve your harness](../../docs/user-guide/evolve-your-harness.rst) describes. After each session, post one
report with `POST /reef/report`: a `score`, optional `feedback`, and the receipts of the session's model calls in
`references`. The score does not have to be a correct answer; a plain 1 for done and 0 for not done is enough.
With `batch_size: 1`, each report starts one step. The step's record is under
`recipes/ace/work/harness/<scenario>/harness/steps/<step>/`: the new tree in `harness/`, the playbook in
`state/playbook.json`, each model call in `calls.json`, and the summary and notes in `result.json`.

To learn from recorded traffic without reports, name the inference stages in the deployment file. Then each
recorded request is one trace, without a score:

```yaml
pipeline:
  record2record:
    module: reef.record2record:InferenceRecord2Record
  record2sample:
    module: reef.record2sample.traces:InferenceSamples
```

To keep a basic check on each new playbook, use `evaluation.policy: floor` with a scorer and a health task: Reef
publishes the playbook only when the agent still passes the task.

### On a fixed task set

Use `evaluation.policy: compare` with a scorer and evaluation tasks. Reef then publishes a new playbook only when
it scores better than the served one on the same tasks. A rejected step keeps its playbook changes in `state/`,
so the next candidate carries them with that step's changes.

## Differences from the paper

- **Bullets used.** In the paper, the Generator names the bullets that it used, and the Reflector tags those. A
  served agent does not name them, so the Reflector reads the whole playbook and tags the bullets that apply.
- **One reflection per trace.** The paper lets the Reflector refine its analysis over several rounds, and its
  offline runs regenerate the answer. Reef does not run the served agent again, so the Reflector answers once.
- **De-duplication.** The method drops only exact duplicates, compared without case and extra spaces. The
  paper's de-duplication by semantic embeddings, and the upstream model merge of similar bullets, are out of
  scope.
- **Refinement by counts.** The paper keeps the playbook within a token budget. This method removes bullets by
  their counts (`prune_margin`) and keeps at most `max_bullets`.
- **Offline epochs.** The paper revisits the training samples for several epochs and keeps the playbook with
  the best validation score. In Reef, the evaluation policy decides which playbook is served.
- **Screens.** The prompts fence trace text as data, and the merge refuses a bullet with the shape of a
  credential or of an instruction override, or with a marker line of the block. The paper has none of these.
- **No labels, weak signal.** The paper notes that without labels or reliable feedback, the playbook can learn
  wrong lessons. The harmful counts and an evaluation policy limit that risk; they do not remove it.
