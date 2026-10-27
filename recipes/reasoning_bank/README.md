# reasoning_bank

A Reef harness evolution method based on [ReasoningBank](https://arxiv.org/abs/2509.25140) (Ouyang et al.). The
method learns from the agent's own runs and needs no ground truth. A judge model labels each unscored run a success
or a failure. An extractor model then writes memory items from the run: strategies from a success, lessons from a
failure. Reef serves each item as an Agent Skill, so the agent opens the items that fit its task.

- Paper: [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- Method: `recipes.reasoning_bank:ReasoningBank` in [method.py](method.py), with its prompts in [prompts/](prompts/)
- Example deployment: [examples/serve.yaml](examples/serve.yaml), online on pi
- Tests: [tests/recipes/reasoning_bank/](../../tests/recipes/reasoning_bank/)

## What one step does

1. **Find the runs.** A trace with a task is a run when it has a score, or when its last step is a reply and not a
   tool call. Unscored traffic arrives one model call at a time. The call that ends with a reply carries the
   transcript of the whole run, so the step skips the calls before it.
2. **Label each run.** A run with a score of at least `success_score` is a success, and a run with a lower score
   is a failure. A run without a score goes to the judge model ([prompts/judge.md](prompts/judge.md)). The judge
   reads the task, the transcript, the final reply and the client's feedback, and returns `success` or `failure`.
3. **Write items.** The extractor model reads the run with the prompt for its outcome
   ([prompts/extract.md](prompts/extract.md) with [prompts/success.md](prompts/success.md) or
   [prompts/failure.md](prompts/failure.md)). It returns at most `max_items_per_trace` items, each with a title,
   a one-sentence description and content. It also sees the titles that the bank holds.
4. **Update the bank.** An item's slug is its title in lowercase ASCII words joined by hyphens. An item with a new
   slug joins the bank, and an item with a slug that the bank holds replaces that item. Past `max_items` items,
   the oldest items leave the bank.
5. **Write the tree.** The step writes one skill per item and removes every other `skills/rb-*` skill. The other
   files of the tree do not change. The step checks the new tree with `tools.check`. When the check fails, neither
   the tree nor the bank changes.

A reply that is not the asked JSON skips its run. An item that carries a credential or an instruction override
stays out of the bank, because every session reads the tree. The step's notes (`method_notes`) count the runs and
the outcomes, name the items that the step added, replaced and dropped, and give the reason for each skip.

The bank is the method's state, `state/memory.json`: a list of items `{slug, title, description, content,
outcome, step}`, oldest first. Reef commits the state at the end of each step, whether or not the step publishes a
tree.

## Memory items as skills

Each item is the skill `skills/rb-<slug>/SKILL.md`. The frontmatter holds the skill name and the item's
description; the body holds the title and the content:

```markdown
---
name: rb-run-the-tests-before-you-report
description: Use when a task asks you to fix code and say that the fix works.
---

# Run the tests before you report

- Run the project's tests after the change.
- Report the fix only when the tests pass, and show their result.
```

The harness does the retrieval. It lists the name and the description of every skill, and the agent opens the
skills that fit its task (progressive disclosure). So the description works as the retrieval key, and the
extractor prompt asks for a description that names the situation, not the solution. Every Reef adapter renders
skills, so the method runs on every adapter.

## Configuration

The settings go under `evolution.method.config`:

| Setting | Default | What it sets |
| --- | --- | --- |
| `judge_model` | `served` | The model that labels a run without a score: `served` or a name under `evolution.models`. |
| `extract_model` | `served` | The model that writes the items of a run. |
| `success_score` | `1.0` | A run with at least this score is a success. A run with a lower score is a failure. |
| `max_items_per_trace` | `3` | The most items that one run adds. |
| `max_items` | `50` | The most items that the bank keeps. The oldest items leave first. `0` sets no limit. |
| `max_traces_per_step` | `0` | The most runs that one step learns from, in trace order. `0` sets no limit. |

Each run costs at most two model calls: one for the judge (only without a score) and one for the extractor. If you
set `evolution.max_model_calls_per_step`, allow two calls per run. A step that runs out of calls fails.

## Deploy it

[examples/serve.yaml](examples/serve.yaml) runs the method online on pi, without ground truth:

- The `pipeline` section names the inference stages. Every model call that a client sends through Reef is a
  trace, and no client has to report a score. The judge labels each run.
- `evolution.evaluation.policy: none` runs no episodes. Every step whose tree passes validation publishes.
- The served model judges and extracts, as in the paper.

The example serves `qwen3-8b` from `http://127.0.0.1:8000`; edit the `inference` section for another model. Start
it from the repository root:

```bash
reef serve -c recipes/reasoning_bank/examples/serve.yaml
```

Then connect pi clients to the scenario, as [Evolve your harness](../../docs/user-guide/evolve-your-harness.rst)
describes. Each published release carries the bank as `skills/rb-*`.

Other setups change one part of the example:

- If the clients report a score for each run (`POST /reef/report`), delete the `pipeline` section. Each trace is
  then a scored run, and its score sets the outcome.
- For a stronger judge, declare the model under `evolution.models` (`url`, `model`, `api_key_env`) and set
  `judge_model` to its name.
- With `evaluation.policy: compare` or `floor`, Reef runs episodes on each new tree before it publishes. The bank
  keeps every item that it learned, whether or not the tree was published, and each step proposes the whole bank
  again.

## Differences from the paper

- **Retrieval.** The paper embeds the task and puts the most similar items (by default one) into the system
  prompt, with their titles and content. Here the agent sees the description of every item and reads the items
  that it chooses.
- **Labels.** The paper judges every run. Here a run with a score takes its outcome from the score, and only an
  unscored run goes to the judge.
- **Consolidation.** The paper adds items and never removes one. Here the extractor sees the bank's titles and can
  replace an item when it reuses the item's title, and the bank keeps at most `max_items` items.
- **Screening.** The paper keeps every item. Here an item with a credential-shaped literal or an instruction
  override stays out of the bank.
- **Inputs.** The judge and the extractor read one JSON view of the run: the task, the transcript without the
  system prompt, the final reply, the feedback and the failure. Past 30,000 characters, the view loses the middle
  of its transcript. The paper's judge for web tasks also reads the final page state.
- **Prompts.** The prompts are Reef's own and ask for JSON. Reef sends no sampling parameters, and the paper
  samples the judge at temperature 0 and the extractor at temperature 1.
- **Scaling.** Memory-aware test-time scaling (MaTTS), which spends more runs on each task to give the extractor
  more varied runs, is not part of this method.
