# reasoning_bank

A Reef harness evolution method based on [ReasoningBank](https://arxiv.org/abs/2509.25140) (Ouyang et al.). The
method learns from the agent's own runs and needs no ground truth. A judge model labels each unscored run a success
or a failure. An extractor model then writes memory items from the run: strategies from a success, lessons from a
failure. By default Reef serves each item as an Agent Skill. On pi and native, you can instead select all items
or the most similar items by embedding, and put them in the system prompt, a message or a tool result.

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
5. **Write the tree.** The step writes skills or a runtime reader, according to `reading`. It replaces the
   method's previous memory files and preserves all other files. The step checks the new tree with
   `services.check`. When the check fails, neither the tree nor the bank changes.
6. **Measure when due.** With `ablate_every: N`, every N valid steps compare the current published tree with a
   copy that has no ReasoningBank memory. The result records whether memory helped on the evaluation tasks.

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
| `ablate_every` | `0` | Compare the published tree with and without memory every N valid method steps. `0` disables this. |
| `reading.reader` | `skills` | `skills` uses the agent's skills; `all` reads the whole bank; `top_k` selects by embedding similarity. |
| `reading.inject` | `system` | For `all` and `top_k`: `system`, `message`, or `tool`. |
| `reading.top_k` | `1` | Maximum items returned by `top_k`. Ties keep the bank's order. |
| `reading.embedding_url` | empty | Full OpenAI-compatible `/v1/embeddings` URL, required only for `top_k`. No credential, query or fragment. |
| `reading.embedding_model` | empty | Embedding model name, required only for `top_k`. |
| `reading.timeout_seconds` | `20` | Positive, finite timeout for one embedding request. |

Each run costs at most two model calls: one for the judge (only without a score) and one for the extractor. If you
set `evolution.max_model_calls_per_step`, allow two calls per run. A step that runs out of calls fails.

## Choose what the agent reads

For pi or native, add this under `evolution.method.config` to retrieve the most relevant item:

```yaml
reading:
  reader: top_k
  inject: system
  top_k: 1
  embedding_url: http://127.0.0.1:8080/v1/embeddings
  embedding_model: your-embedding-model
```

The endpoint must accept `{"model": "...", "input": ["..."]}` and return indexed float vectors in `data`.
Each read embeds the query and every item's title, description and content in one request. Cosine similarity
chooses the items. The reader rejects missing, duplicate, non-finite or zero vectors; it never falls back to
the whole bank. Use `reading: {reader: all, inject: system}` when no embedding service is needed.

The query for system and message injection is the turn's user prompt. For tool injection, the model supplies
the `memory_search` query. The modes behave as follows:

| Mode | pi | native |
| --- | --- | --- |
| `system` | Append selected items before each agent turn. | Retrieve once per turn and reuse the items in each model step's rebuilt system prompt. |
| `message` | Add one custom message before each agent turn. | Add one user message at the turn's first model step. |
| `tool` | Register `memory_search`; read only when the model calls it. | Register the same tool; read only when called. |

Native system mode keeps its cache within the current turn, separately for each subagent. A change to the
bank's contents or reading settings triggers a new read at the next model step. Reloading the same bank
preserves the cache. A new turn retrieves again, even when its prompt matches the previous turn.

The text starts with a fixed sentence identifying memories as untrusted data. Each successful read records
the selected item IDs, reader and injection mode. pi stores a `memory/read` custom session entry. Native stores
`memory_read` in the hook decision or tool result. Native system mode also records these IDs when reusing
the cache. A failed system read records `status: failed` and supplies no memory; it retries only after the
bank changes or a new turn starts. Other failed reads produce pi error entries or native hook/tool errors.
A hook failure leaves the turn without retrieved memory; a tool failure returns an error to the model.

The reader and its bank travel together in the versioned tree, so evaluation and clients run the same code.
ReasoningBank owns `skills/rb-*`, `extensions/reasoning_bank.ts`, `native/hooks/reasoning_bank.py` and
`native/tools/memory_search.py`. Do not put unrelated code at those paths. Changing the reader replaces the
previous delivery files; an empty bank creates no reader.

### Embedding endpoint and credentials

The embedding request runs on the agent host, including during evaluation. That host sends the task query and
bank text to the configured endpoint. Use an endpoint reachable from every client and evaluation worker;
`127.0.0.1` refers to each worker's own host. The endpoint URL and model are published, but credentials are not.

If the endpoint needs authentication, set `REEF_MEMORY_EMBEDDING_API_KEY` in the agent process. The reader uses
it as a bearer token. It does not follow redirects. Client shells can export this variable before launching
pi or native. For sandboxed evaluation, list the variable in `evolution.sandbox.env_from` and permit the
endpoint host in `egress_hosts` where required.

Local evaluation does not inherit this variable. Set `evolution.binary` to a wrapper that reads a protected
credential file and then starts the actual agent, for example:

```sh
#!/bin/sh
set -eu
export REEF_MEMORY_EMBEDDING_API_KEY="$(cat /absolute/private/embedding-key)"
exec /absolute/path/to/pi "$@"
```

Use `reef-native` in the wrapper for native. Native's optional per-tool `bwrap` enforcement removes environment
credentials, so `tool` mode under that enforcement needs a reachable endpoint without bearer authentication.
No reader embeds a key in a snapshot. Embedding requests are separate from the method's chat-call budget;
native `top_k` with `system` injection makes one request per turn, plus one after each change to the bank or reading settings.

## Measure whether memory helps

Set `ablate_every` in the method config and configure evaluation tasks and a scorer. The scorer must give
higher values to better runs. For pi with an existing Harbor task split, use:

```yaml
evaluation:
  policy: none
  task_manifest: /absolute/path/to/tasks/split.json
  tasks_root: /absolute/path/to/tasks
  scorer: reef.evaluation.harness:VerifierRewardScorer
```

This uses the manifest's evaluation split and runs pi in each task's container with the local executor.
The embedding endpoint must be reachable there too. For native, use prompt strings in `evaluation.tasks`
and a prompt scorer implementing `TaskScorer`; native does not install inside Harbor containers.
See [evaluation configuration](../../docs/user-guide/evolve-your-harness.rst) for task and scorer contracts.

With `ablate_every: 10`, every tenth valid method step evaluates the published tree and a copy without its
ReasoningBank files. It uses the same tasks, repeat count, model bindings and scorer on both sides. Other rules,
skills and extensions stay intact. An empty published bank records `empty` without launching tasks.

This also runs under `none` and `floor`, and when the step leaves the tree unchanged. `policy: none` publishes
valid memory updates without evaluating every write; the periodic comparison still runs. Existing successful
runs are reused from the evaluation cache. Allow up to twice the task count times `repeats` in
`evolution.max_runs_per_step`, plus any training or publication evaluations.

The commit's `method_notes.memory_ablation` holds both mean scores, their difference (`with_memory` minus
`without_memory`), pair counts and run IDs. Only successful matching task/repeat pairs enter both means.
No usable pairs gives `inconclusive`. The cadence and last result persist in `state/memory_evaluation.json`,
including across restarts. The comparison describes these tasks and cached runs; it is not a significance test.
A negative difference records `harmful`; it does not automatically delete items or roll back the tree.

## Deploy it

[examples/serve.yaml](examples/serve.yaml) runs the method online on pi, without ground truth:

- The `pipeline` section names the inference stages. Every model call that a client sends through Reef is a
  trace, and no client has to report a score. The judge labels each run.
- `evolution.evaluation.policy: none` runs no task. Every step whose tree passes validation publishes.
- The served model judges and extracts, as in the paper.

The example serves `qwen3-8b` from `http://127.0.0.1:8000`; edit the `inference` section for another model. Start
it from the repository root:

```bash
reef serve -c recipes/reasoning_bank/examples/serve.yaml
```

Then connect pi clients to the scenario, as [Evolve your harness](../../docs/user-guide/evolve-your-harness.rst)
describes. Each published snapshot carries the bank as skills or as the configured runtime reader.

Other setups change one part of the example:

- If the clients report a score for each run (`POST /reef/report`), delete the `pipeline` section. Each trace is
  then a scored run, and its score sets the outcome.
- For a stronger judge, declare the model under `evolution.models` (`url`, `model`, `api_key_env`) and set
  `judge_model` to its name.
- With `evaluation.policy: compare` or `floor`, Reef runs tasks on each new tree before it publishes. The bank
  keeps every item that it learned, whether or not the tree was published, and each step proposes the whole bank
  again.

## Differences from the paper

- **Retrieval.** The default uses agent-selected skills. `reading.reader: top_k` with `inject: system` provides
  embedding selection and system injection, as in the paper. This implementation embeds each item's text;
  it does not keep and embed the original task that produced the item.
- **Labels.** The paper judges every run. Here a run with a score takes its outcome from the score, and only an
  unscored run goes to the judge.
- **Consolidation.** The paper adds items and never removes one. Here the extractor sees the bank's titles and can
  replace an item when it reuses the item's title, and the bank keeps at most `max_items` items.
- **Screening.** The paper keeps every item. Here an item with a credential-shaped literal or an instruction
  override stays out of the bank.
- **Inputs.** The judge and the extractor read one JSON view of the run: the task, the transcript without the
  system prompt, the final reply, the feedback and the failure. The run of an offline step (a run) has no
  transcript in its trace. Past 30,000 characters, the view loses the middle of its transcript. The paper's judge
  for web tasks also reads the final page state.
- **Prompts.** The prompts are Reef's own and ask for JSON. Reef sends no sampling parameters, and the paper
  samples the judge at temperature 0 and the extractor at temperature 1.
- **Scaling.** Memory-aware test-time scaling (MaTTS), which spends more runs on each task to give the extractor
  more varied runs, is not part of this method.
