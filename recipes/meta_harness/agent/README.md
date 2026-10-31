# Meta-Harness with a coding agent as the proposer

This folder is a method made of prompts (`reef.train.harness_backend:AgentMethod`). Each step, the deployment's
coding agent reads the search's history as files, branches from any candidate, writes one complete tree in
`harness/`, and checks and scores it through Reef's tools. This is the upstream Meta-Harness proposer: a coding
agent with the filesystem of the search. `recipes.meta_harness:MetaHarness` is the same search with one model call
per step, which sees an excerpt of the history in its prompt.

`START.md` is the prompt of every step. The search keeps its state in the history, so the folder has no
`RESUME.md`. Select as for `MetaHarness`, with `policy: compare` and `statistic: mean`:

```yaml
evolution:
  method:
    module: reef.train.harness_backend:AgentMethod
    config:
      prompts: recipes/meta_harness/agent
      model: proposer              # an evolution.models entry, never the model under test
  agent:
    adapter: claude                # the coding agent that proposes
    sandbox: bwrap
  evaluation:
    policy: compare
    statistic: mean
```
