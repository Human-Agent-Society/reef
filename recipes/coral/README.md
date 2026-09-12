# CORAL test-time training through Reef and Slime

Runs a real CORAL task with fully attributable inference: CORAL's own runtime
(agent worktrees, gateway, grader daemon) drives coding-agent CLIs whose every
call flows through Reef, evaluator scores return as training data, weights
update, and later attempts serve from the new revision. Implements
[issue #3](https://github.com/Human-Agent-Society/reef/issues/3).

Pinned upstream: [CORAL](https://github.com/Human-Agent-Society/CORAL) commit `0123dfb`.

## Quickstart (2 GPUs)

```bash
cd recipes/coral/examples/coral_demo
pip install -e .[coral]        # demo deps + CORAL at the pinned commit
npm install -g opencode-ai     # or any CORAL runtime CLI; select with --runtime
./run.sh
```

`run.sh` boots the Reef training stack (`serve.yaml`: `CoralRecipe`, Qwen3-8B
LoRA), then `run.py` builds CORAL's `AgentManager` from the real task in
`task/` — CORAL spawns its agent runtimes over `task/seed`, starts its LiteLLM
gateway from `task/litellm_config.yaml` (whose only upstream is the Reef
service), and grades `coral eval` submissions with the packaged grader in
`task/grader`. The reef correlation layer is spliced under the gateway before
any agent spawns; an attempt watcher reports every finalized attempt to Reef
for training. The run auto-stops at its attempt budget
(`--max-attempts`) and writes `work/coral-demo/bundle.json`.

The adapter tests need neither GPU nor CORAL: `python -m pytest tests/test_coral_*.py`.

## Verifying without GPUs

`examples/coral_demo/smoke/run_smoke.sh` runs `run.py` unmodified against the
production Reef service with a canned inference backend
(`smoke/echo_reef_service.py`) and a scriptable CORAL runtime
(`smoke/scripted_runtime.py`). Every wire interaction — worktrees, gateway
key swap, header stamping, receipt capture into the journal, grader-daemon
scoring in the grader venv, exactly-once reporting, the result bundle — is
the real code path; only the model's answers and the agent's "intelligence"
are canned. `coral validate task/` separately proves the task package is a
well-formed CORAL task.

## How the pieces line up

```
CORAL AgentManager (real runtime: worktrees, agent CLIs, per-agent proxy keys,
   |               grader daemon, heartbeats, restarts, attempt budget)
   v
CORAL gateway (identity: x-coral-agent-id, x-coral-session-id)
   v
recipes.coral.middleware      stamps x-reef-scenario + x-reef-tag-coral-{run,agent,commit},
   v                          captures reef receipts -> journal
LiteLLM -> reef serve         stores INFERENCE records with tags,
   v                          answers with x-reef-agent-record-id / receipt SSE frame
CORAL grader daemon finalizes the attempt (.coral/**/attempts/*.json)
   -> recipes.coral.watcher: resolves the attempt's captured references,
      POST /reef/report {score, references, metadata.coral}, exactly once
   -> CoralProcessor groups siblings of one parent commit -> Slime LoRA step
   -> new revision served to the next attempts
```

One discovery problem is one reef scenario; agent and worktree identity live in
tags, so parallel agents share the evolving policy without fragmenting the
scenario.

## Correlation model

The `coral-agent`/`coral-commit` tags reef stores with each INFERENCE record
are the primary correlation key and survive any proxy behavior. The journal
additionally captures reef's response receipts, so reports reference exact
record ids; a stripped receipt degrades to tag-only correlation and is never
fatal. Reports carry a deterministic client-supplied id, so a reporter retry
or crash-replay dedups server-side.

CORAL's gateway stamps each call with the worktree's HEAD at call time — the
*parent* commit the agent was editing, not the commit `coral eval` creates
afterwards. The watcher therefore resolves an attempt's references at the
(agent, parent) coordinate, claiming journal records in order so consecutive
attempts from the same parent (a revert, a retried eval) never share a
reference.

## What gets reported

Real attempts, in their first terminal state, exactly once. CORAL's
`grader_error` attempts (the eval machinery broke, no policy signal) and
`tune` attempts (config sweeps CORAL itself excludes from budgets) are
skipped; archived attempts are ignored.

## How a training step works

The training side treats CORAL's attempt tree as the grouping structure for
grouped relative-reward training. Every attempt names the commit it started
from (`parent_hash`), so the scored siblings of one parent are a set of
rollouts that began from the same code and diverged, which is exactly the
comparison set a relative-reward step needs. First-generation attempts have no parent
and are grouped together: they all diverged from the task seed.

From a report to a gradient:

1. **Grouping.** `CoralProcessor` files each report under its `parent_hash`.
   A group releases once `group_size` scored siblings accrue (`serve.yaml`
   sets 2 for the demo; the default is 4). Parents that never accrue enough
   scored children never train, the barrier is a recipe setting, not a CORAL
   invariant.
2. **Sample assembly.** The report's ordered inference references are
   materialized back into tokens, loss mask, and the rollout-time log
   probabilities reef stored with each INFERENCE record. An attempt is a
   multi-call trajectory, so multi-turn assembly is on by default.
3. **Version guards.** An attempt whose calls span a weight update is
   rejected: its rollout log probs belong to two policies and cannot support
   one importance ratio. A group whose members trained on different
   revisions is discarded whole, because relative rewards only compare
   fairly within one policy version. Both outcomes are visible in the
   processor's `status()` (`never_reasons`, `discarded_groups`).
4. **The step.** A released group becomes one training unit. The recipe
   reuses the `tttd` step preparer (grouped leave-one-out advantages) and
   the `tttd` Slime loss family, CORAL sibling groups have the same shape
   as TTT-Discover steps, so the recipe adds only the group barrier. A group
   where every sibling scored the same still trains as a well-defined
   zero-gradient step rather than starving the barrier.
5. **The loop closes.** The step updates the served LoRA (the demo:
   Qwen3-8B, rank-32 LoRA on `linear_qkv`/`linear_proj`, trained with the
   rollout log probs and a KL term against the base model, see
   `examples/coral_demo/serve.yaml`). Reef serves the new revision, and
   later attempts both run on it and are attributed to it.

The design rationale lives next to the code: `processor.py` (grouping and
guards), `recipe.py` (why `tttd` is reused), `serve.yaml` (deployment knobs).

## The task

`task/` is an ordinary CORAL task — `coral validate task/` accepts it. The
demo problem (implement a stable two-list merge, scored by fraction of checks
passed) is deliberately small so the loop turns over quickly on a 2-GPU
serving stack; swap in any CORAL task by editing `task/` — the wiring does not
change. `task/litellm_config.yaml` is the one reef-specific piece: the
gateway's only upstream is the Reef service, so the served policy is the
agents' only model.

## Known limitations

- `attach_reef_adapter_to_agent_manager` splices under the middleware CORAL's
  manager builds internally. CORAL's gateway now exposes a `header_provider`
  hook, but the reef layer also mirrors headers into request bodies and
  captures response receipts — outside a request-header hook's reach — so the
  splice stays until CORAL grows a response-side hook.
- LiteLLM builds a fresh upstream request, so the middleware mirrors the
  scenario and tags into the body's `extra_headers`; receipt headers are
  matched by suffix because a forwarding proxy prefixes them. Both behaviors
  came out of live GPU runs.
- The demo defaults to the `opencode` runtime; any runtime CLI CORAL registers
  works (`--runtime claude_code`, `codex`, ...), but the CLI must be installed
  and must accept an OpenAI-compatible gateway endpoint for a locally served
  model.
