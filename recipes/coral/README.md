# CORAL test-time training through Reef and Slime

Runs a CORAL discovery task with fully attributable inference: every agent call flows
through Reef, evaluator scores return as training data, weights update, and later
attempts serve from the new revision. Implements
[issue #3](https://github.com/Human-Agent-Society/reef/issues/3).

Pinned upstream: [CORAL](https://github.com/Human-Agent-Society/CORAL) commit `a69cbc2`.

## Quickstart (2 GPUs)

```bash
cd recipes/coral/examples/coral_demo
pip install -e .[coral]     # demo deps + CORAL at the pinned commit
./run.sh
```

`run.sh` boots the Reef training stack (`serve.yaml`: `CoralRecipe`, Qwen3-8B LoRA),
then `run.py` builds CORAL's `GatewayManager` against the Reef endpoint, splices the
correlation layer, and drives two demo agents whose graded attempts train the served
model. The run's result bundle lands in `work/coral-demo/bundle.json`.

The adapter tests need neither GPU nor CORAL: `python -m pytest tests/test_coral_*.py`.

## How the pieces line up

```
CORAL agents (parallel worktrees, per-agent proxy keys)
   v
CORAL gateway (identity: x-coral-agent-id, x-coral-session-id)
   v
recipes.coral.middleware      stamps x-reef-scenario + x-reef-tag-coral-{run,agent,commit},
   v                          captures reef receipts -> journal
LiteLLM -> reef serve         stores INFERENCE records with tags,
   v                          answers with x-reef-agent-record-id / receipt SSE frame
CORAL grader finalizes the attempt
   -> recipes.coral.reporter: POST /reef/report {score, references, metadata.coral}
   -> CoralProcessor groups siblings of one parent commit -> Slime LoRA step
   -> new revision served to the next attempts
```

One discovery problem is one reef scenario; agent and worktree identity live in tags,
so parallel agents share the evolving policy without fragmenting the scenario.

## Correlation model

The `coral-agent`/`coral-commit` tags reef stores with each INFERENCE record are the
primary correlation key and survive any proxy behavior. The journal additionally
captures reef's response receipts, so reports reference exact record ids; a stripped
receipt degrades to tag-only correlation and is never fatal. Reports carry a
deterministic client-supplied id, so grader re-runs dedup server-side. Sibling attempts
run from the same worktree state; resolve their references with the journal cursor
(`size()` before the attempt's calls, `record_ids_since()` after).

## Known limitations

- `attach_reef_adapter` splices under CORAL's middleware object because CORAL builds it
  inline at the pinned commit; CORAL's gateway `header_provider` hook (merged upstream)
  will replace the splice once released.
- LiteLLM builds a fresh upstream request, so the middleware mirrors the scenario and
  tags into the body's `extra_headers`; receipt headers are matched by suffix because a
  forwarding proxy prefixes them. Both behaviors came out of live GPU runs.
- The demo grader is deterministic and local; replace it and the scripted agents with
  CORAL's real task and runtimes for a full deployment — the wiring is unchanged.
