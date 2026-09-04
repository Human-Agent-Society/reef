# The upstream arm

Upstream Meta-Harness, run on the same proposer, target model, executor agent
and sandbox as the Reef arm, so a difference between the two arms is a
difference between the implementations rather than between their settings.

Upstream has no provider seam: `propose_claude` calls
`claude_wrapper.run(model="opus")` directly. Running it on any other proposer
therefore requires modifying it, so this arm is upstream's loop with a
substitution, not stock upstream. `meta_harness.patch` is the whole
difference, and it deliberately touches nothing in the search:

| Change | Why |
| --- | --- |
| `codex_wrapper.py` | A Codex CLI session with the same `run()` signature and the two fields the caller reads, `exit_code` and `stderr`. The proposer writes `pending_eval.json` itself, as before. |
| `import claude_wrapper` -> `codex_wrapper` | The substitution. |
| `model="opus"` -> `META_HARNESS_PROPOSER_MODEL` | Both arms name the same proposer model. |
| `MODEL` reads `HARBOR_MODEL` | Both arms name the same target model. |
| Forward `OPENAI_API_KEY`, `E2B_API_KEY` | Upstream forwards only Anthropic and Runloop credentials. |
| `run_eval.sh` accepts `e2b` | It allowed only `runloop` and `modal`. Harbor's local Docker default collects no verifier reward here. |
| `subset` task set | Runs a named subset instead of all 89, for a bounded comparison. |
| `terminal-bench` git pin dropped | The pinned commit is rewritten history and no longer fetchable; the release artifact it declared, `0.2.18`, is used instead. The earlier reproduction hit this too. |

The frontier rule, the population, the baseline phase, and the evaluation are
untouched. `replay.py` compares this arm's `update_frontier` against Reef's
selector directly, which is where the matching claim comes from.

## One defect found while wiring it

Codex finishes its work and then blocks on `Reading additional input from
stdin...`, so a session never returns and the iteration times out with its
files already written. `codex_wrapper` passes `stdin=DEVNULL`. Reef's own
executor carries the same fix for the same reason.
