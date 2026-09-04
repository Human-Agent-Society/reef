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

## Guards added for a shared, oversubscribed account

The e2b account these runs use is shared, and this project's slice is 32 of
its 100 sandboxes. Other teams routinely hold more than the remainder, so a
job can find no capacity at all. Three changes in `meta_harness.patch` exist
for that:

| Change | Why |
| --- | --- |
| `charge_job` / `SpendCapReached` | Upstream reports cost but never stops. A run left overnight had no ceiling. |
| `check_job_health` / `OutageDetected` | A trial that raises before producing a reward measured nothing, but Harbor still means over whatever completed. One iteration scored 0.033 from 3 completed trials and 57 `RateLimitException`s, and was recorded as "no improvement" against the frontier. |
| `META_HARNESS_ONLY_BASELINE` | Phase 0 prices every baseline. When the run exists to match another arm's executor, the baselines that arm does not use are spend without a comparison to spend it on. |

`run_eval.sh` also drops `--ak temperature=0.7` and adds `--max-retries` with
`--retry-include RateLimitException`. The temperature is not a tuning choice:
`gpt-5.6-luna` rejects function tools unless temperature is 1, and the Reef arm
sends no temperature, so omitting it fixes the request and matches the arms at
once. The retry helps with a brief spike and does not survive a sustained one --
three attempts were exhausted against a full account.
