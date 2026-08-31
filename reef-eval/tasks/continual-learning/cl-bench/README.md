# CL-Bench

[Continual Learning Bench](https://www.continual-learning-bench.com)
([paper](https://arxiv.org/pdf/2606.05661), Apache-2.0): expert-validated
tasks where an agent works through sequential instances of one
environment and should improve by remembering what it saw. Their "system"
(agent plus memory strategy) maps to reef-eval's agent plus carried
`$REEF_EVAL_STATE_DIR`, and their gain metric (stateful minus stateless
reward) is `metrics.transfer` against a plain isolated `reef-eval run` sweep.

All six domains are converted: 301 tasks, the benchmark's full instance
count, committed to this repo. One instance = one task; name order
replays each domain's lifecycle. Every scorer is deterministic and
offline.

| Domain | Tasks | What it is | Reward |
|---|---|---|---|
| `bsm-sNN` | 90 | infer a radio band's transmitter layout across scans | availability IoU (0-1) |
| `sales-iNN` | 12 | yearly 5-year demand forecasts from a shifting data room | WAPE-skill (1 = perfect, can go negative) |
| `cohort-iNN` | 20 | rolling survival meta-analysis across biased studies | information gain in bits over the study baseline |
| `code-iNN` | 19 | sequential real-PR bugfixes in tablib, then tenacity | hidden tests pass = 1.0 |
| `dbx-qNN` | 40 | questions over an unknown SQLite db, 15-query budget | 1 − queries/15 if correct, else 0 |
| `poker-hNNN` | 120 | heads-up hold'em vs scripted exploitable opponents | profit in big blinds |

How the conversions work (`convert_<domain>.py` in this folder):

- bsm, sales, and cohort are self-contained predict-and-score tasks; the
  data rooms and scorers use the upstream code, vendored where practical.
- codebase builds on the upstream-published Docker images, prepares the
  workspace at image build the way upstream does at runtime, and the
  verifier replays the upstream evaluation on the exact test node ids.
- dbx and poker keep their hidden state (the database, the deck) in a
  judge sidecar the agent reaches only over HTTP, so the query budget
  and the deal are enforced out of the agent's hands. Decks match the
  upstream harness card for card.
- Agent containers run offline (`network_mode = "allowlist"`), as
  upstream; harness adapters allowlist their own LLM API hosts.

Every task ships a reference solution with a known exact score: the
truth-derived report (bsm and sales 1.0; cohort scores its per-instance
ceiling, recorded as `oracle_score`), the gold PR patch (codebase 1.0),
the direct correct answer (dbx 1.0), or a simulated check/call line
(poker, per-hand `oracle_reward`).

Deviations from upstream, per domain: sales, cohort, and codebase swap
the upstream step- or action-metered interaction for free shell work
under a time budget (codebase's step-count reward becomes pass/fail, and
cohort's six-tool API becomes direct SQLite access); the persistent
workspace is `$REEF_EVAL_STATE_DIR` rather than a reused `/app`; and where
upstream delivered feedback conversationally, the instructions carry the
same information (dbx includes the previous question's correct answer).
The dbx query budget and the poker deal are not deviations; the sidecar
enforces them exactly.

The tasks are committed, so they run out of the box:

```bash
reef-eval stream cl-bench --agent claude-code   # every domain, in order
reef-eval stream tasks/continual-learning/cl-bench/poker-* --agent claude-code
reef-eval fetch cl-bench bsm sales    # only to regenerate from the pinned sources
```

Regenerating downloads from the pinned upstream commit and HuggingFace
revisions, sha256-verified; the poker domain also needs
`pip install texasholdem==0.11.0` for the oracle simulation. First runs
are heavy in two places: dbx task images download the published database
(~0.4 GB, cached across tasks by Docker layer) and codebase tasks pull
the upstream repo images.
