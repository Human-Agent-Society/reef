# Polyominoes (Frontier-CS algorithmic #0), single agent

Smoke check for the changes in this branch: longest-linear-suffix assembly,
release truncation, `--resume --fresh-sessions`. Qwen3-Coder-30B-A3B-Instruct,
one agent, `group_size: 8`, `group_by: release`, 40-attempt budget. The task
package is CORAL's `examples/frontier_cs_algo/0`; the grader submits to a local
Frontier-CS go-judge and reports the judge's 0-100 score.

| arm | evals | best score | LoRA steps |
|---|---|---|---|
| frozen policy | 43 | 1.08 | 0 |
| trained policy | 39 | 56.84 | 4 |

Paper reference (Table 2, Claude Code + Opus 4.6): SOTA 87.0, CORAL 1-agent 80.2,
4-agent 84.2.

One run per arm; a smoke signal, not a result. The frozen agent never got past
output-format errors (every attempt "Wrong Answer" on all 70 cases). Both arms
were resumed in place with fresh sessions whenever no attempt had been graded
for 45 minutes. The trained arm's last reef status snapshot before shutdown: the suffix
fallback kept 208 of 537 calls across 13 forked attempts, and 2 attempts were
cut to their post-update suffix. `*.events.log` hold every graded
attempt and every report posted to reef.
