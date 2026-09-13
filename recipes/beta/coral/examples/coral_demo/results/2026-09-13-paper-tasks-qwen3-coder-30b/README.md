# CORAL TTT on the CORAL paper's 11 tasks (Qwen3-Coder-30B-A3B)

Tasks from [arXiv:2604.01658](https://arxiv.org/abs/2604.01658) (Table 1 SOTA
column, Table 2 OpenCode + MiniMax M2.5 rows), run with Qwen3-Coder-30B-A3B-Instruct
through reef. Four arms per task: policy frozen or trained during the run, 1 or 4
agents. The paper columns are context (its open-source model is ~230B); the
controlled comparison is frozen vs. trained with everything else equal.

Frozen arms use `serve-coder30b-baseline.yaml` (same stack, group size never
releases). Trained arms use `serve-coder30b.yaml`: rank-32 LoRA, `group_size: 2`,
`group_by: release`. Budget: 40 attempts (1 agent), 120 (4 agents).

| Task | Dir | SOTA | paper M2.5 1-agent | paper M2.5 4-agent | ours 1-agent frozen | ours 1-agent TTT | ours 4-agent frozen | ours 4-agent TTT | evals (frozen/TTT, 1-agent) | evals (4-agent) | TTT steps (1/4-agent) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Circle-Pack. | ↑ | 2.6359 | 2.3531 | 2.5391 | 2.1774 | 2.4426 | 2.4828 | 2.5440 | 43/42 | 124/123 | 16/32 |
| Signal Proc. | ↑ | 0.7429 | 0.7174 | 0.7383 | 0.6407 | 0.5462 | 0.6862 | 0.6589 | 40/41 | 122/123 | 17/28 |
| Erdos Over. | ↓ | 0.38088 | 0.39237 | 0.38311 | 0.39319 | 0.39589 | 0.39201 | 0.38259 | 20/47 | 93/81 | 16/16 |
| MMD-16-2 | ↓ | 12.89 | 12.91 | 12.89 | 16.80 | 18.00 | 13.75 | 12.91 | 44/42 | 128/129 | 19/30 |
| MMD-14-3 | ↓ | 4.16 | 4.53 | 4.19 | 4.78 | 4.98 | 4.73 | 4.53 | 43/43 | 121/122 | 18/30 |
| 3rd-Autocorr. | ↓ | 1.4557 | 1.5337 | 1.4931 | 1.5875 | 1.6437 | 1.5610 | 1.5707 | 51/47 | 129/103 | 13/18 |
| EPLB | ↑ | 0.145 | 0.128 | 0.129 | 0.127 | 0.147 | 0.127 | 0.127 | 42/41 | 31/35 | 17/3 |
| PRISM | ↑ | 26.26 | 25.85 | 26.26 | 23.09 | 22.35 | 22.68 | 23.56 | 44/23 | 20/28 | 11/2 |
| LLM-SQL | ↑ | 0.730 | 0.693 | 0.730 | 0.656 | 0.697 | 0.714 | 0.710 | 41/44 | 31/15 | 12/1 |
| Txn Sched. | ↑ | 4348 | 3704 | 3774 | 3623 | 3497 | 3268 | 3096 | 45/43 | 46/34 | 17/2 |
| Cloudcast | ↓ | 632.70 | 849.40 | 672.80 | 1012.17 | 1035.27 | 999.00 | 1035.27 | 38/41 | 127/122 | 11/29 |

Scores are the graders' normalized scores converted back to the paper's metric
(`../compare.py`). Same-model reading: TTT beats frozen on 3/11 tasks with one
agent (Circle-Pack., EPLB, LLM-SQL) and 5/11 with four (Circle-Pack., Erdos,
MMD-16-2, MMD-14-3, PRISM), ties EPLB and LLM-SQL four-agent, and loses the
rest. Erdos 4-agent TTT (0.38259) passes the paper's M2.5 4-agent number; EPLB
1-agent TTT (0.147) passes the SOTA column.

Caveats: one seed per arm. Runs stop short of budget when the model declares
the task complete and stops submitting; the 4-agent systems arms in particular
got 15 to 46 evals against a 120 budget, and their TTT arms trained only 1 to 3
steps. Two grader outputs are excluded by `compare.py` as unverifiable: the
Cloudcast 4-agent TTT attempt with `total_cost=0`, and a Txn Sched. 4-agent TTT
attempt reporting makespan 121 (score 8197, 1.9x SOTA), since that evaluator
trusts the makespan the program reports. TTT arms' LoRA carried over from earlier
discarded runs of the same task (agents were restarted while fixing the launch
path); a clean rerun should restart the stack with the agent.

Files: `<task>-<arm>.json` (every graded attempt, every report to reef, final
reef status), `table.md`, `sweep-final.txt`; `../collect.py` and `../compare.py`
rebuild them. Launch: `run.py --task <CORAL task package> --scenario <task>
--agents N --max-attempts B --runtime opencode --model reef/reef-policy` against
either serve config. EPLB needs `expert-load.json` and LLM-SQL the `datasets/`
CSVs in the task's `taskdata/` (their graders print the download instructions).
