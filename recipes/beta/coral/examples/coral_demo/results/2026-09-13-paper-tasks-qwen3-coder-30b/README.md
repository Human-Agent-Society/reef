# CORAL TTT on the CORAL paper's task suite (Qwen3-Coder-30B-A3B)

First learning results for the CORAL recipe. Every number here is reproducible
from the retained per-attempt records in this directory; nothing is quoted from
a log that was not kept.

The reference is the CORAL paper ([arXiv:2604.01658](https://arxiv.org/abs/2604.01658)),
Table 1 (SOTA column) and Table 2 (the OpenCode + MiniMax M2.5 rows). The paper's
open-source rows use a ~230B MoE model; our runs use Qwen3-Coder-30B-A3B-Instruct
because it is the largest coding model our training stack serves and trains
colocated on one 8-GPU node. The paper columns are therefore context, not a
like-for-like target. The controlled comparison is between our own arms: the same
model, runtime, task package and budget, with the served policy frozen or trained.

## Arms

| arm | agents | policy | attempt budget |
|---|---|---|---|
| 1-agent frozen | 1 | Qwen3-Coder-30B-A3B, no updates | 40 |
| 1-agent TTT | 1 | same base, LoRA trained by `CoralRecipe` during the run | 40 |
| 4-agent frozen | 4 | frozen | 120 |
| 4-agent TTT | 4 | trained | 120 |

The two frozen arms run the identical serving stack with a group size that never
releases (`serve-coder30b-baseline.yaml`), so the only difference from the TTT
arms is whether weight updates happen. The 4-agent arms cover the six math tasks
and Cloudcast; the four systems tasks EPLB, PRISM, LLM-SQL and Txn Sched. have
1-agent arms only (node budget).

Training configuration (`serve-coder30b.yaml`): rank-32 LoRA on `linear_qkv` and
`linear_proj`, lr 4e-5, KL coefficient 0.1 against the base model, 128k training
sequence length, `group_size: 2`, `group_by: release`. One training step releases
every time two scored attempts have been produced under the current policy
release; the `tttd` step preparer and loss family compute grouped leave-one-out
advantages over that pair.

## Results

Scores are the task graders' normalized scores converted back to the metric the
paper reports (the exact inverse of each grader's normalization; see
`../compare.py`). "evals" is the number of graded submissions, the paper's
"# Evals". "TTT steps" is the number of policy releases the reef service
committed during the run.

| Task | Dir | SOTA | paper M2.5 1-agent | paper M2.5 4-agent | ours 1-agent frozen | ours 1-agent TTT | ours 4-agent frozen | ours 4-agent TTT | evals (frozen/TTT, 1-agent) | evals (4-agent) | TTT steps (1/4-agent) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Circle-Pack. | ↑ | 2.6359 | 2.3531 | 2.5391 | 2.1774 | 2.4426 | 2.4828 | 2.5440 | 43/42 | 124/123 | 16/32 |
| Signal Proc. | ↑ | 0.7429 | 0.7174 | 0.7383 | 0.4788 | 0.5462 | 0.6862 | 0.6589 | 15/41 | 122/123 | 17/28 |
| Erdos Over. | ↓ | 0.38088 | 0.39237 | 0.38311 | 0.39319 | 0.39589 | 0.39201 | 0.38259 | 20/47 | 93/81 | 16/16 |
| MMD-16-2 | ↓ | 12.89 | 12.91 | 12.89 | 16.80 | 18.00 | 13.75 | 12.91 | 44/42 | 128/129 | 19/30 |
| MMD-14-3 | ↓ | 4.16 | 4.53 | 4.19 | 4.78 | 4.98 | 4.73 | 4.53 | 43/43 | 121/122 | 18/30 |
| 3rd-Autocorr. | ↓ | 1.4557 | 1.5337 | 1.4931 | 1.5875 | 1.6437 | 1.5610 | 1.5707 | 51/47 | 129/103 | 13/18 |
| EPLB | ↑ | 0.145 | 0.128 | 0.129 | 0.127 | 0.147 | - | - | 42/41 | -/- | 17/- |
| PRISM | ↑ | 26.26 | 25.85 | 26.26 | 23.09 | 22.35 | - | - | 44/21 | -/- | 11/- |
| LLM-SQL | ↑ | 0.730 | 0.693 | 0.730 | 0.656 | 0.697 | - | - | 41/44 | -/- | 12/- |
| Txn Sched. | ↑ | 4348 | 3704 | 3774 | 3623 | 3497 | - | - | 45/43 | -/- | 17/- |
| Cloudcast | ↓ | 632.70 | 849.40 | 672.80 | 1012.17 | 1035.27 | 999.00 | 1035.27 | 10/41 | 113/122 | 11/29 |

Reading the same-model comparison:

- 1-agent TTT beats 1-agent frozen on 4 of 11 tasks (Circle-Pack., Signal Proc.,
  EPLB, LLM-SQL) and loses on 5 (Erdos, MMD-16-2, MMD-14-3, 3rd-Autocorr., Txn
  Sched.). PRISM and Cloudcast are confounded by eval counts (the PRISM TTT run
  stalled at 21 evals, the Cloudcast frozen run at 10; see caveats).
- 4-agent TTT beats 4-agent frozen on Circle-Pack., Erdos, MMD-16-2 and MMD-14-3,
  and on Erdos the 4-agent TTT arm passes the paper's M2.5 4-agent column
  (0.38259 vs 0.38311) while MMD-16-2 lands next to it (12.91 vs 12.89) with a model an order
  of magnitude smaller. It loses on Signal Proc., 3rd-Autocorr. and Cloudcast.
- EPLB 1-agent TTT (0.147) passes the paper's SOTA column (0.145); the paper's
  Opus 4.6 CORAL run reports 0.149.
- The Cloudcast 4-agent TTT run produced a submission with `total_cost=0`
  (grader score 1.0). That is an evaluator exploit, not a solution, and is
  excluded by `compare.py`; the row reports the best valid attempt.

With one run per arm these are single-seed numbers. The spread between arms on
tasks where nothing should differ (the two frozen arms on Cloudcast, for example)
is a reasonable estimate of run-to-run noise.

## What is in this directory

- `<task>-<arm>.json`: every graded attempt (eval index, commit, score, status,
  time), every report the watcher posted to reef (with reference counts), the
  reef service status at the end of the run (training step, processor
  decisions), and a summary. This is the retained record the table is built from.
- `table.md`: the table above, as produced by `../compare.py`.
- `sweep-final.txt`: the final fleet status line per run.
- `../collect.py`, `../compare.py`: the two scripts that turn run directories
  into these files and this table.

## How the runs were launched

Each arm ran on one 8x H200 node: `serve-coder30b.yaml` (or the `-baseline`
variant) under `reef serve`, then `run.py --task <CORAL task package> --scenario
<task> --agents N --max-attempts B --runtime opencode --model reef/reef-policy`.
Task packages are the CORAL repository's `examples/math/*` and `examples/ADRS/*`
at the pinned commit; EPLB needs `expert-load.json` and LLM-SQL needs the
`datasets/` CSVs placed in the task's `taskdata/` first (their graders print the
download instructions).

## Caveats

- Model: Qwen3-Coder-30B-A3B is far smaller than the paper's models. Absolute
  scores are below the paper's on most tasks; the finding here is about the
  effect of training during the run, not about matching the paper.
- Eval counts differ. The 30B model frequently declares the task complete and
  stops submitting; CORAL restarts the agent and, after the fix described below,
  starts a fresh session after three restarts without a new attempt. Some runs
  still stalled short of budget (PRISM TTT: 21, Cloudcast frozen: 10, Signal Proc.
  frozen: 15, Erdos frozen: 20). Those cells rest on fewer attempts than the rest.
- Restarts. The fleet went through several agent restarts while the launch path
  was being fixed (see below). Each restart starts a new CORAL run from the seed;
  the reported run is the last one per arm. For TTT arms the reef service and its
  LoRA stayed up across agent restarts, so the policy in the final run had been
  trained on the earlier, discarded runs of the same task as well. The frozen
  arms have no such carry-over. This favours the TTT arms and should be removed
  in a rerun by restarting the stack together with the agent.
- 4-agent runs were given a 120-attempt budget after first running at 40, to be
  comparable with the paper's 4-agent eval counts; the 40-attempt runs were
  discarded.

## What had to be fixed to get here

These are in the same pull request as this directory. None of them changes the
training semantics documented in the recipe README except `group_by`.

Reef side:

- `middleware.py`: after replaying the rewritten request body the receive
  wrapper fabricated empty `http.request` messages forever; Starlette's
  streaming response polls `receive()` for the client disconnect, so every
  streamed completion hung. Every opencode call is streamed.
- `watcher.py`: LiteLLM drops reef's receipt frame from streamed responses and
  does not forward the record-id header, so reports carried zero references and
  nothing could train. References are now also resolved from the tags reef
  stores with each record.
- `processor.py`: a coding-agent's call sequence is not one prompt extension
  (the runtime compacts history), so the multi-turn assembler saw a fork and the
  attempt was dropped as a "tensor contract violation". The processor now falls
  back to the terminal call and reports the real violation text. `group_by:
  release` was added because single-agent evolution is a chain with a new parent
  every attempt, so parent-hash groups never reached `group_size`.
- `run.py`: `--task` / `--scenario` so any CORAL task package runs.
- `litellm_config.yaml`: route through LiteLLM's `hosted_vllm` provider (opencode
  speaks the Responses API to anything it thinks is OpenAI) and the
  `qwen3_coder` tool-call parser in `backend-config`.
- Checkpoint retention: `policy: latest` keeps every checkpoint until the
  filesystem hits the free-space floor; with 15 training arms on one shared
  filesystem all of them blocked at the floor together, and each Megatron
  checkpoint of the 30B model is 58 GB. The runs used an external pruner; the
  serve config should set `--reef-checkpoint-max-storage` for shared filesystems.

CORAL side (patch kept separately, against the pinned commit): opencode is given
a custom provider id with an explicit context limit so it compacts instead of
growing sessions past 180k tokens; the resume prompt no longer lets the agent
end on its own "task complete" summary; and after three restarts with no new
attempt the manager starts a fresh session instead of resuming the transcript.
