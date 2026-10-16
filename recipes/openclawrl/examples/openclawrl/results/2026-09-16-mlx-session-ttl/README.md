# Session TTL on the MLX GSM8K stream: 45 s against 1800 s

This result compares two values of
`pipeline.record2record.config.session_ttl_s` on the GSM8K homework stream.
Both runs use the MLX backend on one Apple Silicon Mac. It is not a learning
result: neither run reached the adaptation criterion. The reference learning
result is the seven-GPU run in
[the example README](../../README.md#gsm8k-homework-stream).

## Why the TTL matters

The OpenClaw-RL recipe trains on a reply together with the student reaction
that judges it. The judging stage retires a session after it is idle for
`session_ttl_s` seconds. An expired session cannot bind a reply to its
reaction, so that reply is not trained. Hermes starts each turn as a new
process, so the `x-reef-tag-session` tag is the only link between the turns.

On GPUs a reply comes back in seconds, and a short TTL is safe. On MLX one
generation takes more than two minutes. A training step also blocks serving
while it runs. The gap between a reply and its reaction can therefore be
longer than 45 s.

## Settings

The two runs differ only in `session_ttl_s`. They use the same base weights,
the same KL term, the same candidate evaluation probe, the same timeouts and
the same 72-problem stream in the same order.

| Setting | Value |
| --- | --- |
| Config | [`serve-mlx.yaml`](../../serve-mlx.yaml), with `session_ttl_s` set to 45 or 1800 |
| Task | the 72-session GSM8K homework stream; each run stopped before the end of the stream |
| Model | `mlx-community/Qwen3.5-9B-MLX-4bit` as the policy |
| PRM | `z-ai/glm-5.3` through OpenRouter, 3 votes per turn |
| Student | a hosted model; the model name is not recorded |
| Hardware | one Apple Silicon Mac; the model and memory size are not recorded |
| Batch | 8 judged turns per training step |
| Adapter | LoRA rank 64, scale 2.0, on all 32 layers, `checkpoint_layers` on |
| Adapted projections | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `self_attn.{q,k,v,o}_proj`, `linear_attn.in_proj_qkv` |
| Objective | `openclawrl` loss family, `kl_coef` 0.05, top-4 capture |
| Optimizer | AdamW, lr `3e-5` constant, weight decay 0.01 |
| Responses | up to 8192 tokens, the Hermes client's `max_tokens` |
| Sampling | `training.options.temperature` not set, so the default 1.0 applies; a per-request temperature from Hermes is not recorded |
| Candidate evaluation | the OpenClaw-RL style probe: 8 prompts, 320 tokens each, regression margin 0.17 |

## Result

The longer TTL kept every training pair. It did not change what the policy
learned.

| Measurement | `session_ttl_s: 45` | `session_ttl_s: 1800` |
| --- | --- | --- |
| stream sessions run | 24 | 19 |
| optimizer steps | 27 | 31 |
| optimizer steps by session 4 / 5 / 8 | 1 / 2 / 3 | 6 / 7 / 14 |
| consecutive requests further apart than the TTL | 22 of 54 (40%) | 0 of 54 |
| judging-stage sessions that expired unbound : that bound a turn | 70 : 69 | no unbound expiry reported |
| accepted sessions | 3 (s006, s016, s020) | 0 |

With 1800 s, the largest gap between requests was 1091 s, during a training
step. With 45 s, about half of the judging-stage sessions expired before
they bound a reply to its reaction. The judging stage counts its sessions
separately from the stream sessions. It reports unbound expiries as a
warning, and that warning did not appear in the 1800 s run.

The same agent work produced more training steps with the longer TTL. At
session 8 the 1800 s run had 14 steps and the 45 s run had 3. Over the whole
runs the ratio is smaller: 31 steps in 19 sessions against 27 steps in 24
sessions.

The candidate evaluation probe did not move. The table uses the first 26
steps, which both runs completed, and then all steps of each run.

| Arm | mean `mean_violations`, steps 1–26 | range | mean, all steps | range |
| --- | --- | --- | --- | --- |
| 45 s TTL | 3.303 | 3.125–3.375 | 3.306 (27 steps) | 3.125–3.375 |
| 1800 s TTL | 3.293 | 3.125–3.500 | 3.319 (31 steps) | 3.125–3.500 |

The difference between the arms is 0.010 over steps 1–26. The step-to-step
standard deviation is about 0.09 in both arms. Both series move among the
same few values with no trend. `clean_rate` is 0.0 at every step of both
runs. Every candidate passed the candidate evaluation (`select`).

The accepted sessions are not a usable signal at this scale. Problem s006
was accepted in the 45 s run and failed in the 1800 s run, from the same base
weights. This is consistent with sampling noise.

In `sessions-ttl1800.csv`, six sessions (s009 to s013 and s015) have a turn
count of 0. In `sessions-ttl45.csv`, one session (s004) has a turn count of 0.
The records do not show why these sessions have no turns.

## What this result does not explain

The 1800 s run shows that the policy receives a training signal and that
the adapter changes:

- The batches have contrast. Over 27 steps there were 139 positive and
  76 negative advantages, and only 2 steps had no contrast. These counts
  come from the run logs and are not in the CSV files.
- Every step changes the adapter. The `adapter_delta_l2` column records the
  size of each update.
- About 86% of judged turns are followed by a tool result and about 14% by
  the student, so the judge mostly scores tool progress. The seven-GPU
  reference run has the same split and still learned, so this split does not
  explain the result.

These differences from the reference run remain. No measurement here
isolates one of them:

| | This result | Seven-GPU reference |
| --- | --- | --- |
| Policy | 4-bit `Qwen3.5-9B` | `Qwen3-4B-Thinking-2507` |
| Training | rank-64 LoRA | full parameters, tensor parallel 4 |
| Batch | 8 judged turns | 16 judged turns |
| Learning rate | 3e-5 | 1e-5 |
| PRM and student | hosted models | served locally |

## Files

- `evaluation-ttl45.csv`, `evaluation-ttl1800.csv`: one row per training
  step. The columns are the loss, `kl_coef`, `adapter_delta_l2`, the probe
  metrics (`clean_rate`, `mean_violations`, `answered_rate`, `gold_rate`) and
  the candidate evaluation outcome.
- `sessions-ttl45.csv`, `sessions-ttl1800.csv`: one row per session. The
  columns are the reward, the turn count and the number of style violations.
