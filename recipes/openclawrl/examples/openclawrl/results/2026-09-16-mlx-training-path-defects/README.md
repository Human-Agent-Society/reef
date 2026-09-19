# Three defects in the MLX training path, found by running it

This is a defect record, not a learning result. Running OpenClaw-RL's GSM8K
stream through the merged MLX backend turned up three things the deployment
asked for and the run did not do. Two earlier MLX runs were kept under
`results/` and have been withdrawn: they carry all three, so their numbers were
never properties of the method.

The reference result for this recipe is still the seven-GPU run in
[the example README](../../README.md#gsm8k-homework-stream), which is unaffected —
it does not take the MLX path.

## 1. `kl_coef` never reached the objective

`MLXRuntime._run_training` dispatches on loss family. The `openclawrl` branch
read `self._openclawrl["kl_coef"]`, which defaulted to zero; the deployment's
`kl_coef` was spent only on the *other* branch, as TTT-Discover-style advantage
shaping. `recipes/openclawrl/recipe.py` always names the `openclawrl` family, so
the config key was dead on the only path this recipe takes.

The tell was in the runs' own records: step metrics reported `kl_coef 0.0` for a
deployment configured with `0.05`. Both withdrawn runs trained with no KL term
at all, and one of them named that term among its load-bearing settings.

Fixed by defaulting the objective's coefficient to the deployment's, so it
reaches whichever mechanism the active family carries — the openclawrl loss has
its own KL term, as the slime reference's `kl_loss_coef` does. The Tinker
backend already worked this way: one key, routed to the resolved loss adapter,
and a family with no KL raises rather than ignoring it.

## 2. The style probe stated the rule it was scoring

The candidate gate's probe asked for a worked answer and then added *"do not use
bold, headings, bullet points, or numbered lists"* — a rule the student persona
never states up front. The probe was therefore measuring compliance with its own
instruction, not what the policy had internalised: a model told not to use
bullets can score well on it while opening every session with them.

Uncoached, the baseline falls from 0.125 to **0.0**, which is the number that
agrees with the 67-of-72 markdown rejection rate the withdrawn gated run
recorded at the session level.

The probe also scored truncation rather than style. At `max_tokens: 96` this
model's "here's a thinking process" preamble consumed the budget before the
arithmetic: `no-shown-work` fired on **50 of 50** offline probe replies at 96
against **13 of 50** at 320. At 320 the replies reach a median 1056 characters,
against the 770 a real session's first reply runs to.

Measured offline against base weights and four saved candidates — 100
generations, every reply scored — the corrected probe shows a hard floor rather
than a resolution problem: all three style markers appear in **100 of 100**
replies, base and candidates alike. The evaluator now also reports
`mean_violations`, the distance to the criterion, which can move while
`clean_rate` is pinned at zero.

## 3. `session-ttl-s: 45` discarded about half the training pairs

The method trains on a reply bound to the student reaction that judges it.
`SessionIndex` retires a session after its idle TTL and `ingest` expires before
observing, so an expired session can never bind the two — and Hermes runs each
turn as a fresh process starting from `[system, user]`, so the
`x-reef-tag-session` tag is the only thing carrying that link. Once the session
is gone nothing recovers it.

Every MLX config said 45 against a documented default of 900. That holds on the
reference's timing, where a reply comes back in seconds. Here a single
generation takes over two minutes and a training step blocks the engine longer
still.

Two independent measurements agree at about half:

| | |
| --- | --- |
| consecutive requests further apart than 45s | 15 of 31 (48%) |
| sessions expiring unbound vs binding | 70 : 69 (50%) |

The second figure swings widely while a run is in flight — sessions flush in
bursts, and it read 60:15 at one point and 12:9 at another. The gap
distribution is the stabler measure, and the settled ratio agrees with it.

Raised to 1800, which covers a generation, the student's round trip and a
training step in between. The cost is that a finished session waits that long
before its last turn retires, delaying a step; the short value silently dropped
the data instead.

## Fixing the TTL recovered every severed pair and changed nothing

Two runs differing in one setting, everything else identical — same KL term,
same corrected probe, same timeouts, same base weights, same 72-problem stream
in the same order.

| | `session-ttl-s: 45` | `session-ttl-s: 1800` |
| --- | --- | --- |
| sessions | 24 | 19 |
| optimizer steps | 27 | 31 |
| steps by session 4 / 5 / 8 | 1 / 2 / 3 | 6 / 7 / 14 |
| request gaps exceeding the TTL | 22 of 54 (40%) | **0 of 54** |
| sessions expiring unbound vs binding | 70 : 69 | no warning ever fired |
| sessions with no turn at all | 1 | 0 |

The fix works on its own terms, on two measurements that do not share a failure
mode: no gap in the run comes near the 1800s window (the largest is 1091s, a
training-step block, leaving 40% headroom), and the same agent work yields
roughly five times the optimizer steps.

It did not change what the policy learned. Over the first 26 steps of each arm:

| | mean `mean_violations` | range |
| --- | --- | --- |
| 45s TTL — half the pairs | 3.306 | 3.12–3.38 |
| 1800s TTL — all the pairs | 3.293 | 3.12–3.50 |

A difference of 0.013 against a step-to-step jitter of ±0.13. Both series
oscillate among the same three values with no trend, and `clean_rate` is 0.000
on every step of both. Twenty-six steps on complete data look exactly like
twenty-six on half.

So the TTL was a real defect and was not the reason the policy does not learn.

**Accepts are not a usable signal at this scale.** The 45s arm took three
(s006, s016, s020) and the 1800s arm none. Problem s006 accepted on one arm and
failed on the other from the same base weights, which is sampling noise at
temperature 0.6 rather than a property of either configuration. An earlier draft
of this file reasoned that all three accepts being 2-turn sessions implied an
easy-problem subset; s006 disproves that — the session is short *because* the
reply happened to come out clean, not the other way round.

## What is ruled out, and what is not

Measured, on the 1800s arm:

- The training signal has contrast: 139 positive advantages against 76 negative
  over 27 steps, with only 2 of 27 steps carrying no contrast at all. The
  objective is not starved of direction.
- Adapter displacement from base grows monotonically — `||lora_b||` from 0.84 to
  8.43 over 24 steps, read straight from the safetensors. The updates are real
  and the newly active KL term is not pinning them back, which rules out
  `kl_coef: 0.05` being too strong.
- 86.4% of judged turns are followed by a tool result and 13.6% by the student,
  so the evaluative judge mostly scores tool progress rather than the style the
  task rewards.

That last figure is **not** an explanation for the failure. The seven-GPU
reference run drives the same 72-session stream through the same hermes harness,
so it carries the same split, and it reached adaptation at session 14. A
property shared by the run that learns cannot be why this one does not.

What still differs from the run that learned, none of it isolated by any
measurement here: a `Qwen3-4B-Thinking-2507` policy trained full-parameter
across four tensor-parallel GPUs against a 4-bit quantised `Qwen3.5-9B` trained
through a rank-64 LoRA; 16 judged turns per step against 8; learning rate 1e-5
against 3e-5; a local PRM and student against hosted ones.

Also unfixed: the MLX tool-call parser rejects a sample whose `<tool_call>` never
closes, which happens roughly twice every three sessions on deep transcripts.
The governing cap is the client's `max_tokens: 8192`, not the deployment's, and
whether these samples reach it cannot be read from the records — the streaming
hold buffers everything after the marker by design.

## Files

- `gate-ttl45.csv`, `gate-ttl1800.csv` — per step: loss, `kl_coef`, adapter delta, probe metrics, gate outcome.
- `sessions-ttl45.csv`, `sessions-ttl1800.csv` — per session: reward, turns, violation count.
- [runtime notes](../mlx-runtime-notes.md) — topology, training metrics, capacity.
