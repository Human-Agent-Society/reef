# MLX runtime — operational notes

Runtime-general facts for running Reef's [MLX backend](../../../../../reef/train/mlx_backend)
on Apple Silicon: how it is wired, how to read a step's training metrics, what a
published adapter looks like, and how large a model fits. The numbers here are
measurements, not the gsm8k result — that is in
[the 9B run](2026-09-10-gsm8k-stream-qwen3.5-9b-mlx/README.md).

## Topology

**Colocated, single host, single process, synchronous.** One `MLXEngine` owns
the base model, the LoRA adapter and the optimizer; serving reads those
parameters and training writes them, and the two are kept apart by the
`InferenceAdmissionController` Reef already uses for colocated Slime. Every MLX
operation runs on one dedicated engine thread, because MLX streams are
per-thread.

The optimizer lives with the engine across steps, so Reef's step *N+1* continues
the Adam moments and step counter that step *N* left behind. The `optimizer_step`
metric below is that continuity, made observable.

The frozen-base log-probabilities the KL term needs are computed by zeroing
`lora_b` in place and restoring it — LoRA computes `x @ A @ B * scale`, so a
zeroed `B` *is* the base model, and no second copy of the weights sits in
unified memory.

## Reading a step's training metrics

Each step carries metrics into its commit record:

```
opt_step=1  delta_l2=0.080952  changed=32/32  ratio=0.999434  kl=+6.27e-04  tokens=426
```

- `opt_step` increments across Reef steps — the optimizer is not rebuilt per step.
- `delta_l2` and `changed` are the proof the update is real. The runtime
  **refuses to publish** a candidate whose adapter tensors did not move, so a
  step that trains nothing fails loudly instead of publishing an adapter
  identical to the base with a success record attached.
- `ratio` is `exp(logπθ − logπrollout)` before the update. On an on-policy batch
  it sits at 1.000 ± 0.001, which also confirms the log-probs recorded at
  generation line up token-for-token with the ones recomputed at training time.
  A drift away from 1 means the batch aged behind the serving weights.

Each step publishes a durable adapter artifact through Reef's normal Git-LFS
stack, carrying its compatibility record:

```json
{
  "schema": "reef.mlx.adapter/1",
  "base_model": ".../Qwen3.8-27B-4bit",
  "lora_parameters": {"rank": 256, "scale": 2.0, "keys": ["self_attn.q_proj", "..."]},
  "libraries": {"mlx": "0.32.2", "mlx-lm": "0.31.3"},
  "objective": "openclawrl-topk-opd+kl",
  "scenario_step": 0,
  "source_runtime_load_id": "mlx-37773-1"
}
```

The adapter directory is `mlx-lm`'s own format (`adapters.safetensors` plus
`adapter_config.json`), so `mlx_lm.tuner.utils.load_adapters` loads a published
artifact directly. It is written to a staging directory and renamed into place,
so an interrupted publication leaves either the previous complete adapter or the
new one, never a half-written one.

## How large a model fits

Measured on an M4 Pro with 51.5 GB of unified memory. MLX reports a 40.2 GB
`max_recommended_working_set_size`, but that is a soft wiring hint: a step
peaking at 44.9 GB still ran with linear timing, while one asking for 71 GB took
4.5× longer than the trend — the real cliff is physical RAM.

Weights and generation, 4-bit, LoRA rank 8:

| model | weights | generation |
| --- | --- | --- |
| Qwen2.5-1.5B | 0.87 GB | 130 tok/s |
| Qwen2.5-7B | 4.29 GB | 53 tok/s |
| Qwen3.8-27B | 15.13 GB | ~14 tok/s |
| Qwen2.5-32B | 18.43 GB | 11.5 tok/s |
| Llama-3.3-70B | 39.69 GB | 4.0 tok/s |

Everything above trains. 70B does so with `micro_batch_size: 1` and no headroom,
so treat it as the ceiling rather than a working configuration.

Training peak scales with `micro_batch_size × sequence length`, at roughly
3–5 MB per (sequence × token) — not with model size beyond the weights. Two
sweeps, both one sequence per backward:

| tokens | Qwen2.5-32B | Qwen3.8-27B | Qwen3.8-27B, chunk 1024 |
| --- | --- | --- | --- |
| 1024 | 22.95 GB | 17.58 GB | — |
| 2048 | 28.80 GB | 20.13 GB | — |
| 4096 | 44.35 GB | 25.40 GB | 20.72 GB |
| 8192 | — | 37.31 GB | 27.25 GB |
| 16384 | — | 58.15 GB | 58.82 GB |

`log_probs_chunk_size` removes the `[batch, length, vocab]` logits tensor and is
worth 27% of peak memory at 8192 tokens, for about 40% more time. It buys
nothing at 16384: past roughly 8k the body's own forward graph dominates, so
that length needs a frozen-prefix or gradient-checkpointing change rather than a
smaller chunk.

Long-context *serving* is a different and much easier problem, because the
hybrid architectures keep a KV cache on only a quarter of their blocks:
Qwen3.8-27B prefilled 16384 tokens at 20.56 GB and 65536 at 28.65 GB. Prefill
runs about 100–125 tok/s, which is the binding cost at those lengths, not memory.

## Tuning for your machine

Unified memory scales with the number of sequences held in one backward pass
times their length. In order of effect: lower `micro_batch_size`, then
`max_tokens`, then `lora_layers`.

On a hybrid model (Qwen3.5/3.8: three `GatedDeltaNet` layers to every
full-attention one), `lora_layers` has a second cost. mlx-lm's `GatedDeltaNet`
recurrence runs one token at a time: a Metal kernel without a vjp in eval mode
and a loop of plain ops in training mode, and that loop keeps three to four
fp32 state matrices of 48 × 128 × 128 per token per crossed layer for the
backward — about 10.5 MB a token — while both directions are a chain of
hundreds of tiny kernels. The engine switches only the layers on the
gradient's path into training mode, for the span of one backward, and runs
the recurrence there in its chunkwise form (`reef.train.mlx_backend.gated_delta`,
ported from flash-linear-attention's reference; `recurrence_chunk_size`
tokens a chunk, default 64): matrix products within a chunk, one state
between chunks, and MLX's own autodiff for the backward. On the recurrence
alone at this model's head shape, 700 tokens: 13.5 MB a token and 5.9 s for
mlx-lm's loop, 0.7 MB a token held for the backward and 0.3 s chunkwise.

Past a handful of layers the activations of the transformer blocks themselves
are what fill memory, and `checkpoint_layers: true` recomputes each adapted
layer's forward during the backward instead of keeping them. Measured on
Qwen3.8-27B-4bit, one 700-token row, weights resident at 15.1 GB:

| `lora_layers` | rank | `checkpoint_layers` | peak | backward |
| --- | --- | --- | --- | --- |
| 3, mlx-lm's loop | 256 | off | 30.0 GB | 23 s |
| 8, mlx-lm's loop | 256 | off | killed, out of memory | |
| 8 | 256 | off | 21.7 GB | 17.4 s |
| 64 | 16 | off | 50.5 GB | 744 s |
| 64 | 16 | on | 18.3 GB | 41.6 s |
| 64 | 64 | on | 20.1 GB | 32.2 s |

mlx-lm's loop is linear in length (3 layers: 19.9, 22.2, 24.2, 26.2 GB at
200, 300, 400, 500 tokens). Every layer of the model trains in 20 GB with
checkpointing on; without it the 64-layer row sits at the machine's physical
limit and the clock shows it. Layers below the lowest adapted one keep the
kernel and cost nothing extra, as does serving. Generation dominates step time, so `max_tokens`
is also the main throughput knob.

For training rollouts the recorded behaviour proxy is the model's own log-softmax
at the sampled token, so a tempered sampling distribution makes the importance
ratio approximate rather than exact — keep `temperature` at 1.0 where the
objective relies on an exact ratio. (The gsm8k run here trains at 0.6 because its
signal is the PRM's style verdict, not an importance ratio.)
