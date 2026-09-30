# Small LoRA SFT → OPD AIME experiment (2026-09-30)

This completed run did **not** demonstrate an improvement from OPD.

| Checkpoint | Correct / samples | Mean accuracy | Length-limit cases |
| --- | --- | --- | --- |
| SFT, OPD step 0 | 360 / 480 | 75.0000% | 92 |
| OPD step 30 | 338 / 480 | 70.4167% | 101 |

The change is **−4.5833 percentage points**. The paired question-cluster
bootstrap 95% interval is **[−13.3333, +3.3333] percentage points**
(10,000 resamples, seed 0). It includes zero: this run establishes neither
an improvement nor a statistically clear degradation. The predeclared +3 pp
point-estimate target was not met. This is not the roadmap's full-parameter
+10 pp reproduction, and no alternative checkpoint was selected.

## Protocol and resources

- Frozen Qwen3.5-9B-Base with rank/alpha 32 LoRA; frozen Qwen3.5-9B teacher.
- 4096 OpenThoughts3 math examples; 8192-token maximum, global batch 32,
  one epoch / 128 SFT updates, learning rate 1e-4. SFT took 707.9 seconds.
- Seed-0 shuffled shard order and shuffled rows within the first eligible
  shard: this is shard-subset sampling, not uniform sampling across the full
  dataset. Long solutions were truncated without a fabricated EOS.
  Training used 33,433,845 tokens. Exact normalized question overlap was
  excluded; this does not rule out paraphrase contamination.
- 1920 independent DeepMath prompts; 30 updates × 64 prompts × 4 responses,
  4096 response-token maximum, learning rate 5e-5. All 7680 responses were
  processed; all 30 recorded losses/gradient norms were finite and reported
  trainable base parameters were zero. The SFT adapter was continued.
- AIME'24: the same 30 questions × seeds 0–15, 32768 output-token budget,
  evaluated only at steps 0 and 30. Evaluation took 3279.77 and 3537.95
  seconds respectively. This is mean sampled accuracy, not pass@16.
- Four physical B200 GPUs (2–5), TP4 training and TP1 inference replicas.
  No Tinker. Pipeline and campaign exited 0; GPUs were released after the
  final evaluation. The prior full-parameter experiment remains stopped.

## Saved results and verification

Source commit: `9cc1a8e` (before this result-only documentation commit).
Base revision: `68c46c4b3498877f3ef123c856ecfde50c39f404`.
Teacher revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The runtime and test results are documented in ../../VALIDATION.md.

The compact scored JSONL files preserve every question/seed pair, release,
score, finish reason and token usage. Validation checked 480 unique pairs per
checkpoint, matched pair sets, and rescored every final boxed answer against
the saved expected answer. Compact scores reproduce acceptance.json exactly.
Raw responses, receipts, manifests, logs and checkpoints remain on B200 under
`/raid/x9zou/reef-opd-state/lora-small-20260930`; validation.json records raw
prediction checksums. Adapter validation records retain tensor checksums.

Recompute from the repository root (requires matplotlib):

```bash
python recipes/opd/examples/math/analyze.py \
  recipes/opd/examples/math/results/2026-09-30-lora-small \
  --final-step 30 --eval-every 30 --target-improvement 0.03 --training-mode LoRA
```

No further training or test-guided tuning was launched after this result.
