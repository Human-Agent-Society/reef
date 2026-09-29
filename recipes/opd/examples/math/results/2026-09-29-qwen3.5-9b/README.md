# Qwen3.5-9B OPD experiment: measured controls

The frozen teacher control is complete. The 3,000-step SFT initialization is
still running; its baseline, the 200-step OPD curve and learning acceptance
are pending. This result does not establish the improvement required by
[roadmap #502](https://github.com/Human-Agent-Society/reef/issues/502).

## Frozen teacher result

| Measurement | Observed value |
| --- | --- |
| Model | Qwen/Qwen3.5-9B, frozen |
| AIME'24 questions / samples per question | 30 / 16 |
| Correct / total responses | 406 / 480 |
| Mean accuracy | 84.5833% |
| Responses reaching the output limit | 63 / 480 |
| Formal wall time, including final verification | 10,895.00 s (3 h 1 min 35 s) |
| Hardware | 1 NVIDIA B200, 183,359 MiB total memory |
| Sampled maximum GPU memory | 111,045 MiB |
| Sampled mean GPU utilization / power | 58.86% / 549.26 W |

Accuracy averages the correctness of all 16 samples for each question. It is
not the probability of getting at least one answer right in 16 attempts.
Output-limit cases remain in the denominator and use the same final-answer
scorer. The teacher control measures teacher performance; no training reports
or optimizer updates were submitted.

The formal run lasted from 2026-09-29 13:30:12.998 UTC to 16:31:48 UTC.
Telemetry contains 2,128 samples, with a median interval of 5.12 seconds;
memory is an NVML sampled maximum, not an allocator peak. Startup and preflight
time are excluded. The control container was stopped after validation, and
its GPU coordination lock was released while four-GPU SFT continued.

## Data and verification

`teacher/eval-0000.jsonl` contains all 480 individual question/seed/release
observations: expected and extracted answers, correctness, finish reason,
token counts, receipt ID and response/reasoning checksums. The full generated
responses, receipts, logs and telemetry remain on B200 under
`/raid/x9zou/reef-opd-state/teacher-control-v4`.

The final validator checked the complete 30-by-16 pair set, independently
rescored final boxed answers against the original dataset, and fetched every
receipt. All samples reference the single frozen creation release
`9fef2ede55350af2fae66733465cca9e0faf7b83`.

`teacher/validation.json` records the original prediction and receipt hashes.
`teacher/export.json` records the mapping to compact scores, and
`teacher/export-validation.json` confirms that analysis of the compact and
original files agrees. `teacher/eval-0000.metrics.json` contains the measured
metric. No student learning curve or acceptance value is substituted with
this control.

## Reproduction

Use the environment instructions in [the example README](../../README.md).
The source at launch was `1bf69672a14345e96d471be070f48fcbc4b7a82a`;
`teacher/inputs.json` pins the exact campaign driver, configuration and
prepared data by SHA-256. Source additions after launch include stricter
learning-curve analysis; the original and compact control agree under that
analysis too.

- Teacher revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- AIME dataset: `HuggingFaceH4/aime_2024`, revision
  `2fe88a2f1091d5048c0f36abc874fb997b3dd99a`.
- Immutable image and Slime/SGLang commits: `teacher/deployment.json`.
- Observed Python/package versions: `teacher/environment.json`. This is an
  inventory, not a complete build lock. `runtime-validation.json` records the
  shared training environment's Megatron source check; the teacher control
  itself uses SGLang inference, not Megatron training.
- Sampling: seeds 0 through 15, temperature 1, top-p 1, top-k disabled,
  64,000 output tokens and a 65,536-token context, with deterministic inference.

In a fresh runtime with the pinned teacher and prepared data mounted at
`/work`, copy `teacher/serve.yaml` to `/work/teacher-control-v4/serve.yaml`.
Use one GPU and separate empty state directories. Generate a private service
token, and share its environment with the service and campaign terminals:

```bash
export REEF_TOKEN="$(openssl rand -hex 24)"
export SGLANG_GRPC_PORT=28987
reef serve -c /work/teacher-control-v4/serve.yaml
```

After the service is healthy, run from the repository root:

```bash
python recipes/opd/examples/math/run.py \
  --config /work/teacher-control-v4/serve.yaml \
  --url http://127.0.0.1:28986 --scenario opd-teacher-aime24 \
  --model /work/models/Qwen3.5-9B \
  --train-data /work/data/deepmath.jsonl --eval-data /work/data/aime24.jsonl \
  --output /work/teacher-control-v4/results --steps 0 \
  --eval-repeats 16 --eval-tokens 64000 --eval-every 20 --seed 0 \
  --concurrency 32 --timeout 7200
```

The recorded stack skips unused vision warmup and disables CUDA graphs. Its
OpenAI chat deployment rendered exactly the same token IDs as the OPD
handler for all 30 prompts, with matched effective sampling settings. This
check does not establish bitwise invariance across hardware or batching.
Compare regenerated per-question scores and the aggregate metric; generated
text and release/receipt identifiers need not be identical across deployments.
