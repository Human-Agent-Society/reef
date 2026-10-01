# Qwen3-4B LoRA SFT to OPD comparison (2026-10-01)

The 512-example arm improved from **1.6667% to 9.1667%** in this fixed run.
The paired question-bootstrap 95% interval for its **+7.5 percentage point**
change is **[+1.6667, +15.0000] pp**. This supports a positive effect under
this evaluation, not reliability across training seeds or roadmap acceptance.

| Condition | Correct / samples | Mean sampled accuracy |
| --- | --- | --- |
| Qwen3-4B-Base | 3 / 120 | 2.5000% |
| SFT256 | 2 / 120 | 1.6667% |
| SFT256 + OPD32 | 5 / 120 | 4.1667% |
| SFT512 | 2 / 120 | 1.6667% |
| SFT512 + OPD32 | 11 / 120 | 9.1667% |
| Frozen Qwen3-8B teacher | 87 / 120 | 72.5000% |

| Paired comparison | Change (pp) | Question-bootstrap 95% interval (pp) |
| --- | --- | --- |
| OPD256 minus SFT256 | +2.5000 | [-1.6667, +6.6667] |
| OPD512 minus SFT512 | +7.5000 | [+1.6667, +15.0000] |
| SFT512 minus SFT256 | 0.0000 | [-2.5000, +2.5000] |
| OPD512 minus OPD256 | +5.0000 | [-1.6667, +12.5000] |

## Scope and protocol

This is an **independent Transformers/PEFT training + SGLang sampling
experiment**, not the Reef Slime integration. Different-size 4B/8B teacher
support is not added to Reef by these results. No Tinker was used. The native
Reef Qwen3.5 result remains separately recorded in ../2026-09-30-lora-small/.
Neither experiment establishes the original full-parameter +10 pp roadmap
requirement. The stopped full-parameter and 1024-example campaigns were not
resumed, and no checkpoint was selected on test performance.

Both arms start independently from Qwen3-4B-Base with rank/alpha 16 LoRA on
`down_proj` and `o_proj`, dropout 0, a frozen base, and training seed 0.
SFT uses full math examples at most 8192 tokens, batch 16, AdamW learning
rate 1e-4 with linear decay, one epoch: 16 updates for 256 examples and
32 updates for 512 examples. The latter includes all 256 examples plus
256 seeded samples from the remaining fixed 1024-example pool. That pool
was length filtered from shuffled pinned OpenThoughts3 shards with random
row selection, **not a uniform full-dataset sample**. No result-based data
selection was used. Data manifests retain source pins, selected IDs and
checksums. Exact question exclusion does not rule out paraphrase overlap.

Each arm continues its own SFT adapter for 32 OPD updates on the same
512 independent DeepMath prompts: batch 16, one student response per prompt,
2048 response-token cap, learning rate 5e-5. Frozen Qwen3-8B scores the
student's sampled tokens. The loss uses the detached teacher-minus-rollout
log probability as a token advantage, weighted by the student/rollout
probability ratio and normalized by the batch's response-token count.
There is no correctness reward. Model pins are in each arm's model-pins.json.

All six conditions use the same 30 AIME questions and seeds 0–3 (120 samples),
temperature 1, top-p 1, top-k -1. Native total context is 32768; every sample
has output budget 32768 minus its prompt length, with no context extension.
Evaluation uses the last SFT and OPD adapters only. Accuracy is the fraction
of correct sampled answers, not pass@4. The fixed boxed-integer parser is
preserved; this is not a general symbolic math verifier.
Base results are reused from the cancelled 1024-example campaign with exact
protocol and checksum checks. The teacher is evaluated once in the 256 arm
and reused in the 512 arm. Shared controls are not independent replications.

## Validation and limitations

Both pipelines exited zero. The 512 arm finished at 2026-10-01 00:51:54 UTC.
All six conditions have 120 unique matched question/seed pairs. Every saved
answer was rescored; prompts and effective token budgets matched. The
combined analysis was reproduced exactly, including from the compact score
exports here. Bootstrap uses 10000 question-cluster resamples, seed 0.

Both arms have contiguous SFT/OPD update records with finite losses and
gradient norms, zero trainable base parameters, and all 144 adapter tensors
finite and changed between SFT and final OPD. The earlier two-SFT/one-OPD
smoke was explicitly reused, not rerun for each arm. Physical B200 GPUs 2–5
were released automatically after the pipelines exited. No further training
is scheduled by this comparison.

Only one training seed was tested. Question bootstrap does not capture
training-seed uncertainty; intervals are per comparison without a multiple
comparison adjustment. The two arms change both data amount and SFT update
count. Both SFT baselines are weak and below the sampled Base score; the
positive 512-arm change does not show that a strong SFT model improves or
that teacher-level accuracy is approached. Additional claims need new,
separately specified experiments.

## Saved records and actual code

[Exact executed scripts with SHA256 checksums](https://github.com/Human-Agent-Society/reef/pull/683#issuecomment-5922603188)
include both training drivers and pipelines, control-reuse checks and the
combined scorer. They retain environment-specific paths and require the
prepared model/data directories; they are not a portable Reef example.
`source-record.json` and `runtime-versions.json` identify code and packages.

This packet contains compact paired scores, summary metrics, plans, training
metrics, data selection manifests and validation records. Full generated
text, token IDs/log probabilities and adapters are retained under the two
`/work/qwen3-4b-opd-{256,512}-20260930` directories on B200. They are not
included in this compact repository export. Validation files describe their
observation time; earlier training-validation files still say evaluation was
running. `final-validation.json` and `combined-comparison.json` are final.

This export adds documentation and result data only. It does not change
Reef executable code. Codex assisted with execution, analysis and reporting;
the human contributor remains responsible for reviewing the claims.
