# B200 validation record

Status as of 2026-09-29: integration verified; the full SFT initialization is
running. The AIME'24 baseline, OPD learning curve and final acceptance remain
pending. This record does not claim the roadmap's ten-point improvement.

## Software and data

The runtime image and model/data revisions are listed in `README.md` and
`prepare.py`. Verification used four B200 GPUs, torch 2.11.0+cu129,
Transformers 5.12.1, tokenizers 0.22.2, Megatron Bridge 0.4.2, and the
repository's reviewed Slime and SGLang revisions. The SFT fast path additionally
used flash-linear-attention/fla-core 0.4.2 and causal-conv1d 1.7.0.

- DeepMath: 103,022 prepared prompt-only records.
- SFT: 384,000 prepared and tokenized records; 5,729,403,815 input tokens,
  mean length 14,920.32, minimum 587, maximum 16,384. All input/label lengths
  match. Saved dataset fingerprint: `720ed895b5edeee3`.
- AIME'24: 30 held-out questions. Normalized exact/complete-question substring
  matching found no overlaps with either prepared training dataset. This
  does not exclude paraphrases or other forms of contamination.
- Student and teacher token vocabularies match exactly: 248,077 entries.
  The model embedding/output dimensions are 248,320; the explicit model
  vocabulary setting prevents export from truncating those native rows.

## Real OPD update

A short test used the base student (before SFT), the frozen instruct teacher,
four student-generated responses with a 128-token limit, and one optimizer
update. The smoke driver called the campaign's collection, reporting,
publication-wait and evaluation methods against the live Reef service.

| Observation | Result |
| --- | --- |
| Teacher checkpoint loaded | Qwen3.5-9B |
| Loss / sampled reverse-KL estimate | 0.1642509699 |
| Gradient norm before clipping | 10.3739086634 |
| Learning rate | 5e-5 |
| Committed scenario steps | 1 |
| Subsequent inference release | Changed from the initialization release |
| Exported first decoder MLP 32x32 slice | 860/1024 elements changed |
| Maximum absolute change in that slice | 6.103515625e-5 |

The teacher pass, backward pass, optimizer/HF checkpoint save, weight
publication and subsequent inference completed. A toy arithmetic call only
checked inference after publication; it was not a reasoning benchmark.

The measured optimizer/HF checkpoint pair occupied about 188 GiB. The
corrected 600 GB cap with a 100 GB free-space reserve passed the next-checkpoint
capacity check, including the larger reservation implied by an FP32 SFT
source. A 200 GB cap would not accommodate the protected checkpoint plus the
next reservation.

## SFT initialization and recovery

Eight prepared examples exercised two full-parameter SFT steps at up to
16,384 tokens, global batch four, without adapters. The two steps produced
finite losses (0.9688, 2.5) and gradient norms (6.604, 34.52). Both optimizer
checkpoints were written.

The initial final-export attempt exposed a rank exiting before its peers
finished gathering the full state. Synchronization barriers keep the ranks
alive through export. After that correction, resuming checkpoint 1 completed
step 2 and the final HF export, exiting with code 0. Four 32x32 parameter
slices (embedding, output and first/last decoder MLP) were finite and differed
from the uninterrupted step-2 checkpoint by at most 6.364e-6. This is a sampled
numerical comparison, not a claim of bitwise or whole-model equality.

The complete 3,000-step, batch-128 SFT run is running separately from these
short checks. Its first scheduled checkpoint, `checkpoint-100`, passed the
following validation at 2026-09-29 10:56 UTC, after training continued to
step 101:

- The HF export and FSDP recovery model have matching tensor names and shapes,
  covering 9,409,813,744 parameters.
- The optimizer contains 427 parameter states covering all 8,953,803,264
  trainable text parameters. Parameter groups, moment shapes and optimizer
  step counters agree; the trainer and scheduler also report step 100.
- All four ranks' RNG files load with Python, NumPy, CPU and CUDA state.
- Four selected 8x8 slices from the embedding, output and first/last decoder
  MLP weights are finite and exactly equal between the HF and FSDP files.
  Sampled optimizer moments are also finite.

At 2026-09-29 12:21 UTC, `checkpoint-200` passed the same checks with step-200
trainer, optimizer and scheduler counters, after training continued to step
201. Trainer removed `checkpoint-100` as configured by `save_total_limit=1`;
only `checkpoint-200` remained, with 873.2 GiB free on the checkpoint volume.
The observer performed no deletion or restart.

At 2026-09-29 13:46 UTC, `checkpoint-300` passed the same structural and
sampled numerical validation after training continued to step 301. All 427
optimizer states and the trainer/scheduler counters identify step 300; all
four RNG files load. Only checkpoint 300 remained after automatic retention,
with 869.4 GiB free. The report is `checkpoint-300-validation.json`.

Each checkpoint occupies approximately 137 GiB. These checks cover structure,
shapes, counters and sampled numerical values; they are not full checksums,
actual restores of the formal checkpoints, or evaluation results. The earlier
checkpoint-1 restore test remains the recovery check. No full-run SFT or
AIME result is available yet. Detailed reports are retained under
`/raid/x9zou/reef-opd-state`: `checkpoint-100-validation.json`,
`checkpoint-200-validation.json` and `second-checkpoint-watch-result.json`.

## Campaign driver recovery

CPU fault-injection tests exercise interrupted report submission, a lost
response after the final report was accepted, a published update before its
local record was saved, and interrupted evaluation. Each recovered two-update
campaign has exactly two commits, four unique training reports and one metric
per evaluation boundary; completed samples are reused. Tests also cover a torn
final JSONL record, changed input rejection, missing historical predictions,
unrelated consumed records and concurrent-driver exclusion. The campaign and
Reef report suites passed 52 tests.

This validates driver recovery logic and Reef's existing report idempotency
contract. A real GPU interruption/resume of this updated driver has not yet
been tested. It does not claim recovery of a stopped training service.

## Frozen teacher evaluation preflight

A separate TP1 Qwen3.5-9B deployment on an idle fifth B200 was checked while
SFT continued on its original four GPUs. This control uses Reef's native
SGLang provider deployment; OPD uses Reef's token-native SGLang handler.
The actual server-rendered token IDs matched the OPD handler for all 30
AIME prompts, and effective sampling parameters matched at temperature 1,
top-p 1, top-k disabled, repetition penalty 1 and 64,000 output tokens.

The pinned SGLang sampler ignores request seeds unless deterministic
inference is enabled. The example now enables it, and the live teacher
engine confirmed deterministic inference with PyTorch sampling. Repeating
a toy prompt twice at seed 0 produced identical 658-token completions;
seed 1 produced identical 4,096-token truncated completions. The two seeds
produced different text. All four receipts referenced one creation release.
Completed responses had the correct final boxed answer, while truncated
reasoning was not scored as a final answer. The helper's initial requirement
that every toy response finish within 4,096 tokens was too strict; the saved
validation separately checks repeatability, normal answers and truncation.
These checks establish neither an AIME score nor general bitwise invariance
across hardware, batching or different inference deployments.

The standalone launch skips unused vision warmup, matching Slime's engine
setting, and sets a valid unused `SGLANG_GRPC_PORT`: this SGLang revision
otherwise validates an automatic HTTP port plus 10,000 even with gRPC off.
Earlier startup attempts and the pre-fix seed checks are retained separately;
no formal AIME samples were collected before the correction. Raw preflight
records are under `/raid/x9zou/reef-opd-state/teacher-control-v4`.

After the deterministic-inference configuration change, deployment validation,
recipe deployment and campaign suites passed 69 tests; focused pre-commit
checks passed. Formal teacher evaluation and the SFT/OPD comparison remain
pending.

## Complete evaluation schedule

A regression check reproduced acceptance from only steps 0 and 200, with
all intermediate evaluations missing. Analysis now requires every declared
interval plus the final step, rejects missing/extra/duplicate steps, and
records the evaluation interval in its output. Regression tests also cover
an off-interval final checkpoint and invalid schedule arguments. The complete
suite below was rerun after this correction.

A private score exporter was checked against the completed toy control:
compact scored rows preserve the original analysis result, while exporting
the incomplete real AIME control is refused. This validates the export path;
no benchmark values or learning curve are substituted with toy data.

## Repository checks

- `pre-commit run --all-files`: passed.
- `python -m mypy`: passed for 336 source files.
- `python -m pytest tests --cov=reef --cov-report=term`: 5,382 passed,
  79 skipped, 89.79% coverage (required floor: 80%); rerun after the complete
  evaluation-schedule checks. Duration: 583.55 seconds.
- After example configuration/runtime updates, the campaign, server and
  complete example-config suites: 93 passed.
- Node 22: `npm ci`, `npm run check:docs`, `npm run lint`, `npm run build`: passed.

The full test run used an isolated container with an init process, Node 22
and working IPv6 loopback. Missing npm, unreaped child processes and disabled
IPv6 in the original training container caused unrelated first-run failures;
new-example configuration fixtures also needed updating before the clean run.

Raw logs, receipts, releases, dataset manifests and compact verification JSON
are retained on B200 under `/raid/x9zou/reef-opd-state`. Large disposable smoke
checkpoint copies were removed after validation; the public source models,
prepared training data and validation records remain. PR #683 and experiment
issue #682 track the remaining benchmark work.
