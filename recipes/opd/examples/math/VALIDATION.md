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

The complete 3,000-step, batch-128 SFT run has been started separately from
these short checks. No full-run SFT or AIME result is available yet.

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

## Repository checks

- `pre-commit run --all-files`: passed.
- `python -m mypy`: passed for 336 source files.
- `python -m pytest tests --cov=reef --cov-report=term`: 5,375 passed,
  79 skipped, 89.78% coverage (required floor: 80%); rerun after the campaign
  recovery changes.
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
