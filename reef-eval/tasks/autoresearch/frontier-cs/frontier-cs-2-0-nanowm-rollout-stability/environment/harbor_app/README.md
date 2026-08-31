# nanowm_rollout_stability — agent workspace

Patch the diffusion **sampling** code of the NanoWM checkout at
`/app/nano-world-model` to reduce long-horizon **drift** in a fixed CSGO
long-horizon rollout **at iso-wall-clock** (no extra compute), then submit the
unified diff as `/app/solution.patch`.

The exact rollout length and which frames are scored are fixed by the judge and
**not disclosed** (the scored horizon is drawn per run), so a solution must reduce
drift *generally* — keying behaviour off an assumed rollout length or a hardcoded
frame index will not transfer to the scored run.

Workflow:
1. Edit the diffusion sampler under `nano-world-model/src/diffusion/**`.
2. `bash /app/public_test.sh`  — static patch-policy check on your diff.
3. `bash /app/make_submission.sh`  — writes `/app/solution.patch` from your edits.
4. `bash /app/submit.sh`  — enqueue for the black-box judge (quick set).

Rules (see the task statement): Python-only; allowed = `src/diffusion/**.py`;
the model/VAE/metric/harness/data AND the VAE-decode/video plumbing
(`src/sample/sampling_utils.py`) are frozen and denied; no env-var/benchmark/
timing tricks. The judge fixes the rollout call
(length / context / steps / scheduling) — you change the sampler internals to
make the long rollout drift less per step.

Scored on the relative reduction in **tail-frame LPIPS-vs-GT** drift (the drifted
tail of the long rollout) vs the unpatched baseline, gated by a **wall-clock
guardrail** (patched generation time may rise at most ~10% over baseline — you
cannot buy lower drift with more compute; that is the speedup task's axis). The
length of the rollout and the exact tail window scored are not disclosed and vary
per scored run, so target *real* drift reduction, not a fixed frame range. See
`../readme`.
