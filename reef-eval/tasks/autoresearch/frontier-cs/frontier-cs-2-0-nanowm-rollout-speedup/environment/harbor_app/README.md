# nanowm_rollout_speedup — agent workspace

Patch the diffusion **sampling** code of the NanoWM checkout at
`/app/nano-world-model` to speed up the fixed CSGO 50-frame rollout at
iso-quality, then submit the unified diff as `/app/solution.patch`.

Workflow:
1. Edit the diffusion sampler under `nano-world-model/src/diffusion/**`.
2. `bash /app/public_test.sh`  — static patch-policy check on your diff.
3. `bash /app/make_submission.sh`  — writes `/app/solution.patch` from your edits.
4. `bash /app/submit.sh`  — enqueue for the black-box judge (quick set).

Rules (see the task statement): Python-only; allowed = `src/diffusion/**.py`;
the model/VAE/metric/harness/data AND the VAE-decode/video plumbing
(`src/sample/sampling_utils.py`) are frozen and denied; no env-var/benchmark/
timing tricks. The judge fixes the rollout call —
you change the sampler internals.

Scored on wall-clock speedup vs the unpatched baseline, gated by an LPIPS-vs-GT
quality guardrail (≤3% rise). See `../readme` and `../DESIGN.md`.
