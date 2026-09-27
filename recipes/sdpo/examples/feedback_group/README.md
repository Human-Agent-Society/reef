# SDPO feedback-group smoke

This Harbor task makes eight on-policy attempts at one arithmetic question.
Its private grader returns a score and text feedback for each attempt. The
driver waits until the group is complete, uses `recipes.sdpo.prepare_group`
to create eight reports, waits for one Reef/Slime training release, then runs
one more Harbor trial to record post-update attempts. The two trials are
stored by reef-eval's `Lab` under `work/lab`.

This is a **wiring smoke**, not a paper benchmark. It has one arithmetic
question, one update, no held-out test set and no learning claim. The
SciKnowEval, ToolAlpaca and LiveCodeBench v6 campaigns and controls are
separate work.

## Run on a Linux GPU host

Use a Reef GPU environment with four GPUs capable of serving and fully
training Qwen3-8B, the pinned Slime runtime and Docker/Harbor available.
Place `Qwen/Qwen3-8B` at `/root/models/Qwen3-8B`, install this checkout's
Reef and the example harness, then run:

```bash
cd recipes/sdpo/examples/feedback_group
./run.sh --campaign my-first-sdpo-smoke
```

`run.sh` starts Reef, waits for readiness, runs the two reef-eval trials and
stops the service. `work/reef.log`, `work/*attempts.json`, `work/lab` and
Reef's training/checkpoint directories hold the run records. A
`work/<campaign>-summary.json` records the active response count, pre/post
artifact versions and scores; the runner fails if post-training inference
still uses the old version. Use a new campaign
name for a fresh run. This configuration uses 8 responses per step,
top-K 100 plus tail, JSD, EMA rate 0.05, token-level IS clipped at 2 and
per-response token means averaged over the batch. The model request caps responses at 256 tokens so the task
finishes quickly. For the paper's first track, use 32 questions per step,
8 responses per question, the actual benchmark scorers and the 8192-token
response budget.

This example has not been executed on a GPU in the current workspace. The
moving teacher's EMA state is currently not persisted on restart; run this
single-step smoke without resuming an interrupted step.
