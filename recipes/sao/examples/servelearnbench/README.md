# ServeLearnBench Retail L1 with SAO

This example runs the Retail L1 scenario of ServeLearnBench through Reef the way the benchmark's own runner does.
Every window runs in order: a window's serving tasks are trained on in batches, then its held-out test tasks run against the frozen weights, and the weights carry into the next window.
The summary reports the official Hidden and Fully Specified test scores separately.
It also reports the combined mean over all test tasks.
It implements the Retail part of [issue #893](https://github.com/Human-Agent-Society/reef/issues/893); the other scenarios and the OPD comparison are outside this example.

The benchmark dependency is pinned in `pyproject.toml`.
Its official agent loop owns environment resets, tools, budgets, termination, and verification.
The harness connects the model transport to Reef and reports the official scalar reward.
Hidden rules, ground-truth actions, and verifier details never enter SAO feedback.

## Layout

```text
servelearnbench/
  run.sh          starts reef serve, then runs the campaign
  run.py          the campaign: the ordered split, training batches, checkpoint resume, results.jsonl
  batching.py     keeps each declared task batch together, including window tails
  serve.yaml      the Reef stack: Qwen3-8B actor and critic, SAO, exact-token episodes
  harness/
    config.py     shared batch, micro-batch, warmup, and context settings
    campaign.py   durable results and category scores
    agent.py      the benchmark's LLM interface over reef_client and its episode loop with a turn limit
    report.py     the score report against an episode's inference receipts
```

## Experiment settings

| Setting | Pilot |
| --- | --- |
| Actor | Dense Qwen3-8B, full-parameter updates |
| Critic | Qwen3-8B value model; self-attention frozen, other parameters trainable |
| Sample / batch | One complete task trajectory; up to 64 tasks per training batch; micro batch at most 16 |
| Warmup | First training batch: critic only |
| Critic cadence | Two critic updates per batch, including warmup |
| Actor cadence | One update per batch after warmup |
| Learning rates | Actor `1e-6`; critic `5e-6` |
| Evaluation | Each window's tests against the frozen weights at that window's end; no test reports or updates |
| Context / per-call output | 40,960 / 16,384 tokens |
| Episode budgets | Official 100 turns, 50 world-tool calls, 256,000 output tokens |

This is a dense, batch-64 adaptation of SAO. It is not a reproduction of the paper's MoE, batch-128 experiment.
Critic warmup is not critic pretraining. No pretrained value checkpoint is assumed.
The existing two-critic-update timing is preserved; this example does not claim the paper specifies that timing exactly.

All methods use the same generation settings and task split.
The context and per-call output limits above differ from the benchmark's default per-call limit of 65,536 tokens.
The default episode-level budgets remain unchanged.
Use `--max-steps` to set a different model-turn limit for every serving and test task.
The results record that limit. Apply the same limit to every compared method.
A length-limited model turn can continue through the official agent loop.
Task-level budget exhaustion and context overflow retain the official reward and ending reason.
The driver neither drops these trajectories by length nor adds value bootstrapping.
An unassemblable training trajectory fails explicitly.

Reef's chat handler serves each episode from its exact sampled token IDs (`inference.handler-config.episode_continuation`).
A later call appends only the new observations and turn delimiters as masked context.
This avoids Qwen3 history re-rendering, which otherwise removes earlier reasoning from the training mask.
Blind, Oracle, and SAO use the same setting.

## Install and run

Use Python 3.12 and Reef's supported Slime/SGLang environment from [docker/README.md](../../../../docker/README.md).
Install Reef and its runtime pin as described there. From the repository root, install this example:

```bash
uv pip install -e recipes/sao/examples/servelearnbench
```

Keep the code in project storage. Put model files, checkpoints, logs, and runtime caches on scratch storage.
Set paths to a downloaded Qwen3-8B checkpoint and two fresh, empty directories:

```bash
export MODEL_DIR=/scratch/your-project/models/Qwen3-8B
export CHECKPOINT_DIR=/scratch/your-project/retail-w1/sao-seed42
export REEF_WORK=/scratch/your-project/retail-w1/runs/sao-seed42
export SAO_GPUS=4
bash recipes/sao/examples/servelearnbench/run.sh
```

`SAO_GPUS` sets tensor parallelism for the actor, critic, and inference engine.
The actor and critic share training GPUs. Serving reserves a separate set of GPUs after the training-component refactor (#922).
Allocate `2 * SAO_GPUS` GPUs in total; `SAO_GPUS=1` needs two GPUs.
The default is a launch configuration, not a measured minimum GPU requirement.
Allocate the GPUs before running the script. Do not run the service on a login node.

The defaults are 100 model calls and at most 16,384 output tokens per call.
Thinking and the visible reply share the output budget; the configured context limit is 40,960 tokens.
An episode that reaches the limit ends with the official context-overflow ending and keeps its score.

For baselines, run `METHOD=blind` or `METHOD=oracle` with fresh directories and a separate stack.
Both baselines use the same stack without reports; their training backend stays idle.
Oracle receives the official window bulletin; Blind receives no extra context.
Each stack starts from the same base checkpoint. No method receives test feedback for training.

```bash
METHOD=blind bash recipes/sao/examples/servelearnbench/run.sh
# Change REEF_WORK and CHECKPOINT_DIR before the next run.
METHOD=oracle bash recipes/sao/examples/servelearnbench/run.sh
```

Each trainable serving task submits one report containing every ordered inference receipt.
The driver collects up to 64 tasks within one window, saves the batch, and submits its reports.
Each report names the complete task list for its batch. The record stage waits for every declared task.
The pipeline takes that complete group as one job; each trajectory keeps its own reward.
The grouping does not create prompt comparison groups or change the SAO objective.
It then waits for training and weight publication.
Job submission does not count as a completed update. The driver waits for a selected checkpoint, the job's completion, and reopened inference admission.
No task spans a weight update. Test tasks use the final published actor and never submit reports.
Keep `save-interval: "1"` so each completed batch publishes weights.
A training timeout or service error aborts the run.

The shared defaults are `SAO_BATCH=64`, `SAO_MICRO_BATCH=16`, `SAO_WARMUP=1`, and `SAO_CONTEXT=40960`.
The launcher and result records use the same configuration.
`SAO_CONTEXT` cannot exceed 40,960 because this example does not enable YaRN.
The configured full batch must be divisible by the micro batch; this stack uses data parallel size one.
A window tail uses its actual sample count. Its final micro-batch may be smaller than 16.
For example, a 57-task tail uses micro-batches of 16, 16, 16, and 9. Loss normalization uses 57 samples.
Warmup counts training batches across the whole stream and does not restart at window boundaries.
Each warmup batch performs two critic updates and no actor update.
This warmup is an implementation choice, not value pretraining.

A batch never spans two windows. Every window's remaining serving tasks train as one smaller final batch.
The driver waits for that batch's checkpoint and serving weights before starting the window's tests.
Retail L1 has six windows with 79, 140, 140, 120, 120 and 121 serving tasks, plus 384 test tasks.
At batch 64, all 720 serving tasks train in 14 batches; 208 tasks belong to the six smaller final batches.
The default run has one critic-only batch and 13 actor updates. W0 trains 64 tasks for warmup, then updates the actor on 15 tasks.
A W1-only run has three batches of 64, 64, and 12 tasks: one critic-only batch and two actor updates.
A small final batch is a full optimizer step at the configured learning rate, so it changes the update count and gradient noise.
`--windows 1` selects W1. `--windows 1,2` selects two windows and carries training state between them.
Window indices must be unique and increasing. Gaps are allowed.
The launcher checks the plan before starting the GPU service.
A SAO plan with no actor update fails with an error, including smoke plans.
Use `--check-only` with `run.py` to inspect a plan without starting Reef:

```bash
python recipes/sao/examples/servelearnbench/run.py --method sao --output /scratch/your-project/results.jsonl --check-only
```
The summary records per-window test scores, trained task counts and ending reasons, including budget and context endings.

## Low-cost smoke checks

First run the CPU tests from the repository root:

```bash
python -m pytest tests/recipes/sao/test_servelearnbench.py
```

These tests run the official Retail verification with scripted inference and the real SAO sample assembly.
They need the benchmark installed and do not establish GPU training or benchmark performance.

For the exact-token history check, point Reef's handler test at a downloaded Qwen3 tokenizer:

```bash
REEF_TEST_QWEN_TOKENIZER="$MODEL_DIR" python -m pytest tests/inference/test_chat_episode_continuation.py
```

For a GPU wiring check, allocate two GPUs, use Qwen3-0.6B, and use fresh directories.
Use a short allocation, including a preemptable allocation when available:

```bash
export MODEL_DIR=/scratch/your-project/models/Qwen3-0.6B
export SAO_GPUS=1
export SAO_BATCH=2 SAO_MICRO_BATCH=2 SAO_WARMUP=1
bash recipes/sao/examples/servelearnbench/run.sh --smoke --max-steps 20 --max-tokens 64
```

This runs the first window only: three serving tasks and two test tasks.
Smoke selects up to `max(2, 2 * SAO_BATCH - 1)` serving tasks to exercise a partial batch.
Each task has at most 20 model calls, with at most 64 output tokens per call.
The first two serving tasks warm up the critic in one batch; the one-task tail also updates the actor.
The output explicitly marks the run as smoke. Do not compare its score with the 8B pilot.
Restore `SAO_BATCH=64 SAO_MICRO_BATCH=16 SAO_WARMUP=1` and the 8B model for the pilot.

## Results and recovery limits

`REEF_WORK/results.jsonl` contains configuration, scored episodes, committed training releases, and a summary.
Episode rows include traces, token usage, inference receipts, ending reasons, and producing weight versions.
Training rows record the actual `batch_size`. The summary counts `partial_batch_tasks`; `untrained_tail_tasks` is zero.
`test_score` is the combined mean of all test rewards, multiplied by 100; `window_scores` holds the same per window.
These combined means mix categories and are not the official category scores.
`category_scores` and `window_category_scores` contain the Hidden and Fully Specified means and task counts.
Each category uses a mean over its test tasks, not an unweighted mean of window scores.
An absent category has count zero and score `null`.
Serving rewards never contribute to test scores.
`REEF_WORK/reef.log` contains service and training diagnostics.

Every update saves actor and critic checkpoints, including optimizer state.
The example caps retained checkpoint storage at 1 TiB with a 10 GiB free-space floor.
Adjust these options in `serve.yaml` if your allocation differs.
Copy checkpoints you need to retain before your scratch purge deadline.

To resume, retain the original `REEF_WORK`, `CHECKPOINT_DIR`, `MODEL_DIR`, and scenario name.
Keep the same window selection, model, sampling configuration, batch sizes, and warmup.
Append `--resume` to the original command:

```bash
bash recipes/sao/examples/servelearnbench/run.sh --windows 1 --resume
```

Reef restores the model, optimizer, training progress, and inference weights from its saved state.
The campaign checks its configuration and committed release IDs against that state.
Completed batches and test tasks are skipped. An unreported partial batch is collected again.
Its previous result rows are saved in `results.interrupted-*.jsonl` for inspection.
A fully collected pending batch keeps its original receipts; identical reports can be sent again without duplicate training.
A restart can assign a new runtime load ID to the same restored weights.
The campaign accepts that ID only after checking the committed training history.

Use `--stop-after-batches N` to stop after N total committed training batches.
Remove that option when resuming. For a small checkpoint-resume check, use fresh directories and the 0.6B smoke configuration:

```bash
bash recipes/sao/examples/servelearnbench/run.sh --smoke --max-steps 20 --max-tokens 64 --stop-after-batches 1
bash recipes/sao/examples/servelearnbench/run.sh --smoke --max-steps 20 --max-tokens 64 --resume
```

Resume requires campaign format 3, which records batch membership and trains window tails.
Older score-only-tail campaigns cannot resume under this changed training schedule.
The driver does not repair incomplete optimizer checkpoints.
If Slime reports an ambiguous interrupted training job, follow the [backend recovery instructions](../../../../docs/user-guide/troubleshooting.rst).
A fresh run still requires empty directories. Never start two Reef services on the same storage.
