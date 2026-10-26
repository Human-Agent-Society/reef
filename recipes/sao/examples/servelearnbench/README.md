# ServeLearnBench Retail L1 with SAO

This example runs the Retail L1 scenario of ServeLearnBench through Reef the way the benchmark's own runner does.
Every window runs in order: a window's serving tasks are trained on in batches, then its held-out test tasks run against the frozen weights, and the weights carry into the next window.
The summary's test score is the mean reward over all test tasks times 100, the benchmark's setting score for the scenario.
It implements the Retail part of [issue #893](https://github.com/Human-Agent-Society/reef/issues/893); the other scenarios and the OPD comparison are outside this example.

The benchmark dependency is pinned in `pyproject.toml`.
Its official agent loop owns environment resets, tools, budgets, termination, and verification.
The harness connects the model transport to Reef and reports the official scalar reward.
Hidden rules, ground-truth actions, and verifier details never enter SAO feedback.

## Layout

```text
servelearnbench/
  run.sh          starts reef serve, then runs the campaign
  run.py          the campaign: the fixed split, batches of eight, the training barrier, results.jsonl
  serve.yaml      the Reef stack: Qwen3-8B actor and critic, SAO, exact-token episodes
  harness/
    agent.py      the benchmark's LLM interface over reef_client and its episode loop with a turn limit
    report.py     the score report against an episode's inference receipts
```

## Experiment settings

| Setting | Pilot |
| --- | --- |
| Actor | Dense Qwen3-8B, full-parameter updates |
| Critic | Qwen3-8B value model; self-attention frozen, other parameters trainable |
| Sample / batch | One complete task trajectory; eight tasks per training batch; micro batch one |
| Warmup | First 10 training batches: critic only |
| Critic cadence | Two critic updates per batch, including warmup |
| Actor cadence | One update per batch after warmup |
| Learning rates | Actor `1e-6`; critic `5e-6` |
| Evaluation | Each window's tests against the frozen weights at that window's end; no test reports or updates |
| Context / per-call output | 40,960 / 16,384 tokens, the model's native context |
| Episode budgets | Official 100 turns, 50 world-tool calls, 256,000 output tokens |

This is a dense, batch-eight adaptation of SAO. It is not a reproduction of the paper's MoE, batch-128 experiment.
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

`SAO_GPUS` sets tensor parallelism for the colocated actor, critic, and inference engine.
The default is a launch configuration, not a measured minimum GPU requirement.
Allocate the GPUs before running the script. Do not run the service on a login node.

The defaults are 100 model calls and at most 16,384 output tokens per call.
Thinking and the visible reply share the output budget; the context limit is the model's native 40,960 tokens.
Retail episodes stay near 20,000 tokens. An episode that reaches the limit ends with the official context-overflow ending and keeps its score.

For baselines, run `METHOD=blind` or `METHOD=oracle` with fresh directories and a separate stack.
Both baselines use the same stack without reports; their training backend stays idle.
Oracle receives the official window bulletin; Blind receives no extra context.
Each stack starts from the same base checkpoint. No method receives test feedback for training.

```bash
METHOD=blind bash recipes/sao/examples/servelearnbench/run.sh
# Change REEF_WORK and CHECKPOINT_DIR before the next run.
METHOD=oracle bash recipes/sao/examples/servelearnbench/run.sh
```

Each serving task submits one report containing every ordered inference receipt.
The driver collects eight tasks before waiting for training and weight publication.
No task spans a weight update. Test tasks use the final published actor and never submit reports.
Keep `save-interval: "1"` so each completed batch publishes weights.
A training timeout or service error aborts the run.

`SAO_BATCH=8` and `SAO_WARMUP=10` are independent configuration values.
The warmup counts optimizer steps across the whole stream, so the first ten batches of the first window update only the critic.
This warmup is an implementation choice, not value pretraining.

A batch never spans two windows. A window's serving tasks that do not fill a batch are served and scored but not reported for training; their `untrained_tail` rows make that explicit.
Retail L1 has six windows with 79, 140, 140, 120, 120 and 121 serving tasks and 384 test tasks in all; at `SAO_BATCH=8`, 16 serving tasks fall into tails.
`--windows 1,2` runs a subset of the windows, in that order.
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

For a GPU wiring check, use one GPU, Qwen3-0.6B, and fresh directories.
Use a short allocation, including a preemptable allocation when available:

```bash
export MODEL_DIR=/scratch/your-project/models/Qwen3-0.6B
export SAO_GPUS=1
export SAO_BATCH=1 SAO_WARMUP=1
bash recipes/sao/examples/servelearnbench/run.sh --smoke --max-steps 20 --max-tokens 64
```

This runs the first window only: two serving tasks and two test tasks.
Each task has at most 20 model calls, with at most 64 output tokens per call.
The first serving task warms up the critic; the second also updates the actor.
The output explicitly marks the run as smoke. Do not compare its score with the 8B pilot.
Restore `SAO_BATCH=8 SAO_WARMUP=10` and the 8B model for the pilot.

## Results and recovery limits

`REEF_WORK/results.jsonl` contains configuration, scored episodes, committed training releases, and a summary.
Episode rows include traces, token usage, inference receipts, ending reasons, and producing weight versions.
The summary's `test_score` is the arithmetic mean of every test reward across the windows run, multiplied by 100; `window_scores` holds the same per window.
`REEF_WORK/reef.log` contains service and training diagnostics.

Every update saves actor and critic checkpoints, including optimizer state.
The example caps retained checkpoint storage at 1 TiB with a 10 GiB free-space floor.
Adjust these options in `serve.yaml` if your allocation differs.
Copy checkpoints you need to retain before your scratch purge deadline.

Reef owns checkpoint recovery. This campaign driver does not yet resume a partially completed benchmark stream.
It refuses nonempty run directories and previously trained scenarios to prevent mixing fresh tasks with old weights.
If interrupted, retain its logs and checkpoints, and start a separate pilot run.
CPU checkpoint tests cover Reef's save and recovery contracts; a real interrupted GPU run needs separate validation.
