# ServeLearnBench Retail L1/W1 with SAO

This pilot trains on 140 Retail L1/W1 serving tasks, then evaluates on 64 held-out test tasks.
W1 is the second, zero-based window. The driver preserves task order and excludes all other windows.
It implements the Retail pilot from [issue #893](https://github.com/Human-Agent-Society/reef/issues/893).
It does not implement the complete benchmark or its OPD comparison.

The benchmark dependency is pinned in `pyproject.toml`.
Its official agent loop owns environment resets, tools, budgets, termination, and verification.
The adapter connects model transport and sends the official scalar reward to Reef.
Hidden rules, ground-truth actions, and verifier details never enter SAO feedback.

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
| Evaluation | Frozen final actor, no test reports or updates |
| Context / per-call output | 131,072 / 16,384 tokens; static YaRN factor four |
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

The example's inference handler retains exact sampled token IDs across turns.
It appends only new observations and turn delimiters as masked context.
This avoids Qwen3 history re-rendering, which can otherwise remove earlier actions from the training mask.
Blind, Oracle, and SAO use this same history policy.
It retains prior reasoning instead of applying the tokenizer's usual history cleanup.
The handler accepts one serial campaign and nonstreaming requests only.

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

The launcher prepares a separate model view under `CHECKPOINT_DIR/model-128k`.
It links the original weights and writes a Qwen3 configuration with YaRN factor four and original context 32,768.
The source checkpoint stays unchanged. SGLang reads this configuration; the example provider installs matching YaRN in both training roles.
The provider also sets the attention magnitude and disables incompatible fused RoPE.
All published HF checkpoints must retain this model configuration.

The defaults are 100 model calls and at most 16,384 output tokens per call.
Thinking and the visible reply share the output budget; the cumulative context limit is 131,072 tokens.
A 128k limit does not force a task to generate 128k tokens or establish a GPU capacity result.

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

`SAO_BATCH=8` and `SAO_BATCH=1 SAO_WARMUP=10` are independent configuration values.
Together, they use the first 80 serving tasks for 20 critic updates and no actor updates.
The actor starts on batch 11. This warmup is an implementation choice, not value pretraining.

The fixed split contains 140 serving tasks. Seventeen complete batches train on 136 tasks.
The final four tasks are still served and scored but are not reported for training.
Their `untrained_tail` rows make this remainder policy explicit; no samples are duplicated or borrowed from test.
The summary records trained task counts and ending reasons, including budget and context endings.

## Low-cost smoke checks

First run the CPU tests from the repository root:

```bash
python -m pytest tests/recipes/sao tests/slime_backend/test_critic_metrics.py \
  tests/slime_backend/test_worker_hooks.py
```

These tests include official Retail verification with scripted inference and real SAO sample assembly.
The CP test uses two CPU/Gloo ranks. They do not establish GPU training or benchmark performance.

For the history check, point the test at a downloaded Qwen3 tokenizer:

```bash
REEF_TEST_QWEN_TOKENIZER="$MODEL_DIR" python -m pytest tests/recipes/sao/test_retail_inference.py
```

This checks that tool serialization and length-limited turns retain every sampled action token.

For a GPU wiring check, use one GPU, Qwen3-0.6B, and fresh directories.
Use a short allocation, including a preemptable allocation when available:

```bash
export MODEL_DIR=/scratch/your-project/models/Qwen3-0.6B
export SAO_GPUS=1
export SAO_BATCH=1 SAO_WARMUP=1
bash recipes/sao/examples/servelearnbench/run.sh --smoke --max-steps 20 --max-tokens 64
```

This runs two serving tasks and two test tasks.
Each task has at most 20 model calls, with at most 64 output tokens per call.
The first serving task warms up the critic; the second also updates the actor.
The output explicitly marks the run as smoke. Do not compare its score with the 8B pilot.
Restore `SAO_BATCH=8 SAO_WARMUP=10` and the 8B model for the pilot.

## Results and recovery limits

`REEF_WORK/results.jsonl` contains configuration, scored episodes, committed training releases, and a final test score.
Episode rows include traces, token usage, inference receipts, ending reasons, and producing weight versions.
The test score is the arithmetic mean of the 64 official rewards, multiplied by 100.
`REEF_WORK/reef.log` contains service and training diagnostics.

Every update saves actor and critic checkpoints, including optimizer state.
The example caps retained checkpoint storage at 1 TiB with a 10 GiB free-space floor.
Adjust these options in `serve.yaml` if your allocation differs.
Copy checkpoints you need to retain before your scratch purge deadline.

Reef owns checkpoint recovery. This campaign driver does not yet resume a partially completed benchmark stream.
It refuses nonempty run directories and previously trained scenarios to prevent mixing fresh tasks with old weights.
If interrupted, retain its logs and checkpoints, and start a separate pilot run.
CPU checkpoint tests cover Reef's save and recovery contracts; a real interrupted GPU run needs separate validation.
