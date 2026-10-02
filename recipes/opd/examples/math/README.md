# OPD on mathematical reasoning

## LoRA experiment and length-budget rerun

The completed [small LoRA experiment](results/2026-09-30-lora-small/README.md)
used 4096-token training responses and a 32768-token evaluation budget. AIME
mean sampled accuracy changed from 75.0% to 70.4167%; this did not demonstrate
an improvement. The historical result and its actual configuration remain
unchanged in that result packet.

The commands below configure the [requested rerun](https://github.com/Human-Agent-Society/reef/pull/683#issuecomment-5923244778)
with 16384-token training responses and a 64000-token evaluation budget. Use
a fresh run directory and the same initial SFT adapter, prompts and seeds;
do not continue from the negative run's final OPD adapter. Re-evaluate the SFT
baseline and final checkpoint under the same larger budget. No result for this
rerun is available yet; changing these limits does not establish an improvement.
The 65536-token serving context covers the evaluation budget plus the prompt,
and the 32768-token training/teacher limits cover each full training sequence.

The standalone Qwen3-4B/8B study is tracked separately in
[experiment #682](https://github.com/Human-Agent-Society/reef/issues/682).
It is not a Reef reproduction or evidence of a strong SFT baseline.

This LoRA protocol tests whether OPD improves an SFT student. The previous full-parameter SFT was stopped at the contributor's
request after step 1239/3000 (last saved checkpoint: 1200). Its original
roadmap acceptance remains separate; this LoRA experiment does not establish
the original approximately ten-point, full-parameter reproduction target.

- Keep the pinned Qwen3.5-9B-Base student and frozen Qwen3.5-9B teacher below.
- Randomly select 4096 OpenThoughts3 math demonstrations using seed-0 shuffled
  shards and seed-0 shuffled rows within each shard. This samples a shard subset,
  not uniformly across the whole dataset. Truncate to 8192 tokens including
  the prompt; long reasoning/answers may be incomplete. Do not invent EOS.
  Exclude normalized whole-question AIME overlap, and deduplicate questions.
  This is not a paraphrase contamination audit. Preserve the selection manifest.
- Train one epoch: 128 updates, global batch 32, initial learning rate 1e-4
  with linear decay. Freeze the base; use LoRA rank/alpha 32, dropout zero,
  on all language-model MLP down projections and full-attention output
  projections (18,874,368 trainable parameters for this model).
- Continue the same adapter with 30 OPD updates, 64 independent DeepMath
  prompts per update and four student responses per prompt. Responses are
  capped at 16384 tokens; learning rate is 5e-5. No answer rewards are used.
- Evaluate the SFT adapter and final update 30 only, using all 30 AIME'24
  questions and seeds 0 through 15 with a 64000-output-token budget.
  Report any observed gain separately from the operational 3-percentage-point
  target and paired question-bootstrap uncertainty. Improvement is not assured.
  Record the pinned teacher control and exact evaluation settings separately.

`prepare_small.py` consumes the pinned source files from `prepare.py` and writes
`manifest.json`, the selected JSONL files, and a length-limited tokenized dataset:

```bash
python recipes/opd/examples/math/prepare_small.py --random-truncate --source /work/data --output /work/lora-small/data
torchrun --standalone --nproc-per-node=4 recipes/opd/examples/math/sft.py \
  --data /work/lora-small/data/sft.jsonl --tokenized /work/lora-small/data/tokenized \
  --output /work/lora-small/sft --steps 128 --global-batch-size 32 \
  --max-length 8192 --save-steps 64 --lora-rank 32 --lora-alpha 32
```

For deployment, point `OPD_MODEL_PATH` to the unchanged pinned Base weights
with the teacher's tokenizer/chat template, `OPD_ADAPTER_PATH` to the SFT
`final` adapter, and `OPD_RUN_DIR` to a fresh experiment directory. Export
`REEF_TOKEN`, then use `serve.lora-small.yaml`. The importer validates all 80
adapter tensors and splits row-parallel tensors across TP4 before creating
optimizer master weights. Only these unfused Qwen3.5 projections and pipeline
parallel size one are supported. Frozen teacher/reference passes clear the
student adapter before backing up their weights.

```bash
reef serve -c recipes/opd/examples/math/serve.lora-small.yaml
# In another terminal with the same environment:
python recipes/opd/examples/math/run.py \
  --config recipes/opd/examples/math/serve.lora-small.yaml \
  --url http://127.0.0.1:28994 --scenario opd-lora-small \
  --train-data /work/lora-small/data/deepmath.jsonl \
  --eval-data /work/lora-small/data/aime24.jsonl --output /work/lora-small/results \
  --steps 30 --prompts-per-step 64 --samples-per-prompt 4 \
  --train-tokens 16384 --eval-tokens 64000 --eval-repeats 16 --eval-every 30 --concurrency 64
python recipes/opd/examples/math/analyze.py /work/lora-small/results \
  --final-step 30 --eval-every 30 --target-improvement 0.03 --training-mode LoRA
```

## Original full-parameter protocol

This example targets roadmap [#502](https://github.com/Human-Agent-Society/reef/issues/502):
full-parameter on-policy distillation from `Qwen/Qwen3.5-9B` into an
OpenThoughts3-SFT initialization of `Qwen/Qwen3.5-9B-Base`. It runs on local GPU
workers through Reef, Slime/Megatron and SGLang. No Tinker credentials or API
are used. Experiment tracking is in [#682](https://github.com/Human-Agent-Society/reef/issues/682).
See [VALIDATION.md](VALIDATION.md) for completed checks and pending acceptance.
The [frozen teacher control](results/2026-09-29-qwen3.5-9b/README.md) completed
with 84.5833% mean AIME'24 accuracy over 480 samples. Student learning results
remain pending.

## Protocol and acceptance

The model pair follows cookbook commit `dfe4d77e8e8c`. The original 2025 blog
used Qwen3-8B and reported AIME'24 moving from approximately 60% to 70%.
Results with the 9B pair must identify that difference.

- Student base: `Qwen/Qwen3.5-9B-Base`, revision `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- Frozen teacher: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Initialization if no matching public full-parameter checkpoint is available:
  384,000 OpenThoughts3 examples, one pass, batch 128, 3,000 steps, 16,384-token
  sequences, full language-model SFT at initial learning rate 1e-4 with linear
  decay. The unused vision encoder is frozen; no adapters are used. The
  Transformers/FSDP initializer replaces the hosted reference trainer.
- OPD: DeepMath prompts in dataset order, truncated to 1,024 prompt tokens as
  in the reference; 512 prompts times four responses, one optimizer update
  per batch, 200 steps, learning rate 5e-5, temperature 1, response limit 16,384.
  The teacher sees the exact student sequence and remains frozen. No answer
  labels, correctness rewards or teacher-only context are used for training.
- Evaluation: the 30 AIME'24 questions, 16 samples per question at the fixed
  seeds 0 through 15, temperature 1, top-p 1, top-k disabled, 64,000 output-token
  budget, the same boxed-answer instruction and scorer for all checkpoints.
  Enable SGLang deterministic inference in every deployment: this pinned
  runtime otherwise ignores per-request sampling seeds. Evaluate at
  initialization, every 20 steps, and step 200. Evaluation receipts
  are never reported for training. This repeated-sampling protocol must be
  reported separately from a one-sample benchmark score.
- Acceptance is predeclared: the final scheduled OPD checkpoint should improve
  mean AIME'24 accuracy over the frozen SFT initialization by approximately
  ten percentage points; the operational target here is at least 0.10 absolute.
  Report the full curve and uncertainty, including a failed target. Do not
  select the best test-set checkpoint and call it the final result.

A running service, a smoke update or a falling KL loss does not meet that
acceptance criterion. No AIME improvement has yet been established by this
example. Preserve each run's raw predictions, receipts, releases and metrics.

## Environment

Use the repository's supported Slime GPU image and development environment
(`docker/README.md`). The example uses four GPUs: a TP4 actor and four TP1
rollout engines colocated on those devices. The teacher pass temporarily
loads the frozen weights into the actor and then restores the student. The
full training and evaluation lengths require substantially more memory and
time than a short startup check.

Mount model/data/output storage at `/work`. Provide at least enough disk for
the base, teacher, SFT checkpoint and two optimizer checkpoints. The example
limits OPD checkpoint storage to 600 GB and reserves 100 GB of free space.
The measured B200 optimizer/HF checkpoint pair occupied about 188 GiB;
the cap must fit the protected current checkpoint plus the next reservation,
not just one checkpoint. An FP32 SFT export also increases the initial
reservation estimate. Set an appropriate cap and reserve for your filesystem. State and
ports must belong to this experiment, not another running Reef deployment.

The B200 integration check used base image digest
`slimerl/slime@sha256:8851cdff296ce6e569fed9b427aab05a72b55e803ccf73ecb6697c261621fd02`,
with the repository's reviewed Slime and SGLang pins, torch 2.11.0+cu129,
Transformers 5.12.1, tokenizers 0.22.2 and Megatron Bridge 0.4.2. The SFT
initializer also needs `flash-linear-attention==0.4.2`, `fla-core==0.4.2`
and `causal-conv1d==1.7.0` built against the selected CUDA/PyTorch environment.
It refuses to train if Transformers would use its slow Gated DeltaNet fallback.
Install these only in the GPU environment, preserving its existing torch:

```bash
uv pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2
uv pip install --no-deps --no-build-isolation causal-conv1d==1.7.0
```

From the repository root, with its environment activated:

```bash
hf download Qwen/Qwen3.5-9B-Base --revision 68c46c4b3498877f3ef123c856ecfde50c39f404 --local-dir /work/models/Qwen3.5-9B-Base
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir /work/models/Qwen3.5-9B
python recipes/opd/examples/math/prepare.py --output /work/data --tokenizer /work/models/Qwen3.5-9B
```

The preparation script pins all three datasets and writes file checksums.
It refuses to overwrite existing inputs. The SFT shuffle buffer follows the
reference's 384,000-example buffer; preparation takes significant host memory.
The scripts require the packages in this example's `pyproject.toml` plus the
GPU training environment. They are a direct `reef_client` campaign and do not
require Harbor or reef-eval.

## SFT initialization

Prefer a published compatible SFT checkpoint with documented data and
training settings. If one is unavailable, prepare and train the initialization:

```bash
python recipes/opd/examples/math/sft.py --tokenize-only --data /work/data/sft.jsonl --tokenized /work/sft-data --output /work/sft
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --standalone --nproc-per-node=4 recipes/opd/examples/math/sft.py --data /work/data/sft.jsonl --tokenized /work/sft-data --output /work/sft
```

The initializer keeps the latest optimizer checkpoint (the previous one may
coexist while the next is being written). It masks the observed prompt, preserves reasoning tokens, and
does not append a false EOS when a response is truncated. It saves optimizer
checkpoints for explicit `--resume /work/sft/checkpoint-N` and exports the
final Hugging Face checkpoint to `/work/sft/final`. The exported tokenizer
comes from the pinned teacher, whose vocabulary is shared by the student.

## OPD and evaluation

Start the service with a fresh run directory and a private service token:

```bash
export OPD_MODEL_PATH=/work/sft/final
export OPD_RUN_DIR=/work/opd
mkdir -p "$OPD_RUN_DIR"
export REEF_TOKEN="$(openssl rand -hex 24)"
reef serve -c recipes/opd/examples/math/serve.yaml
```

In another terminal with the same `REEF_TOKEN`, after `/healthz` is ready:

```bash
python recipes/opd/examples/math/run.py --config recipes/opd/examples/math/serve.yaml --train-data /work/data/deepmath.jsonl --eval-data /work/data/aime24.jsonl --output /work/opd/results
```

The driver collects a complete batch before submitting any training reports,
checks that all responses came from one release, then waits for exactly one
committed update. Its batch must match both recipe and trainer configuration.
It refuses an existing output directory or scenario unless `--resume` is
explicitly supplied. To recover an interrupted **driver process**, rerun the
same command with `--resume`, retaining the same live Reef service, scenario
and output directory. Completed responses are reused; only missing samples
from the current release are regenerated. Reports have deterministic IDs, so
retrying after a lost HTTP response does not enqueue a duplicate update.

Recovery checks input/configuration/driver SHA-256 hashes, sampling settings,
release history and each commit's exact consumed receipts/report IDs. It
refuses missing historical predictions, unrelated updates, rollback or pending
publication histories. Only one driver may own an output directory at a time.
Network timeout and concurrency may change on resume; the statistical protocol
may not. An incomplete final JSONL append is discarded before collecting the
missing sample; malformed complete records are rejected.

This option does not restore a stopped Reef trainer or its optimizer. If the
service or GPU runtime fails, preserve the run and inspect its checkpoint and
recovery state before proceeding. In particular, do not start a fresh service
from baseline weights and label it a resumed trained model.

Use `--steps 0` on an independently launched frozen SFT or teacher deployment
for a control evaluation. Keep the evaluation data, seeds, token budget,
prompt and scorer identical. Do not report control receipts for training.
For a smoke run, copy the YAML and reduce **both** batch sizes together with
`--prompts-per-step`, `--samples-per-prompt` and token budgets. Label those
results as integration checks.

## Output

`config.json` records the driver settings and `inputs.json` their input/driver
checksums; `train-*.jsonl` and `eval-*.jsonl`
record responses, receipts, sampled release IDs, seeds, finish reasons and
usage. `metrics.jsonl` records evaluation accuracy and truncations per step.
`releases-*.json` records publication history, and `commit-*.json` identifies the
exact records consumed by each update. Evaluation timing covers the attempt
that completed the evaluation, not time spent in previous interrupted attempts. Keep the service logs and exact
resolved stack configuration beside these files. Include the actual SFT
checkpoint identifier and software image digest in an experiment report.

After the final scheduled evaluation, generate the curve and acceptance record:

```bash
python recipes/opd/examples/math/analyze.py /work/opd/results
```

The analysis requires the same question/seed pairs at every checkpoint and
all scheduled evaluations, including the final step. Missing, duplicate or
unexpected steps prevent an acceptance result. For a different predeclared
schedule, pass matching `--final-step` and `--eval-every` values. Its
95% bootstrap interval resamples whole questions, retaining repeated samples
and baseline/final pairing. `acceptance.json` reports whether the final point
estimate reaches the predeclared target; it does not equate that threshold
with statistical significance. `learning-curve.png` includes every evaluated
checkpoint. The scorer accepts the final boxed integer (including a simple
`\boxed{\text{42}}` form); symbolic equivalents are not silently converted.
