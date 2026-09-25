# SDPO on SciKnowEval Chemistry

This example runs the generalization sweep of
[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
through Reef: the Chemistry split of SciKnowEval, trained with the `sdpo`
recipe (`recipes/sdpo/`). Each step samples a complete 32 by 8 grid from one
policy release, lets an EMA teacher reread each question with a successful
sibling's response, and distils that teacher into the student in one optimizer
step. Every five steps the served model is scored avg@16 on the held-out split,
and that series is the learning curve.

The data, the prompts, the scoring rule and the hyperparameters are the pinned
reference's ([lasgroup/SDPO](https://github.com/lasgroup/SDPO) at
`7c457fc1b1f6`), which the task image clones.

```text
run.sh                          starts the Reef stack, then runs the episode
run.py                          reef-eval runs the harbor task under harness:HarborAgent
harness/agent.py                runs the stage in the task container; makes no model calls itself
harbor/chemistry/
  instruction.md                what the task trains and how
  task.toml                     timeouts and resource limits
  environment/
    Dockerfile                  clones the reference at its pin, installs reef-client
    chemistry.py                the split, the prompts, the reference's scorer, the Reef calls
    stage.py                    the runner: samples each grid, reports it, waits, evaluates
  tests/grade.py                the verifier: the last avg@16 from the runner's curve
serve.yaml                      the training stack: Qwen3-8B on four GPUs, the sdpo loss family
```

## The protocol

The reference's settings, resolved from `verl/trainer/config/sdpo.yaml` and
`user.yaml` at the pin:

| Setting | Value |
| --- | --- |
| Data | SciKnowEval Chemistry, 1890 train and 210 test questions, the reference's split |
| Step | 32 questions x 8 on-policy samples, one optimizer step |
| Windows | 2048-token prompt, 8192-token response, 18944-token model window |
| Teacher | EMA, update rate 0.05; reprompt capped at 10240 tokens |
| Divergence | generalized JSD, beta 0.5, student top-100 plus the remaining mass |
| Correction | detached per-token importance weights, capped at 2 |
| Demonstration | the first successful sibling in rollout order, excluding the rollout itself |
| Feedback | disabled, as the generalization sweep has it |
| Optimizer | AdamW, LR 1e-5 after 10 warmup steps, weight decay 0.01, clipping at 1 |
| Evaluation | every 5 steps, avg@16 at temperature 0.6 and top-p 0.95 |

A rollout's score is the reference's own: correct when the letter between the
response's last `<answer>` tags matches the dataset's. The score does not enter
the loss. It picks which sibling demonstrates, and a rollout whose question no
sibling solved keeps the plain request and a sample weight of 0, so the step
still takes its optimizer step.

Evaluation samples are not recorded, so they never become training data.

## The model, and what this does not reproduce

The paper reports Chemistry on both `Qwen/Qwen3-8B` and
`allenai/Olmo-3-7B-Instruct`, and this example uses Qwen3-8B, which is also the
reference's own default. **The recorded OLMo-3 comparison in the SDPO
pull request was produced by the author's verl implementation, not by Reef, and
cannot be run here**: the pinned Slime ships no OLMo architecture, and OLMo-3
also needs interleaved sliding-window attention and YaRN rope scaling in
Megatron. An OLMo-3 arm needs that backend support first.

So the comparison this example supports is a Reef Qwen3-8B curve against an
author-implementation Qwen3-8B curve on the same split, seed and budget. That
second curve has not been run either.

## Run it

In a configured Reef GPU environment with four GPUs and Docker:

```bash
hf download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
./run.sh
```

`run.sh` mints a token per run directory, waits for `/healthz`, and runs the
episode in an ephemeral `uv` environment. State goes to `$SDPO_RUN_DIR`
(default `./work`): the trial's rows, the runner's log, the curve, and the Reef
log all land there.

The runner's settings come from the host environment and the harness forwards
them into the container:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `SDPO_STEPS` | 0 | a ceiling on the steps; 0 runs the whole schedule |
| `SDPO_TRAINING_HOURS` | 0 | stop after the first evaluation past this training time; 0 disables the budget |
| `SDPO_GROUPS_PER_STEP` | 32 | questions per grid; MUST equal `serve.yaml` |
| `SDPO_ROLLOUTS_PER_GROUP` | 8 | samples per question; MUST equal `serve.yaml` |
| `SDPO_EVAL_EVERY` | 5 | steps between evaluations |
| `SDPO_EVAL_SAMPLES` | 16 | samples per test question, the paper's avg@16 |
| `SDPO_SEED` | 42 | fixes the question order |

Start with `SDPO_STEPS=2` to check the stack end to end before committing to a
budget. Four H100s at this window are tight: a step runs three forwards over
the batch (the student's top-K selection, the teacher, and the training
forward), so `max-tokens-per-gpu` and `mem-fraction-static` in `serve.yaml` are
the first knobs to tune.

## Results

None recorded yet. A run is complete when it carries the raw per-evaluation
values, the seed, the model revision and the resolved configuration, following
[Reef #22](https://github.com/Human-Agent-Society/reef/issues/22). The
acceptance criterion from the roadmap is SDPO above GRPO at matched generations,
reaching GRPO's final accuracy in fewer, as the paper reports.
