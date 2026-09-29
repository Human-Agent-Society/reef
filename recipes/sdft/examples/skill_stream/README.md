# SDFT on a skill stream

This example reproduces the sequential experiment in Figure 3 of
[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897).
One model learns Tool Use first and Science Q&A second. Both skills are scored
during both stages. A drop in the skill that is not being trained is
forgetting.

The model trains with the `sdft` recipe (`recipes/sdft/`) and is compared with
an SFT control on the same demonstrations. Each stage runs as a
[reef-eval](https://github.com/Human-Agent-Society/reef-eval) episode and a
judge scores the served model on both test splits.


```text
run.py                runs the two stages in order and starts each one from the previous stage's weights
harness/agent.py      the Harbor agent that runs the stage runner in the task container
harbor/tooluse/       the Tool Use stage
harbor/science/       the Science Q&A stage
  environment/
    skills.py         the two datasets with their scorers and the Reef calls
    stage.py          the stage runner that samples 32 prompts and reports them and waits for the training step
    score.py          scores the served model on both test splits
    judge_server.py   the reef-eval template judge
  tests/grade.py      the verifier that returns the judge's final score
serve.yaml            the training stack config
docker-compose.yaml   the stack in the reef image on four GPUs
run.sh                checks the setup and runs run.py
results/              the learning curve of the recorded run
```

## The protocol

The data comes from the reference implementation
([idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)
at `d77573212fa0`).

- **Tool Use** is ToolAlpaca with 4046 training prompts and 97 test prompts.
  Each prompt holds a tool's documentation and a user request in the ReAct
  format. The demonstration is the dataset's golden response. A test answer
  is correct when its API call equals the golden call.
- **Science Q&A** is the Chemistry L-3 subset of SciKnowEval with 2674
  training prompts and 507 test prompts. Each prompt is a four-option
  question. The demonstration is GPT-4o's response. A test answer is correct
  when the text in its last `<answer>` tag matches exactly.


The training settings are the ones the authors gave for this experiment in
issue 9 of the reference. The learning rate is 1e-5 with 10 warmup steps and
a cosine schedule over the stage. Each step takes 32 prompts with one
on-policy sample per prompt and a stage runs for two epochs. The loss is the forward KL
with truncated importance sampling capped at 2 and it skips the first three
response tokens.

The teacher is a frozen copy (EMA=0) of the stage's initial weights
(`sdft-teacher-update-rate: 0`). With non-zero EMA, we did observe SDFT training collapse with model drifting.

`run.py` starts each stage from the previous stage's HF export and the first
stage starts from the base model. The stack uses four GPUs with the actor and
four rollout engines colocated. `stage.py` sends each step's 32 prompts
through Reef and reports each demonstration as the report's
`teacher_context`. It waits for the training step before it samples again so
every sample is on policy. The judge scores the served model before the first
step and every ten steps and after the last step.


## Setup (once)

The training stack needs the GPU environment described in
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst) as the
`reef` image. On the host:

```bash
pip install uv
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
```

## Run

```bash
cd recipes/sdft/examples/skill_stream
./run.sh   # Tool Use for 252 steps and then Science Q&A for 167
```

The run reads these environment variables:

- `REEF_IMAGE` is the stack image and defaults to `reef`.
- `MODEL_DIR` holds the model and defaults to `~/models`.
- `RUN_DIR` holds the Lab store and each stage's checkpoints and defaults to
  `./work`.
- `CUDA_VISIBLE_DEVICES` names the stack's four GPUs and defaults to `0,1,2,3`.
- `REEF_PORT` is the stack's host port and defaults to `28902`.
- `SDFT_STEPS` caps the steps of each stage.

`run.py` takes `--stream` and `--seed`. The stream names the run in the Lab
store. Recorded stages are skipped so a crashed stream resumes and a new name
starts over. The stream also names the stack so two streams can run side by
side on different GPUs and ports.

## Results

![Both skills' test accuracy against gradient steps for SDFT and the SFT control](results/2026-09-19-skill-stream-qwen2.5-7b/learning_curve.png)

The figure shows one run of each method on Qwen2.5-7B-Instruct.
The curves follow the Science Q&A stage through step 120.

Tool Use training comes first and both methods learn it to about the same
level. SFT pushes Science Q&A below the base model and SDFT does not. SFT's
Science Q&A score falls from 32% to 28% while SDFT's rises from 30% to 36%.

Science Q&A training comes second and both methods learn the new skill. SDFT
forgets only a little Tool Use and goes from 67% to 61%. SFT forgets much
more and goes from 70% to 56%.
