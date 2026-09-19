# SDFT on a skill stream

This example reproduces the sequential experiment of
[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
(Figure 3): one model learns a stream of skills in turn, and each skill's test
accuracy is followed through the whole stream, so the score on a skill after
the next skill's training is the forgetting. The stream here is Tool Use, then
Science Q&A, the first two of the paper's three (the Medical split is not
published), trained with the `sdft` recipe (`recipes/sdft/`) and compared with
an SFT control on the same demonstrations. Every stage runs as a
[reef-eval](https://github.com/Human-Agent-Society/reef-eval) episode: a Harbor
task whose judge scores the served model on both skills, so the curves are
the Lab store's trace rows.

The [`sdft` recipe page](../../../../docs/user-guide/recipes/sdft.rst) documents
the recipe; [Evolve your model](../../../../docs/user-guide/evolve-your-model.rst)
walks through the training stack. This README records the protocol, its
distance from the paper's, and the numbers.

```text
run.py             the stream: per stage, the Reef stack from the previous stage's weights, then lab.run
harness/agent.py      the Harbor agent: runs the stage runner in the task container with the host's SKILLS_* settings
harbor/tooluse/       the Tool Use stage: ToolAlpaca's training split through Reef, the reference's regex scorer
harbor/science/       the Science Q&A stage: the Chemistry L-3 split through Reef, exact match on the answer tag
  environment/
    skills.py         the two skills: the reference datasets, their scorers, the Reef calls (shared by both tasks)
    stage.py          the stage runner: 32 prompts, 32 samples, 32 reports, one step, wait for the release, repeat
    score.py          the judge's rule: both skills' test accuracy of the served model, this task's as the reward
    judge_server.py   reef-eval's template judge, recording the scores and reporting the last submission
  tests/grade.py      the verifier: the judge's final result as the reward, its score log as the trace
serve.yaml            the stack config: full fine-tuning, the reference's Figure 3 settings
docker-compose.yaml   the stack in the reef image: four GPUs, the engines colocated with the actor
plot.py               Figure 3 from the Lab store: both skills' accuracy against gradient steps, the SFT control beside
run.sh                checks the setup, mints the token, runs run.py in an ephemeral uv environment
```

## The protocol

The reference implementation ([idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)
at `d77573212fa0`) ships both splits under `data/`. Tool Use is ToolAlpaca:
4046 training prompts, each a tool's documentation and a user request in the
ReAct format, with the dataset's golden response as the demonstration; 97
test prompts scored by `eval_tooluse.py` (the multiset of `Action:` names and
the merged `Action Input:` JSON must both equal the golden API call). Science
Q&A is the Chemistry L-3 subset of SciKnowEval: 2674 training prompts, each a
system message fixing the `<reasoning>`/`<answer>` format and a four-option
question, with GPT-4o's response as the demonstration; 507 test prompts
scored by `eval_science.py` (exact match of the text inside the last
`<answer>` tag). Both scorers decode greedily, Tool Use in a 1024-token
window and Science Q&A in 2048.

The paper's Figure 3 trains one model through the skills in sequence, each
skill a single-task run started from the previous one's weights. The
settings are the ones the authors gave for these runs (issue 9 of the
reference): learning rate 1e-5 with a cosine schedule and 10 warmup steps,
32 prompts per optimizer step for two epochs, one on-policy sample per
prompt in a 2048-token window, truncated importance sampling capped at 2,
the first three response tokens skipped, and the teacher a copy of the
stage's initial weights moving 2% toward the policy after every step. The
KL is the forward one (`sdft-divergence: forward`), the reference's default
and the paper's setting. That issue's run switched to the reverse KL; on
this stack the reverse KL left Tool Use at its baseline for 180 steps while
its training KL fell from 0.32 to 0.08 (its gradient on the demonstration's
action token scales with the student's own probability of it, so a student
that rarely picks that action barely moves toward it), whereas the forward
KL's gradient, the difference of the two distributions, carries it over
directly.

`run.py` keeps that protocol with Reef in the trainer's place. Each stage
starts the stack from the previous stage's HF export (the base model for the
first) with the learning-rate schedule spanning exactly the stage's steps
(252 for Tool Use, 167 for Science Q&A: two shuffled epochs cut into steps
of 32, the tail dropped), then runs the stage as a Harbor task. The stack is
four GPUs, the actor (tensor parallel 4) colocated with four rollout
engines. In the task container, `stage.py` sends each step's 32 prompts
through Reef at temperature 1.0, reports each demonstration as the report's
`context` against the sample's receipt, and waits for the step's training
release before sampling the next step, so every sample is on policy. Before
the first step, every ten steps, and after the last, it submits the step
number to the task's judge, which scores the served model on both test
splits and records both accuracies; the verifier's reward is this stage's
skill after the last step, and the judge's log becomes the trace rows
`plot.py` draws.

## The SFT control

Figure 3 compares SDFT with supervised fine-tuning on the same
demonstrations. The control is not part of this example's code: it ran on
the same tasks, stack and optimizer with an SFT recipe that rendered each
demonstration as the assistant turn of the recorded request and trained
those tokens with Slime's stock `sft_loss`, ignoring the student's samples,
so the stage runner and the judge drove both runs unchanged (the same
prompts in the same order, the same 32-prompt steps). Reef does not ship
that recipe; `plot.py --control` takes its judge scores as a CSV in the
columns `plot.py` writes and draws them beside the SDFT stream.

What differs from the reference: sampling goes through SGLang instead of
vLLM (the same settings: temperature 1, top-p 1, no top-k, no repetition
penalty), the trainer is Megatron instead of TRL on one GPU, the teacher
copy's update is accumulated in float32 where the reference mixes bfloat16
weights, and the steps run across the epoch boundary with the tail dropped
where TRL's dataloader ends each epoch on a partial batch.

## Setup (once)

The training stack needs the GPU environment described in
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst), as the
`reef` image. On the host:

```bash
pip install uv
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
```

## Run

```bash
cd recipes/sdft/examples/skill_stream
./run.sh                                     # Tool Use (252 steps), then Science Q&A (167)
SKILLS_STEPS=2 SKILLS_GPUS=4,5,6,7 SKILLS_PORT=28903 ./run.sh --stream smoke   # two steps per stage, beside a full run
uv run --no-project --python 3.12 --with reef-eval --with matplotlib plot.py \
    --lab work/lab --out figure3
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (default
`~/models`), `RUN_DIR` (default `./work`: the Lab store and trials under
`lab/`, each stage's stack state and checkpoints under
`<stream>/sdft/<task>/`). `run.py` takes `--stream` (the stream's name in
the Lab store; rows already recorded are skipped, so a crashed stream
resumes and a new name starts over) and `--seed`; `SKILLS_GPUS` (four
comma-separated ids, default `0,1,2,3`) and `SKILLS_PORT` (default `28902`)
place the stack, and the stream names it, so a smoke stream runs beside the
full one on the other four GPUs of an eight-GPU host. The stage runner reads
`SKILLS_EPOCHS`, `SKILLS_PROMPTS_PER_STEP` (must equal the recipe's batch
size in `serve.yaml`), `SKILLS_MAX_TOKENS`, `SKILLS_EVAL_EVERY` and
`SKILLS_STEPS`, forwarded from the host by the harness.

## Results

The Figure 3 runs (the SDFT stream and the SFT control, seed 42) are in
progress; their scores and the figure follow.
