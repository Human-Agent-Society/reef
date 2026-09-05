# OpenClaw-RL on Reef

This example implements the harness side of
[OpenClaw-RL](https://arxiv.org/abs/2603.10165)'s personal-agent experiment.
OpenClaw-RL trains an agent from its ordinary usage: each turn is scored by
what the user did next, a follow-up that moves on counting as acceptance and
a complaint as rejection, and the policy updates while it keeps serving. The
method itself is the `openclawrl` recipe package (`recipes/openclawrl/`). Its
processor rebuilds sessions from the traffic Reef already records, judges
every completed turn with a PRM on a private worker, and turns accepted
hindsight hints into a training signal. No report call is required. The
processor prefers a stable, conversation-unique `x-reef-tag-session` value
and otherwise falls back to matching transcripts; this example's header shim
supplies the tag for the unmodified Hermes agent.

This directory is the experiment around that recipe: a simulated student
brings GSM8K homework problems to a real Hermes agent, one session after
another, and the metric is how many sessions the agent needs before its
answers match the student's taste. The student wants solutions that do not
look AI-written (natural prose, no bold, no lists) but still show every step
of a correct solution. It never states this preference, so the agent has to
learn it from the reactions.

The [`openclawrl` recipe page](../../../../docs/user-guide/recipes/openclawrl.rst)
documents the recipe's configuration. This README records the stream's
implementation details and the learning curve of a completed run.

```text
harbor-tasks/
  gsm8k-s000/ ... gsm8k-s071/   one stock Harbor task per session, in dataset order
    task.toml                    metadata, timeouts, agent network allowlist (judge only)
    instruction.md               what the agent is told
    environment/
      Dockerfile                 hermes-agent, pinned to a commit
      Dockerfile.judge           FROM the shared sidecar image, plus this session's problem.json
      docker-compose.yaml        main container + judge service
      problem.json               this session's question and gold answer
    tests/test.sh                copies the judge's /final verdict into the verifier output
harness/
  agent.py                       HermesStreamAgent: the reef-eval agent that runs one session
user_sim/
  personas.py                    the student persona and the strict acceptance criterion
  student_server.py              the judge sidecar (HTTP), scripted or LLM-backed
  pyproject.toml                 packaged as openclawrl-user-sim
  Dockerfile                     the shared sidecar image, built by run.sh
results/
  learning_curve.py              per-session accept and style metrics, logged to W&B during a stream or exported afterwards
  2026-08-27-gsm8k-stream-qwen3-4b-thinking/
                                 the learning curve of a complete run
serve.yaml                       the paper's training stack: Reef, Slime, the PRM, the student model
docker-compose.yaml              the reef container: GPUs, mounts, host networking, the health check
run.sh                           builds the sidecar image, starts the stack, runs the stream
restamp.sh                       re-pins the 72 tasks to user_sim/'s content hash after a change there
pyproject.toml                   makes harness/ importable
```

## The ordinary agent

The agent is a stock `hermes-agent` install inside each task container. It
is driven through its command line, one `hermes -z` turn per student
message, with its home directory on the stream's state mount. Hermes reads
its model endpoint from its own config and sends OpenAI-compatible chat
requests; it knows nothing about Reef, scenarios, or training.

The judge sidecar plays the student. `student_server.py` runs the persona
from `personas.py`, reacts to each reply, and records the session. The
reactions come from the Qwen3-32B persona served by the stack, named by
`OPENCLAWRL_USER_LLM_URL` and `OPENCLAWRL_USER_LLM_MODEL`; the task
environments require both. A scripted line is the fallback when the LLM call
fails or when the LLM student starts dictating the math, which the persona
forbids: a styled reply gets the complaint, a clean one gets the request to
save the file.

## The homework stream

The stream is a [reef-eval task stream](https://github.com/Human-Agent-Society/reef-eval):
an ordered folder of ordinary Harbor tasks that reef-eval runs strictly in order,
with one directory (`$REEF_EVAL_STATE_DIR`) mounted into every position. The
agent carries state across sessions, so the sessions cannot be run as
independent trials.

The committed stream has the original OpenClaw-RL paper's 72 GSM8K problems
as 72 tasks. Each task is a stock Harbor task whose environment carries only
its own `problem.json`; the judge sidecar is a compose service built from one
shared image.

The sidecar scores each session on the agent's first solution reply, under
the strict criterion in `personas.py`: no AI-style markers (bold, headers,
bullet or numbered lists, horizontal rules, tables, `\boxed`, "final
answer:"), at least two visible calculation steps, and the gold answer
present.

Weights and memory are two separate ways the agent can adapt. The paper's
numbers are for weights only, so `hermes_memory` is off by default.

## The changes needed for Reef

`HermesStreamAgent` (`harness/agent.py`) owns one session per stream
position. Around the unmodified agent it makes three changes:

1. It runs a small header shim (`reef_client.serve`) on the host and points
   hermes's model endpoint at it. Hermes cannot set the `x-reef-scenario`
   or `x-reef-tag-session` header itself, so the shim attaches the scenario
   and stable conversation tag, then forwards the request to Reef.
2. It mints a Reef scenario id at position 0 and writes it to
   `$REEF_EVAL_STATE_DIR`. One stream is one scenario, which is one chain of
   runtime load IDs in Reef.
3. It writes the hermes config on first use with context compression turned
   off.

The runtime flow is:

```text
reef-eval starts the task container and the judge sidecar for the next session
  -> the harness reads the student's message from the sidecar
  -> hermes runs one turn; its model calls go through the shim to Reef
  -> the SGLang backend records the sampled tokens, loss mask, log-probabilities, and top-K capture
  -> the harness posts hermes's reply to the sidecar, and the student reacts
  -> on the next model call, the processor uses the session tag to bind the preceding call to its following tool result or user reaction
  -> the PRM judges that next state and may propose a hindsight hint
  -> batch_size judged turns form one batch; the top-K select loss trains the policy
  -> Megatron performs one optimizer step and synchronizes weights to SGLang
  -> the next session runs on the updated weights
```

## Setup (once)

You need a GPU host with seven available GPUs, Docker, and `uv` (for `uvx`,
which runs reef-eval). In `docker-compose.yaml`, `device_ids` defines the
container's pool once: the default exposes physical GPUs 1-7, leaving GPU 0 free.
Ray assigns device IDs within that pool: the PRM and student model each reserve
one GPU through `services[].resources.num_gpus`; Slime reserves five more for
the Megatron actor (tensor parallel 4) and policy rollout engine (one GPU).
The inference services start before Slime allocates its group. The CPU-only
`slime-driver` does not reserve GPUs itself, and individual services do not
set CUDA visibility. This is a single-host stack, not a multi-node deployment.

There is no `ray-head` service or fixed Ray port to configure. The first GPU
service selects the Ray executor and Reef starts one shared local runtime.
Reef passes its actual address to Slime and subsequent services, and stops
the runtime after the services exit. Set `RAY_ADDRESS` only to connect to an
external cluster; Reef does not stop that cluster. An unavailable external
address is an error, not a reason to start a different local cluster.
When launching directly instead of through Compose, restrict the available
pool once at the deployment boundary, for example
`CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 reef serve -c recipes/openclawrl/examples/openclawrl/serve.yaml`.

```bash
docker build -f docker/Dockerfile.reef -t reef-openclawrl .
hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir ~/models/Qwen3-4B-Thinking-2507   # policy and PRM
hf download Qwen/Qwen3-32B --local-dir ~/models/Qwen3-32B                             # the student
pip install uv
```

Qwen3-4B-Thinking-2507 serves as both the policy and the PRM, as in the
reference run script. The PRM is the frozen base model on its own engine;
the policy is trained. Qwen3-32B plays the student.

Two more things to check on a host you did not set up yourself. reef-eval needs
Python 3.12 or newer; if `uvx` picks an older interpreter, set
`UV_PYTHON=3.12`. And the repository root must not contain a stale
`reef.egg-info`, because it is mounted into the container and shadows the
installed package's entry points, which stops the sglang plugin from loading.

## Run

```bash
OPENCLAWRL_USER_LLM_URL=http://<host-ip>:30001 \
OPENCLAWRL_USER_LLM_MODEL=qwen3-32b-user-llm \
bash recipes/openclawrl/examples/openclawrl/run.sh
```

This builds the sidecar image, boots the training stack in Docker (about six
minutes on B200s), waits for it to become healthy, then runs the 72-session
stream through reef-eval. The stack keeps running after the stream ends.
Re-running the same command resumes the stream where the lab left off and
reuses the healthy stack.

The paths and names are constants at the top of `run.sh`: the models under
`~/models`, checkpoints under `~/reef-run`, and the task list (all 72 GSM8K
tasks). A few settings come from the environment, because the task
environments and other processes read them:

- `OPENCLAWRL_USER_LLM_URL` and `OPENCLAWRL_USER_LLM_MODEL` name the student
  model the stack serves on port 30001. The task environments require both,
  and reef-eval rejects every task when either is unset. Use the host's
  address, not localhost, because the sidecar calls it from inside a
  container.
- `WANDB_API_KEY=...` is forwarded into the stack for live training curves
  (also set `observability.wandb.enabled: true` in `serve.yaml`) and starts
  `results/learning_curve.py`, which logs the per-session verdicts into the same
  W&B group.

A reef process trains one scenario for its lifetime. Pointing a different
stream name, or a fresh run directory, at a stack that has already trained is
rejected with "training is already bound to scenario ...". Stop the stack
with `docker compose down` in this directory before switching streams or
starting a variant.

### Reading a run

The reef-eval lab is at `~/reef-run/lab`. Each session's verdict is in
`trials/gsm8k-sNNN__*/verifier/final.json`, with the reward, the list of
violated rules, and the turn count. `results/learning_curve.py` turns those
verdicts into the stream's learning curve: whether the first reply was
accepted, which style markers the judge flagged (bold, bullets, numbered
lists), the cumulative accept count. While a stream runs with `WANDB_API_KEY`
set as mentioned above, `run.sh` runs the script in logging mode and each
session appears as one point in an `eval` run of the same W&B group as the
training run. After a stream, the same script exports those points to a CSV
and draws the figure the README keeps:

```bash
uv run --no-project --with matplotlib \
    recipes/openclawrl/examples/openclawrl/results/learning_curve.py \
    --lab ~/reef-run/lab --csv results/<run>/learning_curve.csv --plot results/<run>/learning_curve.png
```

## Results

### GSM8K homework stream

`results/2026-08-27-gsm8k-stream-qwen3-4b-thinking/` is a run result from the above.

| Setting | Value |
| --- | --- |
| Task | the 72-session GSM8K homework stream, hermes memory off |
| Model | `Qwen3-4B-Thinking-2507` as the policy and as the PRM |
| Student | the `Qwen3-32B` persona |
| Hardware | the seven-GPU layout of `serve.yaml`: a tensor-parallel-4 actor, one rollout engine, one PRM engine, one engine for the Qwen3-32B student |
| Batch | 16 judged turns per training step |
| Responses | up to 8192 tokens in a 65,536-token context |
| Objective | top-K select loss, PPO clip 0.2 / 0.28, `w_rl` 1.0, `w_opd` 1.0, top-4 capture, `sequence_optimal` hint selection |
| Sampling | temperature 0.6, top-p 0.95, top-k 20 |
| Optimizer | Adam, lr `1e-5` constant, weight decay 0.1, betas 0.9 / 0.98 |
| Checkpointing | one checkpoint per version, weights hot-swapped after each step |
| Tracking | per-session verdicts logged to W&B online; training metrics not tracked |

![Accumulated accepts and the rolling bold and list rates over the first 36 sessions of the stream](results/2026-08-27-gsm8k-stream-qwen3-4b-thinking/learning_curve.png)

A session passes when the agent's first solution reply already matches the student's preferences, with the work shown and the correct answer. The student wants homework that does not look AI-written, so a reply that uses bold text or a bullet or numbered list draws a complaint. The bold rate and the list rate measure this habit over the run, as the fraction of the last ten sessions whose first reply still contains bold text or a list. From the curves we see both fall as training goes on, and the run reaches the paper's adaptation criterion (three passed sessions in a row) at session 14.

![A failing and a passing session replayed from the run](results/2026-08-27-gsm8k-stream-qwen3-4b-thinking/demo.gif)

The demo above replays two sessions from this run. In session 1 the student
rejects a formatted reply, reef keeps the training going, and by session 16
the first reply passes directly.
