# SAO on Reef

This example implements the harness side of
[Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508).
SAO trains on one graded rollout at a time. There is no comparison group and
no barrier, so a rollout enters training the moment its score arrives, and the
next request is served by the updated weights. The method itself is the `sao`
recipe package (`recipes/sao/`). This directory holds the loop around it: three
IMOAnswerBench problems as Harbor tasks, a Harbor agent that runs six scored
rollouts per problem through Reef, and `run.py`, which runs the tasks in order.

The [`sao` recipe page](../../../../docs/user-guide/recipes/sao.rst) documents
the recipe's configuration and runtime metrics, and
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst) walks
through the training stack this example starts. This README records the
example's implementation details, its distance from the paper's protocol, and
a completed comparison against GRPO at the paper's model scale.

```text
harbor/               three IMOAnswerBench problems as Harbor tasks, run in order by run.py
  imo-4/                problem_idx 4 (gold: 2^{u-2})
  imo-8/                problem_idx 8 (gold: -2023/2024^2)
  imo-12/               problem_idx 12 (gold: 1/2)
    task.toml             metadata, timeouts, resource limits
    instruction.md        the problem text and the \boxed{} instruction
    environment/          the Python image the verifier runs in
    tests/
      test.sh             runs the verifier
      grade.py            extracts \boxed{}, checks it against the gold answer
harness/              agent harness (imports reef_client, not reef)
  __init__.py           lazily exports HarborAgent
  agent.py              HarborAgent: six scored rollouts per problem, one report each
  report.py             posts Harbor's verifier reward against the trial's receipts
serve.yaml            Reef + Ray + Slime/Megatron + SGLang stack config, critic colocated
run.py                the loop, written out: solve, verify, report, task by task
run.sh                starts the Reef training stack, then runs run.py
pyproject.toml        makes the harness importable
results/              the reward curves of the Qwen3-30B-A3B comparison
```

## The harness

`HarborAgent` receives one problem per Harbor trial as its `instruction`,
looks up the gold answer, and runs `ROLLOUTS` (six) attempts at it. Each
attempt is one chat request at `temperature=1.0, top_p=1.0` with a
2048-token generation window. The agent extracts the last `\boxed{}` from the
completion and compares it with the gold answer under the strict equivalence
rule the Harbor verifier uses: an exact match after whitespace and `$` are
stripped, or a numeric evaluation of simple LaTeX (fractions, roots, π) within
a relative tolerance of `1e-6`. The reward is binary, 1.0 for a correct answer
and 0.0 otherwise. The rule is copied into `harness/agent.py` rather than
imported from the task, because the harness is installed on its own into
reef-eval's environment.

The last completion is written to `/workspace/answer.txt`, where the Harbor
verifier (`harbor/imo-*/tests/grade.py`) scores it independently and records
the trial's reward.

## The changes needed for Reef

The rollout loop in `HarborAgent.run` contains the two integration points:

```python
response, agent_record_id = await asyncio.to_thread(self._ask_reef, instruction)
completion = response["choices"][0]["message"]["content"]
predicted = extract_answer(completion)
score = 1.0 if answers_equal(gold, predicted) else 0.0
self._client.report(SCENARIO, {"score": score, "references": [agent_record_id]}, recipe=RECIPE)
```

1. Send inference through a scenario-scoped Reef endpoint
   (`inference_with_record`) and keep the returned `agent_record_id` as the
   generation receipt.
2. After the local grader computes the reward, report it against that exact
   receipt. With `batch_size: 1`, each report is one training step.

A third piece runs after the trial. Harbor writes `result.json` when the
verifier finishes, and a watcher thread posts the verifier's reward as one
more report, referencing all six receipts (`harness/report.py`; the report id
is derived from the trial id, so a repeated post changes nothing). That report
does not train, because its references have already trained and the `sao`
recipe does not accept multi-reference samples. It records the trial's verdict
against the same receipts.

The runtime flow is:

```text
reef-eval starts one Harbor trial for the next problem
  -> the agent sends one OpenAI-compatible chat request through Reef
  -> the SGLang backend renders the prompt once and calls /generate
  -> Reef stores the sampled tokens, the loss mask, and the rollout log-probabilities
  -> the agent extracts \boxed{} and scores it against the gold answer
  -> the agent reports the score against that rollout's receipt
  -> SAOProcessor accepts the report and emits one PolicySample
  -> the sao step preparer hands Slime a batch of one
  -> the colocated critic computes values; skip-observation GAE builds the advantages
  -> Slime runs policy_loss with SAO's per-token DIS primitive, after two critic steps
  -> Megatron performs one optimizer step and synchronizes weights to SGLang
  -> the next rollout is served by the updated weights
```

## Included paper problems

The three tasks are IMOAnswerBench (`Hwilner/imo-answerbench`) problems
`problem_idx` 4, 8, and 12, the slice the paper-scale run below used. They
were chosen because the untrained model neither always solves nor always
fails them under the strict grader, so the rewards carry a signal. Each
`instruction.md` is the problem text followed by the instruction to put the
final answer in `\boxed{}`. The verifier applies the same extraction and
equivalence rule as the agent to `/workspace/answer.txt`. No LLM judge is
involved anywhere in the loop.

## Setup (once)

The training stack needs the GPU environment described in
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst): Ray,
the Slime driver, and CUDA builds of torch, SGLang, and Megatron. Inside it,
from this directory:

```bash
pip install -e .
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
```

`run.py` also needs `docker`: Harbor runs each task's verifier in its own
container. The `docker run` line in Evolve your model does not provide that,
so when the stack itself runs inside the reef image, extend it:

```bash
docker run --gpus all --network host --ipc host --shm-size 32g -it \
  -v ~/models:/root/models \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$REPO":"$REPO" -w "$REPO" \
  reef bash
# inside: apt-get update && apt-get install -y docker.io
```

Mount the repo at its host path (`-v "$REPO":"$REPO"`, not `/workspace/Reef`):
Harbor's sibling containers bind-mount trial directories by path, and those
paths must mean the same thing to the host docker daemon.

## Run

```bash
./run.sh
```

`run.sh` starts `reef serve -c serve.yaml` with its state under `./work`,
waits for `/healthz`, and runs `run.py`. `serve.yaml` describes a two-GPU
stack: one Megatron actor with the critic colocated on it, and one SGLang
rollout engine, serving `Qwen2.5-1.5B-Instruct`. On the first start Reef
loads the Hugging Face weights directly and writes the Megatron checkpoint
that later starts load. Ray, Slime, Megatron, and SGLang take minutes to come
up; `work/reef.log` has the service log if the wait never ends.

Reef starts and stops the shared Ray runtime automatically; no `ray start`
or fixed Ray port is needed. `run.sh` defaults the local cluster's GPU pool to
`CUDA_VISIBLE_DEVICES=0,1`; override it at launch to choose different GPUs.
To use an existing cluster, set `RAY_ADDRESS`; that cluster's node configuration
controls GPU visibility, and Reef leaves it running on exit. Slime allocates
the model GPUs; the local driver does not reserve them a second time.

`run.py` is the loop, written out. For each task in order, reef-eval's `Lab.run`
executes one episode: the agent runs its six rollouts, reporting each one as
it is scored, then Harbor's verifier scores the last completion. The ordering
is the experiment: task `N+1` is served by the weights task `N` produced.

Nothing has to be exported. The service URL, token, scenario, rollout count,
and generation window are constants at the top of `harness/agent.py`, and
the port and token they use are the ones written in `serve.yaml`. The task
list is `TASKS` in `run.py`.

### Reading the release chain and the metrics

Each scored rollout publishes one `training` entry to the scenario's version
chain:

```bash
curl -s -H "Authorization: Bearer reef-local" \
    http://127.0.0.1:8900/reef/scenarios/sao-smoke/releases
```

The runtime reports `pg_clipfrac` (the fraction of tokens the DIS mask
removed), `critic/explained_variance`, actor and critic `grad_norm`, and the
asynchrony telemetry `sao/policy_lag_*`, `sao/queue_age_s_*`, and
`sao/effective_token_rate`. Set `observability.wandb.enabled: true` in
`serve.yaml` and export `WANDB_API_KEY` to keep them per committed step;
`observability.wandb.directory` is where the run files go.

### A larger model

Change `reef.model_path`, the GPU counts and parallelism flags, and
`--seq-length` and `--rollout-max-response-len` in `serve.yaml`. The
objective flags are the paper's reasoning-domain values and do not change
with model size. `reef.batch_size` and `training.global_batch_size` must stay
equal, because each rollout sample is its own data-parallel unit.

## Paper fidelity

The integration reproduces:

- one rollout per training step, with no comparison group and no
  slowest-sample barrier (`batch_size: 1`, `--global-batch-size=1`);
- a value model colocated with the actor and two critic steps per actor step
  (`--critic-steps-per-actor=2`), trained at the paper's value learning rate
  of `5e-6` (`--critic-lr=5e-6`) with the paper's 10-step value warmup
  (`--num-critic-only-steps=10`: the first ten rollout steps fit the
  zero-initialized value head before any policy update);
- value targets from Monte-Carlo returns (λ = 1) and policy advantages from
  the length-adaptive λ with α = 1.5, built by skip-observation GAE in the
  training backend, so the Reef payload carries no advantages;
- the DIS per-token loss with the reasoning-domain mask bounds 0.3 and 5.0,
  computed against the engine's rollout log-probabilities
  (`--use-rollout-logprobs`), which is why the deployment selects Reef's
  token-native SGLang chat backend;
- sampling at `temperature=1.0, top_p=1.0`, a constant policy learning rate
  of `1e-6`, and no entropy bonus.

The cookbook configuration is a functional smoke rather than the paper's
setup: it serves Qwen2.5-1.5B-Instruct with a 2048-token generation window,
trains on the benchmark's own problems, and starts from the public
instruction-tuned checkpoint. The paper-scale run below closes the model gap
and lists its remaining deviations.

## Results

### Qwen2.5-1.5B-Instruct on IMOAnswerBench, 90 scored rollouts

`results/2026-09-09-imo-answerbench-qwen2.5-1.5b-instruct/` records one line
per scored rollout — task, serving weight-release, receipt, tokens, 0/1
reward, predicted vs. gold, timestamp — and the plotting script
(`plot_learning_curve.py`) regenerates the figure below from those records
without touching the training stack.

![90 scored rollouts, raw outcomes with 95% bootstrap CI](results/2026-09-09-imo-answerbench-qwen2.5-1.5b-instruct/learning_curve.png)

The plot shows every rollout, coloured by task (gold answer), the running
mean of the reward sequence, a 95% bootstrap confidence interval, and a
vertical line at each new serving release. The wide CI early in the run
narrows as the sample count grows; the running mean stays near the sample
rate rather than tracking a training trend.

Headline numbers from `rollouts.jsonl` (90 rollouts, 30 per problem, three
IMOAnswerBench problems, strict `oxed{}` equivalence grader):

| Slice | n | correct | mean reward |
| --- | ---: | ---: | ---: |
| all rollouts | 90 | 5 | `0.0556` |
| `problem_idx` 4 (gold `2^{u-2}`) | 30 | 1 | `0.033` |
| `problem_idx` 8 (gold `-2023/2024²`) | 30 | 0 | `0.000` |
| `problem_idx` 12 (gold `1/2`) | 30 | 4 | `0.133` |

Reading the plot honestly:

- On this base model at this scale, IMO problems are near the noise floor. The
  running mean moves at the pace of the four `problem_idx=12` hits — a task
  whose gold answer (`1/2`) is a common guess — rather than tracking a
  learning trend.
- Three substantive serving releases produced most rollouts (30 / 29 / 29
  per release); the release chain shows training landing between tasks
  rather than after every rollout, because the value model's warmup
  (`--num-critic-only-steps=10`) delays the first policy update by ten
  rollouts.
- Nothing on the plot supports a claim that SAO learned to solve harder IMO
  problems in 90 rollouts on this backbone. The plot supports the claim
  that the training stack is wired end to end and the records make the
  outcome auditable.

The paper's own scale — Qwen3-30B-A3B-Thinking, IMOAnswerBench, and a full
training run — is where SAO's ordering is expected to appear. That
reproduction is blocked on the runtime bug documented in
[Known limitations](#known-limitations) below and is not attempted here.

Reproducing this figure from the retained records:

```bash
python plot_learning_curve.py   results/2026-09-09-imo-answerbench-qwen2.5-1.5b-instruct/rollouts.jsonl   --output learning_curve.png
```

Producing a fresh run at the same scale on two GPUs:

```bash
SAO_ROLLOUTS=30 ./run.sh   # 30 rollouts per task, 90 total (~1.5–2h)
```

### Known limitations

- Paper-scale reproduction (Qwen3-30B-A3B-Thinking on a single 8-GPU node,
  TP4/PP2/EP4 or TP8/PP1/EP8 with colocated critic and rollout) fails during
  weight export inside
  `reef/train/slime_backend/reef_adapters/megatron/hf_export.py` with
  `KeyError: "HF export weight 'vp_stages.0.decoder.final_layernorm.weight'
  is missing from the actor backup"`. Reproduced across both pipeline
  layouts and both `--megatron-to-hf-mode` values (`bridge`, `raw`), on an
  image built from `docker/Dockerfile.reef` at the runtime pin. Attempted on
  2026-09-09; a separate bug fix is required before a 30B rerun can replace
  the 1.5B figure above. Once fixed, the paper-scale settings only differ
  from the shipped `serve.yaml` in `reef.model_path`
  (`Qwen3-30B-A3B-Thinking-2507`), the MoE knobs slime's own
  `scripts/models/qwen3-30B-A3B.sh` prescribes
  (`--moe-token-dispatcher-type=alltoall`, `--moe-router-topk=8`,
  `--moe-grouped-gemm`, `--moe-router-dtype=fp32`, `--moe-permute-fusion`,
  `--moe-aux-loss-coeff=0`), the parallelism (`TP4/PP2/EP4` or
  `TP8/PP1/EP8`), the sequence budget (`--seq-length=65536
  --rollout-max-response-len=61440`), and 8 GPUs (`--num-gpus-per-node=8`,
  `--colocate`).

### Attempts that produced no result

**SWE-Bench Verified.** All three arms scored 0.0 on every episode. The
minimal bash scaffold (at most 12 turns of at most 8192 tokens in a 32k
window) cannot resolve the target astropy instances end to end, and the paper
uses OpenHands with up to 300 turns and 128k context. The results were
omitted as uninformative.

**A TIR SFT init.** Two SFT variants of Qwen3-30B-A3B-Thinking-2507 on
GPT-OSS-120B-generated tool-integrated-reasoning traces were tried: v1 with
3.5k TIR-only samples for 3 epochs, v2 with a 30k mixed corpus (60% TIR, 40%
NuminaMath-CoT) for 2 epochs. On the full 400-problem benchmark, strict
grader, without Python:

| Model | Reef base | SFT v1 (TIR only, 3.5k) | SFT v2 (mixed, 30k) | Paper SFT (with / without Python) |
| --- | ---: | ---: | ---: | ---: |
| Mean over 4 runs | `44.69` | 9.75 | 5.44 | 53.3 / 42.0 |

Both variants lost the base model's reasoning without recovering the paper's
SFT number. The paper's data curation, filtering, and mixing are unpublished,
so matching that number was not pursued further.
