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
REEF Eval's environment.

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
REEF Eval starts one Harbor trial for the next problem
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

`run.py` is the loop, written out. For each task in order, REEF Eval's `Lab.run`
executes one episode: the agent runs its six rollouts, reporting each one as
it is scored, then Harbor's verifier scores the last completion. The ordering
is the experiment: task `N+1` is served by the weights task `N` produced.

Nothing has to be exported. The service URL, token, scenario, rollout count,
and generation window are constants at the top of `harness/agent.py`, and
the port and token they use are the ones written in `serve.yaml`. The task
list is `TASKS` in `run.py`.

### Reading the version chain and the metrics

Each scored rollout publishes one `training` entry to the scenario's version
chain:

```bash
curl -s -H "Authorization: Bearer reef-local" \
    http://127.0.0.1:8900/reef/scenarios/sao-smoke/versions
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
  (`--critic-steps-per-actor=2`);
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

### Qwen3-30B-A3B on IMOAnswerBench, 48 scored rollouts per arm

Two arms were trained on the same three problems, from the same checkpoint,
at the same rollout budget, and only the objective differed: one arm ran the
`sao` recipe, the other a GRPO control with the same DIS mask
(`--advantage-estimator=grpo`, groups of four). The runs were made in August 2026 with an earlier version of
this harness, on the tree before the `recipes/` layout; the drivers and the
recipe have been ported since, and every number below was read from the run
records and the version endpoints captured at the time. The per-rollout
records are in the repository history
(`examples/sao/bench/deployments/qwen3_30b_a3b_main/results/` at commit
`375e0036`); the figure below is plotted from them.

This is a budget-limited comparison. Every scored rollout became a report
against the training bridge (SAO commits once per rollout, GRPO once per
filled group), but 48 rollouts per arm is far short of a training run, so the
table answers whether the paper's ordering holds at this budget, not where
the methods converge.

| Setting | Value |
| --- | --- |
| Task | IMOAnswerBench `problem_idx` 4, 8, 12; one rollout per task in rotation (SAO), one group of four per task (GRPO) |
| Model | `Qwen3-30B-A3B-Thinking-2507`, converted to Megatron `torch_dist` |
| Actor, with the SAO critic | one 8-GPU node, TP4 / PP2 / EP4, sequence parallel, full recompute, CPU-offloaded precision-aware Adam |
| Rollout | a second 8-GPU node, SGLang TP8, temperature 1.0, top-p 1.0 |
| Generation window | 61,440 tokens in a 65,536-token sequence |
| Objective | DIS 0.3 / 5.0, two critic steps per actor step, length-adaptive λ with α 1.5, policy lr `1e-6`, value lr `5e-6` |
| Budget | 48 scored rollouts per arm; batch 1 (SAO), group 4 (GRPO) |
| Reward | strict `\boxed{}` equivalence against the gold answer, no LLM judge |

| Arm | Scored rollouts | Training commits | Mean reward | imo-4 | imo-8 | imo-12 | Wall-clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base (untrained; 16 runs per problem, same grader and window) | 48 | 0 | `0.458` | 15/16 | 3/16 | 4/16 | n/a |
| SAO | 48 | 47 | `0.479` | 15/16 | 4/16 | 4/16 | 22,623 s |
| GRPO(+DIS) | 48 | 12 | `0.417` | 16/16 | 2/16 | 2/16 | 8,055 s |

The ordering matches the paper at this budget: SAO (0.479) above the base
rate (0.458) above GRPO (0.417). An earlier summary of these runs listed
GRPO's imo-12 count as 3/16; the run records give 2/16, which is the count
the mean of 0.417 (20 of 48) corresponds to. SAO's gain is on imo-8, with imo-12 at the
base rate and imo-4 close to saturated. GRPO(+DIS) ends below the base rate:
12 filled groups over 48 rollouts, and most of those groups uniform (all four
correct or all four wrong), which leaves no gradient. The paper's per-step
batch is 128 over a full training run.

![Cumulative mean reward over the 48 scored rollouts per arm](results/2026-08-15-imo-answerbench-qwen3-30b-a3b/learning_curve.png)

The curve is the cumulative mean reward against scored rollouts, per arm,
with the base rate as the dashed line. The runs predate Reef's experiment tracking, so no per-step W&B
history exists for them. A new run with `observability.wandb.enabled: true`
records the step metrics that the TTT-Discover and OpenClaw-RL results keep
(mean reward, response length, KL, step time).

Wall-clock includes a per-step serving-weight export whose cost is the same
at every step, so it penalizes SAO's four times higher step count; compare
step times on the reward-versus-rollout records rather than on wall-clock.

Deviations from the paper's protocol:

- **No TIR SFT init.** The paper's math arm starts from an unpublished SFT on
  GPT-OSS-120B tool-integrated-reasoning data; these runs start from the
  public Thinking checkpoint. The paper's absolute number (74.0 accuracy
  after full training) is not reachable from this init at any budget.
- **Generation window 61,440 tokens** (paper: 128k). Some rollouts truncate at
  the cap; the cap is the same for both arms.
- **Budget 48 rollouts per arm, batch 1 (SAO) / group 4 (GRPO)**; the paper
  trains with batch 128 over a full run.
- **Trained on the benchmark's own problems**; the paper trains on a separate
  math corpus and evaluates on the benchmark.

For a base number comparable with the paper's Table 1, the untrained model
was also run on the full 400-problem IMOAnswerBench, four runs per problem,
temperature 1.0, top-p 1.0, in a 65,536-token window:

| Grader | Reef base | Paper base (without Python) |
| --- | ---: | ---: |
| Strict answer equivalence | `44.69` | 55.3 |

The paper does not specify its grading protocol for IMOAnswerBench. The
strict rule is a lower bound, an LLM-judge upgrade path measured earlier is
an upper bound at 63.44, and the paper's 55.3 falls between the two. Compare
arms against the Reef base column, not against the paper's number.

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
