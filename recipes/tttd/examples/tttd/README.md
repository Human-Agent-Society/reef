# TTT-Discover on Reef

This example implements the harness side of
[Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175). Its
code is intentionally split so the Reef integration is visible and removable:

The [TTT-Discover guide](../../../../docs/user-guide/recipes/tttd.rst) explains the runtime
sequence, a reduced smoke, trajectory configuration, and recovery behavior. This
README records the example's implementation details, paper fidelity, and completed
reproduction results.

```text
harbor/               self-contained REEF Eval/Harbor task definitions
  erdos_min_overlap/
  circle_packing_26/
  circle_packing_32/
    task.toml            metadata, timeouts, resource limits
    instruction.md       exact task prompt shown to the model
    environment/         container, judge service, canonical verifier
    tests/test.sh        asks the judge for the final result, writes reward
harness/              agent harness (PUCT search + Reef adapter)
  search.py             Scorer callable + PUCT archive + search algorithm
  scorer.py             JudgeScorer (HTTP) + ProgramScorer (direct sandbox)
  sandbox.py             isolated subprocess execution for generated programs
  agent.py              ReefTTTDiscoverHarness rollout/report adapter
  run_controller.py     training barrier + paired PUCT resume state
  harbor_agent.py       Harbor BaseAgent (imports harbor package)
serve.yaml            Reef + Ray + Slime/Megatron + SGLang stack config
run.py                one REEF Eval episode owning the complete TTT trajectory
run.sh                starts the reef training stack, then runs run.py
pyproject.toml        makes the harness importable
results/              formal result data, generated programs, and plots
```

## The ordinary harness

`TTTDiscoverHarness` accepts the model call that an ordinary harness already
has. Evaluation stays local:

```python
from recipes.tttd.examples.tttd import TTTDiscoverHarness

harness = TTTDiscoverHarness(call_model, MyProblem(), "task instruction", model="my-model")
best = harness.run(steps=50)
```

Its single-rollout path is:

```python
response = self.generate(self._request_payload(parent))
return self._evaluate_response(parent, response)
```

There is no report hook, transport protocol, receipt wrapper, or other
Reef-shaped abstraction in the ordinary harness.

## The changes needed for Reef

Use the concrete Reef harness instead:

```python
from reef_client import ReefClient
from recipes.tttd.examples.tttd import ReefTTTDiscoverHarness

harness = ReefTTTDiscoverHarness(
    ReefClient("http://reef-host:8900", token="secret"),
    MyProblem(),
    "task instruction",
    scenario="my-single-discovery-problem",
    recipe="tttd",
    artifact_version="checkpoint-v42",  # optional starting checkpoint
    model="reef",
)
best = harness.run(steps=50)
```

`ReefTTTDiscoverHarness` reuses the request construction, evaluation, archive,
and PUCT mechanics. Its overridden `_rollout` method contains the four actual
integration changes in one place:

1. Send inference through a scenario-scoped Reef endpoint.
2. Retain the returned `agent_record_id` as the generation receipt.
3. After the existing evaluator computes a reward, send a Reef report that
   references that exact inference receipt.
4. Include the rollout group's `comparison_set` so the scenario processor can
   construct grouped training data.

The ordinary harness and problem do not import `ReefClient`, know the scenario
name, retain inference IDs, or construct Reef report payloads.

The runtime flow is:

```text
PUCT chooses G parents
  -> harness submits G × R OpenAI-compatible chat requests
  -> the TTTD inference backend renders each prompt once and calls SGLang /generate
  -> Reef stores exact tokens, response loss mask, and rollout log-probabilities
  -> sandbox executes each generated program; the official verifier gives rewards
  -> harness reports every reward against its exact inference receipt
  -> TTTDProcessor waits for the complete G × R step and removes constant groups
  -> Slime prepares adaptive-beta entropic leave-one-out advantages from the reserved batch
  -> Slime applies frozen-base token KL and un-clipped importance sampling
  -> Megatron performs one optimizer step and synchronizes weights to SGLang
  -> valid children update the local PUCT archive
  -> the Harbor controller waits for the durable scenario commit and new weight version
  -> the post-step PUCT archive is atomically paired with that committed version
```

## Included paper problems

`harbor/erdos_min_overlap/environment/score.py` adapts the
[official Erdős minimum-overlap environment](https://github.com/test-time-training/discover/tree/main/examples/erdos_min_overlap),
one of the paper's mathematics tasks. It keeps the same:

- step-function representation using samples over `[0, 2]`;
- constraints `0 ≤ h[i] ≤ 1` and `sum(h) = n/2`;
- full-correlation calculation of the upper bound `C₅`;
- continuous reward `1 / (1e-8 + C₅)`.

As in the reference implementation, the model writes a Python search program
whose `run()` function returns `(h_values, c5_bound, n_points)`. The program is
run in a subprocess with a wall-clock timeout, CPU/thread limits, network
access governed by the deployment, and an isolated temporary working
directory. The task prompt, seed construction, program contract, normalization
semantics, verifier, continuous reward, and PUCT search value are ported from
[`test-time-training/discover@6c40e82`](https://github.com/test-time-training/discover/tree/6c40e82dab9d5de7416ac873ad5cd3106084aaed).

Acknowledgement: this task is adapted from the MIT-licensed TTT-Discover
implementation by Mert Yuksekgonul and collaborators. Full attribution and
license terms are included below.

`harbor/circle_packing_26/` and `harbor/circle_packing_32/` adapt the two
circle-packing tasks from the same paper. In each task, the model writes
`run_packing()`, which returns circle centers, radii, and a reported sum. The
judge independently requires the selected number of circles, checks square
boundaries and pairwise non-overlap, rejects non-finite values, and computes
the reward directly as `sum(radii)`. The reported sum is never trusted. Both
task prompts preserve their corresponding initial TTT-Discover prompt byte for
byte.

## Setup (once)

```bash
git submodule update --init third_party/reef-client
pip install -e ./third_party/reef-client
pip install -e . "reef-eval[harbor]"
```

```bash
./run.sh
```

That runs `harbor/erdos_min_overlap` on the paper grid: 8 groups of 64
rollouts, one optimizer step, thinking enabled, two GPUs. Nothing has to be
exported first except `TTTD_TASK` — `run.py`, `harness/harbor_agent.py`, and
`serve.yaml` each write out the values they use.

### Another problem, or a smaller grid

`harbor/circle_packing_26` and `harbor/circle_packing_32` are bundled as well.
`TTTD_TASK` selects one:

```bash
TTTD_TASK=circle_packing_26 ./run.sh
```

`run.sh` gives each task its own scenario and `work/<task>/` state directory,
and sizes the memory limits for it — the packing tasks need a longer context
(`32768`) on the same GPUs, so they get a smaller per-GPU token budget.

The step grid lives in two places and both must agree, or Reef waits for
coordinates the harness never sends: `GROUPS_PER_STEP` and
`ROLLOUTS_PER_GROUP` in `harness/harbor_agent.py`, and `groups_per_step`,
`rollouts_per_group`, and `--global-batch-size` (their product) in
`serve.yaml`. A one-step plumbing smoke sets both sides to 2 x 2 and
`enable_thinking = False`, so a short completion is not spent entirely in the
reasoning channel before it emits a program.

`work/erdos_min_overlap/` holds this problem's checkpoints, artifacts,
scenario records, and PUCT state; a second problem needs its own directory so
the two cannot mix. `work/erdos_min_overlap/tttd-search-state.json` records a
pending archive immediately after a search step and marks it committed only
after Reef's durable training transaction advances the scenario and publishes
a new serving weight version.
Restarting with the same scenario, task instruction, and sampling settings
resumes that paired state; a missing or mismatched step fails closed instead
of silently restarting PUCT against a later model checkpoint. Serving weight
versions are session scoped, so a correctly restored step may rebind the saved
archive to the new engine version.

Use a fresh scenario for each discovery problem: TTT-Discover fine-tunes on one
test problem rather than learning a general task policy.

To add another problem, create a sibling under `harbor/`, write a `score.py`
with a `grade(artifact)` function, and point the values in the table above at
it. The harness scores every generated solution by POSTing it to the task's
judge, so no task-specific imports enter the shared harness.

## Paper fidelity and training ownership

The integration reproduces:

- 8 comparison groups per training step and 64 rollouts per group by default;
  both cardinalities are configurable and the reference setting uses 8 × 64;
- one PUCT-selected initial state shared by each group;
- rank-prior PUCT with maximum-child `Q`, ancestor visit backpropagation, and
  full-lineage diversity blocking;
- the top 2 children per expansion and a top-1000 archive that retains seeds;
- invalid-action reward 0 by default;
- the complete-step barrier and post-barrier removal of constant-reward groups;
- adaptive-beta entropic leave-one-out advantages, using the reference Torch
  float32 bisection procedure;
- frozen-base centered token KL in Slime;
- Tinker's full-batch, token-sum, un-clipped importance-sampling policy loss
  rather than PPO clipping or per-trajectory token averaging;
- Tinker's sampling defaults (`temperature=1`, `top_p=1`, `top_k=-1`)
  explicitly, rather than SGLang's model-specific defaults;
- the reference Adam settings (`lr=4e-5`, betas `0.9/0.95`, `eps=1e-8`,
  zero weight decay, and no gradient clipping);
- raw continuous rewards linked to exact inference tensors rather than
  reconstructed from ordinary HTTP chat JSON;
- Megatron Bridge LoRA with the paper configuration (`rank=32`, `alpha=32`)
  on QKV, attention output, and both MLP projections. The base parameters remain
  frozen in both runtimes. Reef synchronizes the serving-native `lora_A` and
  `lora_B` adapter tensors to SGLang.

The default HTTP backend cannot manufacture training tensors from response
text. The deployment therefore selects Reef's token-native SGLang chat
backend. The public harness route remains `/v1/chat/completions`; inside the
inference backend, Reef renders the prompt once, calls SGLang `/generate`, and
records the engine's sampled token IDs and rollout log-probabilities. Reef's
weight surface names the scenario's own adapter revision on every request;
the harness never selects one and the backend never injects a fallback.

New Slime configs should use the
`recipes.tttd.slime.objective.*` custom-function paths.
The former `reef_adapters.tttd.*` paths
were removed; existing deployment configs must migrate to the backend-owned
module path.

## Earlier Erdős 50-step experiment

We ran a result-level reproduction of the Qwen3-8B Erdős experiment from
[Learning to Discover at Test Time](https://test-time-training.github.io/discover.pdf).
The run used 50 optimizer steps, Qwen3-8B thinking, the two-phase completion
policy, a 26,000-token phase-one budget, the grouped entropic objective,
rank-32 LoRA, and the paper's Adam learning rate. It used eight groups of 32
rollouts on one four-B200 node. The paper's experiment used 64 rollouts per
group, so this run used half of its rollout budget.

### Setup

| Setting | Value |
| --- | --- |
| Task | Erdős minimum-overlap program discovery |
| Model | `Qwen/Qwen3-8B`, thinking enabled |
| Runtime | Reef, Slime, Megatron, and SGLang LoRA serving |
| Hardware | `4 × NVIDIA B200`, using TP2 × DP2 |
| Search grid | 8 groups × 32 rollouts |
| Training | 50 steps, LoRA rank and alpha 32, Adam learning rate `4e-5` |
| Sequence policy | 30,000-token window, 26,000-token phase-one budget, and two-phase completion |
| Evaluation | 1,000-second program budget and 1,100-second timeout |
| Checkpointing | One checkpoint per step, retaining the latest checkpoint |

### Results

| Completed steps | Runtime | Best certified C₅ upper bound | Best reward |
| ---: | ---: | ---: | ---: |
| 50/50 | approximately 22.6 hours | `0.380916` | `2.625249` |

Table 2 of the paper reports `0.380932` for TTT-Discover with Qwen3-8B on the
same Erdős task. The verifier minimizes this bound. Search outcomes depend on
sampling, so the two numbers describe separate trajectories with different
rollout counts.

## Formal circle-packing results

The Packing 26 and Packing 32 runs each completed 50 steps with the bundled
harness and task judges. They used `Qwen/Qwen3-8B` with thinking enabled on two
NVIDIA B200 GPUs. Every step contained eight groups of 64 rollouts, and Reef
trained a rank-32 LoRA adapter after receiving the complete grid.

| Task | Completed | Certified sum of radii | TTT-Discover reported | Target |
| --- | ---: | ---: | ---: | ---: |
| Packing 26 | 50/50 | `2.6359830849177777` | `2.635983` | `2.636` |
| Packing 32 | 50/50 | `2.9395727712074926` | `2.939572` | `2.940` |

These results match the published TTT-Discover values at six decimal places.
The targets come from the task instructions and do not establish global
optima.

![Verified circle-packing configurations](results/formal-8x64-v3-packing/packing_configurations.png)

W&B recorded 50 committed rows for each task. The mean rollout reward reached
its maximum at step 18 for Packing 26 and step 17 for Packing 32. Both search
archives contained their reported six-decimal result by step 20. Sampled-policy
KL increased later in both trajectories. The plot reports grid-level training
metrics, so it should be read separately from the best certified program stored
in each search archive.

New runs also record the full grid's minimum, maximum, mean, population standard
deviation, zero-reward fraction, and constant-group filtering counts under the
`tttd` W&B namespace. Each `tttd_step_committed` harness event includes the
archive size and its best reward. These fields were added after the two formal
runs, so they are not present in the stored history below.

![W&B metrics from the two formal runs](results/formal-8x64-v3-packing/wandb_training_metrics.png)

The [result directory](results/formal-8x64-v3-packing/README.md) contains the
numeric W&B history, generated programs, certified milestone summaries,
coordinates, provenance hashes, and plotting scripts. Each task has one formal
trajectory, so the stored results do not estimate variance across seeds.

## Attribution and license

Parts of this example are adapted from
[`test-time-training/discover@6c40e82`](https://github.com/test-time-training/discover/tree/6c40e82dab9d5de7416ac873ad5cd3106084aaed),
including the TTT-Discover search procedure, adaptive-entropic objective,
two-phase completion behavior, Erdős minimum-overlap task, and circle-packing
tasks for `n=26` and `n=32`.

<details>
<summary>Upstream MIT license</summary>

> MIT License
>
> Copyright (c) 2025 Mert Yuksekgonul
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

</details>
