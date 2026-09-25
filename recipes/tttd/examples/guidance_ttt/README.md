# Guidance-TTT on four discovery tasks

Guidance-TTT trains a guidance model while a frozen executor writes candidate
programs. The task judge supplies the reward. Only the guidance inference
receipt reaches Reef's grouped TTTD training recipe.

The [user guide](../../../../docs/user-guide/recipes/guidance-ttt.rst) explains
the loop, configuration, four paper results, search curves, and resume behavior.

| `GUIDANCE_TASK` | Candidate | Evaluator | Archive ranking |
|---|---|---|---|
| `polyomino_packing` (default) | C++17 | FrontierCS, 70 cases | Maximum packing score |
| `lasso_path` | Python wrapper with C++17 `CPP_CODE` | SimpleTES, 17 cases | Maximum inverse geometric-mean solve time |
| `ahc058` | C++20 | Official tester, 150 public cases | Maximum mean raw score |
| `trimul` | Python/Triton | H100, 18 correctness tests and seven timing cases | Minimum latency |

## Run

Use Linux, Docker, and the supported Reef GPU environment. From the repository root:

```bash
git submodule update --init third_party/reef-client
python -m pip install -e ./third_party/reef-client
python -m pip install -e . -e recipes/tttd/examples/guidance_ttt
cd recipes/tttd/examples/guidance_ttt
export GUIDANCE_TASK=lasso_path
python prepare.py "$GUIDANCE_TASK"
```

Start the selected [judge](judges/README.md), then check its bootstrap:

```bash
python smoke.py
```

Choose the frozen executor and start the training stack:

```bash
# Local OpenAI-compatible GPT-OSS-120B server:
export GUIDANCE_EXECUTOR=local
export GUIDANCE_EXECUTOR_URL=http://127.0.0.1:8000/v1
./run.sh
```

For GLM-5.2, set `GUIDANCE_EXECUTOR=openrouter` and provide
`OPENROUTER_API_KEY` through your environment. The script reads the key only
for executor requests. Do not put its value in configuration or source files.

Set `GUIDANCE_EXECUTOR_MAX_TOKENS=16384` to bound each executor response during a smoke run.
Otherwise, the provider selects its default output limit. Executor service errors stop the step; they are not candidate rewards.

`serve.yaml` defaults to one update with two groups of four rollouts.
Set the grid to 8 by 16, `global-batch-size` to 128, and `steps` to 30 for
the paper's search budget. Set `GUIDANCE_MODEL=Qwen/Qwen3-14B` for the paper's
AHC058 and TriMul configurations. Size the GPU allocation for that model.

`GUIDANCE_CONFIG` selects an alternative YAML file. `GUIDANCE_STATE_DIR`
selects an isolated state root. `GUIDANCE_MODEL_PATH` selects existing weights.
The launcher passes absolute paths to the runtime. The host judge URL is
`GUIDANCE_JUDGE_URL`; the Docker verifier uses `GUIDANCE_JUDGE_CONTAINER_URL`.
See the user guide for defaults and the other configuration values.

## What the harness guarantees

1. The guidance model reads the task, parent summary, and verifier score.
2. The frozen executor reads the parent program and new guidance.
3. Each score references the exact guidance receipt and step-grid coordinate.
4. Invalid programs do not enter the executable archive. AHC partial rewards remain trainable.
5. Raw metrics determine archive ranking. Training rewards can use a different scale.
6. Judge infrastructure failures stop the step instead of generating synthetic rewards.
7. A committed archive is restored only with its matching training identity and checkpoint.

The harness makes one guidance attempt per rollout. Malformed guidance receives
zero reward without format repair. The executor returns a full `<solution>` and
a canonical `<summary>`. Search uses rank-prior PUCT, `best_child` Q, ancestor
visit updates, two children per expansion, and a top-1000 archive.

The example uses Reef's existing TTTD recipe, adaptive entropic advantages,
frozen-base KL, and rank-32 LoRA publication. The final Harbor verifier's reward
is evaluation-only, because every training rollout already reports its own reward.

## Results

The [paper records](results/paper/README.md) contain four main results:
91.89 Polyomino, 0.1739 Lasso, 850,082,731 AHC058, and 1,129 microseconds TriMul.
They include the models, evaluation protocols, all displayed baseline rows,
and source checksums. These are historical paper results, not new runs of this port.

The older [Reef run records](results/README.md) remain separate. Their Polyomino
14B and TriMul measurements describe earlier configurations and evaluations.

## Local checks

From the repository root:

```bash
python -m pytest tests/test_guidance_ttt.py tests/test_guidance_ttt_tasks.py -q -o addopts=
cd docs/site
npm ci
npm run check:docs
npm run lint
npm run build
```

CPU tests check task selection, reward semantics, archive direction, receipt
linkage, invalid candidates, infrastructure errors, and configuration consistency.
A real bootstrap check requires its external evaluator. A complete training test
also requires the frozen executor and a supported GPU stack.

## Files

- `harbor/<task>/`: instruction, task contract, container, and final verifier.
- `harness/`: guidance search, executor client, judge adapter, archive, and controller.
- `prepare.py`: pinned external benchmark and bootstrap downloads.
- `judges/`: official evaluator adapters and setup instructions.
- `run.sh`, `run.py`, `serve.yaml`: one complete Reef and Harbor trajectory.
- `results/`: historical measurements and paper result records.

## Source licenses

Reef's harness and adapters use the repository's Apache-2.0 license.
The [benchmark notices](harbor/NOTICE.md) identify external task sources.
Preparation keeps external code and its original license under ignored `work/` paths.
