# Task evaluators

Run these commands from `recipes/tttd/examples/guidance_ttt/`.
Start one judge per port. Set `GUIDANCE_JUDGE_URL` for the host harness and
`GUIDANCE_JUDGE_CONTAINER_URL` for Harbor's Docker container.
The latter defaults to `http://host.docker.internal:8082` for the three new tasks.
The judge must listen on an interface reachable through that host gateway.

The services accept multipart `POST /submit` with `pid`, `lang`, and `code`.
They return a submission ID in `sid`. `GET /result/<sid>` returns the result.
`GET /health` checks the HTTP service. Bootstrap verification checks the evaluator itself.

| Field | Meaning |
|---|---|
| `status` | `done`, `error` for a rejected candidate, or `environment_error` |
| `valid` | All required correctness checks passed |
| `score` | Finite, nonnegative training reward |
| `scoreUnbounded` | Raw task metric used to rank valid archive entries |
| `trainingRewardOnInvalid` | Whether an invalid candidate retains a partial training reward |

An infrastructure error stops the training step. It never becomes a candidate reward.
Run candidate evaluators on dedicated workers without model-provider credentials.
The Lasso adapter uses a fresh network-disabled Docker container for each candidate.
AHC delegates execution limits to ALE-Bench. TriMul uses a separate H100 worker.

## Polyomino Packing

Use the pinned FrontierCS judge:

```bash
git clone https://github.com/FrontierCS/Frontier-CS.git work/Frontier-CS
git -C work/Frontier-CS checkout 6d597dfb60be9e592881aef051b94e30d197c436
docker compose -f work/Frontier-CS/algorithmic/docker-compose.yml up -d --build
```

The default URL is `http://127.0.0.1:8081`. Problem `0` uses the 70-case suite.
The harness re-evaluates the bundled seed before a new search.

## Lasso Path

Fetch the pinned source and build its evaluator container:

```bash
python prepare.py lasso_path
docker build -f judges/lasso.Dockerfile -t reef-guidance-lasso judges
python judges/lasso_server.py --host 0.0.0.0 --port 8082
```

The upstream SimpleTES checkout is pinned to
`47d3413da1d85dc24341219d47452d2601e56a57`.
Its evaluator owns the 17 cases, float64 requirement, `1e-6` objective tolerance,
and inverse geometric-mean timing score. The candidate is a Python wrapper
with `CPP_CODE`, not a standalone C++ file. The adapter imposes a 600-second
wall-clock limit, four CPU cores, and 16 GiB of memory per candidate.
Hardware and thread settings affect timing. These defaults do not guarantee
that a new run reproduces the paper's absolute timing.

SimpleTES is AGPL-3.0 licensed. Preparation downloads its source and license
into the ignored `work/` directory. Reef does not vendor its solver or evaluator.

## AHC058

Install the pinned ALE-Bench dependency in a separate evaluator environment:

```bash
python prepare.py ahc058
python -m venv work/ale-venv
work/ale-venv/bin/pip install -r judges/requirements-ale-bench.txt
docker pull yimjk/ale-bench:cpp20-202301
docker tag yimjk/ale-bench:cpp20-202301 ale-bench:cpp20-202301
work/ale-venv/bin/python judges/ale_bench_server.py \
  --task ahc058 --host 0.0.0.0 --port 8082 \
  --candidate-workers 1 --case-workers 8
```

The adapter checks the SHA-256 hashes of the released 150 inputs and tester.
Each case retains its two-second and 1 GiB limits.
Training reward is the public mean raw score divided by 3,000,000.
Failed cases can produce a partial reward. Only candidates accepted on every
case enter the archive. This public-suite score is distinct from an AtCoder
submission score.

## TriMul

Prepare the MIT-licensed TTT-Discover evaluator and bootstrap:

```bash
python prepare.py trimul
python -m pip install 'modal>=1.0,<2'
modal setup
modal deploy judges/trimul_modal.py
python judges/trimul_server.py --modal-app reef-guidance-trimul --port 8082
```

The Modal function uses one H100, Torch 2.7.1, and Triton 3.3.1. It loads
18 correctness tests and seven benchmark cases from the pinned TTT-Discover
checkout. It returns the geometric-mean latency in microseconds and the
reward `1500 / latency_us`. Modal deployment and evaluation use your account.

For an existing compatible HTTP evaluator, use:

```bash
export TRIMUL_JUDGE_URL=https://your-evaluator.example/evaluate
# Supply TRIMUL_JUDGE_TOKEN through your credential store if required.
python judges/trimul_server.py --endpoint "$TRIMUL_JUDGE_URL" --port 8082
```

The endpoint accepts `{"solution": "...", "runner_timeout_s": 1100}`.
Its response contains `report.all_correct`, `report.score_us`,
`report.test_count`, `report.benchmark_count`, and `report.benchmarks`.
The adapter rejects missing, non-finite, or incompatible measurements.

## Check one bootstrap

After the judge starts, run:

```bash
GUIDANCE_TASK=lasso_path python smoke.py
```

Repeat with the selected task. This command performs one real evaluator call
and creates a verified seed archive. It does not call a model or start training.
A failed bootstrap returns a nonzero exit status.
