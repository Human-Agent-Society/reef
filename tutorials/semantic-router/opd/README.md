# OPD routing feasibility test

Can Reef route a request to a student, train on that student's answer, and serve the
next request with the updated weights? This example checks that loop over two
on-policy distillation (OPD) updates. It uses a separate frozen teacher for scoring
and for two short chat requests.

The routed Reef service runs the base record-only recipe. For each student request,
the example adapter calls the native [OPD service](../../../recipes/opd/examples/math/README.md)
and includes its receipt in the response. The driver checks the routing record and
reports the **native student receipt** for training. A teacher chat goes directly
to the teacher's serving endpoint and produces no student training sample.

## Run

Use a Linux host with Docker Compose, the [pinned Semantic Router image](../README.md#1-build-the-semantic-router-image)
and the [Reef Slime GPU environment](../../../docker/README.md). The small profile
uses eight GPUs: four single-GPU student serving replicas, two GPUs for the teacher,
and two for the student trainer. The native service and routing frontend must be
able to reach each other on loopback.

From the repository root, prepare the model directories, four DeepMath prompts,
and a fresh directory for native checkpoints. Set `REEF_TOKEN` to the same private
value in both shells.

```bash
export OPD_MODEL_PATH=/work/models/Qwen3-0.6B
export OPD_TEACHER_PATH=/work/models/Qwen3-1.7B
export OPD_RUN_DIR=/work/opd-routing-native
export REEF_TOKEN="$(openssl rand -hex 32)"

hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir "$OPD_MODEL_PATH"
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir "$OPD_TEACHER_PATH"
python recipes/opd/examples/math/prepare.py --tokenizer "$OPD_MODEL_PATH" --prompts 4 --output /work/opd-routing-data
bash tutorials/semantic-router/opd/serve-native.sh
```

Keep the native service running. In a second shell on the same host, activate the
Reef environment, export the same `REEF_TOKEN`, and wait for
`curl -fsS http://127.0.0.1:8900/healthz` to succeed. Set `SR_IMAGE` to the
reference of your locally built image, including its repository digest:

```bash
export SR_IMAGE='reef-semantic-router@sha256:<digest>'
bash tutorials/semantic-router/opd/run.sh /work/opd-routing-data/deepmath.jsonl
```

## What to check

| Request | Expected result |
| --- | --- |
| Four `auto` student samples per update | SR picks the default student. All four samples come from the same native weight version. Only their native receipts are reported. |
| One `auto` teacher control per update | The keyword rule selects the teacher, which answers directly. Separately, OPD scores the student's generated tokens to train the student. |
| One explicit student request after update two | It uses the second published version. The driver leaves this final receipt unreported. |

The launcher prints its output directory. Each `results/driver/step-*.json`
contains the selected targets, receipts, weight artifact IDs and phase timings.
`results/driver/completed.json` identifies the weights used by the final student
request. The launcher cleans up its routing services. Stop the native OPD service
in the first shell after the test finishes.

## Measured feasibility run

On eight RTX A6000 GPUs, Qwen3-0.6B completed two rank-128 LoRA updates with
Qwen3-1.7B as the teacher. Each batch contained two prompts sampled twice, with
a limit of 256 generated tokens. All eight student rollouts reached that limit.
The second batch used the first update's weights, and the explicit student request
used the second update's weights. All 196 LoRA B tensors changed at each update.

| Wall time per update | First | Second |
| --- | ---: | ---: |
| Teacher chat control | 96.48 s | 2.03 s |
| Collect four student responses in parallel | 33.45 s | 20.18 s |
| Check the batch and submit reports | 0.39 s | 0.41 s |
| Wait for the new student weights after reporting | 123.47 s | 113.20 s |
| **Total** | **253.80 s** | **135.82 s** |

These measurements used the PR frontend at `c2fb4b18` and an experiment build
of the native OPD backend based on `06c07815`. The commands above target the
current checkout's native API. The ten `auto` requests had a median SR preview
time of 3.91 ms and a maximum of 227.70 ms. The step totals exclude service
startup and shutdown.

This run tested routing, training and serving the updated weights. It did not
measure routing accuracy, convergence, or throughput during concurrent teacher
chat and scoring. Teacher-only and student-only benchmark runs and shadow
execution (both models answering the same request) were outside its scope.
