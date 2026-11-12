# Route between two Qwen2.5 models on one GPU

This example starts two vLLM servers, sends chat requests through Reef and
changes the default target twice while both models stay loaded. The student
serves Qwen2.5-0.5B-Instruct and the teacher serves Qwen2.5-1.5B-Instruct.
Both models generate answers. The example exercises serving only.

## Run

Use a Linux host or Ubuntu under WSL2 with Docker Compose, GPU access from
Docker, and enough free GPU memory for both servers. The model settings are
sized for one 32 GiB RTX 5090. Activate the Python environment containing
this Reef checkout. The update check uses Linux process descriptors for
`SIGHUP`. If your Reef Python lacks `signal.pidfd_send_signal`, it uses
Ubuntu's `/usr/bin/python3` for this one call.

From the repository root, run:

```bash
bash tutorials/semantic-router/qwen2.5/run.sh
```

The first run builds the pinned SR image, as described in the
[routing tutorial](../README.md#1-the-semantic-router-image), and later runs
reuse it. To use an image you already have instead, set `SR_IMAGE` to its
repository digest (`repository@sha256:...`) or local image ID (`sha256:...`).

The script uses GPU `0` by default. Set `GPU_DEVICE` to select a different
GPU. On the first run it needs access to the pinned public Qwen weights,
vLLM image and SR build sources. An offline run needs the vLLM image, the model
weights and the SR image already on the host: the image an earlier run built,
or a local image named by `SR_IMAGE`. Then set `HF_HUB_OFFLINE=1`; it stops
model downloads, not Docker's network access for an SR image build. You can
point `STUDENT_MODEL_DIR` and `TEACHER_MODEL_DIR`
at the corresponding local model snapshots. Set `WORK_DIR` to choose where
the script creates its per-run directory. It keeps the results and logs there
and shuts down the example's services when it exits.

## What to look for

| Check | Expected result |
| --- | --- |
| Send an ordinary prompt with `model: auto` | The student answers. |
| Send a prompt containing `reef_teacher_required` with `model: auto` | The keyword rule sends the request to the teacher. |
| Send the same keyword with an explicit student target | The student answers because the explicit target takes precedence. |
| Change the default to the teacher and send `SIGHUP` | The same ordinary prompt now gets a teacher answer. |
| Restore the student default during a teacher stream | A new `auto` request gets a student answer while the original teacher stream is still producing tokens. Each receipt points to its own policy. |

Read `result.json` for the returned models, receipts and update timings. The
time from `SIGHUP` to the first answer under the new policy includes update
preparation and model response time. The public admission probes measure
the shorter interval during which Reef refuses new inference. By default the
teacher generates 384 tokens. To compare multiple stream lengths on the same
warm deployment, run `bash tutorials/semantic-router/qwen2.5/run.sh --stream-tokens 384 768`.
Each length runs one idle update and one update during generation.
