Train on Apple Silicon with MLX
==============================

The optional ``mlx`` backend serves and trains one LoRA adapter on an arm64
Mac. It uses `MLX <https://github.com/ml-explore/mlx>`__ and
`mlx-lm <https://github.com/ml-explore/mlx-lm>`__. One Reef process holds the
base model, the adapter and the optimizer. That process answers inference
requests and runs the training steps. The backend does not start Ray, does not
need CUDA, and does not use a separate inference engine. The engine loads the
base model from ``inference.model-path``, a local directory or a Hugging Face
Hub ID.

Use this backend to run a weight-training recipe on one Mac. It trains on the
traffic that Reef records, as the GPU backends do. It is not a distributed
trainer: serving and training take turns on the same weights.

Install
-------

The backend needs macOS on Apple Silicon (arm64) and Python 3.12 or newer.
MLX publishes wheels only for arm64 macOS, so the ``mlx`` extra does not
install on other platforms. Run the command from a source checkout with its
submodules:

.. code:: bash

   uv pip install -e ".[mlx]" -e ./third_party/reef-client

The extra pins ``mlx`` and ``mlx-lm`` together. If you select the backend
without the extra, ``reef serve`` stops at startup and tells you to install
``reef-infra[mlx]``.

Configure and run
-----------------

Select ``training.backend: mlx``, set the base model in
``inference.model-path``, and set ``training.options.checkpoint_dir``. The
example below trains the OpenClaw-RL recipe. That recipe also needs its
``pipeline`` section, which configures the PRM judge; see the `OpenClaw-RL
recipe page <recipes/openclawrl.rst>`__.

.. code:: yaml

   reef:
     ready-timeout: 1800
   inference:
     model-path: mlx-community/Qwen3.5-9B-MLX-4bit
   training:
     backend: mlx
     options:
       checkpoint_dir: work/mlx/checkpoints
       lora_rank: 16
       learning_rate: 1.0e-5
       capture_topk: 4   # the openclawrl loss needs the generation top-K
   recipe:
     implementation: recipes.openclawrl.recipe:OpenClawRLRecipe
   storage:
     agent-record-dir: work/mlx/agent-record
     artifact-repository: work/mlx/artifacts.git
     artifact-work-dir: work/mlx/artifact-work
     artifact-cache-dir: work/mlx/artifact-cache

The complete example config is described in `Example: OpenClaw-RL on a
Mac`_. Start the service with ``reef serve -c <file>``. Reef loads the model inside the
service process, so ``reef.ready-timeout`` must cover the model download and
load. Its default of 30 seconds is too short for most models.

The deployment checks these conditions before it loads the model:

- ``training.options.checkpoint_dir`` is required. Each trained candidate is
  written to a directory below it.
- Ray, executor and inference-engine settings are refused:
  ``training.ray-address``, ``training.ray-namespace``,
  ``training.ray-actor-name``, ``training.colocate``, ``inference.url``,
  ``inference.backend``, ``inference.handler-factory``,
  ``inference.handler-config``, ``inference.num-gpus``,
  ``inference.tensor-parallel-size``, ``inference.options``, and the
  ``execution.training`` and ``execution.rollout`` sections.
- ``training.options`` cannot set ``type``, ``model_path``,
  ``inference_timeout_s``, ``train_timeout_s`` or ``connect``. Reef sets
  these fields itself.

``training.options`` keys accept ``-`` or ``_``, and
``--training.options.<key>`` on the command line overrides the YAML value.

Training options
----------------

All keys are under ``training.options``. Only ``checkpoint_dir`` is required.

.. config::

   training.options.checkpoint_dir | required | directory for the candidate adapters
   training.options.lora_layers | 8 | number of transformer blocks that get a LoRA adapter, counted from the last block
   training.options.lora_rank | 8 | LoRA rank
   training.options.lora_scale | 2.0 | LoRA scale
   training.options.lora_dropout | 0.0 | LoRA dropout
   training.options.lora_keys | [self_attn.q_proj, self_attn.v_proj] | projections that get an adapter, as mlx-lm module paths
   training.options.learning_rate | 1e-5 | AdamW learning rate, constant for the whole run
   training.options.weight_decay | 0.01 | AdamW weight decay
   training.options.kl_coef | 0.0 | weight of the frozen-base KL term, as described under Loss families below
   training.options.loss_options | {} | settings of the selected loss family; for ``openclawrl``: ``w_rl`` 1.0, ``w_opd`` 1.0, ``eps_lo`` 0.2, ``eps_hi`` 0.28, ``diff_clip`` 1.0 (null for no limit), ``hint_selection`` sequence_optimal or shortest, ``native_k`` 20, ``kl_coef`` (default: ``training.options.kl_coef``); ``tttd`` reads only ``kl_coef``
   training.options.max_tokens | 256 | response token limit when a request sets none
   training.options.temperature | 1.0 | sampling temperature when a request sets none; must be positive
   training.options.top_p | 1.0 | top-p for every request
   training.options.seed | 0 | MLX random seed
   training.options.capture_topk | 0 | top-K candidates recorded for each generated token; the ``openclawrl`` loss needs a value above 0
   training.options.chat_template_kwargs | {} | default chat-template values for every request, such as ``enable_thinking``
   training.options.micro_batch_size | 8 | maximum sequences in one backward pass; a larger batch accumulates over several passes
   training.options.checkpoint_layers | false | recompute each adapted block's forward pass during the backward pass, to save memory
   training.options.log_probs_chunk_size | 0 | response positions scored in one pass over the output head; 0 scores all at once
   training.options.cache_limit_gib | 0.0 | upper limit in GiB of the MLX buffer cache; 0 keeps the MLX default
   training.options.prefill_step_size | 0 | prompt tokens in one prefill pass; 0 processes the prompt whole
   training.options.recurrence_chunk_size | 64 | chunk size for GatedDeltaNet layers during training, a power of two; 0 uses mlx-lm's per-token loop
   training.options.adapter_name | reef-mlx | adapter name reported by the serving side

Reef refuses an unknown key and a value of the wrong type at startup.
``lora_keys`` must be a list of module paths. ``prefill_step_size`` above 0
needs ``lora_layers: 1`` and no adapted key, value or ``linear_attn``
projection. With any other adapter, a cached prompt prefill would drop part
of the gradient. ``chat_template_kwargs`` cannot set
``tokenize``, ``add_generation_prompt``, ``conversation``, ``messages`` or
``tools``, because the engine controls how a prompt is rendered.

``inference.timeout-s`` (default 300 seconds) bounds how long a training
step or a candidate probe waits for in-flight requests to finish before it
takes the engine.

Loss families
-------------

Each training job trains the objective that its recipe selected. The backend
implements two loss families:

- ``openclawrl`` trains the OpenClaw-RL objective: a PPO clipped term on the
  sampled tokens and on-policy distillation onto the captured top-K
  candidates. Each sample must carry its generation top-K, so set
  ``capture_topk``. Each sample must also carry the teacher candidates that
  the OpenClaw-RL judging stage attaches. ``kl_coef`` adds a frozen-base KL
  term to this loss.
- ``tttd`` trains a token-level policy-gradient objective with one advantage
  per sample. When ``kl_coef`` is not 0, each token's advantage also gets the
  centered frozen-base KL penalty of the TTT-Discover reference.

The backend refuses these jobs before training:

- a job whose loss family is not ``openclawrl`` or ``tttd``;
- a job that selects a learning-rate schedule;
- a schedule with ``epochs`` other than 1, with ``shuffle``, or with an
  integer ``batch_size``. The reserved batch is always one optimizer step.

A training step that leaves every adapter tensor unchanged fails. Reef does
not publish an adapter that is identical to its predecessor.

Each step adds these values to the training metrics of its job:
``optimizer_step``, ``loss``, ``response_tokens``, ``active_gib``,
``adapter_delta_l2`` and ``adapter_tensors_changed``. The ``tttd`` family
also reports ``importance_ratio`` and ``ppo_kl``. ``optimizer_step`` counts
up across steps, because the optimizer state stays in memory for the life of
the process. On an on-policy batch, ``importance_ratio`` is close to 1.0.

The importance ratio uses the model's untempered log probability of each
sampled token. With a sampling temperature other than 1.0, the ratio is an
approximation. Keep ``temperature`` at 1.0 when the objective depends on an
exact ratio.

Serving
-------

The backend serves ``/v1/chat/completions``. Other routes, such as the
Anthropic routes, return HTTP 404. The backend reads these request fields:

- ``messages``, which must be a non-empty list;
- ``max_tokens`` or ``max_completion_tokens``;
- ``temperature``;
- ``tools`` and ``tool_choice``;
- ``chat_template_kwargs``, which overrides the deployment default.

``training.options.top_p`` applies to every request. A request that sets
``top_p`` to another value fails with HTTP 400. The request also fails when
it sets ``n`` above 1, or sets any of ``top_k``, ``min_p``, ``seed``,
``stop``, ``frequency_penalty``, ``presence_penalty``,
``repetition_penalty``, ``logit_bias``, ``logprobs`` or ``top_logprobs``.
``max_tokens`` must be a positive integer, and ``temperature`` must be finite
and not negative.

The chat template renders the declared tools into the prompt. The reply's
tool calls are parsed with the mlx-lm parser that matches the model's
template, and returned as ``tool_calls``. If a template has no mlx-lm parser,
the reply stays text. ``tool_choice: none`` keeps the tools in the prompt and
does not parse calls. A reply that opens a tool call and does not close it
fails the request. This occurs when the response reaches its token limit
inside a call.

A reasoning model's ``<think>`` block is returned as ``reasoning_content``.
Those tokens are still response tokens, so they train.

With ``stream: true``, the response is sent as OpenAI SSE chunks while the
engine generates. Tool-call text is held back until the call is complete.
Closing the stream stops the generation.

The engine generates one sequence at a time. Concurrent requests wait in
turn; they are not batched. While a training step runs, inference admission
is closed, so new requests wait for the step to finish.

Candidates and the published artifact
-------------------------------------

Each training step produces a candidate. The step trains on the live
adapter, exports the result, and then restores the weights that were served
before the step. Serving changes only when Reef selects the candidate. A
rejected candidate is discarded, and serving never used its weights.

A candidate evaluator can generate from a candidate before Reef decides
whether to publish it. The OpenClaw-RL style probe uses this. The probe
closes inference admission while it runs.

The published artifact is an mlx-lm adapter directory:

- ``adapters.safetensors``: the adapter weights;
- ``adapter_config.json``: the mlx-lm adapter configuration;
- ``reef-adapter.json``: the base model, the LoRA settings, the ``mlx`` and
  ``mlx-lm`` versions, the objective and loss family, the scenario step, the
  training job ID, and a checksum of each adapter file.

``mlx_lm.tuner.utils.load_adapters`` loads the directory directly. The
directory is written to a temporary location and then renamed, so an
interrupted write leaves the previous complete adapter.

The artifact holds the adapter weights only. After a restart, Reef loads the
committed adapter into the engine before it serves. The optimizer state
starts again from zero.

Size and tune the run
---------------------

These measurements come from an M4 Pro with 51.5 GB of unified memory. Use
them as a starting point for your own machine.

Weights and generation speed, 4-bit, LoRA rank 8:

+---------------+---------+------------+
| Model         | Weights | Generation |
+===============+=========+============+
| Qwen2.5-1.5B  | 0.87 GB | 130 tok/s  |
+---------------+---------+------------+
| Qwen2.5-7B    | 4.29 GB | 53 tok/s   |
+---------------+---------+------------+
| Qwen3.8-27B   | 15.1 GB | ~14 tok/s  |
+---------------+---------+------------+
| Qwen2.5-32B   | 18.4 GB | 11.5 tok/s |
+---------------+---------+------------+
| Llama-3.3-70B | 39.7 GB | 4.0 tok/s  |
+---------------+---------+------------+

All of these models train. The 70B model trains only with
``micro_batch_size: 1`` and no free memory, so treat it as the limit. MLX
reports a recommended working set of 40.2 GB on this machine, but that is
not the limit. A step with a 44.9 GB peak ran at normal speed. A step that
needed 71 GB, more than the physical memory, took 4.5 times longer than
expected.

Training memory grows with ``micro_batch_size`` times the sequence length.
It is about 3–5 MB per sequence token, in addition to the weights. Peak
memory for one sequence per backward pass:

+--------+--------------+-------------+------------------------------+
| Tokens | Qwen2.5-32B  | Qwen3.8-27B | Qwen3.8-27B, chunk size 1024 |
+========+==============+=============+==============================+
| 1024   | 23.0 GB      | 17.6 GB     | not measured                 |
+--------+--------------+-------------+------------------------------+
| 2048   | 28.8 GB      | 20.1 GB     | not measured                 |
+--------+--------------+-------------+------------------------------+
| 4096   | 44.4 GB      | 25.4 GB     | 20.7 GB                      |
+--------+--------------+-------------+------------------------------+
| 8192   | not measured | 37.3 GB     | 27.3 GB                      |
+--------+--------------+-------------+------------------------------+
| 16384  | not measured | 58.2 GB     | 58.8 GB                      |
+--------+--------------+-------------+------------------------------+

The last column sets ``log_probs_chunk_size: 1024``.

When a step runs out of memory, change these options in this order:

1. Set ``checkpoint_layers: true``. This matters most when many blocks are
   adapted. On Qwen3.8-27B with all 64 blocks adapted (rank 16, one
   700-token row), peak memory fell from 50.5 GB to 18.3 GB, and the
   backward pass fell from 744 s to 41.6 s.
2. Lower ``micro_batch_size``.
3. Lower ``max_tokens``.
4. Lower ``lora_layers``.

``log_probs_chunk_size`` reduces peak memory by about 27% at 8192 tokens,
for about 40% more step time. It does not help at 16384 tokens.

Generation takes most of a step's time, so ``max_tokens`` also sets the
throughput. Long prompts for serving need less memory than training: the
hybrid Qwen3.8-27B prefilled 65,536 tokens in 28.7 GB, at about 100–125
prompt tokens per second.

Example: OpenClaw-RL on a Mac
-----------------------------

`serve-mlx.yaml <../../recipes/openclawrl/examples/openclawrl/serve-mlx.yaml>`__
runs the OpenClaw-RL GSM8K stream with the policy on MLX. The PRM judge and
the simulated student are hosted models. The config adapts all 32 blocks of
``Qwen3.5-9B-MLX-4bit`` at rank 64 with ``checkpoint_layers`` on. Run
``reef serve -c`` with this file on the host, not in Docker, and point the
harness at the machine's LAN address.

Serving on a Mac is slower than on the reference GPUs. Two settings depend on
that speed:

- ``OPENCLAWRL_TURN_TIMEOUT_S`` sets the harness's limit for one agent turn,
  600 seconds by default. In a 72-session stream with Qwen3.5-9B on an M4 Pro,
  the median turn took 397 s and the slowest took 1141 s. Turns that reach
  the limit are recorded as failed sessions. Raise the limit on this backend.
- ``pipeline.record2record.config.session_ttl_s`` must be longer than the gap
  between a reply and the reaction that judges it. ``serve-mlx.yaml`` sets
  1800. The `session TTL result
  <../../recipes/openclawrl/examples/openclawrl/results/2026-09-16-mlx-session-ttl/README.md>`__
  shows that 45 seconds lost about half of the training pairs.

Verification scope
------------------

The tests are in ``tests/mlx_backend/``. The runtime contract tests use a fake
engine and run on any machine. They cover step preparation, the refused
schedules and loss families, candidate export, selection, rejection,
rollback and the serving responses. The engine tests need the ``mlx`` extra
and skip without it. CPU CI does not install the extra, so it does not run
the MLX engine. Real generation, optimizer steps and memory limits are
checked only on Apple Silicon.
