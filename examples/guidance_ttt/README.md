# Guidance-TTT on Reef

This example ports the `summary_only` Guidance-TTT loop to Reef while keeping
the execution model outside the trainable policy:

```text
PUCT archive selects G parent candidates
  -> Qwen3-8B sees each parent's canonical summary and verifier score
  -> Reef records R guidance generations and their exact SGLang tokens/logp
  -> a frozen external model sees parent code + guidance and writes a candidate
  -> the selected task's official verifier returns a reward and authoritative raw score
  -> each score references the exact Reef guidance receipt that produced it
  -> grouped TTT-Discover (tttd) advantages update only Qwen's rank-32 LoRA adapter
  -> valid candidates become PUCT children for the next step
```

The executor is therefore an ordinary OpenAI-compatible service. Its prompts,
tokens, and weights are not part of Reef training. The Reef-specific boundary
is limited to the guidance call and the score report linked to that call.

## What was added

The integration is intentionally an example rather than a new Reef core mode:

```text
state.py / puct.py / library.py  Guidance-TTT archive and search mechanics
prompts.py                       summary-only policy and execution prompts
execution.py                     frozen OpenAI-compatible executor adapters
harness.py                       receipt-linked rollout and verification loop
tasks/                           task registry plus Polyomino, VLIW, and TriMul contracts
verifier/                        official task adapters and provider-neutral HTTP service
deployments/qwen3_8b_lora/       Qwen3-8B LoRA runner and direct-node CLI
seeds/                           verified runnable bootstrap candidate
```

It reuses Reef's existing `tttd` recipe, token-native SGLang capture,
TTT-Discover processor/backend preparation, Slime/Megatron optimizer, checkpoint protocol,
and serving-native LoRA publication. No execution-model-specific behavior is
added to Reef's inference engine or training runtime.

## Summary-only semantics

Only `summary_only` is exposed here:

- The guidance actor receives the problem, the selected candidate's canonical
  whole-solution summary, and its verifier result. It never receives source
  code.
- The frozen executor receives the same problem, the selected candidate's full
  runnable source, and the new guidance.
- The executor returns a complete `<solution>` and a new canonical `<summary>`.
  That summary is the only implementation context shown to the guidance actor
  if this child is selected later.
- Only the Qwen guidance response mask is trainable. Executor output is used
  for verification and archive evolution, not as an RL trajectory.

The policy gets exactly one generation attempt. A response is accepted only if
it contains one non-empty terminal `<guidance>...</guidance>` block. Malformed
guidance receives reward zero, skips the executor, still counts as a PUCT visit,
and is reported against its original Reef receipt. There is no format repair,
fallback generation, or retry. Transient HTTP transport failures may be retried
by the executor client; they never cause the guidance policy to be sampled
again.

## Search and optimization

The archive preserves the Discover-compatible Guidance-TTT settings from
[`open-ttt-verl@ea47140`](https://github.com/Chonghe-Jiang/open-ttt-verl/tree/ea47140ca7aea324d89ea5585afffb03c2c01522):

- one PUCT-selected parent shared by every rollout in a comparison group;
- rank-prior PUCT with `best_child` Q and ancestor visit backpropagation;
- top two children per expansion and a top-1000 archive;
- invalid or failed candidates counted as visits but not added as executable
  search nodes;
- configurable groups and rollouts, with a complete `G × R` step barrier;
- adaptive-beta entropic leave-one-out advantages and frozen-base token KL;
- un-clipped importance sampling against captured rollout log-probabilities;
- Qwen3-8B thinking with explicit temperature, top-p, top-k, and per-rollout
  seeds (defaults remain 1, 1, -1, and unset; source-matched kernel runs use
  0.9, 0.95, -1, and seed 0);
- Adam at `4e-5`, rank/alpha `32/32` LoRA, frozen base parameters, and
  serving-native adapter publication to SGLang.

Three tasks use the same summary-only training path:

| Task ID | Candidate | Authoritative raw score | Direction | Default seed |
|---|---|---|---|---|
| `polyomino_packing` | complete C++17 solver | FrontierCS score | maximize | verified GPT-OSS-120B parent |
| `vliw_kernel_optimization` | complete EdgeBench `solution.py` | simulator cycles | minimize | official pinned starter, verified before use |
| `trimul` | complete Triton `submission.py` | H100 geometric-mean runtime | minimize | fixed independently generated scratch seed |

VLIW uses a dense `1,000,000 / cycles` training reward while preserving cycles
for PUCT comparison and reporting. TriMul uses the TTT-Discover reward
`1,500 / runtime_us`. Invalid candidates receive zero reward; verifier service
failures are recorded separately as environment errors. Task selection,
score direction, seed digest, and non-secret verifier configuration are
immutable resume fields.

The kernel assets and evaluators are pinned to the source revision documented
under `tasks/assets/`. Hidden EdgeBench cases remain only in the official judge
image. The vendored TriMul evaluator is byte-hash tested against its upstream
MIT-licensed files.

## Verifier boundary

Polyomino continues to use FrontierCS/go-judge. VLIW can invoke its pinned
official Docker judge directly or through the included HTTP verifier service.
TriMul uses that HTTP boundary to run the official evaluator on an isolated
H100 worker. The service is infrastructure-neutral: it has no cloud SDK or
provider launcher, accepts only a complete candidate and timeout, and returns
official task artifacts.

Keep verifier workers on a trusted private network. Candidate programs are
untrusted code, so run the service in a disposable container or sandbox with
no model/API credentials and require its bearer token unless the network is
otherwise isolated. TriMul subprocesses receive only an allowlisted
compiler/CUDA environment; service credentials are explicitly removed.

## Execution backends

Two frozen backends are built in:

| Backend | Model | Configuration |
|---|---|---|
| Local | `openai/gpt-oss-120b` | OpenAI-compatible endpoint, temperature 0, high reasoning effort, 1,200s timeout with no retries |
| OpenRouter | `z-ai/glm-5.2` | `OPENROUTER_API_KEY`, high reasoning effort, up to six transient-error retries |

The local endpoint defaults to `http://127.0.0.1:8000/v1` and can be replaced
with `--executor-base-url`. GPT-OSS reasoning effort is explicit and defaults to
`high`; use `--gpt-oss-reasoning-effort` to select `low`, `medium`, or `high`.
The OpenRouter key is read only from the environment
and is never serialized into the library, resume state, result, or logs. The
key/account provider policy must allow a provider serving `z-ai/glm-5.2`;
request-level routing cannot override an account-level provider allowlist.

## Running it

Use the [direct-node deployment guide](deployments/qwen3_8b_lora/README.md) to
build the existing TTTD LoRA image, select a task, start its official verifier,
and run a strict one-step test with either backend. A local GPT-OSS-120B
deployment needs three GPUs in the tested layout: two visible to Qwen/Reef and
one visible only to the executor. An API executor needs only the two Qwen/Reef
GPUs; TriMul additionally needs an isolated H100 verifier worker.

The launcher defaults to a small `2 × 4`, 12,288-token qualification. Groups,
rollouts, sequence limits, sampling, concurrency, LoRA rank, and total steps
are CLI arguments rather than algorithm constants. Verifier timeouts default
per task: 340s for Polyomino, 120s for VLIW, and 1,160s for TriMul. Add
`--require-nonconstant-step`
for a strict qualification that requires at least one verified candidate and
at least one nonconstant reward group. Long experiments normally omit that
flag and record any stochastic degenerate step instead of aborting it.

Checkpoint and archive state advance together after each successful optimizer
transaction. Resume with the same run name and settings plus `--resume`; the
requested `--steps` is the final total, not an additional count.

Use `--frozen-policy-control` for the matched search-only control. It keeps the
same policy inference, executor, verifier, seeds, and archive evolution but
does not report scores for LoRA training. The runner proves that the serving
version remains unchanged and that zero optimizer steps complete.

## Tests

From the repository root:

```bash
ruff check examples/guidance_ttt tests/test_guidance_ttt.py tests/test_guidance_ttt_kernel_tasks.py
PYTHONPATH=. pytest -q tests/test_guidance_ttt.py tests/test_guidance_ttt_kernel_tasks.py
```

The tests cover strict parsing, summary-only code isolation, exact receipt
linkage, executor skipping on malformed guidance, dynamic cardinalities,
Discover-compatible PUCT/archive behavior, task identity/minimize semantics,
secret and candidate-environment hygiene, official asset hashes, all three
verifier adapters, frozen-control semantics, deterministic sampling seeds, and
deployment validation. The adaptive deployment runner additionally checks the
real optimizer gradient, frozen-base status, LoRA-B update, checkpoint,
adapter checksum publication, and served weight-version advance.

## Credits and license

`state.py`, `puct.py`, and `library.py`, together with the prompt/search design
used by this example, are adapted from
[`Chonghe-Jiang/open-ttt-verl@ea47140`](https://github.com/Chonghe-Jiang/open-ttt-verl/tree/ea47140ca7aea324d89ea5585afffb03c2c01522)
and were modified for Reef's receipt/report and TTTD training interfaces. The
upstream work and Reef are licensed under the Apache License 2.0; the repository
root [`LICENSE`](../../LICENSE) applies. The upstream NOTICE entry is:

> Copyright 2023-2024 Bytedance Ltd. and/or its affiliates

The VLIW and TriMul task adaptations are pinned to
[`open-ttt-verl@8e79cd3`](https://github.com/Chonghe-Jiang/open-ttt-verl/commit/8e79cd3d5cad7ea3f018d5ef001e6bd2962286d9).
EdgeBench starter attribution and the complete TTT-Discover MIT notice are in
their respective `tasks/assets/*/SOURCE.md` files.
