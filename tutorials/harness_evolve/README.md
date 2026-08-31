# Harness evolution quickstart

This example is the smallest full run of the harness evolution mechanism: the agent harness configuration is a composition tree of nodes, a proposal is one gated tree mutation, and a winning mutation publishes as a versioned artifact any client can pull. The proposer is the served model itself: it reads the current skill nodes and its own failing requests and proposes one skill mutation as a strict JSON object. `evaluate` grades real headless episodes on a small fixed task set by exact final answer, so a proposal only publishes when it makes previously failing tasks pass their episodes.

The pinned paper reproduction built on this mechanism lives at `recipes/skillclaw/`.
## Directory layout

```text
harness_evolve/
  serve.yaml     the stack (reef service) plus the harness_evolve recipe
                 sections: tasks, seed composition, dotted method references
  harness/       the method package serve.yaml's dotted refs name
    evolution.py the method: propose (self proposer over failures, skill
                 nodes only) and evaluate (exact last-line answer grading)
  run.py         the loop, written out: record -> report -> evolve -> pull
  run.sh         copies the recipe config out of serve.yaml, starts
                 reef serve, waits for /healthz, runs run.py
  1_evolve_your_harness.ipynb
                 the same pass cell by cell, with the service managed as a
                 subprocess from the notebook
  pyproject.toml makes harness/ an installable package
  README.md      this file
```

## Setup (once)

The client side needs only the stdlib SDK; the service side runs from the repo checkout via `run.sh`:

```bash
pip install reef-client
```

You also need the pi coding agent binary on your PATH (serve.yaml's `evolution.binary`) and an OpenAI-compatible endpoint for the model under test.

## Quick start

```bash
./run.sh
```

serve.yaml carries the endpoint (`upstream_url: http://127.0.0.1:8000`, no /v1 suffix) and the model (`qwen3-8b`) as literals; edit them there to point at your own. The one value it does not hold is the provider key: `export REEF_UPSTREAM_API_KEY=...` if your endpoint needs one.

## Notebook

`1_evolve_your_harness.ipynb` walks the same pass cell by cell and manages the service as a subprocess, so one kernel holds the whole loop. Set the endpoint, model, and key in its first code cell; the notebook patches both serve.yaml bindings (the `reef` section's upstream values and the recipe's `model.path`) into `work/serve-notebook.yaml` and materializes the recipe config from the patched text. `run.py` stays the reference implementation of the loop; the notebook mirrors it.

Pick a model that fails at least one task and still writes the strict JSON mutation `propose` expects. A model that passes all three tasks reports no failures, so no evolve step ever runs (the DeepSeek result below); one that cannot author the JSON commits its step as `skipped: no proposal`, visible in `work/agent-record/*.commits.jsonl`. The committed outputs are a full local pass with no GPU: ollama `qwen2.5:7b` on a Mac mini scored 1.0 / 0.0 / 1.0 on the recorded pass, the self proposer updated `answer-style`, and the gate published on 1 win, 0 losses, 2 ties.

## What one run does

1. `run.sh` copies serve.yaml's recipe sections into `work/recipes/harness_evolve.yaml` and starts Reef. The recipe boots with the seed composition: one starter `answer-style` skill node. The endpoint and model live only on serve.yaml's `reef` section (`upstream_url`, `upstream_api_key`, `upstream_model`); Reef hands them to `propose` as a model binding and renders them into each evaluation episode, so neither the method nor the published tree ever names them.
2. `run.py` sends each of the three exact-answer coding tasks once through reef inference; reef serves the reply and records the exchange. The reply is graded the same way the evolve gate grades episodes, and the score is reported against the receipt.
3. Only failures batch (`max_score: 0.0`, the SkillClaw window), and `batch_size: 1` makes every failing report one evolve step: `propose` sends the failing requests and current skills back to the same model through `models.served.chat`, which answers with one skill mutation; the candidate and current compositions each run one episode per task; the mutation publishes only on a gate win.
4. `run.py` pulls `GET /reef/harness` and prints the gate metrics and the evolved `SKILL.md` files. Point any pi at the pulled tree, with its model set to Reef, and it carries the learned skill.

## Results

Measured 2026-08-17 on one B200 node, pi 0.84.2 as `evolution.binary`, vllm 0.27.0 at `http://127.0.0.1:8000`, one fresh scenario per run.

DeepSeek-V4-Flash-0731 as `deepseek-v4-flash` (tensor parallel 4, fp8 kv cache) answered every task exactly on the recorded pass (`9592`, `2880067194370816120`, `30` alone on the last line) in two runs of 3 s and 4 s each, so no report entered the `max_score: 0.0` window, nothing batched, and both runs printed `every task passed: nothing batched, no evolve step runs`. No proposal was made and no gate ran: the evolve path only opens when the recorded pass fails at least one task.

Qwen3-8B as `qwen3-8b` (tensor parallel 1) opens that window. Serve it with tool calling enabled (`--enable-auto-tool-choice --tool-call-parser hermes`); without those flags vllm rejects pi's `tool_choice: "auto"` requests with a 400, every episode scores as failed, and every gate dead ties. One run, 63 s wall clock end to end:

| pass | [sieve] | [fib] | [csv] |
|------|---------|-------|-------|
| recorded | 1.0 | 0.0 | 1.0 |
| gate: current | 1.0 | 0.0 | 1.0 |
| gate: candidate | 1.0 | 1.0 | 1.0 |

The one failing report (`[fib]`) batched and triggered one evolve step. The self proposer (qwen3-8b) proposed `create direct-answer`, a new skill beside the starter; the gate scored candidate 3.0 against current 2.0 (1 win, 0 losses, 2 ties, 0 episode failures) and published artifact `0fe30330c6ddacd400fbb905113711c3940f7071`. The pulled tree carries the proposed skill verbatim:

```text
--- pi-agent/skills/direct-answer/SKILL.md ---
# direct-answer

Enforces answers to be exact plain integers without additional text. For numerical problems requiring precise values (e.g., Fibonacci numbers, large computations), this skill ensures the response contains only the final integer result on the last line, adhering strictly to the format specified in the query.
```
