<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reef-logo-dark.svg">
  <img src="docs/assets/reef-logo-light.svg" alt="Reef" width="220">
</picture>

<h3>Continual learning infra for self-improving agents</h3>

[![CI](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml)
[![Docker](https://github.com/Human-Agent-Society/reef/actions/workflows/docker.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/docker.yml)
[![Docs Build](https://github.com/Human-Agent-Society/reef/actions/workflows/docs-build.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/docs-build.yml)

[![Docs](https://img.shields.io/badge/docs-reefinfra.ai-0E7490?logo=readthedocs&logoColor=white)](https://reefinfra.ai)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/8k4WeVuTb)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Reef is infrastructure that serves an entire continual learning backend. Reef
exposes standardized http endpoints so that you can download agents just like
how you download `codex` or `opencode` using `curl`, and so that your agent can
send its model requests to Reef's inference endpoint instead of the provider's.

The only difference is that, Reef constantly evaluates your agent behavior
and improves the served harness and model weights in the backend. You keep getting
better and better results without having to do anything.

</div>


## How it works

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loop-animation-dark.svg">
  <img src="docs/assets/loop-animation-light.svg" alt="Reef serves requests, records feedback, produces updates, and commits accepted updates to a version history." width="76%">
</picture>
</div>

Reef processes each learning cycle in four steps. The table also shows which
modules implement each step.

| Step | What happens | Core code |
|---|---|---|
| 1&nbsp;·&nbsp;Serve | Forward a request to the model provider and store the interaction. | [`service/`](reef/service) implements HTTP routes, authentication, and streaming. [`runtime/`](reef/runtime) connects Reef to the model provider. |
| 2&nbsp;·&nbsp;Observe | Associate each report with the stored interactions referenced by its receipts. | [`core/`](reef/core) defines the stored record types. [`dispatcher.py`](reef/dispatcher.py) sends each record to its scenario. |
| 3&nbsp;·&nbsp;Grow | Build a batch from eligible records and use it to update model weights or harness files. | [`sao/`](recipes/sao), [`tttd/`](recipes/tttd), [`openclawrl/`](recipes/openclawrl), [`harness_evolve/`](reef/train/cordis_backend) are the learning methods, one package each; [`infra/recipe/`](reef/recipe) is the contract they implement and [`infra/train/`](reef/train) prepares batches and runs training jobs. |
| 4&nbsp;·&nbsp;Commit | Gate the candidate against your tasks, then record an accepted update as a new version and make it available for serving. | [`train/evaluation/`](reef/train/evaluation) evaluates each candidate and decides whether to accept it. [`artifact/`](reef/artifact) stores the Git-backed version history. [`surface/`](reef/surface) delivers each artifact to the serving runtime or harness client. |


## Using Reef

Reef supports two learning surfaces: model **weights** and agent **harnesses**.
The recipe bound to a scenario determines which surface it updates.

### 1 · Serve

The following example starts the SAO (arXiv:2607.07508) example deployment. Run it
from a Reef checkout in an environment that satisfies the GPU requirements in
[Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/).

```bash
pip install -e ".[slime]" && pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen3-8B"
export REEF_TOKEN="reef-local"

reef serve -c recipes/sao/examples/sao/serve.yaml \
  --reef.model_path "$MODEL_PATH" \
  --reef.port "8900"

curl -f http://127.0.0.1:8900/healthz          # ready to serve
```

### 2 · Train weights

Send inference requests through Reef and report a score for each response. The
SAO recipe uses each eligible scored rollout to run a training step.

#### Send an inference request and report feedback

Reef's inference endpoint is OpenAI- and Anthropic-compatible: `/v1/chat/completions`
and `/v1/messages` take the provider's own request body. The first request for a
new scenario must include both Reef headers. Reef stores the recipe binding and
rejects later attempts to change it.

The response body uses the provider's OpenAI-compatible format. Reef adds the
`x-reef-agent-record-id` response header. Its value is the **receipt** that a
later report uses to identify this interaction. A report can contain a numeric
`score`, textual or structured `feedback`, and the receipts it evaluates. This
example reports both a score and a short explanation.

```python
import os
import httpx

reef = httpx.Client(
    base_url="http://127.0.0.1:8900",
    headers={"Authorization": f"Bearer {os.environ['REEF_TOKEN']}", "x-reef-scenario": "hello-reef"},
    timeout=300,
)

# Inference using Open-AI compatible format
response = reef.post(
    "/v1/chat/completions",
    json={
        "model": os.environ["MODEL_PATH"],
        "messages": [{"role": "user", "content": "Return exactly: reef is ready"}],
    },
)

receipt = response.headers["x-reef-agent-record-id"]
answer = response.json()["choices"][0]["message"]["content"]

# Sending report about the inference
matched = answer.strip() == "reef is ready"

reef.post(
    "/reef/report",
    json={"score": float(matched), "feedback": "matched" if matched else "wrong answer", "references": [receipt]},
).raise_for_status()
```

`feedback` carries the richer signal, plain text or a structured object,
for recipes that read more than a scalar. The endpoint will validate the 
**report schema** ([`reef/core/reports/`](reef/core/reports)).


#### Watch it learn and grow

Once the recipe has enough feedback, it runs a training step and synchronizes
the updated weights to the serving runtime. Later inference requests use the
current version without restarting Reef.

### 3 · Evolve your harness

The `harness_evolve` recipe updates a harness tree that may contain rules,
skills, configuration, prompts, and extensions. It builds a candidate from
reported interactions, evaluates the current and candidate harnesses on the
configured tasks, and publishes the candidate only when it wins that
comparison. Harness scenarios do not share data or versions.

#### Install Reef harness that grows with you

You can install Reef harness like how you install most coding agents.
The following is an example. A new scenario will be automatically created
and bundled with the downloaded harness.

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash

reef-pi -p "fix the bug"
```

You can also retrieve an evolved harness by supplying its scenario in the header.
For example, if you have a scenario `harness-evolve-code-repair`, you can install its harness via the following.

```bash
curl -fsS -H 'x-reef-scenario: harness-evolve-code-repair' \
  -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash
```

#### Report a task result

`reef-pi` stores the receipts from a run, so its `report` command only needs
the result you want to associate with the preceding interaction:

```bash
reef-pi -p "fix the failing test in auth.py"

# ... run your tests, grade the result ...

reef-pi report --score 0 --feedback "missed the empty-token case"
# reef-pi: reported 1 receipt(s) to harness-evolve-code-repair
```

Reef batches eligible reports according to the recipe configuration. When
version checking is enabled, the adapter checks for a newer published version
the next time it starts. Interactive sessions offer **Update with …** and
**Skip** before accepting input; choosing update runs the installer directly.
Headless sessions print the instruction instead.
The [harness evolution guide](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)
describes the proposal, evaluation, and publication process.


## Cookbook recipes

Choose a recipe based on the feedback available from the workload and the
artifact that should be updated. These implementations live in this
repository's `recipes/` cookbook; they are selected by dotted class reference
and do not ship in the Reef wheel.

| Workload | Method and recipe | Updated artifact | Documentation |
|---|---|---|---|
| A stream of tasks scored by tests or a verifier | SAO, `sao` | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/sao/) · [Example](recipes/sao/examples/sao/README.md) |
| Agent traffic with useful next-state signals and no explicit reports | OpenClaw-RL, `openclawrl` | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) · [Example](recipes/openclawrl/examples/openclawrl/README.md) |
| Repeated, scored attempts at one problem | TTT-Discover, `tttd` | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [Example](recipes/tttd/examples/tttd/README.md) |
| Scored code search with a trainable guidance model and a frozen executor | Guidance-TTT, `tttd` | Guidance-model weights | [Example and Reef results](recipes/tttd/examples/guidance_ttt/README.md) |
| Scored agent interactions used to improve prompts, rules, skills, or configuration | Harness evolution, `harness_evolve` | Harness tree; no GPU required | [Guide](https://reefinfra.ai/docs/user-guide/evolve-your-harness/) · [Example](tutorials/harness_evolve/README.md) |


## How is Reef different?

Reef builds the infra for AI that grows:

| Ability | Inference engine (vLLM, SGLang, …) | RL training framework (slime, veRL, AReaL, …) | **Reef** |
|---|:---:|:---:|:---:|
| Serves live traffic | ✅ | ❌ | ✅ |
| Trains weights | ❌ | ✅ | ✅ |
| Version management | ❌ | ❌ | ✅ |
| Stays live through updates | ❌ | ❌ | ✅ |
| Evolves beyond weights (skills, harness) | ❌ | ❌ | ✅ |


## Learn more

The [documentation](https://reefinfra.ai/docs/) is organized in the following order:

- [Quickstart](https://reefinfra.ai/docs/getting-started/quickstart/): install Reef, connect a client, and inspect the version history
- [HTTP API](https://reefinfra.ai/docs/reference/http-api/): use the HTTP API and report feedback
- [Write a recipe](https://reefinfra.ai/docs/developer-guide/write-a-recipe/): configure how Reef processes data and produces updates
- [Evolve your harness](https://reefinfra.ai/docs/user-guide/evolve-your-harness/): evolve a harness instead of model weights
- [Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/): configure and operate a training deployment
- [Recipes](https://reefinfra.ai/docs/user-guide/recipes/): additional references on
  the cookbook implementations in this repository
- [Architecture](https://reefinfra.ai/docs/getting-started/architecture/): Overall architecture of Reef
- [Glossary](https://reefinfra.ai/docs/reference/glossary/): Explanation of the terminologies used

## Contribute

- Start with the [contribution guide](CONTRIBUTING.md)
- Write RFCs directly in [GitHub issues](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml) for design discussions.
- Report suspected vulnerabilities privately by following the [security policy](SECURITY.md).
