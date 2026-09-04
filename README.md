<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reef-logo-dark.svg">
  <img src="docs/assets/reef-logo-light.svg" alt="Reef" width="220">
</picture>

<h3>Continual learning infra for self-improving agents</h3>

[![CI](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml)
[![PyPI package: reef-infra](https://img.shields.io/pypi/v/reef-infra?label=PyPI%3A%20reef-infra&logo=pypi&logoColor=white)](https://pypi.org/project/reef-infra/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

English | [中文](README.zh.md)

<div align="left">

Reef is infrastructure that serves an entire continual learning backend. Reef
exposes standardized http endpoints so that you can download agents just like
how you download `codex` or `opencode` using `curl`, and so that your agent can
send its model requests to Reef's inference endpoint instead of the provider's.

The only difference is that, Reef constantly evaluates your agent behavior
and improves the served harness and model weights in the backend. You keep getting
better and better results without having to do anything.

</div>

**[Get started](https://reefinfra.ai/docs/getting-started/quickstart/) |
[Roadmap](https://github.com/Human-Agent-Society/reef/issues/25) |
[Launch post](https://x.com/ao_qu18465/status/2094867930081337730) |
[Join Discord](https://discord.gg/5y8e5f937k)**

</div>


## Installation

> 💡 **Note**
>
> Reef's artifact and checkpoint functionality requires the `git-lfs` system
> package. Reef initializes Git LFS locally for its artifact repositories.

We recommend [uv](https://docs.astral.sh/uv/) for managing packages, and the
commands below use it.

### From PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install reef-infra
python3 -c "import reef; print(reef.__version__)"
```

### From source

```bash
git lfs install
git clone https://github.com/Human-Agent-Society/reef.git
cd reef
uv venv && source .venv/bin/activate
uv pip install -e .
python3 -c "import reef; print(reef.__version__)"
```

Use the source checkout for development and for the training examples below.


## How it works

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loop-animation-dark.svg">
  <img src="docs/assets/loop-animation-light.svg" alt="Reef serves requests, records feedback, produces updates, and commits accepted updates to a version history." width="76%">
</picture>
</div>

Reef processes each learning cycle in four steps. The table also shows which
modules implement each step.

| Step | What happens | Where it lives |
|---|---|---|
| **1&nbsp;·&nbsp;Serve** | Serve agent requests and record interactions. | [`service/`](reef/service) — agent requests and interaction records<br>[`runtime/`](reef/runtime) — inference and artifact updates |
| **2&nbsp;·&nbsp;Observe** | Match feedback to recorded interactions. | [`records.py`](reef/records.py) — stored interactions and feedback<br>[`train/processors/`](reef/train/processors) — feedback matching and eligibility |
| **3&nbsp;·&nbsp;Grow** | Produce an update from eligible records. | [`recipe/`](reef/recipe) — recipe integration<br>[`train/`](reef/train) — batches and update jobs |
| **4&nbsp;·&nbsp;Commit** | Evaluate and publish accepted updates. | [`train/evaluation/`](reef/train/evaluation) — candidate evaluation<br>[`artifact/`](reef/artifact) — version history<br>[`surface/`](reef/surface) — artifact delivery |


## Using Reef

Reef supports two learning surfaces: model **weights** and agent **harnesses**.
The deployment's recipe determines which surface its scenarios update.

### 1 · Serve

The following example starts the SAO (arXiv:2607.07508) example deployment. Run it
from a Reef checkout in an environment that satisfies the GPU requirements in
[Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/).

```bash
uv pip install -e ".[slime]" && uv pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
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
and `/v1/messages` take the provider's own request body. A request includes the
`x-reef-scenario` header; a new name creates a scenario using the deployment's
configured recipe. Requests do not select recipes.

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

| Workload | Recipe module | Updated artifact | Documentation |
|---|---|---|---|
| A stream of tasks scored by tests or a verifier | <code>recipes.sao.recipe:SAORecipe</code> | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/sao/) · [Example](recipes/sao/examples/sao/README.md) |
| Agent traffic with useful next-state signals and no explicit reports | <code>recipes.openclawrl.recipe:OpenClawRLRecipe</code> | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) · [Example](recipes/openclawrl/examples/openclawrl/README.md) |
| Repeated, scored attempts at one problem | <code>recipes.tttd.recipe:TTTDRecipe</code> | Model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [Example](recipes/tttd/examples/tttd/README.md) |
| Scored code search with a trainable guidance model and a frozen executor | <code>recipes.tttd.recipe:TTTDRecipe</code> | Guidance-model weights | [Guide](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [Example](recipes/tttd/examples/guidance_ttt/README.md) |
| Agent feedback used to evolve its skill pool | <code>recipes.skillclaw.recipe:SkillClawRecipe</code> | Skill pool (harness tree); no GPU required | [Guide](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) · [Example](recipes/skillclaw/README.md) |


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

## Community & Contributing

Working on continual self-improving agent?

- [Join Discord](https://discord.gg/5y8e5f937k) to share your recipes, ask implementation questions, and discuss new features.
- Join the [GitHub Discussions](https://github.com/orgs/Human-Agent-Society/discussions) to ask questions, share ideas, and connect with the community.
- Start contributing with the [contribution guide](CONTRIBUTING.md).
- Propose designs through an [RFC issue](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml).
- Report suspected vulnerabilities privately by following the [security policy](SECURITY.md).

If Reef looks useful to you, please give it a ⭐ — it helps the community to discover and contribute to the project.


## The Team

Reef is built by the following team members, listed in alphabetical order by last name:

[Wenhao Chai](https://github.com/wenhaochai),
[Shuangrui Ding](https://github.com/Mark12Ding),
Hao He,
Haoze He,
[Chonhe Jiang](https://github.com/Chonghe-Jiang),
[Nan Jiang](https://github.com/nanjiangwill),
Xuan Jiang,
[Xiaochen Li](https://github.com/SeuperHakkerJa),
[Paul Liang](https://github.com/pliang279),
[Bo Liu](https://github.com/Benjamin-eecs),
Boyuan Long,
[Qiuyang Mang](https://github.com/joyemang33),
[Zhenting Qi](https://github.com/zhentingqi),
[Ao Qu](https://github.com/quao627),
Mingruo Qu,
Zhaokai Wang,
Xuezhi Yan,
[Hanfei Yu](https://github.com/hanfeiyu),
[Haofei Yu](https://github.com/lwaekfjlk),
[Simon Yu](https://github.com/simonucl),
[Han Zheng](https://github.com/MikeZheng777),
[Kaichen Zhou](https://github.com/kaichen-z),
[Zijian Zhou](https://github.com/BobbyZhouZijian),
[Jiacheng Zhu](https://github.com/Jiacheng-Zhu-AIML),
Dingyi Zhuang.


## Acknowledgements

We are particularly grateful to these projects which power important parts of Reef:

- [SGLang](https://github.com/sgl-project/sglang) — high-performance inference
- [slime](https://github.com/THUDM/slime) — model weight training
- [cordis](https://github.com/cordiverse/cordis) — harness evolution
