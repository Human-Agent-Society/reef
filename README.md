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

Reef is the first open-source infrastructure for continual self-improving agents.
It connects agent inference, feedback, learning, and versioned delivery. Use it
to train model weights with Slime and SGLang, or improve an agent's harness, including its prompts, rules, and skills.


</div>

**[Get started](https://reefinfra.ai/docs/getting-started/quickstart/) |
[Roadmap](https://github.com/Human-Agent-Society/reef/issues/25) |
[Launch post](https://x.com/ao_qu18465/status/2094867930081337730) |
[Join Discord](https://discord.gg/5y8e5f937k) |
[Join WeChat Group](docs/community/wechat.md)**

</div>


## When to use Reef

Use Reef when you want your agent to keep improving simply by learning from how you interact with your agent.

| Your goal | Learning path | What you need |
|---|---|---|
| Keep getting stronger model designed for you | Model weight training | A trainable model, a supported GPU stack, and feedback your recipe can use |
| Get your harness to self-improve | Harness optimization | A model endpoint, representative tasks, and an evaluator; no local training GPUs |
| Scientific discoveries | Test-time training | An execution environment, a correctness checker, and a measurable objective |


## How Reef fits your stack

| Ability | Inference engine (vLLM, SGLang, …) | RL training framework (Slime, veRL, AReaL, …) | **Reef** |
|---|:---:|:---:|:---:|
| Serves live traffic | ✅ | ❌ | ✅ |
| Trains weights | ❌ | ✅ | ✅ |
| Version management | ❌ | ❌ | ✅ |
| Stays live through updates | ❌ | ❌ | ✅ |
| Evolves beyond weights (skills, harness) | ❌ | ❌ | ✅ |


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
| **2&nbsp;·&nbsp;Observe** | Match feedback to recorded interactions. | [`storage/records.py`](reef/storage/records.py) — stored interactions and feedback<br>[`train/processors/`](reef/train/processors) — feedback matching and eligibility |
| **3&nbsp;·&nbsp;Grow** | Produce an update from eligible records. | [`recipe/`](reef/recipe) — recipe integration<br>[`train/`](reef/train) — batches and update jobs |
| **4&nbsp;·&nbsp;Commit** | Apply the configured selection policy and publish accepted updates. | [`train/evaluation/`](reef/train/evaluation) — candidate evaluation<br>[`artifact/`](reef/artifact) — version history<br>[`surface/`](reef/surface) — artifact delivery |


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


## Using Reef

Reef supports two learning surfaces: model **weights** and agent **harnesses**.
The deployment's recipe determines which surface its scenarios update.

As a minimal example, start Reef as a pure inference server:

```bash
uv run reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
```

### Weight-training deployment

#### Start the deployment

The following example starts the SAO (arXiv:2607.07508) example deployment. Run it
from a Reef checkout in an environment that satisfies the GPU requirements in
[Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/).

```bash
uv pip install -e ".[slime]" && uv pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
export REEF_TOKEN="reef-local"

reef serve -c recipes/sao/examples/sao/serve.yaml \
  --inference.model-path "$MODEL_PATH" \
  --reef.port "8900"

curl -f http://127.0.0.1:8900/healthz          # ready to serve
```

#### Send an inference request and report feedback

Send inference requests through Reef and report a score for each response. The
SAO recipe uses each eligible scored rollout to run a training step.

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

# Send a provider-compatible inference request
response = reef.post(
    "/v1/chat/completions",
    json={
        "model": os.environ["MODEL_PATH"],
        "messages": [{"role": "user", "content": "Return exactly: reef is ready"}],
    },
)

response.raise_for_status()
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

### Harness-evolving deployment

Improve harness skills using a model API instead of GPUs.

The harness evolve recipe carries its own profile; specify the provider URL
and model. From your Reef checkout and activated Python environment:

```bash
reef serve --recipe harness-evolve \
  --inference.upstream-url http://127.0.0.1:11434 \
  --inference.upstream-model gemma4:26b
```

The example connects to a local Ollama server. For another provider, change
`--inference.upstream-url` and `--inference.upstream-model`, and set
`REEF_UPSTREAM_API_KEY` if authentication is required. The profile listens on
`127.0.0.1:8900` with no token and keeps its state under `.reef/harness-evolve/`. To change anything
else, copy [the profile](reef/service/profiles/harness-evolve.yaml) and pass
your copy with `-c`.

In another terminal with the same Python environment activated (the install
bakes that terminal's `python3` into `reef-pi`), install the harness and run a task:

```bash
curl -fsS 'http://127.0.0.1:8900/reef/harness/install?adapter=pi' | bash
reef-pi -p "fix the failing test in auth.py"

# After running your tests, report the actual result:
reef-pi report --score 0 --feedback "missed the empty-token case"
```

To change the model, restart `reef serve` with another `--inference.upstream-model`
and rerun the install command before `reef-pi`: installation writes the model ID into the
local harness configuration.

Failed reports trigger a candidate skill update. Reef evaluates it against the
current harness on the tutorial's three coding tasks and publishes it only if
it wins. See the [tutorial](tutorials/evolve-your-harness/README.md) to customize the
tasks and evaluation.

To ask for a harness change in plain words and see the whole path from the ask to the install, run the [Reefine tutorial](tutorials/reefine/README.md).

Reefine ships with `reef-infra`: start it with `reef serve --recipe reefine --model ollama/gemma4:26b`.

## Recipes and examples

Pick a recipe by the **task type** of your workload and by **what it should
evolve**: model weights or the agent harness. Recipes that evolve model
weights need the GPU training stack; harness recipes need only a model
endpoint. Each recipe below links to its guide, and each measured benchmark
links to its results page; the [recipe catalog](https://reefinfra.ai/docs/user-guide/recipes/)
adds the code and example for every recipe. Reefine ships with `reef-infra`;
the other implementations live in this repository's `recipes/` cookbook, are
selected by dotted class reference, and do not ship in the Reef wheel.

| Task type | Task shape | Evolves the model | Evolves the harness | Standard benchmarks |
|---|---|---|---|---|
| Scientific discovery | One hard problem, repeated attempts, a measurable objective | [TTT-Discover](https://reefinfra.ai/docs/user-guide/recipes/tttd/), [Guidance-TTT](recipes/tttd/examples/guidance_ttt/README.md) | None yet | Measured: [TriMul](recipes/tttd/examples/guidance_ttt/results/README.md), [circle packing (n = 26, 32)](recipes/tttd/examples/tttd/README.md#formal-8x64-results), [Erdős minimum overlap](recipes/tttd/examples/tttd/README.md#formal-8x64-results). |
| Continual learning on a task stream | A stream of independent tasks, each scored by a verifier | [SAO](https://reefinfra.ai/docs/user-guide/recipes/sao/) | [Meta-Harness](recipes/meta_harness/README.md), [GEPA](https://reefinfra.ai/docs/user-guide/recipes/gepa/), [Harness evolve](https://reefinfra.ai/docs/user-guide/evolve-your-harness/), [Reefine](docs/user-guide/recipes/reefine.rst) | Measured: [AIME 2025](recipes/gepa/examples/aime/README.md), [IMOAnswerBench](recipes/sao/examples/sao/README.md#results), [Terminal-Bench 30-task subset](recipes/meta_harness/RESULTS.md). Proposed: CEO-Bench |
| Learning from usage | Real interaction with no explicit score, or delayed feedback | [OpenClaw-RL](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) | [SkillClaw](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) | Measured: the [OpenClaw-RL simulated student](recipes/openclawrl/examples/openclawrl/README.md#results). |

[`recipes/basic/`](recipes/basic/) is the record-only starting stack and stays
outside the catalog. For a small walkthrough of feedback, candidate edits, and
publication, start with [the coding harness tutorial](tutorials/evolve-your-harness/README.md).
Each result page documents its task, evaluation setup, measurements, and
limitations.


## Architecture

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img src="docs/assets/architecture-light.svg" alt="Reef architecture: harness requests flow through a scenario to inference. Receipt-linked feedback feeds records and recipe training; artifact evaluation selects updates for versioned publication. Rejected candidates leave the current release serving." width="1200">
</picture>
</div>

## Learn more

The [documentation](https://reefinfra.ai/docs/) is organized in the following order:

- [Quickstart](https://reefinfra.ai/docs/getting-started/quickstart/): install Reef, connect a client, and inspect the version history
- [HTTP API](https://reefinfra.ai/docs/reference/http-api/): use the HTTP API and report feedback
- [Write a recipe](https://reefinfra.ai/docs/developer-guide/write-a-recipe/): configure how Reef processes data and produces updates
- [Evolve your harness](https://reefinfra.ai/docs/user-guide/evolve-your-harness/): evolve a harness instead of model weights
- [Evolve your model](https://reefinfra.ai/docs/user-guide/evolve-your-model/): configure and operate a training deployment
- [Recipes](https://reefinfra.ai/docs/user-guide/recipes/): the catalog of cookbook
  recipes by task type, with the benchmarks each has measured
- [The core loop](https://reefinfra.ai/docs/getting-started/core-loop/): The core loop of Reef
- [Glossary](https://reefinfra.ai/docs/reference/glossary/): Explanation of the terminologies used

## Community & Contributing

Working on continual self-improving agent?

- [Join Discord](https://discord.gg/5y8e5f937k) to share your recipes, ask implementation questions, and discuss new features.
- [Join the WeChat group](docs/community/wechat.md): the group is full, so add the assistant and it will invite you.
- Join the [GitHub Discussions](https://github.com/orgs/Human-Agent-Society/discussions) to ask questions, share ideas, and connect with the community.
- Start contributing with the [contribution guide](CONTRIBUTING.md).
- Propose designs through an [RFC issue](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml).
- Report suspected vulnerabilities privately by following the [security policy](SECURITY.md).

If Reef looks useful to you, please give it a ⭐ — it helps the community to discover and contribute to the project.


## The Team

Reef brings together people exploring how agents can learn from experience and
improve over time. The people below help turn that idea into working infrastructure.

This list is non-exhaustive, with team members listed alphabetically by last name:

[Wenhao Chai](https://github.com/wenhaochai),
[Shuangrui Ding](https://github.com/Mark12Ding),
[Hao He](https://github.com/hehaodele),
[Haoze He](https://github.com/HectorHHZ),
[Chonghe Jiang](https://github.com/Chonghe-Jiang),
[Nan Jiang](https://github.com/nanjiangwill),
[Xuan Jiang](https://github.com/Xuan-1998),
[Xiaochen Li](https://github.com/SeuperHakkerJa),
[Paul Liang](https://github.com/pliang279),
[Bo Liu](https://github.com/Benjamin-eecs),
[Boyuan Long](https://github.com/BoyuanLong),
[Qiuyang Mang](https://github.com/joyemang33),
[Zhenting Qi](https://github.com/zhentingqi),
[Ao Qu](https://github.com/quao627),
[Mingruo Qu](https://github.com/workhardforcoding),
[Zhaokai Wang](https://github.com/wzk1015),
[Xuezhi Yan](https://github.com/yanxz),
[Hanfei Yu](https://github.com/hanfeiyu),
[Haofei Yu](https://github.com/lwaekfjlk),
[Simon Yu](https://github.com/simonucl),
[Han Zheng](https://github.com/MikeZheng777),
[Kaichen Zhou](https://github.com/kaichen-z),
[Zijian Zhou](https://github.com/BobbyZhouZijian),
[Jiacheng Zhu](https://github.com/Jiacheng-Zhu-AIML),
[Dingyi Zhuang](https://github.com/ZhuangDingyi).


## Star History

<a href="https://star-history.com/#Human-Agent-Society/reef&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Human-Agent-Society/reef&type=date&legend=top-left&sealed_token=z8QelisjJA7wNSk0E_tcfZ8YzFIYY9czZQTvqRy51kdbOVVAvadCE0iKIhrM6qPqkxdDrdRUQOLxKLlazXbTU8-l5Oxj-pYCcAF-d2erPCw3RjKZ5dJXBFd2bgPhBu65TZVZxZReP9lznlTpnGvAynSWUsO1CjapS8nXUqALToFUAHraMIapsjhfWECk&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Human-Agent-Society/reef&type=date&legend=top-left&sealed_token=z8QelisjJA7wNSk0E_tcfZ8YzFIYY9czZQTvqRy51kdbOVVAvadCE0iKIhrM6qPqkxdDrdRUQOLxKLlazXbTU8-l5Oxj-pYCcAF-d2erPCw3RjKZ5dJXBFd2bgPhBu65TZVZxZReP9lznlTpnGvAynSWUsO1CjapS8nXUqALToFUAHraMIapsjhfWECk" />
    <img alt="Reef Star History Chart" src="https://api.star-history.com/chart?repos=Human-Agent-Society/reef&type=date&legend=top-left&sealed_token=z8QelisjJA7wNSk0E_tcfZ8YzFIYY9czZQTvqRy51kdbOVVAvadCE0iKIhrM6qPqkxdDrdRUQOLxKLlazXbTU8-l5Oxj-pYCcAF-d2erPCw3RjKZ5dJXBFd2bgPhBu65TZVZxZReP9lznlTpnGvAynSWUsO1CjapS8nXUqALToFUAHraMIapsjhfWECk" />
  </picture>
</a>


## Acknowledgements

We are particularly grateful to these projects which power important parts of Reef:

- [SGLang](https://github.com/sgl-project/sglang) — high-performance inference
- [slime](https://github.com/THUDM/slime) — model weight training
- [cordis](https://github.com/cordiverse/cordis) — harness evolution
