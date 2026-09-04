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
[Join Discord](https://discord.gg/5y8e5f937k)**

</div>


## When to use Reef

Use Reef when you want your agent to keep improving simply by learning from how you interact with your agent.

| Your goal | Learning path | What you need |
|---|---|---|
| Keep getting stronger model designed for you | Model weight training | A trainable model, a supported GPU stack, and feedback your recipe can use |
| Get your harness to self-improve | Harness optimization | A model endpoint, representative tasks, and an evaluator; no local training GPUs |
| Scientific discoveries | Test-time training | An execution environment, a correctness checker, and a measurable objective |


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

### 1 · Start a weight-training deployment

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

### 3 · Evolve your harness

The `harness_evolve` recipe updates a harness tree that may contain rules,
skills, configuration, prompts, and extensions. It builds a candidate from
reported interactions, evaluates the current and candidate harnesses on the
configured tasks, and publishes the candidate only when it wins that
comparison. Harness scenarios do not share data or versions.

#### Install a versioned harness

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


## Recipes and examples

Choose a recipe based on your workload's feedback and the artifact you want to
update. The implementations live in this repository's `recipes/` cookbook,
are selected by dotted class reference, and do not ship in the Reef wheel.

| Workload | Recipe guide | Updated artifact | Examples and results |
|---|---|---|---|
| A stream of tasks scored by tests or a verifier | [SAO](https://reefinfra.ai/docs/user-guide/recipes/sao/) | Model weights | [Example](recipes/sao/examples/sao/README.md) · [Results](recipes/sao/examples/sao/README.md#results) |
| Agent traffic with useful next-state signals and no explicit reports | [OpenClaw-RL](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) | Model weights | [Example](recipes/openclawrl/examples/openclawrl/README.md) |
| Repeated, scored attempts at one problem | [TTT-Discover](https://reefinfra.ai/docs/user-guide/recipes/tttd/) | Model weights | [Example](recipes/tttd/examples/tttd/README.md) · [Results](recipes/tttd/examples/tttd/README.md#formal-8x64-results) |
| Scored code search with a trainable guidance model and a frozen executor | [Guidance-TTT / TTTD](https://reefinfra.ai/docs/user-guide/recipes/tttd/) | Guidance-model weights | [Example](recipes/tttd/examples/guidance_ttt/README.md) · [Results](recipes/tttd/examples/guidance_ttt/results/README.md) |
| Agent feedback used to evolve its skill pool | [SkillClaw](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) | Harness skills; no training GPUs | [Example](recipes/skillclaw/README.md) |
| Scores and transcripts used to improve prompts and instructions | [GEPA](https://reefinfra.ai/docs/user-guide/recipes/gepa/) | Harness; fixed model weights | [Example and results](recipes/gepa/examples/aime/README.md) |

For a small walkthrough of feedback, candidate edits, and publication, start with
[the coding harness tutorial](tutorials/harness_evolve/README.md). Each result
page documents its task, evaluation setup, measurements, and limitations.


## Architecture

```mermaid
sequenceDiagram
    accTitle: How Reef serves, records, trains, evaluates, and publishes
    autonumber
    participant H as Harness
    participant S as Scenario
    participant I as Inference
    participant T as Trainer
    participant G as Training*
    participant E as Artifact evaluation

    opt Harness recipe: pull the served tree
      H->>S: GET /reef/harness for scenario
      S-->>H: Harness tree and release
      Note over H: Agent runs on that tree
    end
    Note over H,I: Serve and record each request
    H->>S: Inference request for scenario
    S->>S: Freeze current release
    S->>I: Provider-native request
    I-->>S: Provider response
    S->>S: Validate frozen release and store record
    S-->>H: Response and receipt
    H->>S: Feedback quotes the receipt
    S->>T: Eligible record
    opt Processor has a batch
      Note over S,E: Produce, evaluate, and select a candidate
      T->>G: Prepared step
      G-->>T: Candidate artifact ready
      T->>E: evaluate(candidate)
      E-->>T: Evaluation result
      T->>E: decide(candidate, result)
      E-->>T: Select or reject
      alt Candidate selected
        T->>S: Commit new release
      else Candidate rejected
        Note over S,I: Previous release keeps serving
      end
    end
```

The inference runtime is always required. The training runtime is needed only for
recipes that update model weights. Harness recipes run their step in Reef’s own
process, with no training GPU.

Artifact evaluation measures each candidate artifact and decides whether to
select or reject it before publication. The default weight-recipe selector is
`AlwaysSelect`; configure a custom evaluator to apply your quality criteria.

See the [architecture guide](https://reefinfra.ai/docs/getting-started/architecture/)
for the request path, scenarios, and release lifecycle.

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
