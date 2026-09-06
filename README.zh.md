<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reef-logo-dark.svg">
  <img src="docs/assets/reef-logo-light.svg" alt="Reef" width="220">
</picture>

<h3>面向自我进化 Agent 的持续学习基础设施</h3>

[![CI](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml)
[![PyPI package: reef-infra](https://img.shields.io/pypi/v/reef-infra?label=PyPI%3A%20reef-infra&logo=pypi&logoColor=white)](https://pypi.org/project/reef-infra/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 中文

<div align="left">

Reef 是首个面向持续自我进化 Agent 的开源基础设施。它连接 Agent 推理、反馈、学习与
版本化交付。你可以用它配合 Slime 和 SGLang 训练模型权重，也可以改进 Agent 的
harness，包括提示词、规则和技能。


</div>

**[快速上手](https://reefinfra.ai/docs/getting-started/quickstart/) |
[路线图](https://github.com/Human-Agent-Society/reef/issues/25) |
[发布文章](https://x.com/ao_qu18465/status/2094867930081337730) |
[加入 Discord](https://discord.gg/5y8e5f937k)**

</div>


## 何时使用 Reef

如果你希望 Agent 通过与你的日常交互不断学习、持续进化，就适合使用 Reef。

| 你的目标 | 学习路径 | 所需条件 |
|---|---|---|
| 持续获得更贴合自身需求的强大模型 | 模型权重训练 | 可训练模型、受支持的 GPU 栈，以及 recipe 可利用的反馈 |
| 让 harness 自我进化 | Harness 优化 | 模型端点、有代表性的任务和评估器；无需本地训练 GPU |
| 进行科学发现 | 测试时训练 | 执行环境、正确性检查器和可度量的目标 |


## Reef 在技术栈中的位置

| 能力 | 推理引擎（vLLM、SGLang…） | RL 训练框架（Slime、veRL、AReaL…） | **Reef** |
|---|:---:|:---:|:---:|
| 承接线上流量 | ✅ | ❌ | ✅ |
| 训练权重 | ❌ | ✅ | ✅ |
| 版本管理 | ❌ | ❌ | ✅ |
| 更新期间持续服务 | ❌ | ❌ | ✅ |
| 可进化权重以外的部分（技能、harness） | ❌ | ❌ | ✅ |


## 工作原理

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loop-animation-dark.svg">
  <img src="docs/assets/loop-animation-light.svg" alt="Reef 响应请求、记录反馈、产出更新，并将通过的更新提交至版本历史。" width="76%">
</picture>
</div>

Reef 的每个学习周期分为四步，下表同时列出各步骤对应的模块。

| 步骤 | 说明 | 对应模块 |
|---|---|---|
| **1&nbsp;·&nbsp;Serve** | 响应 Agent 请求，记录每次交互。 | [`service/`](reef/service) — Agent 请求与交互记录<br>[`runtime/`](reef/runtime) — 推理与 artifact 更新 |
| **2&nbsp;·&nbsp;Observe** | 将反馈匹配到已记录的交互。 | [`records.py`](reef/records.py) — 已存储的交互与反馈<br>[`train/processors/`](reef/train/processors) — 反馈匹配与条件判定 |
| **3&nbsp;·&nbsp;Grow** | 从符合条件的记录中产出一次更新。 | [`recipe/`](reef/recipe) — recipe 接入<br>[`train/`](reef/train) — 批次与更新任务 |
| **4&nbsp;·&nbsp;Commit** | 应用配置的选择策略并发布通过的更新。 | [`train/evaluation/`](reef/train/evaluation) — 候选评估<br>[`artifact/`](reef/artifact) — 版本历史<br>[`surface/`](reef/surface) — artifact 分发 |


## 安装

> 💡 **注意**
>
> Reef 的 artifact 和 checkpoint 功能依赖系统的 `git-lfs` 包。Reef 会在本地为自身的
> artifact 仓库初始化 Git LFS。

推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖，下文命令均基于 uv。

### 通过 PyPI 安装

```bash
uv venv && source .venv/bin/activate
uv pip install reef-infra
python3 -c "import reef; print(reef.__version__)"
```

### 通过源码安装

```bash
git lfs install
git clone https://github.com/Human-Agent-Society/reef.git
cd reef
uv venv && source .venv/bin/activate
uv pip install -e .
python3 -c "import reef; print(reef.__version__)"
```

开发或运行下文的训练示例时，请使用源码安装。


## 使用 Reef

Reef 支持两类学习载体：模型**权重**和 Agent 的 **harness**。每个部署使用的 recipe
决定其 scenario 更新哪一种载体。

### 模型权重训练部署

#### 启动部署

下面的示例启动 SAO（arXiv:2607.07508）示例部署。请在 Reef 源码目录下运行，并确保
运行环境满足[进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)中的 GPU 要求。

```bash
uv pip install -e ".[slime]" && uv pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
export REEF_TOKEN="reef-local"

reef serve -c recipes/sao/examples/sao/serve.yaml \
  --reef.model_path "$MODEL_PATH" \
  --reef.port "8900"

curl -f http://127.0.0.1:8900/healthz          # ready to serve
```

#### 发送推理请求并上报反馈

将推理请求发送至 Reef，并为每个响应上报分数。SAO recipe 使用每条符合条件的带分
rollout 执行一次训练。

Reef 的推理端点兼容 OpenAI 和 Anthropic：`/v1/chat/completions` 与 `/v1/messages`
直接接收相应模型提供商的请求体。请求需包含 `x-reef-scenario` 请求头；新的名称会使用
部署配置的 recipe 创建 scenario。请求本身不选择 recipe。

响应体使用模型提供商的 OpenAI 兼容格式。Reef 会添加 `x-reef-agent-record-id` 响应头，
其值是后续报告用来标识本次交互的**回执**。报告可以包含数值 `score`、文本或结构化
`feedback`，以及它所评估的回执。下面的示例同时上报分数和简短说明。

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

部分 recipe 需要的不止一个分数，`feedback` 用于承载更丰富的信号，可以是纯文本或
结构化对象。端点会校验**上报 schema**（[`reef/core/reports/`](reef/core/reports)）。


#### 观察学习与进化

反馈积累到一定数量后，recipe 会执行一次训练，并将更新后的权重同步至推理运行时。
后续推理请求直接使用当前版本，无需重启 Reef。

### Harness 进化部署

使用模型 API 改进 harness 技能，无需 GPU。将
[deployment.yaml](tutorials/evolve-your-harness/configs/deployment.yaml) 中的 `model.path` 设置为你的
模型名称，然后在 Reef 源码目录及已激活的 Python 环境中运行：

```bash
export REEF_UPSTREAM_URL="https://api.openai.com"  # No /v1 suffix
export REEF_UPSTREAM_API_KEY="your-openai-api-key"
reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml
```

如使用其他模型提供商，请填写对应的基础 URL、模型名称和 API key。此配置在 `8901`
端口部署 Reef，并使用 `reef-local` 作为访问 token。

在另一个终端中安装 harness 并运行任务：

```bash
export REEF_TOKEN="reef-local"   # the script writes it into the installed harness's
                                 # model binding; keep it exported for `report`
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8901/reef/harness/install?adapter=pi' | bash
reef-pi -p "fix the failing test in auth.py"

# After running your tests, report the actual result:
reef-pi report --score 0 --feedback "missed the empty-token case"
```

失败报告会触发候选技能更新。Reef 会在教程的三个编程任务上对候选技能和当前 harness
进行评估，仅在候选胜出时才发布。如何自定义任务和评估方式，请参阅
[教程](tutorials/evolve-your-harness/README.md)。


## Recipes 与示例

请根据工作负载可提供的反馈和需要更新的 artifact 选择 recipe。这些实现位于本仓库的
`recipes/` cookbook 中，通过带点号的类路径指定，不随 Reef wheel 发布。

| 工作负载 | Recipe 指南 | 更新的 artifact | 示例与结果 |
|---|---|---|---|
| 由测试或校验器打分的任务流 | [SAO](https://reefinfra.ai/docs/user-guide/recipes/sao/) | 模型权重 | [示例](recipes/sao/examples/sao/README.md) · [结果](recipes/sao/examples/sao/README.md#results) |
| 具备可用的下一状态信号、但无显式上报的 Agent 流量 | [OpenClaw-RL](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) | 模型权重 | [示例](recipes/openclawrl/examples/openclawrl/README.md) |
| 对同一问题的多次带分尝试 | [TTT-Discover](https://reefinfra.ai/docs/user-guide/recipes/tttd/) | 模型权重 | [示例](recipes/tttd/examples/tttd/README.md) · [结果](recipes/tttd/examples/tttd/README.md#formal-8x64-results) |
| 带分数的代码搜索：引导模型可训练，执行器冻结 | [Guidance-TTT / TTTD](https://reefinfra.ai/docs/user-guide/recipes/tttd/) | 引导模型权重 | [示例](recipes/tttd/examples/guidance_ttt/README.md) · [结果](recipes/tttd/examples/guidance_ttt/results/README.md) |
| 使用 Agent 反馈进化其技能池 | [SkillClaw](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) | Harness 技能；无需训练 GPU | [示例](recipes/skillclaw/README.md) |
| 使用分数和交互记录改进提示词与指令 | [GEPA](https://reefinfra.ai/docs/user-guide/recipes/gepa/) | Harness；模型权重不变 | [示例与结果](recipes/gepa/examples/aime/README.md) |

如果想快速了解反馈、候选修改和发布流程，可以从[编程 harness
教程](tutorials/evolve-your-harness/README.md)开始。每个结果页面都会说明任务、评估设置、
测量结果和局限性。


## 架构

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

请求路径、scenario 和 release 生命周期详见[架构指南](https://reefinfra.ai/docs/getting-started/architecture/)。

## 进一步了解

[文档](https://reefinfra.ai/docs/)按以下顺序组织：

- [快速上手](https://reefinfra.ai/docs/getting-started/quickstart/)：安装 Reef，接入客户端，查看版本历史
- [HTTP API](https://reefinfra.ai/docs/reference/http-api/)：使用 HTTP API 并上报反馈
- [编写 recipe](https://reefinfra.ai/docs/developer-guide/write-a-recipe/)：配置 Reef 如何处理数据、产出更新
- [进化你的 harness](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)：不训练权重，改进 harness
- [进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)：配置并运维训练部署
- [Recipes](https://reefinfra.ai/docs/user-guide/recipes/)：本仓库 cookbook 实现的进一步说明
- [架构](https://reefinfra.ai/docs/getting-started/architecture/)：Reef 的整体架构
- [术语表](https://reefinfra.ai/docs/reference/glossary/)：文档所用术语的解释

## 社区与贡献

你是否也在研究持续自我进化的 Agent？

- 加入 [Discord](https://discord.gg/5y8e5f937k)，分享 recipe、交流实现细节、讨论新功能。
- 在 [GitHub Discussions](https://github.com/orgs/Human-Agent-Society/discussions) 提问、分享想法、与社区交流。
- 参与开发请从[贡献指南](CONTRIBUTING.md)开始。
- 设计方案请通过 [RFC issue](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml) 提出。
- 发现疑似漏洞请按[安全策略](SECURITY.md)私下反馈。

如果 Reef 对你有帮助，欢迎点个 Star ⭐，让更多人发现并参与进来。


## 团队

Reef 汇聚了一群探索 Agent 如何从经验中学习、持续进化的人。以下成员共同将这一想法
变成可用的基础设施。

这份名单并未列尽所有团队成员，以下按姓氏字母顺序排列：

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
    <img alt="Reef Star 增长历史图" src="https://api.star-history.com/chart?repos=Human-Agent-Society/reef&type=date&legend=top-left&sealed_token=z8QelisjJA7wNSk0E_tcfZ8YzFIYY9czZQTvqRy51kdbOVVAvadCE0iKIhrM6qPqkxdDrdRUQOLxKLlazXbTU8-l5Oxj-pYCcAF-d2erPCw3RjKZ5dJXBFd2bgPhBu65TZVZxZReP9lznlTpnGvAynSWUsO1CjapS8nXUqALToFUAHraMIapsjhfWECk" />
  </picture>
</a>


## 致谢

以下项目支撑了 Reef 的关键部分，在此感谢：

- [SGLang](https://github.com/sgl-project/sglang) — 高性能推理
- [slime](https://github.com/THUDM/slime) — 模型权重训练
- [cordis](https://github.com/cordiverse/cordis) — harness 进化
