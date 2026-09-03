<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reef-logo-dark.svg">
  <img src="docs/assets/reef-logo-light.svg" alt="Reef" width="220">
</picture>

<h3>面向自我进化智能体的持续学习基础设施</h3>

[![CI](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml)
[![PyPI package: reef-infra](https://img.shields.io/pypi/v/reef-infra?label=PyPI%3A%20reef-infra&logo=pypi&logoColor=white)](https://pypi.org/project/reef-infra/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 简体中文

<div align="left">

Reef 是一套完整的持续学习后端基础设施。Reef 提供标准化的 HTTP 端点，让你可以像
用 `curl` 下载 `codex` 或 `opencode` 那样下载智能体，也让你的智能体把模型请求发送到
Reef 的推理端点，而不是直接发给模型提供方。

唯一的区别在于，Reef 会持续评估你的智能体行为，并在后端不断改进所服务的
harness 和模型权重。你什么都不用做，就能持续获得越来越好的结果。

</div>

**[快速上手](https://reefinfra.ai/docs/getting-started/quickstart/) |
[路线图](https://github.com/Human-Agent-Society/reef/issues/25) |
[发布博文](https://x.com/ao_qu18465/status/2094867930081337730) |
[加入 Discord](https://discord.gg/5y8e5f937k)**

</div>


## 安装

> 💡 **注意**
>
> Reef 的 artifact 与 checkpoint 功能依赖系统包 `git-lfs`。Reef 会为其 artifact
> 仓库在本地初始化 Git LFS。

我们推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖包，下面的命令均基于 uv。

### 从 PyPI 安装

```bash
uv venv && source .venv/bin/activate
uv pip install reef-infra
python3 -c "import reef; print(reef.__version__)"
```

### 从源码安装

```bash
git lfs install
git clone https://github.com/Human-Agent-Society/reef.git
cd reef
uv venv && source .venv/bin/activate
uv pip install -e .
python3 -c "import reef; print(reef.__version__)"
```

如果要进行开发，或运行下文的训练示例，请使用源码方式安装。


## 工作原理

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loop-animation-dark.svg">
  <img src="docs/assets/loop-animation-light.svg" alt="Reef 服务请求、记录反馈、产出更新，并把被采纳的更新提交到版本历史。" width="76%">
</picture>
</div>

Reef 的每个学习周期分为四步。下表同时列出了实现各步骤的模块。

| 步骤 | 具体做什么 | 代码位置 |
|---|---|---|
| **1&nbsp;·&nbsp;服务（Serve）** | 服务智能体请求并记录交互。 | [`service/`](reef/service) — 智能体请求与交互记录<br>[`runtime/`](reef/runtime) — 推理与 artifact 更新 |
| **2&nbsp;·&nbsp;观察（Observe）** | 把反馈匹配到已记录的交互。 | [`records.py`](reef/records.py) — 存储的交互与反馈<br>[`train/processors/`](reef/train/processors) — 反馈匹配与资格判定 |
| **3&nbsp;·&nbsp;生长（Grow）** | 从符合条件的记录中产出一次更新。 | [`recipe/`](reef/recipe) — recipe 接入<br>[`train/`](reef/train) — 批次与更新任务 |
| **4&nbsp;·&nbsp;提交（Commit）** | 评估并发布被采纳的更新。 | [`train/evaluation/`](reef/train/evaluation) — 候选评估<br>[`artifact/`](reef/artifact) — 版本历史<br>[`surface/`](reef/surface) — artifact 分发 |


## 使用 Reef

Reef 支持两种学习载体（learning surface）：模型**权重（weights）**和智能体
**harness**。具体由部署所使用的 recipe 决定其场景更新哪一种载体。

### 1 · 服务

下面的示例会启动 SAO（arXiv:2607.07508）示例部署。请在 Reef 源码检出目录中运行，
且运行环境需满足[进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)
中列出的 GPU 要求。

```bash
uv pip install -e ".[slime]" && uv pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
export REEF_TOKEN="reef-local"

reef serve -c recipes/sao/examples/sao/serve.yaml \
  --reef.model_path "$MODEL_PATH" \
  --reef.port "8900"

curl -f http://127.0.0.1:8900/healthz          # 服务已就绪
```

### 2 · 训练权重

把推理请求发送给 Reef，并为每个响应上报一个分数。SAO recipe 会用每一条符合条件、
带分数的 rollout 执行一次训练步。

#### 发送推理请求并上报反馈

Reef 的推理端点同时兼容 OpenAI 与 Anthropic：`/v1/chat/completions` 和 `/v1/messages`
直接接受各自提供方的请求体。请求中需带上 `x-reef-scenario` 请求头；使用一个新名字时，
会按部署所配置的 recipe 创建一个新场景（scenario）。请求本身不能选择 recipe。

响应体使用提供方的 OpenAI 兼容格式。Reef 会额外添加 `x-reef-agent-record-id` 响应头，
其值就是**回执（receipt）**，之后上报时用它来标识这次交互。一次上报可以包含数值
`score`、文本或结构化的 `feedback`，以及它所评价的回执列表。下面的示例同时上报了
分数和一句简短说明。

```python
import os
import httpx

reef = httpx.Client(
    base_url="http://127.0.0.1:8900",
    headers={"Authorization": f"Bearer {os.environ['REEF_TOKEN']}", "x-reef-scenario": "hello-reef"},
    timeout=300,
)

# 使用 OpenAI 兼容格式进行推理
response = reef.post(
    "/v1/chat/completions",
    json={
        "model": os.environ["MODEL_PATH"],
        "messages": [{"role": "user", "content": "Return exactly: reef is ready"}],
    },
)

receipt = response.headers["x-reef-agent-record-id"]
answer = response.json()["choices"][0]["message"]["content"]

# 上报本次推理的反馈
matched = answer.strip() == "reef is ready"

reef.post(
    "/reef/report",
    json={"score": float(matched), "feedback": "matched" if matched else "wrong answer", "references": [receipt]},
).raise_for_status()
```

对于需要读取标量之外信息的 recipe，`feedback` 承载了更丰富的信号，可以是纯文本，
也可以是结构化对象。该端点会校验**上报模式（report schema）**
（[`reef/core/reports/`](reef/core/reports)）。


#### 看着它学习和成长

当 recipe 积累到足够的反馈后，它会执行一次训练步，并把更新后的权重同步到服务运行时。
之后的推理请求无需重启 Reef 就会使用最新版本。

### 3 · 进化你的 harness

`harness_evolve` recipe 会更新一棵 harness 树，其中可以包含规则、技能、配置、提示词
和扩展。它基于上报的交互构建候选 harness，在配置的任务上评估当前 harness 与候选
harness，只有当候选胜出时才会发布。各 harness 场景之间不共享数据和版本。

#### 安装与你共同成长的 Reef harness

你可以像安装大多数编程智能体那样安装 Reef harness。下面是一个示例。系统会自动创建
一个新场景，并与下载的 harness 绑定。

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash

reef-pi -p "fix the bug"
```

你也可以在请求头中指定场景，来获取一个已经进化过的 harness。例如，如果你已有场景
`harness-evolve-code-repair`，可以用下面的命令安装它的 harness。

```bash
curl -fsS -H 'x-reef-scenario: harness-evolve-code-repair' \
  -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash
```

#### 上报任务结果

`reef-pi` 会保存一次运行产生的回执，因此它的 `report` 命令只需要提供你想关联到
前一次交互的结果：

```bash
reef-pi -p "fix the failing test in auth.py"

# ... 运行你的测试，给结果打分 ...

reef-pi report --score 0 --feedback "missed the empty-token case"
# reef-pi: reported 1 receipt(s) to harness-evolve-code-repair
```

Reef 会按 recipe 配置对符合条件的上报进行批处理。启用版本检查后，adapter 会在下次
启动时检查是否有更新发布的版本。交互式会话会在接收输入前提供 **Update with …** 和
**Skip** 选项；选择更新会直接运行安装程序。无头（headless）会话则改为打印相应提示。
[harness 进化指南](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)
详细说明了提案、评估和发布的完整流程。


## Cookbook recipes

请根据工作负载能提供的反馈类型，以及需要更新的 artifact 来选择 recipe。下列实现位于
本仓库的 `recipes/` cookbook 中；它们通过带点号的类引用来选择，且不随 Reef wheel 一起发布。

| 工作负载 | Recipe 模块 | 更新的 artifact | 文档 |
|---|---|---|---|
| 由测试或校验器打分的任务流 | <code>recipes.sao.recipe:SAORecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/sao/) · [示例](recipes/sao/examples/sao/README.md) |
| 带有有用的下一状态信号、但没有显式上报的智能体流量 | <code>recipes.openclawrl.recipe:OpenClawRLRecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) · [示例](recipes/openclawrl/examples/openclawrl/README.md) |
| 针对同一个问题的多次带分数尝试 | <code>recipes.tttd.recipe:TTTDRecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [示例](recipes/tttd/examples/tttd/README.md) |
| 带分数的代码搜索，含一个可训练的引导模型和一个冻结的执行器 | <code>recipes.tttd.recipe:TTTDRecipe</code> | 引导模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [示例](recipes/tttd/examples/guidance_ttt/README.md) |
| 用智能体反馈来进化其技能池 | <code>recipes.skillclaw.recipe:SkillClawRecipe</code> | 技能池（harness 树）；无需 GPU | [指南](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) · [示例](recipes/skillclaw/README.md) |


## Reef 有什么不同？

Reef 为会成长的 AI 构建基础设施：

| 能力 | 推理引擎（vLLM、SGLang…） | RL 训练框架（slime、veRL、AReaL…） | **Reef** |
|---|:---:|:---:|:---:|
| 服务线上流量 | ✅ | ❌ | ✅ |
| 训练权重 | ❌ | ✅ | ✅ |
| 版本管理 | ❌ | ❌ | ✅ |
| 更新期间保持在线 | ❌ | ❌ | ✅ |
| 进化范围超越权重（技能、harness） | ❌ | ❌ | ✅ |


## 了解更多

[文档](https://reefinfra.ai/docs/)按以下顺序组织：

- [快速上手](https://reefinfra.ai/docs/getting-started/quickstart/)：安装 Reef、接入客户端并查看版本历史
- [HTTP API](https://reefinfra.ai/docs/reference/http-api/)：使用 HTTP API 并上报反馈
- [编写 recipe](https://reefinfra.ai/docs/developer-guide/write-a-recipe/)：配置 Reef 如何处理数据并产出更新
- [进化你的 harness](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)：进化 harness 而非模型权重
- [进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)：配置并运维一个训练部署
- [Recipes](https://reefinfra.ai/docs/user-guide/recipes/)：本仓库 cookbook 实现的更多参考资料
- [架构](https://reefinfra.ai/docs/getting-started/architecture/)：Reef 的整体架构
- [术语表](https://reefinfra.ai/docs/reference/glossary/)：文中所用术语的解释

## 社区与贡献

你也在做持续自我进化的智能体吗？

- [加入 Discord](https://discord.gg/5y8e5f937k)，分享你的 recipe、提出实现问题、讨论新特性。
- 加入 [GitHub Discussions](https://github.com/orgs/Human-Agent-Society/discussions)，提问、分享想法、与社区交流。
- 从[贡献指南](CONTRIBUTING.md)开始参与贡献。
- 通过 [RFC issue](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml) 提出设计方案。
- 按照[安全策略](SECURITY.md)私下报告疑似漏洞。

如果 Reef 对你有帮助，欢迎点一个 ⭐ —— 这能帮助更多人发现并参与这个项目。


## 致谢

我们特别感谢以下项目，它们支撑了 Reef 的重要组成部分：

- [SGLang](https://github.com/sgl-project/sglang) —— 高性能推理
- [slime](https://github.com/THUDM/slime) —— 模型权重训练
- [cordis](https://github.com/cordiverse/cordis) —— harness 进化
