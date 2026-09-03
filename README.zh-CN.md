<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reef-logo-dark.svg">
  <img src="docs/assets/reef-logo-light.svg" alt="Reef" width="220">
</picture>

<h3>让 Agent 自我进化的持续学习基础设施</h3>

[![CI](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml/badge.svg)](https://github.com/Human-Agent-Society/reef/actions/workflows/ci.yml)
[![PyPI package: reef-infra](https://img.shields.io/pypi/v/reef-infra?label=PyPI%3A%20reef-infra&logo=pypi&logoColor=white)](https://pypi.org/project/reef-infra/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 简体中文

<div align="left">

Reef 是一整套持续学习的后端基础设施，全部包在标准的 HTTP 接口后面：你可以像用
`curl` 装 `codex`、`opencode` 那样把 agent 装到本地，也可以把 agent 的模型请求
指向 Reef 的推理接口，而不是模型厂商的。

不一样的地方在于，Reef 会一直盯着 agent 的表现，在后端把 harness 和模型权重
越调越好。你这边什么都不用做，效果自己会变好。

</div>

**[快速上手](https://reefinfra.ai/docs/getting-started/quickstart/) |
[路线图](https://github.com/Human-Agent-Society/reef/issues/25) |
[发布博文](https://x.com/ao_qu18465/status/2094867930081337730) |
[加入 Discord](https://discord.gg/5y8e5f937k)**

</div>


## 安装

> 💡 **注意**
>
> artifact 和 checkpoint 功能依赖系统里的 `git-lfs`，Reef 会在本地给自己的
> artifact 仓库初始化 Git LFS。

下面的命令都用 [uv](https://docs.astral.sh/uv/) 装包，我们也推荐这么用。

### 从 PyPI 装

```bash
uv venv && source .venv/bin/activate
uv pip install reef-infra
python3 -c "import reef; print(reef.__version__)"
```

### 从源码装

```bash
git lfs install
git clone https://github.com/Human-Agent-Society/reef.git
cd reef
uv venv && source .venv/bin/activate
uv pip install -e .
python3 -c "import reef; print(reef.__version__)"
```

做开发、跑下面的训练示例，都用源码装。


## 它是怎么跑的

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loop-animation-dark.svg">
  <img src="docs/assets/loop-animation-light.svg" alt="Reef 响应请求、记录反馈、产出更新，把通过的更新提交进版本历史。" width="76%">
</picture>
</div>

一个学习周期分四步，下表也标出了每步对应的模块。

| 步骤 | 做什么 | 代码在哪 |
|---|---|---|
| **1&nbsp;·&nbsp;Serve** | 响应 agent 请求，记下每次交互。 | [`service/`](reef/service) — agent 请求和交互记录<br>[`runtime/`](reef/runtime) — 推理和 artifact 更新 |
| **2&nbsp;·&nbsp;Observe** | 把反馈对回之前记下的交互。 | [`records.py`](reef/records.py) — 存下来的交互和反馈<br>[`train/processors/`](reef/train/processors) — 反馈匹配和达标判断 |
| **3&nbsp;·&nbsp;Grow** | 用达标的记录产出一次更新。 | [`recipe/`](reef/recipe) — recipe 接入<br>[`train/`](reef/train) — 批次和更新任务 |
| **4&nbsp;·&nbsp;Commit** | 评估更新，通过的就发布。 | [`train/evaluation/`](reef/train/evaluation) — 候选评估<br>[`artifact/`](reef/artifact) — 版本历史<br>[`surface/`](reef/surface) — artifact 分发 |


## 怎么用

Reef 能学两种东西：模型**权重**和 agent 的 **harness**。一个 scenario 到底更新哪种，
看这套部署用的是什么 recipe。

### 1 · 起服务

下面的例子会启动 SAO（arXiv:2607.07508）示例部署。要在 Reef 源码目录里跑，机器还得
满足[进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)里写的 GPU 要求。

```bash
uv pip install -e ".[slime]" && uv pip install --no-deps --group runtime

export MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
export REEF_TOKEN="reef-local"

reef serve -c recipes/sao/examples/sao/serve.yaml \
  --reef.model_path "$MODEL_PATH" \
  --reef.port "8900"

curl -f http://127.0.0.1:8900/healthz          # 服务起来了
```

### 2 · 训练权重

把推理请求发给 Reef，再给每个回复打个分。SAO recipe 拿每条达标的带分 rollout 走一次
训练。

#### 发一个推理请求，再报反馈

Reef 的推理接口同时兼容 OpenAI 和 Anthropic：`/v1/chat/completions` 和 `/v1/messages`
直接收各家原本的请求体。请求要带 `x-reef-scenario` 头；写一个没用过的名字，就会按这套
部署配好的 recipe 新建一个 scenario。recipe 不能在请求里选。

响应体沿用各家的 OpenAI 兼容格式，Reef 会多塞一个 `x-reef-agent-record-id` 响应头。
这个值就是**回执**（receipt），之后上报时靠它认出是哪次交互。一次上报可以带分数
`score`、文本或结构化的 `feedback`，还有它评价的那些回执。下面的例子分数和说明都报了。

```python
import os
import httpx

reef = httpx.Client(
    base_url="http://127.0.0.1:8900",
    headers={"Authorization": f"Bearer {os.environ['REEF_TOKEN']}", "x-reef-scenario": "hello-reef"},
    timeout=300,
)

# 用 OpenAI 兼容格式发推理请求
response = reef.post(
    "/v1/chat/completions",
    json={
        "model": os.environ["MODEL_PATH"],
        "messages": [{"role": "user", "content": "Return exactly: reef is ready"}],
    },
)

receipt = response.headers["x-reef-agent-record-id"]
answer = response.json()["choices"][0]["message"]["content"]

# 报一下这次推理的结果
matched = answer.strip() == "reef is ready"

reef.post(
    "/reef/report",
    json={"score": float(matched), "feedback": "matched" if matched else "wrong answer", "references": [receipt]},
).raise_for_status()
```

有些 recipe 不只看一个分数，`feedback` 就是给它们准备的，纯文本或结构化对象都行。
接口会校验**上报的 schema**（[`reef/core/reports/`](reef/core/reports)）。


#### 看着它长大

反馈攒够以后，recipe 会跑一次训练，把新权重同步给推理运行时。之后的请求直接用上新版本，
不用重启 Reef。

### 3 · 进化你的 harness

`harness_evolve` 这个 recipe 更新的是一棵 harness 树，里面可以放规则、技能、配置、
提示词和扩展。它拿上报的交互拼一个候选 harness 出来，在配好的任务上跟当前版本比一比，
赢了才发布。各个 harness scenario 之间数据和版本互不相通。

#### 装一个会跟着你长的 Reef harness

装 Reef harness 跟装大多数编程 agent 差不多，比如下面这样。过程中会自动建一个新
scenario，跟装下来的 harness 绑在一起。

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash

reef-pi -p "fix the bug"
```

想拿一个已经进化过的 harness，在请求头里指定 scenario 就行。比如手上已经有
`harness-evolve-code-repair` 这个 scenario：

```bash
curl -fsS -H 'x-reef-scenario: harness-evolve-code-repair' \
  -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash
```

#### 报一次任务结果

`reef-pi` 自己存着一次运行留下的回执，所以 `report` 只要给出你想挂到上一次交互上的
结果就够了：

```bash
reef-pi -p "fix the failing test in auth.py"

# ... 跑你的测试，给结果打个分 ...

reef-pi report --score 0 --feedback "missed the empty-token case"
# reef-pi: reported 1 receipt(s) to harness-evolve-code-repair
```

Reef 按 recipe 的配置把达标的上报攒成批。开了版本检查的话，adapter 下次启动会看有没有
新发布的版本：交互式会话在等你输入之前会给 **Update with …** 和 **Skip** 两个选项，选
更新就直接跑安装脚本；无头会话只打印一句提示。提案、评估、发布这一整套流程，见
[harness 进化指南](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)。


## Cookbook recipes

选哪个 recipe，看两件事：你的负载能给出什么反馈，以及你想更新哪个 artifact。下面这些
实现都放在本仓库的 `recipes/` 里，用带点号的类路径指定，不会打进 Reef 的 wheel 包。

| 负载 | recipe 模块 | 更新什么 | 文档 |
|---|---|---|---|
| 靠测试或校验器打分的任务流 | <code>recipes.sao.recipe:SAORecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/sao/) · [示例](recipes/sao/examples/sao/README.md) |
| 有可用的下一状态信号、但没人显式上报的 agent 流量 | <code>recipes.openclawrl.recipe:OpenClawRLRecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/openclawrl/) · [示例](recipes/openclawrl/examples/openclawrl/README.md) |
| 同一个问题反复试、每次都有分 | <code>recipes.tttd.recipe:TTTDRecipe</code> | 模型权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [示例](recipes/tttd/examples/tttd/README.md) |
| 带分数的代码搜索：引导模型可训练，执行器冻结 | <code>recipes.tttd.recipe:TTTDRecipe</code> | 引导模型的权重 | [指南](https://reefinfra.ai/docs/user-guide/recipes/tttd/) · [示例](recipes/tttd/examples/guidance_ttt/README.md) |
| 用 agent 反馈进化它自己的技能池 | <code>recipes.skillclaw.recipe:SkillClawRecipe</code> | 技能池（harness 树），不用 GPU | [指南](https://reefinfra.ai/docs/user-guide/recipes/skillclaw/) · [示例](recipes/skillclaw/README.md) |


## Reef 有什么不一样

Reef 想做的是让 AI 能长大的那层基础设施：

| 能做什么 | 推理引擎（vLLM、SGLang…） | RL 训练框架（slime、veRL、AReaL…） | **Reef** |
|---|:---:|:---:|:---:|
| 接线上流量 | ✅ | ❌ | ✅ |
| 训练权重 | ❌ | ✅ | ✅ |
| 版本管理 | ❌ | ❌ | ✅ |
| 更新时服务不断 | ❌ | ❌ | ✅ |
| 权重之外也能进化（技能、harness） | ❌ | ❌ | ✅ |


## 想了解更多

[文档](https://reefinfra.ai/docs/)按下面的顺序编排：

- [快速上手](https://reefinfra.ai/docs/getting-started/quickstart/)：装好 Reef，接上客户端，看看版本历史
- [HTTP API](https://reefinfra.ai/docs/reference/http-api/)：接口怎么调，反馈怎么报
- [写一个 recipe](https://reefinfra.ai/docs/developer-guide/write-a-recipe/)：配置 Reef 怎么处理数据、怎么产出更新
- [进化你的 harness](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)：不训权重，改进 harness
- [进化你的模型](https://reefinfra.ai/docs/user-guide/evolve-your-model/)：训练部署怎么配、怎么运维
- [Recipes](https://reefinfra.ai/docs/user-guide/recipes/)：本仓库 cookbook 实现的更多说明
- [架构](https://reefinfra.ai/docs/getting-started/architecture/)：Reef 的整体架构
- [术语表](https://reefinfra.ai/docs/reference/glossary/)：文档里各种术语是什么意思

## 社区与贡献

也在折腾会自我进化的 agent？

- 来 [Discord](https://discord.gg/5y8e5f937k) 聊聊：分享你的 recipe，问实现细节，聊想加的功能。
- 去 [GitHub Discussions](https://github.com/orgs/Human-Agent-Society/discussions) 提问、发想法、认识同好。
- 想动手，从[贡献指南](CONTRIBUTING.md)开始。
- 有设计上的提案，走 [RFC issue](https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml)。
- 发现疑似漏洞，按[安全策略](SECURITY.md)私下报给我们。

觉得 Reef 有用的话，点个 ⭐ 吧，能让更多人发现它、一起来建设。


## 致谢

Reef 有几块关键的地方靠这些项目撑着，特别感谢：

- [SGLang](https://github.com/sgl-project/sglang) — 高性能推理
- [slime](https://github.com/THUDM/slime) — 模型权重训练
- [cordis](https://github.com/cordiverse/cordis) — harness 进化
