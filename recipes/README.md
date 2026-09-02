# Examples

Setup, once per example directory (installs the example's harness and its
declared dependencies, including `reef-client` and, where used,
[reef-eval](https://github.com/Human-Agent-Society/reef-eval)):

```bash
pip install -e .
```

Then `./run.sh` — it starts Reef (the example's stack YAML) and runs the loop
(`run.py`).

[Basic](basic/) is everything on the core, record-only `recipe` — the
deployment that learns nothing, and the smallest complete loop around it.
Its two stack files are where a deployment starts before it picks a method,
and what the quickstart serves:

- `external-provider.yaml` — no GPU, no local model: one Reef process
  proxying to an HTTP provider (`reef serve -c recipes/basic/external-provider.yaml`).
- `local-sglang.yaml` — local inference: an SGLang server plus Reef, no
  training.

Each is complete and runnable: a flat `reef:` section (translated into the
frozen `ServiceSettings` by
[`reef/service/deploy/settings.py`](../reef/service/deploy/settings.py)) plus
a `services:` list the orchestrator starts in dependency order, with `${VAR}`
environment and `${dotted.path}` config interpolation. Copy one and adapt it;
`reef serve -c <stack> --<key> <value>` overrides any `reef.*` setting. The
`recipe` they bind is the base contract in
[`reef/recipe/base.py`](../reef/recipe/base.py); a stack that binds a method
lives with that method (`recipes/<method>/examples/<example>/serve.yaml`; the
smallest weight-training one is
[`recipes/sao/examples/sao/serve.yaml`](sao/examples/sao/serve.yaml)).
Two contracts hold the set honest:
[`test_training_server.py`](../tests/reef_service/test_training_server.py)
boots the internal service from every cookbook stack, and
[`docs/site/scripts/check-doc-contracts.mjs`](../docs/site/scripts/check-doc-contracts.mjs)
derives the documented port and health route from `local-sglang.yaml`.

Around those stacks, the loop on the Harbor task standard: a
[Harbor](https://github.com/laude-institute/harbor) task (`harbor/`), a
Harbor agent harness that records its model call through Reef and reports
the verifier reward back at trial end (`harness/`), the loop written out
(`run.py` — [reef-eval](https://github.com/Human-Agent-Society/reef-eval)'s
`Lab.run`, one episode), and a launcher (`run.sh`) that starts Reef from
`external-provider.yaml` with local overrides and runs it.

[SAO](sao/examples/sao/README.md) is the functional smoke for the cookbook
SAO recipe, the smallest weight-updating loop. Three IMOAnswerBench problems
run in order by `run.py`, each driving six scored rollouts through Reef with a
verifiable binary reward, and every scored rollout is one training step. Its
README keeps a comparison against GRPO(+DIS) at Qwen3-30B-A3B scale.

[TTT-Discover](tttd/examples/tttd/README.md) separates a normal, service-agnostic rollout
harness from its Reef adapter. It demonstrates grouped discovery rollouts,
continuous evaluation, exact inference-to-report references, and
paper-faithful PUCT state reuse. Its README keeps the formal circle-packing
runs and an Erdős run, with the stored W&B history.

[Guidance-TTT](tttd/examples/guidance_ttt/README.md) trains a summary-only Qwen guidance
policy while a frozen external execution model writes verifier-scored
programs. It demonstrates how to attach an execution model without adding it
to Reef's training or inference-token capture path.

[Harness-Evolve quickstart](../tutorials/harness_evolve/README.md) runs the smallest skill
evolution on the harness evolution mechanism: the served model proposes one
skill mutation over its own failing traffic, gated real episodes on three
exact-answer coding tasks decide it, and the winning composition publishes
for client pull via `GET /reef/harness`. Setup here is just
`pip install reef-client`: the loop drives `reef_client` directly,
no Harbor task or reef-eval.

[GEPA](gepa/examples/aime_harness_evolve/README.md) reproduces the upstream
AIME reflective Pareto search and then runs the same search over fixed Reef
harness compositions. It keeps a frozen control, a rules-only conformance
cell, a multi-node rules-plus-skill cell, sealed held-out scoring, resumable
cost accounting, and durable publication of the selected composition.

[SkillClaw](skillclaw/README.md) rebuilds the SkillClaw
reproduction as a method package on the same mechanism: `propose` is the
sealed night (one decision per skill group plus the no-skill bucket) mapped
to one composite mutation sequence, `selection: always` publishes every
non skip night as the paper's ungated regime does, and the method ships its
own delivery - a recipe surface that injects the served pool's catalog into
every proxied request. The campaign driver embeds the Reef service, runs
the frozen 60-task WildClawBench day in docker, pulls the published pool
from `GET /reef/harness`, and seals rounds for the preregistered gain
criterion carried verbatim from the sealed campaign. Its `harbor/` is one
WildClawBench task vendored in the standard Harbor format (self-contained
image, the benchmark's own programmatic grader), and `run.py solve` is the
one-episode reef-eval smoke over it.

[OpenClaw-RL](openclawrl/examples/openclawrl/README.md) runs the paper's
personal-agent experiment as a reef-eval task stream: a simulated student brings
72 GSM8K homework problems to a Hermes agent whose model calls go through
reef, and the metric is the number of sessions before the agent's answers
match the student's taste. The method (session correlation, PRM judging, the
hint-conditioned teacher) is the `openclawrl` cookbook package, so the example
contains only the harness side: the task stream, the Hermes agent wrapper,
the student sidecar, and the analysis scripts. Its README keeps the learning
curve and training curves of a complete run.
