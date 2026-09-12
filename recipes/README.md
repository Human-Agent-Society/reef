# Recipes and examples

Setup, once per example directory (installs the example's harness and its
declared dependencies, including `reef-client` and, where used,
[reef-eval](https://github.com/Human-Agent-Society/reef-eval)):

```bash
pip install -e .
```

Then `./run.sh` — it starts Reef (the example's stack YAML) and runs the loop
(`run.py`).

The catalog below groups recipes by the **task type** they serve, then by
**what they evolve**: model weights or the agent harness. Recipes that evolve
model weights need the GPU training stack; harness recipes need only a model
endpoint. Reefine ships with `reef-infra`; every other recipe here is a
cookbook package. [Basic](#basic) is the record-only starting stack and stays
outside the catalog, and [beta recipes](#beta-recipes) join it once they
publish learning results. The root [README](../README.md#recipes-and-examples)
and the [recipes guide](../docs/user-guide/recipes.rst) present the same
catalog.

## Scientific discovery

One hard problem, repeated attempts, and a measurable objective. The recipe
trains on the attempts it generates itself, at test time.

| Recipe | Evolves | Code | Docs | Example |
|---|---|---|---|---|
| TTT-Discover | model weights | [`recipes/tttd/`](tttd/) | [TTT-Discover](../docs/user-guide/recipes/tttd.rst) | [TTT-Discover on circle packing and Erdős minimum overlap](tttd/examples/tttd/README.md) |
| Guidance-TTT | guidance-model weights; the executor stays frozen | [`recipes/tttd/`](tttd/) | [Guidance-TTT](tttd/examples/guidance_ttt/README.md) | [Guidance-TTT on TriMul](tttd/examples/guidance_ttt/README.md) |

No recipe evolves the harness for this task type yet. CORAL TTT targets it and
is in [beta](#beta-recipes).

[TTT-Discover](tttd/examples/tttd/README.md) separates a normal, service-agnostic rollout
harness from its Reef adapter. It demonstrates grouped discovery rollouts,
continuous evaluation, exact inference-to-report references, and
paper-faithful PUCT state reuse. Its README keeps the formal circle-packing
runs and an Erdős run, with the stored W&B history.

[Guidance-TTT](tttd/examples/guidance_ttt/README.md) trains a summary-only Qwen guidance
policy while a frozen external execution model writes verifier-scored
programs. It demonstrates how to attach an execution model without adding it
to Reef's training or inference-token capture path.

## Continual learning on a task stream

Independent tasks, each scored by a verifier. The recipe learns from the
feedback on each task as the stream goes by.

| Recipe | Evolves | Code | Docs | Example |
|---|---|---|---|---|
| SAO | model weights | [`recipes/sao/`](sao/) | [SAO](../docs/user-guide/recipes/sao.rst) | [SAO on IMOAnswerBench](sao/examples/sao/README.md) |
| GEPA | harness tree: rules, skills, and agent commands | [`recipes/gepa/`](gepa/) | [GEPA](../docs/user-guide/recipes/gepa.rst) | [GEPA on AIME 2025](gepa/examples/aime/README.md) |
| Meta-Harness | harness: complete compositions | [`recipes/meta_harness/`](meta_harness/) | [Meta-Harness](meta_harness/README.md) | [Meta-Harness on Terminal-Bench](meta_harness/README.md) |
| Reefine | harness: skills, rules, agent commands, and pi extensions | [`reef/recipe/reefine/`](../reef/recipe/reefine/) | [Reefine](../docs/user-guide/recipes/reefine.rst) | [Reefine on reef-pi](../tutorials/reefine/README.md) |

[SAO](sao/examples/sao/README.md) is the functional smoke for the cookbook
SAO recipe, the smallest weight-updating loop. Three IMOAnswerBench problems
run in order by `run.py`, each driving six scored rollouts through Reef with a
verifiable binary reward, and every scored rollout is one training step.

[GEPA](gepa/examples/aime/README.md) rebuilds reflective prompt evolution as a
method package on the same mechanism: `propose` is one GEPA iteration - Pareto
sample a parent from the method's own archive, reflect on one component with a
stronger model over the served composition's failing traffic, and accept the
child only if it beats its parent on the minibatch - and `selection` publishes
only on a strict mean improvement over the full validation set. Nothing in it
imports the upstream package. Its AIME example is the validation: the driver
embeds the Reef service, runs the quickstart's 45 training problems through it
three at a time, and seals the two 150-problem test passes against the retained
official record (26.67% to 38.67% on AIME 2025, seed 0); the method's own seed-0
run reflected from the same parents on the same problems and reached 46.67%, and
its seed-1 run gained the official 12 points.

[Meta-Harness](meta_harness/README.md) searches complete harness compositions
using all retained candidates and scores. It selects strict mean-score
improvements and commits the population with Reef's serving state. See the
[Terminal-Bench results](meta_harness/RESULTS.md) and selected harness.

[Reefine](../docs/user-guide/recipes/reefine.rst) is the built-in recipe that
turns a person's plain-language request into a harness update: the served
model proposes the change, the gate scores it, and code extensions wait for a
promote before they run. `reef serve --recipe reefine` starts its profile
without a checkout; the [Reefine tutorial](../tutorials/reefine/README.md)
records a bug-fix flow, a research loop, and a measurement of which requests
won the gate.

## Learning from usage

Real interaction with no explicit score, or delayed feedback. The recipe reads
the signal out of the traffic it already serves.

| Recipe | Evolves | Code | Docs | Example |
|---|---|---|---|---|
| OpenClaw-RL | model weights | [`recipes/openclawrl/`](openclawrl/) | [OpenClaw-RL](../docs/user-guide/recipes/openclawrl.rst) | [OpenClaw-RL on the GSM8K homework stream](openclawrl/examples/openclawrl/README.md) |
| SkillClaw | harness skill pool | [`recipes/skillclaw/`](skillclaw/) | [SkillClaw](../docs/user-guide/recipes/skillclaw.rst) | [SkillClaw on WildClawBench](skillclaw/README.md) |

[OpenClaw-RL](openclawrl/examples/openclawrl/README.md) runs the paper's
personal-agent experiment as a reef-eval task stream: a simulated student brings
72 GSM8K homework problems to a Hermes agent whose model calls go through
reef, and the metric is the number of sessions before the agent's answers
match the student's taste. The method (session correlation, PRM judging, the
hint-conditioned teacher) is the `openclawrl` cookbook package, so the example
contains only the harness side: the task stream, the Hermes agent wrapper,
the student service, and the analysis scripts.

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

## Basic

[Basic](basic/) is everything on the core, record-only `recipe` — the
deployment that learns nothing, and the smallest complete loop around it.
Its two stack files are where a deployment starts before it picks a method,
and what the quickstart serves:

- `external-provider.yaml` — no GPU, no local model: one Reef process
  proxying to an HTTP provider (`reef serve -c recipes/basic/external-provider.yaml`).
- `local-sglang.yaml` — local inference: an SGLang server plus Reef, no
  training.

Each is complete and runnable: a flat `reef:` section (translated into the
frozen `ServiceConfig` by
[`reef/service/deploy/service_config.py`](../reef/service/deploy/service_config.py)) plus
a `services:` list the orchestrator starts in dependency order, with `${VAR}`
environment and `${dotted.path}` config interpolation. `${REEF_PYTHON}`
defaults to the interpreter running `reef serve`, so Python services that use
it share Reef's environment without changing the meaning of literal `python`
commands. Copy one and adapt it;
`reef serve -c <stack> --<section.field> <value>` overrides the matching YAML setting,
for example `--inference.model-path /models/demo` or `--reef.port 9000`. The
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

## Beta recipes

[CORAL TTT](beta/coral/README.md) and its
[`coral_demo`](beta/coral/examples/coral_demo/) example are beta. They live
under `recipes/beta/coral/` until complete, reproducible learning results are
published. Integration and smoke tests validate the wiring; they do not
establish learning performance. See the recipe's
[validation instructions](beta/coral/README.md#verifying-without-gpus).

[CORAL TTT](beta/coral/README.md) runs a
[CORAL](https://github.com/Human-Agent-Society/CORAL) discovery task — parallel
coding agents in git worktrees, graded attempts on one problem — with every
agent call served and attributed through Reef. CORAL's gateway traffic carries
Reef receipts into an append-only call journal; a watcher reports each
finalized attempt exactly once with its exact inference references, and
sibling attempts of one parent commit train as one grouped relative-reward
step (reusing the TTT-Discover preparer and loss family). Its example is a
real CORAL task driven by CORAL's own runtime, plus a no-GPU smoke lane that
runs the whole loop against the production Reef service with a canned model.
