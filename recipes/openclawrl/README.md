# openclawrl

Reproduction of [OpenClaw-RL](https://arxiv.org/abs/2603.10165)'s personal-agent experiment as a Reef weight-training recipe. OpenClaw-RL trains an agent from its ordinary usage: each turn is scored by what the user did next, a follow-up that moves on counts as acceptance and a complaint as rejection, and the policy updates while it keeps serving. The agent only has to use Reef as its inference endpoint; the package holds the method, and the simulated-student experiment around it lives in [examples/openclawrl](examples/openclawrl/README.md).

- Paper: [arXiv:2603.10165](https://arxiv.org/abs/2603.10165)
- Pins: the repo's `third_party/slime` submodule for training; the completed run used a Qwen3-4B thinking student, a hermes-agent image pinned to a commit in the example's Dockerfile, and a 72-session GSM8K homework stream
- Claim scope: one recorded run over the first 36 sessions of the stream. The run reached the paper's adaptation criterion, three passed sessions in a row, at session 14, and the bold and list rates fall over the run. One student persona, one task family; this is an adaptation result, not a benchmark-wide reproduction.

## Layout

```text
openclawrl/
  recipe.py        OpenClawRLRecipe: training spec, loss family "openclawrl"
  processor.py     computed feedback: rebuilds sessions from recorded traffic, judges turns
  preparer.py      builds the policy batch from accepted turns and hindsight hints
  sessions.py      session reconstruction from the records Reef already keeps
  prm.py           the PRM judge the processor runs on a private worker
  slime/           the training-plane objective and the hint-conditioned teacher
  examples/openclawrl/  the runnable experiment: student simulator, Harbor sessions, results
```

## Where the rest is documented

[The openclawrl recipe page](../../docs/user-guide/recipes/openclawrl.rst) covers configuration and the serving stack, and the [example README](examples/openclawrl/README.md) records implementation details, the stream protocol, and the recorded run.
