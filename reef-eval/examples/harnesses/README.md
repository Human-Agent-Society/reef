# Benchmark harness adapters

These adapters run three long-horizon systems against the same reef-eval task and
judge. They do not replace or wrap the scorer: every candidate still goes to
`$JUDGE_URL/submit`, the submission limit is enforced by the task, and Harbor's
verifier produces the final trusted reward.

All three subclass the one base class, `TideHarnessBase` in
[`base.py`](base.py), which encodes the standard operating procedure as its
`run` template, in order: `setup` (install the pinned tool), `_prepare`
(upload configs and seeds), `_launch` (the run itself, where a timeout is a
normal ending), `_finalize` (submit the final artifact if the judge saw zero
submissions), and `_collect_usage` (meter tokens). OpenEvolve and
CORAL subclass it directly and use its command-pipeline utilities; Codex
lists it first and delegates `_launch` to Harbor's built-in Codex agent.

All three populate Harbor's standard input, cached-input and output-token
fields, which reef-eval stores on the episode row as `used_n_input_tokens`,
`used_n_cache_tokens`, and `used_n_output_tokens`.

Run an adapter from the repository root (Docker and `reef-eval[harbor]` are
required):

```bash
export OPENAI_API_KEY=...

python examples/run_harness.py openevolve --model gpt-5-mini --iterations 100
python examples/run_harness.py codex --model gpt-5.6-terra
python examples/run_harness.py coral --model gpt-5.6-terra --agents 2
```

The example is wired to `tasks/autoresearch/first-party/circle-packing`, which
makes the three methods directly comparable in one Lab. Change `TASK` in
[`../run_harness.py`](../run_harness.py) to point at another task. Codex
consumes the task instruction directly through Harbor's standard
non-interactive agent. Replace OpenEvolve's candidate program and CORAL's seed
`solution.json` when adapting those harnesses to a new solution format.

| Adapter | Integration | Pinned version |
|---|---|---|
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | evolves `initial_program.py`; its evaluator executes the candidate and returns the reef-eval judge score | adapter 0.1.1 + OpenEvolve 0.3.2 |
| [Codex](https://developers.openai.com/codex/noninteractive/) | reuses Harbor's built-in `codex exec --json` agent, including trajectory and usage collection; after the agent stops, submits the final `solution.json` to the judge if the run never submitted (otherwise the verifier grades an empty log) | Codex CLI 0.147.0 |
| [CORAL](https://github.com/Human-Agent-Society/CORAL) | launches a two-agent organization; `coral eval` calls a packaged `TaskGrader` that submits `solution.json` to reef-eval | adapter 0.1.0 + CORAL 0.7.16 |

The adapters deliberately pin their tool versions so a benchmark record has a
meaningful harness version. Update the constants in each harness's `agent.py`,
the table above, and the protocol tests together.

## Credentials and network access

`run_harness.py` passes an environment-variable placeholder to Harbor, not the
API key value, so the key is resolved only when the agent starts. Harbor keeps
Codex's temporary authentication and session data outside task artifacts.

The task must permit the model provider endpoint through its agent-phase network
policy. The judge itself remains the only scoring authority.
