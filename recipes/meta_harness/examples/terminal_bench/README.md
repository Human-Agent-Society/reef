# Meta-Harness through Reef harness evolution

This experiment adapts Meta-Harness's persistent-population search to
declarative Reef compositions. OpenAI Codex proposes complete candidate trees
inside a durable workspace. Reef validates and renders each tree for Pi,
evaluates it with the Terminal-Bench verifier, retains every valid and invalid
attempt, and publishes the selected provider-free composition. It tests the
full-history mechanism through Reef; it does not reproduce Meta-Harness's
reported model setting or benchmark score exactly.

The experiment has three equal target-episode-budget cells:

1. a frozen genesis composition;
2. an incumbent-only control whose proposer sees the current winner and a
   compact summary; and
3. full-history Meta-Harness whose proposer can inspect and branch from every
   retained candidate, train trajectory, dev score, and prior proposer session.

The proposer runs locally under a restricted filesystem profile. Candidate
trees are limited to declarative `rules`, `skill`, and `agent_command` nodes.
`config` and executable `code_extension` nodes are rejected so an evaluated
candidate cannot load host resources or inspect controller files and held-out
task IDs.

## What maps to Reef

The candidate is a content-addressed sequence of the three allowed Reef node
kinds. It contains no provider or credential. For each trial, Reef renders that
composition for Pi and adds a transient model binding. A fixed,
controller-owned runtime extension replaces Pi's `bash` tool and forwards
commands to Harbor's task container; Harbor, not the harness or proposer, runs
the Terminal-Bench verifier.

The full-history cell gives Codex a durable filesystem containing every
candidate, raw train trajectories, aggregate-only dev feedback, rejected
attempts, and prior proposer sessions. The incumbent-only cell gets a separate
view with only the current winner and compact score history. The outer loop
validates proposals, runs train and dev evaluation, promotes strict dev-score
improvements, and withholds test evaluation until selection is fixed.
An evaluated non-winner is explicitly recorded as `rejected` by selection but
remains `retained` in the population, so a later full-history round may still
branch from it.

## Directory layout

```text
terminal_bench/
  harness/       candidate, workspace, proposer, Harbor bridge, search, reports
  run.py         three-cell controller, pin checks, resume, and publication
  run.sh         locked launcher from the Reef repository root
  pyproject.toml standalone Harbor and Terminal-Bench dependencies
  uv.lock        exact Python dependency resolution
  README.md      protocol, commands, trust boundary, and retained records
```

## Pinned adaptation

- Reef base commit:
  [`6b0d9a1345191a0df1a7e324fa09875e8581a83f`](https://github.com/Human-Agent-Society/reef/commit/6b0d9a1345191a0df1a7e324fa09875e8581a83f)
- Meta-Harness:
  [`95175f70c758dd1145b395edfe8b67e6f9d80fbd`](https://github.com/stanford-iris-lab/meta-harness/commit/95175f70c758dd1145b395edfe8b67e6f9d80fbd)
- Terminal-Bench historical source recorded in the
  [pinned reference environment](https://github.com/stanford-iris-lab/meta-harness/blob/95175f70c758dd1145b395edfe8b67e6f9d80fbd/reference_examples/terminal_bench_2/pyproject.toml):
  `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b`; that rewritten-history commit
  is no longer fetchable, so the runnable lock uses its declared
  `terminal-bench==0.2.18` release artifact
- Harbor: `0.3.0`
- Terminal-Bench: `0.2.18`, `terminal-bench@2.0` hard subset. Harbor's
  registry currently resolves all 30 task paths to
  [`69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`](https://github.com/laude-institute/terminal-bench-2/commit/69671fbaac6d67a7ef0dfec016cc38a64ef7a77c),
  and trials fetch that immutable Git tree directly
- Pi: `0.84.2`
- target model: `gpt-5.4-mini-2026-03-17`
- target-model pricing snapshot (observed 2026-08-30):
  [$0.75/M input, $0.075/M cached input, and $4.50/M output tokens](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- proposer: the locally installed Codex CLI, with its exact version captured
  in every live run
- proposer model request: mutable Codex alias `gpt-5.5`, high reasoning
  effort; the CLI version and requested alias are retained, but backend alias
  resolution is not an immutable model snapshot
- search: two rounds and two trials per task

This differs materially from the pinned upstream release: it uses OpenAI
models instead of the released Claude setting, a fixed 10/10/10 partition of
the 30-task hard subset instead of the full upstream evaluation, two rounds,
one proposal per round, and one strict dev-score incumbent. The three cells are
a Reef-specific ablation, not an official-reference arm. Pi uses the pinned
model provider's request defaults; reasoning effort, temperature, and output
limit are not overridden. A single run validates implementation behavior and
reports score differences descriptively; it does not establish a statistically
reliable full-history advantage.

## Validate without model calls

From the Reef repository root:

```bash
recipes/meta_harness/examples/terminal_bench/run.sh --dry-run
recipes/meta_harness/examples/terminal_bench/run.sh \
  --cell full_history \
  --smoke \
  --dry-run
uv run --locked \
  --project recipes/meta_harness/examples/terminal_bench \
  --with-editable . \
  --with pytest \
  python -m pytest -q tests/test_meta_harness_reproduction.py
```

The dry run checks the locked Harbor and Terminal-Bench releases, resolves all
30 named tasks through Harbor's registry and verifies their exact Git commit,
then checks the Pi version, Docker daemon, Codex binary, and Reef base commit.
It also proves that the pinned Codex permission profile can write inside a
temporary proposal surface while denying reads from its sibling, and verifies
Git LFS before publication can be reached. It prints the exact plan without
reading a target-model credential or starting a Codex proposer turn. With the
default 10/10/10 task split, two rounds, and two trials, every cell receives 140
target episodes; all three cells plan 420 target episodes and four Codex
proposer turns.

`--smoke` selects the first pinned task from each existing train, dev, and test
split. It deliberately keeps two search rounds and two trials per task, so a
full-history smoke still exercises 14 target episodes, two Codex proposer
turns, later-round history access, held-out test sealing, and publication. An
all-cell smoke plans 42 target episodes and four proposer turns. This is a
bounded end-to-end implementation check, not a benchmark result: the complete
10/10/10 split remains the adaptation experiment.

## Run the experiment

The Codex proposer uses the local Codex login. Set `REEF_CODEX_BINARY` when
`codex` is not on `PATH`; the macOS app binary is the final fallback. The Pi
target needs a fresh OpenAI project credential in `OPENAI_API_KEY`. Supply it
outside chat and set a hard project budget in the OpenAI dashboard before
running. Then launch one cell or all cells:

```bash
# Bounded end-to-end validation before the full experiment.
recipes/meta_harness/examples/terminal_bench/run.sh \
  --cell full_history \
  --smoke \
  --max-observed-cost-usd 5 \
  --output-dir /absolute/path/to/meta-harness-smoke

# Complete full-history adaptation cell. The cap is a stop threshold, not an estimate.
recipes/meta_harness/examples/terminal_bench/run.sh \
  --cell full_history \
  --max-observed-cost-usd 25 \
  --output-dir /absolute/path/to/meta-harness-run

recipes/meta_harness/examples/terminal_bench/run.sh \
  --cell all \
  --max-observed-cost-usd 75 \
  --output-dir /absolute/path/to/meta-harness-run
```

`--max-observed-cost-usd` is a resumable local stop threshold. The example
values above are ceilings, not projected costs. The runner prices each
completed Pi trial from its measured uncached-input, cached-input, and output
tokens using the pinned snapshot above; it does not trust a custom provider's
possibly absent native cost metadata. Once those estimates reach the limit,
the runner starts no new trial. It can overshoot by the cost of one in-flight
trial and does not price the subscription-backed Codex proposer. A provider
failure before Pi returns token usage cannot be charged to this ledger, so the
guard is not a replacement for the external project budget. Pi's
provider-reported cost is kept separately in Harbor metadata for auditing. The
target credential is withheld from Codex, retained records, and the published composition. Codex
tool commands run with a filesystem profile that exposes only minimal system
files and the current proposal surface; source files containing test IDs and
sibling cells are outside that readable view.

Cells may be staged into one compatible output directory. Finish every staged
experiment with the exact `--cell all` invocation shown above: completed
cells are integrity-checked and skipped, then `plan.json` and `results.json`
are regenerated for the authoritative three-cell comparison. Changing smoke
mode, rounds, trials, task splits, models, source commit, or dependency pins
requires a new output directory.

## Retained records

The two search cells write `workspace/`: proposer-visible population,
candidate history, round journal, frozen proposals, train history, aggregate
dev scores, and Codex JSONL sessions. Every cell writes:

- `private-evaluations/`: all Harbor trial records and the held-out result,
  never exposed through the evolution workspace;
- `summary.json`, `learning-curve.json`, and `candidate-parents.dot`: scores,
  candidate counts, status counts, token use, observed cost, and wall time;
- `workspace/proposal-events.jsonl`: invalid, selection-rejected, selected,
  duplicate, and failed proposal decisions, including retention disposition and
  the incumbent before and after each evaluated proposal;
- `published-composition/` and `artifacts.git/`: the selected provider-free
  tree published through Reef; and
- `done.json`: written only after the equal episode budget, held-out test,
  publication, and—for full history—later-round history-access proof all
  pass validation. It hashes every retained cell file outside transient Git
  work/cache directories, so a changed or incomplete result is not skipped.

Runs are restartable. Existing trial identities, completed search rounds,
Codex sessions, publication metadata, and observed costs are reused rather
than silently repeated. If evaluation stops partway through a search round,
the frozen proposal and its proposer session remain pending; the next run
continues that same proposal from its cached Harbor trials instead of asking
Codex for a replacement. Interrupted proposal validation also resumes without
another proposer turn, while a failed proposer turn remains retryable and does
not consume its round. Harbor infrastructure failures are retained as separate
attempt records and retried rather than silently scored as benchmark failures.
Failures with returned usage are charged; failures before usage is available
are visible in the attempt log but absent from the local spend ledger.

## Status

The provider-independent adaptation, locked environment, dry run,
deterministic tests, and real Pi-to-Harbor bridge smoke work without provider
calls. No paid benchmark result is recorded yet. The first paid run is an
exploratory implementation validation; neutral and negative results are
retained and reported unchanged.
