# Agent Instructions for Reef

These instructions apply to AI-assisted work in `Human-Agent-Society/reef`.
`AGENTS.md` is the shared source of truth; `CLAUDE.md` is a relative symlink to
this file. Edit this file when updating shared instructions.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for project policy. Follow any more
specific `AGENTS.md` in the area you change. Explicit user instructions take
precedence over repository guidance.

## Contribution workflow

- Inspect the working tree before editing and preserve existing user changes.
  Keep each change focused on the requested problem.
- Before opening an issue or pull request, search existing issues and PRs for
  overlapping work. Follow the RFC criteria in `CONTRIBUTING.md` for changes to
  architecture, public contracts, persistence, or project policy.
- Reproduce bugs and inspect the relevant implementation before changing it.
  Avoid speculative fixes, unrelated formatting, and unnecessary abstractions.
- Use [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).
  Explain the problem, resulting behavior, compatibility impact, and actual
  verification results. Disclose non-trivial AI assistance. The human
  contributor remains responsible for reviewing and understanding the change.
- Keep credentials and private transcripts out of logs, fixtures, and commits.
  Follow [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Codebase and boundaries

Reef connects inference, feedback, learning, and versioned delivery for model
weights and agent harnesses. The distribution is `reef-infra`; imports use
`reef`. Use the [codebase map](docs/contributing/codebase-structure.rst) and the
affected package's `__init__.py` docstring to find the owner of a change.

| Location | Responsibility |
| --- | --- |
| `reef/core/` | Shared value types, wire contracts, and errors |
| `reef/dispatcher.py`, `reef/scenario/` | Coordination, scenario state, commit ordering, and recovery |
| `reef/service/` | HTTP, authentication, streaming, and deployment |
| `reef/recipe/`, `reef/train/` | Recipe contracts, processors, training, evaluation, and backend integrations |
| `reef/runtime/` | Backend-neutral runtime contracts, scheduling, and publication coordination |
| `reef/inference/` | Concrete inference integrations, engine control, and weight reception |
| `reef/surface/` | Delivery of published artifacts |
| `reef/artifact/` | Versioned artifacts and repositories |
| `reef/storage/` | Record storage contracts, persistence, and retention |
| `reef/harness/` | Harness adapters, rendering, runners, and trajectories |
| `recipes/`, `tutorials/` | Method implementations, runnable examples, and tutorials |
| `tests/`, `docs/`, `docker/` | Verification, documentation, and deployment environments |

- Keep shared mechanisms in `reef/` and method-specific policy in `recipes/`.
  The core must not import cookbook methods; `recipes/` does not ship in the
  Reef wheel.
- Keep `reef-client` a separate, dependency-free protocol client. Harnesses
  consume `reef_client`; they should not need Reef's service or training stack.
- Keep the base installation usable on CPU. GPU dependencies belong to the
  supported training environment, and concrete adapters belong under their
  integration. Declare or pin third-party dependencies instead of copying them.
- Preserve provider request bodies, scenario isolation, receipt-to-feedback
  linkage, and artifact publication/recovery contracts.

## Development environment

Use `uv` and the repository virtual environment for Python work. Reuse an
existing environment; for a new checkout, the usual setup is:

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" -e ./third_party/reef-client
pre-commit install
```

Reef supports Python 3.10 and newer; CI tests 3.10, 3.11, and 3.12. Keep source
syntax compatible with 3.10. Git LFS is required for artifact/checkpoint work.

Training-related work may also need:

```bash
uv pip install -e ".[slime]"
uv pip install --no-deps --group runtime
```

The `slime` extra supplies Python-side adapter dependencies; the `runtime`
group pins Slime itself. Preserve `--no-deps` so the installation does not
replace the container's CUDA-compatible stack. Follow
[development](docs/contributing/development.rst) and [docker/README.md](docker/README.md)
for the environment required by the selected backend.

## Python style and design

- Follow `pyproject.toml`: Black and isort format at 119 columns; Ruff checks
  code and naming; mypy checks `reef`. Match nearby code and add types to new
  or changed interfaces.
- Prefer focused functions, data classes for values, and cohesive objects for
  state and lifecycle. Use composition and explicit abstract base classes for behavior.
- Do not use `typing.Protocol`, `typing_extensions.Protocol`, or `runtime_checkable`.
  Define an `ABC` and inherit it explicitly. CI checks all first-party Python files,
  including tutorials, Docker/docs/CI scripts, and root files, without baseline exceptions.
  Third-party code, local dependency/build trees, vendored benchmarks, published result
  programs, and golden fixtures are excluded; see the paths in
  [.github/scripts/check_python_design.py](.github/scripts/check_python_design.py).
- Do not use `TYPE_CHECKING`. Fix dependency direction or move shared contracts
  so annotation imports work at runtime.
- Do not model long-lived behavior as `Callable` constructor arguments,
  callable-valued fields, or callback containers. Use an explicit interface. Keep unavoidable
  third-party dynamic behavior local to its boundary adapter.
- Do not use `assert` or `del` statements in `reef/`; use explicit validation,
  exceptions, and mutation APIs. Test assertions are allowed.
- Use descriptive domain names and short comments that explain intent or
  constraints. Follow the Google-style documentation guidance in
  `CONTRIBUTING.md`.
- Do not bypass checks by adding broad suppressions or growing
  `.github/python-design-baseline.txt` to accommodate new violations.

### Code structure and readability

Apply these rules when writing or reviewing code. The
[Code Review Style Guide](https://github.com/zhaochenyang20/sglang-diffusion-routing/issues/32)
is a reference; the rules here take precedence where it differs.

- **Avoid fragmented functions.** Inline tiny helpers used only once or twice
  when they merely split up a continuous operation. Extract functions for a
  meaningful responsibility or substantial reuse, not to meet an arbitrary
  line limit. Keep related logic readable in one place.
- **Use ordinary identifier names.** Only functions nested inside other
  functions should use a leading underscore. Start other variables, fields,
  constants, module-level functions, and methods with an English letter.
  Preserve Python-required special names such as `__init__`.
- **Handle realistic failures.** Validate types and data at entry points, then
  rely on those contracts internally. Keep `try/except` blocks narrow and catch
  specific, expected failures only when there is a meaningful recovery or error
  translation. Do not add speculative fallbacks, repeated checks, or tests for
  impossible states. Tests should be concise and cover observable behavior and
  the operation's main failure modes.
- **Keep constants with their consumer.** If a constant is defined in one
  module solely to be imported by one other module, define it in the consuming
  module instead. Share constants when they have actual shared use; avoid
  unnecessary imports and separate modules for single-use values.
- **Describe types explicitly.** Do not use `Any` fields or annotations to
  bypass pre-commit, mypy, or other checks. Model the actual value types and
  interfaces instead of weakening annotations to silence failures.
- **Name the actual quantity or concept.** Variable and function names must
  express their domain, scientific, or physical meaning. Include units or
  representation where needed, such as `timeout_seconds`, `token_count`, or
  `weight_dtype`; avoid arbitrary abbreviations and vague placeholder names.
- **Make branches complete and shallow.** Use explicit `else` branches when
  choosing values, assigning variables, or returning alternative results.
  Guard clauses that return, raise, or continue early do not need `else`.
  Check types and invalid data early to keep the main path clear and avoid
  excessive branching or deeply nested `if/else` blocks.
- **Use explicit attributes and interfaces.** Do not use `getattr`, `hasattr`,
  or dynamic attribute mutation to guess object capabilities or add defensive
  defaults. Replace these patterns in code being changed with typed attributes,
  direct access, and explicit interfaces; fix the contract instead of probing
  for attributes at runtime.

## Naming and terminology

Prefer simple, conventional terminology already used in this repository.

Do not introduce abstract or uncommon terminology when a simpler name is sufficient.
In particular, avoid terms such as:

- ledger
- sidecar
- provenance
- evidence

unless the term is already established in the codebase or is the standard technical term for the concept.

Prefer concrete alternatives such as:

- metadata
- manifest
- record
- source information
- checksum
- validation result
- training metadata

When adding new concepts, reuse existing repository vocabulary before inventing new terminology.

Prefer concrete names over metaphors in code, UI text, logs, and documentation:

- Use `check` for a condition, `validation` for checking validity, and
  `evaluation` for running and scoring tasks instead of `gate`.
- Use `result` for an outcome, `selection_result` for a candidate selection,
  and `review_result` for a review instead of `verdict`.
- Name the actual operation or quantity: `evaluation_score`, `evaluation_tasks`,
  `result_html`, and "passed the checks" are clearer than metaphorical names.
- Apply terminology changes to identifiers, serialized keys, producers, consumers,
  tests, and documentation. New records should use the preferred keys. Preserve
  older records and callers with targeted fallback reads or API aliases where
  compatibility is needed; keep those legacy names at the boundary rather than
  spreading them through new code or displayed labels.
- Keep standard technical names such as neural-network gates and third-party identifiers.

## Tests and checks

Start with the smallest existing suite that exercises the changed behavior.
Extend nearby tests and fixtures; check observable results and failure cases.
Public interface changes need contract tests. For training or performance
changes, include relevant evaluation or baseline comparisons.

```bash
# Focused test example; substitute the suite relevant to the change.
.venv/bin/python -m pytest tests/reef_service/test_reef_artifacts.py -q

# Python checks (activate .venv first for pre-commit's local hooks).
pre-commit run --files path/to/changed_file.py
.venv/bin/python -m mypy

# Full checks before requesting review for Python changes.
pre-commit run --all-files
.venv/bin/python -m pytest tests/
```

The full suite needs training dependencies even for collection. Some tests
skip unavailable optional runtimes; others import Slime and torch directly.
Use the [testing guide](docs/contributing/testing.rst) and
[CI configuration](.github/workflows/ci.yml) to reproduce the needed environment.
CI uses `GIT_CONFIG_GLOBAL=/dev/null` and `GIT_CONFIG_SYSTEM=/dev/null` for
Git LFS test isolation. Record skipped or unavailable checks honestly.

For full-suite coverage on Python 3.12, use
`.venv/bin/python -m pytest tests --cov=reef --cov-report=term`.
The configured coverage floor applies to the whole package, not a focused run.

Pre-commit includes repository-wide Python design, statement, and README
pairing checks even when invoked with `--files`. Review formatter edits and
keep unrelated working-tree changes intact.

## Area-specific guides and documentation

- Before adding a component, read
  [adding-components.rst](docs/contributing/adding-components.rst).
- For recipes and examples, read [recipes/AGENTS.md](recipes/AGENTS.md) and the
  method's README. Keep examples self-contained.
- For harness adapters, read
  [harness-adapters.rst](docs/developer-guide/harness-adapters.rst).
  Golden harness trees under `tests/reef_service/data/harness_goldens/` are
  test fixtures, including their instruction files; preserve exact output.
- For documentation site changes, read [docs/site/AGENTS.md](docs/site/AGENTS.md).
  With Node.js 22, run `npm ci`, `npm run check:docs`, `npm run lint`, and
  `npm run build` from `docs/site/`, matching the docs CI job.
- Keep `README.md` and `README.zh.md` synchronized. After reviewing both, run
  `.venv/bin/python .github/scripts/check_readme_i18n.py --write`, then
  `.venv/bin/python .github/scripts/check_readme_i18n.py`, and include the
  updated `README.i18n.yaml` with the change.
- Update affected API, configuration, and user documentation alongside code.
  Keep this guide concise and link detailed rules to their owning documents.
