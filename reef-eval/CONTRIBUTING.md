# Contributing to REEF Eval

reef-eval stays small, so contributions are reviewed against a short list of
design rules more than against style.

## The rules PRs are reviewed against

1. **One frozen interface.** `Lab.run`'s signature and the results-table schema
   don't change; columns may be added, never renamed or retyped. If your
   change needs to break this, open an issue first: it means the design is
   wrong somewhere, and that is the thing to fix.
2. **Tasks stay stock Harbor.** No reef-eval-specific fields in `task.toml`,
   ever. Benchmark structure lives in scripts *around* tasks.
   `tests/test_task_suite.py` validates every committed `task.toml`
   under Harbor's `TaskConfig`.
3. **Abstractions are earned.** A helper enters the library when the same
   shape has appeared in at least two real scripts. Until then it lives in
   `examples/`.
4. **Dependency direction.** Catalog conversion scripts depend on published
   formats, never reef-eval's runtime internals. Metrics import pandas, never reef_eval.
5. **Anti-reward-hacking measures are tested.** Anything claiming to stop
   an agent from gaming the score needs a test that actually cheats and
   fails. The
   pattern is the zero-reward cases each task declares in
   `tests/grader_tests.json`, which `tests/test_task_suite.py` runs.
6. **Measurement code fails loudly.** A missing metric column, a
   duplicate store key, or a corrupt benchmark line raises. Only submitted
   solutions are handled leniently: they score 0 with a reason instead of
   raising.
7. **Unbuilt work lives in the roadmap.** Docs state what exists; where a
   gap must be mentioned, link the [Reef issue tracker](https://github.com/Human-Agent-Society/reef/issues)
   instead of writing "not built yet" in place.
8. **The product name is REEF Eval.** Use `reef-eval` for the distribution and
   command, and `reef_eval` for the Python import package, as in
   `from reef_eval import Lab`.

## What's welcome

- **Autoresearch tasks**: copy `tasks/_template`, work the TODO markers,
  and `tests/test_task_suite.py` picks it up (scoring cases, zero-reward
  cases, stock-Harbor validation) with zero test code written. The
  containerized oracle score is gated separately by `scripts/e2e_oracle.py`.
- **Benchmark converters**. A converter PR should include: the conversion
  script, one converted sample task checked in as a test fixture, and a
  catalog row update with accurate status.
- **Metrics**: one pure function + docstring declaring expected columns +
  a small-frame test. Nothing else is required.
- **Executors** for new backends, implementing the `Executor` protocol.
- **Example protocols** under `examples/`: real autoresearch scripts are
  how library helpers get earned.

## Workflow

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/mypy
```

CI runs the tests, lint, and the offline examples on every PR. Harbor-
dependent tests skip when Harbor isn't installed; if your change touches the
Harbor mapping, install it locally (`uv pip install -e path/to/harbor`) and
run `tests/test_harbor_integration.py`.
