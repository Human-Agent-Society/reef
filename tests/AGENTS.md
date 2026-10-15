# Test organization

Choose the suite that owns the behavior under test. Extend an existing relevant
module before adding another. Keep new test modules out of the `tests/` root.

## Directory map

| Directory | Responsibility |
| --- | --- |
| `artifact/`, `storage/`, `surface/` | Artifact repositories, record persistence, and artifact delivery |
| `core/`, `scenario/` | Shared value contracts, scenario lifecycle, dispatcher coordination, and commit recovery |
| `runtime/`, `inference/`, `train/` | Runtime coordination, inference integrations, and the training pipeline |
| `sandbox/` | Shared execution, environment contracts, isolation, file transfer, and cleanup |
| `slime_backend/`, `tinker_backend/` | Contracts specific to a training backend |
| `record2record/`, `record2task/`, `task2batch/` | The corresponding stages of the training pipeline |
| `evaluation/`, `harness/`, `observability/` | Candidate evaluation, harness execution, and tracking |
| `reef_recipe/` | Shared recipe contracts and built-in recipes from `reef/recipe/` |
| `recipes/<recipe>/` | Cookbook methods and their examples; use the method name for beta recipes too |
| `reef_service/` | HTTP contracts, deployment, and service workflows across components |
| `cli/`, `reef_client/` | CLI behavior and the separate `reef_client` package |
| `contributing/` | Repository policy, documentation, and checks across shipped examples |
| `plugin_contracts/`, `packaging/` | Plugin boundaries and installed-package verification |
| `smoke/` | Opt-in runs against real harnesses and external services |
| `support/` | Shared fixtures and helpers; no collected test modules |

Keep component tests with their owner even when they use a dispatcher or a
service fixture. A test belongs in `reef_service/` when it checks the service
contract or the service's coordination of several components.

Use markers for `unit`, `integration`, `acceptance`, and `sandbox`; do not add a
second directory hierarchy for those categories. A mock-based test stays with
its component, even when the real component needs a GPU.

## Fixtures and imports

- Keep fixtures in the nearest applicable `conftest.py`. Reuse scoped support
  modules for helpers shared across suites.
- Do not introduce imports from collected test modules. Extract the shared
  helper when adding a new consumer; avoid unrelated helper rewrites.
- Import test helpers through `tests.<owner>`. Pytest uses importlib mode, so
  test directories do not shadow production or third-party packages.
- Keep test data beside its consuming suite. Harness golden trees are exact
  fixtures; their instruction files are data, not repository instructions.

## Moving and verifying tests

Update imports, resource paths, fixture scope, CI selection, and documented
commands together. Compare collected test IDs through an old-to-new path map,
including parameters and markers. Keep assertions and fixture contents intact
during a directory-only migration.

`packaging/wheel-tests.txt` lists the modules run against installed wheels.
Update it when moving a listed module or adding a wheel-compatible contract.
These tests run outside the checkout and must not import Reef from source.
The source and sandbox jobs still collect all of `tests/`.

Run `python .github/scripts/check_test_layout.py` from the repository root.
The check rejects misplaced test modules and invalid wheel-suite entries; it
also runs through pre-commit in CI. Update this map and the check together when
adding an owner directory.

Apply placement rules to new tests. Keep moves of existing tests in focused
cleanup changes rather than unrelated feature PRs.
