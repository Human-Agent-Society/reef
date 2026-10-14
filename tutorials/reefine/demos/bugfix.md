# Bug fix flow

The request `run.py bugfix` files, the text of the fenced block below, one line:

```text
when I ask you to fix a bug: reproduce it first with a failing test, fix it, run the tests, then have a second agent review the diff before you tell me it is done
```

The candidate must make pi perform the requested workflow. A rule that lists the steps is tested by running the changed harness on the application task. Formal evaluation uses a fresh copy of the fixture, independent model judgments and trusted arithmetic checks. A missing failing test, failed test run or failed second-agent review rejects the candidate. The evaluator also compares two protected tasks with the current release.

## The workspace fixture

`workspace/` is a tiny Python project with one failing test: `adder.py` carries `sum_to`, the sum of the integers from 1 to n inclusive, written with `range(1, n)`, so it stops one short; `test_adder.py` expects `sum_to(4) == 10` and fails with 6. Formal evaluation uses its own copy, retained under `work/deployment/steps/reefine-demo/<step>/reefine-evaluation/request/workspace/`. `run.py` creates another copy under `work/bugfix-<timestamp>/workspace/` for the show session. The committed fixture stays as it is.

The show session runs `reef-pi -p "fix the bug in adder.py"` in that copy. What it should do under the change: run `pytest` and see the failure first (or write a test that shows it), edit `adder.py`, run `pytest` again and see it pass, then have a second agent review the diff successfully before it says it is done. `run.py` prints the session's tool calls in order, read from the receipts the wrapper spools at exit, so the order is on the record either way.

The fixture copy has its own Git repository. The evaluation and show sessions use separate copies. `run.py` exits with an error when evaluation or the show workflow fails; it saves the result before exiting.
