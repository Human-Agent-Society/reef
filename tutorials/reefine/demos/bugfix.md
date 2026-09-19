# Bug fix flow

The request `run.py bugfix` files, the text of the fenced block below, one line:

```text
when I ask you to fix a bug: reproduce it first with a failing test, fix it, run the tests, then have a second agent review the diff before you tell me it is done
```

A working answer is a `rules` entry or a skill that names the four steps in order (reproduce with a failing test, fix, run the tests, review), or an `agent_command` such as `/fix-bug`, or a `code_extension` that runs a second pi session over the diff before the reply. The gate scores whatever the proposer wrote on the recipe's three arithmetic tasks, which a workflow change does not move; see the README's known limitations.

## The workspace fixture

`workspace/` is a tiny Python project with one failing test: `adder.py` carries `sum_to`, the sum of the integers from 1 to n inclusive, written with `range(1, n)`, so it stops one short; `test_adder.py` expects `sum_to(4) == 10` and fails with 6. `run.py` copies the directory into `work/bugfix-<timestamp>/workspace/` before the show session, so the committed fixture stays as it is.

The show session runs `reef-pi -p "fix the bug in adder.py"` in that copy. What it should do under the change: run `pytest` and see the failure first (or write a test that shows it), edit `adder.py`, run `pytest` again and see it pass, then review the diff before it says it is done. `run.py` prints the session's tool calls in order, read from the receipts the wrapper spools at exit, so the order is on the record either way.
