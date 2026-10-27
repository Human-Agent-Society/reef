# Bug fix flow

The request `run.py bugfix` files, the text of the fenced block below, one line:

```text
when I ask you to fix a bug: reproduce it first with a failing test, fix it, run the tests, then have a second agent review the diff before you tell me it is done
```

A working answer is a rules file or a skill that names the four steps in order (reproduce with a failing test, fix, run the tests, review), or an `agent_command` such as `/fix-bug`, or a `code_extension` that runs a second pi session over the diff before the reply. The evaluation runs whatever the proposer wrote on the recipe's health task, which checks that the harness still works and not that it follows the workflow; the design and the review on the step's page say whether the change answers the request.

## The workspace fixture

`workspace/` is a tiny Python project with one failing test: `adder.py` carries `sum_to`, the sum of the integers from 1 to n inclusive, written with `range(1, n)`, so it stops one short; `test_adder.py` expects `sum_to(4) == 10` and fails with 6. `run.py` copies the directory into `work/bugfix-<timestamp>/session/workspace/` before the show session, so the committed fixture stays as it is.

The show session runs `pi -p "fix the bug in adder.py"` on the installed tree in that copy. What it should do under the change: run `pytest` and see the failure first (or write a test that shows it), edit `adder.py`, run `pytest` again and see it pass, then review the diff before it says it is done. `run.py` prints the session's tool calls in order, read from pi's own session log, so the order is on the record either way.
