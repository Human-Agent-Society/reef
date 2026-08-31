# _template: copy this folder to make a task

The template **is a complete working task** (maximize `x` in `[0, 1]`):
the test suite is green the moment you copy it. Replace one piece at a
time and keep the suite green:

```bash
mkdir -p tasks/autoresearch/my-suite       # your own benchmark folder
cp -r tasks/_template tasks/autoresearch/my-suite/my-task
pytest tests/test_task_suite.py            # your copy is picked up automatically
```

Work through the `TODO(task)` markers:

1. `task.toml`: name, description, budget (`[agent] timeout_sec`)
2. `instruction.md`: the problem, the solution format, the submission protocol
3. `environment/score.py`: THE scoring rule, run by the judge on every submission
4. `environment/judge_config.json`: the submission budget
5. `tests/grader_tests.json`: one case per scoring rule: solution → expected reward → why
6. `solution/solve.sh`: the reference solution (submits once; proves the pipeline)
7. `environment/Dockerfile`: anything the agent's container should contain

Optionally add `environment/final.py` (same signature as score.py): the
**final judge**: hidden tests, run exactly once on the best submission
when the verifier finalizes. Put held-out data there; the submission
budget can never probe it.

You never edit `judge_server.py`, `docker-compose.yaml`,
`Dockerfile.judge`, or `tests/`; they are the generic judge plumbing.
Prove the task in containers with
`harbor trial start -p tasks/autoresearch/my-task`.

Full reference for the judge protocol, the scoring contract, the
`grader_tests.json` schema, network policy, and GPU tasks:
[docs/authoring-tasks.md](../../docs/authoring-tasks.md).
