# Examples

The first two run with zero setup; the rest need Docker and
`pip install reef-eval[harbor]`.

| Script | What it shows | Needs |
|---|---|---|
| [`quickstart.py`](quickstart.py) | the Lab API: episodes, resume, the results table (agents are simulated) | nothing |
| [`stream_quickstart.py`](stream_quickstart.py) | a stream: carried state, the learning curve, resume | nothing |
| [`random_search.py`](random_search.py) | the judge protocol from the method's side, in ~20 lines: submit, read the score, stop at the budget | a running judge (`reef-eval run ... --local`) |
| [`minimal_harness.py`](minimal_harness.py) | the smallest complete harness: a ~25-line `BaseAgent` that puts `random_search.py` in the container and runs it | Docker |
| [`llm_harness.py`](llm_harness.py) | the autoresearch loop with a model in it: propose, submit, and put the judge's score into the next prompt | Docker + an API key |

`minimal_harness.py` and `llm_harness.py` are the two halves worth
reading together. Both are the same integration, an adapter that reaches
the container over `environment.exec`; they differ only in what proposes
the candidates.

## Reference harness adapters

[`harnesses/`](harnesses) holds comparable OpenEvolve, Codex, and CORAL
adapters, all submitting to the same judge, with `run_harness.py` as the
entry point. These are baselines for comparison, not a starting point;
begin with `minimal_harness.py`.
