# Guidance-TTT Reef results

This directory records two completed Guidance-TTT searches run through Reef.
Both used Qwen3-14B as the trainable guidance model, GLM-5.2 as the frozen
executor, 8 groups of 16 rollouts per update, and 30 training updates.

| Task | Seed | Search-time best | Fixed-candidate check |
|---|---:|---:|---:|
| Polyomino Packing | 27.8105 | 89.7965 | Evaluated by the same deterministic 70-case suite |
| TriMul | 10,177.40 µs | 1,110.85 µs | 1,158.46 ± 3.76 µs over three H100 repeats |

The search-time best is the best score observed while the archive was being
built. The TriMul repeat check holds the final kernel and software stack fixed,
then runs the evaluator three times. It is the better number to use when
reporting stable latency. Polyomino's evaluator is deterministic, so it does
not need a timing repeat.

These are single completed runs, not estimates over random seeds. The records
show what happened in these runs; they do not by themselves establish that
every gain was caused by policy training. The case studies make a narrower
claim: they pair a guidance message with the child produced from that parent
and the score returned by the verifier.

## Files

- [`runs.json`](runs.json) contains configurations, summary metrics, selected
  archive identifiers, reevaluation values, and SHA-256 provenance.
- [`polyomino.md`](polyomino.md) follows one guidance-to-candidate transition in
  the packing run.
- [`trimul.md`](trimul.md) follows one guidance-to-kernel transition in the
  Triton run.
- [`check_results.py`](check_results.py) checks the run set and the TriMul repeat
  statistics.

Run the compact-record checks from this directory:

```bash
python3 check_results.py
```

The full committed archives are hundreds of megabytes, so they are not copied
into Git. Their hashes in `runs.json` bind these compact records to the remote
run artifacts. The historical Polyomino score `91.8907` is deliberately absent:
its artifact came from an `open-ttt-verl` run rather than a Reef run.
