# Serving a local reasoning model behind the official code

The official graders, the openclaw tools, and the ported evolver size
their completion budgets for a model whose whole completion is the
answer. A reasoning model counts its reasoning tokens against the same
budgets, so the official values return empty content deterministically:
the judge at 1200 and 2048, the openclaw image and pdf tools at 4096,
and the evolver stages at 8192. The official code stays at its pin, so
every accommodation lives in `judge_proxy.py`, a reverse proxy between
the harness and the model server:

- `max_tokens` below 32768 is raised to 32768. The measured need of the
  largest improve reply is about 15k tokens of reasoning plus content.
- `max_completion_tokens` below 16384 is raised to 16384. The floor is
  chosen so the openclaw main path budget of 32000 is never touched.
- The evolver decide, create, and merge stages are constrained to
  `json_object`. The model drops one closing brace on long JSON replies
  (finish stop, depth short by one), which the official zero tolerance
  parser turns into a silent skip; constrained decoding removes the
  failure. Verified against captured failing requests: 6 of 6 parse.
- Streamed responses pass through incrementally; night stage calls are
  also mirrored to `night-tap.jsonl` for audit, bytes unchanged.

Point `REEF_UPSTREAM_URL` and `REEF_SC_JUDGE_BASE` at the proxy port
(see `serving.env.template`) and run both runs through it. The proxy
must serve BOTH runs: routing only one run through it changes budgets
on the measured quantity and voids the comparison.

Known limitation, measured: about 2 to 3 percent of image tool and
night decide calls burn the whole raised budget on reasoning and still
return empty content. The failure is a heavy tail draw, not input
determined: the same requests replayed 63 times offline never exceeded
7k reasoning tokens. Both runs share the loss.
