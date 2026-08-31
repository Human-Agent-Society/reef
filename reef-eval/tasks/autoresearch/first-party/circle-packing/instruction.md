# Circle packing (n=3)

Place **3 non-overlapping circles inside the unit square** to maximize the
**sum of their radii**. Circles must lie entirely inside the square and must
not overlap (tangency is allowed).

## Solution format

```json
{"circles": [[x1, y1, r1], [x2, y2, r2], [x3, y3, r3]]}
```

Your score is the sum of radii, computed in exact arithmetic — overlap or
out-of-bounds beyond tolerance 1e-9 scores 0.

## How you are scored

Submit your solution file to the judge — your final score is your **best
submission** (`$JUDGE_URL` is set in your environment):

```bash
curl -s -X POST --data-binary @solution.json "$JUDGE_URL/submit"
# → {"n": 0, "score": ..., "reason": "...", "best": ..., "remaining": ...}
curl -s "$JUDGE_URL/status"
```

You have a **limited number of submissions** (see `remaining`) — evaluate
candidates yourself however you like, and spend submissions on the ones
worth scoring. Being stopped at the deadline is normal; your best
submission so far is what counts.

The known optimum for n=3 is ≈ 1.007626. Beat the greedy baseline (0.75)
and get as close to the optimum as you can within the budget.
