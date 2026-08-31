# Traveling salesman: shorten the tour

`/app/cities.json` holds 40 fixed cities in the unit square. Find a short
**closed tour** visiting every city exactly once.

**Reward = (identity-tour length) / (your tour length).** Visiting cities in
file order scores exactly 1.0; every improvement pushes the reward above 1.0.
Good heuristics (nearest-neighbor + 2-opt and beyond) reach ~1.8–2.0.

## Solution format

```json
{"tour": [0, 17, 5, ...]}
```

A permutation of 0..39 — invalid permutations score 0.

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
