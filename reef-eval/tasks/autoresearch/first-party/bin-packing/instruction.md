# Bin packing

`/app/items.json` holds 60 item sizes and a bin capacity of 100. Pack every
item into as few bins as possible.

**Reward = (first-fit bin count) / (your bin count).** The classic first-fit
heuristic scores exactly 1.0; every bin you save pushes the reward above 1.0.
(First-fit-decreasing and smarter searches do save bins here.)

## Solution format

```json
{"bins": [[0, 7, 12], [1, 3], ...]}
```

Item indices, each appearing exactly once; every bin's sizes must sum to
≤ 100. Invalid packings score 0.

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
