# Function minimization

Minimize the Levi N.13 function over real (x, y):

```
f(x, y) = sin²(3πx) + (x−1)²(1 + sin²(3πy)) + (y−1)²(1 + sin²(2πy))
```

It is deceptive: gradient descent from most starts lands in one of many local
minima. Your **reward is 1 / (1 + f(x, y))** — the global optimum scores 1.0.

## Solution format

```json
{"x": 0.0, "y": 0.0}
```

Coordinates must be finite and within |v| ≤ 100.

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

Baseline for reference: the origin (0, 0) scores 1/3.
