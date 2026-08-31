# Symbolic regression

`/app/train.json` holds 40 noiseless samples (x, y) of an unknown function
on roughly x ∈ [−3, 3]. Recover the formula.

## Solution format

```json
{"expr": "2*x + cos(x)"}
```

Allowed: numbers, `x`, `+ - * / **`, unary minus, and
`sin cos tan exp log sqrt abs`. Anything else scores 0. Max 2000 chars.

## Scoring — read this carefully

The judge scores each submission as 1/(1+RMSE) on the **training** points.
A **final judge** then scores your best submission ONCE on **held-out
points you never see, on a wider range that lies OUTSIDE the training
interval** — that number is your grade. An expression that memorizes the
training set (e.g. a high-degree fitted polynomial) will score near 1.0 in
session and collapse on the final grade, because a polynomial that matches
the samples still diverges from the true function once you extrapolate.
Only the true structure generalizes — the exact formula scores 1.0.

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
