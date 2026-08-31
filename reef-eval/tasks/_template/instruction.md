# Maximize x   <!-- TODO(task): replace the problem; keep the protocol -->

Find the largest `x` in `[0, 1]`. Your score is `x` itself.

(This placeholder problem is trivial on purpose — the template ships as a
complete working task so you can copy it, see the suite pass, then replace
the problem while keeping the protocol below.)

## Solution format

A JSON file:

```json
{"x": 0.5}
```

## How you are scored

Submit your solution file to the judge; your final score is your **best
submission** (`$JUDGE_URL` is set in your environment):

```bash
curl -s -X POST --data-binary @solution.json "$JUDGE_URL/submit"
# → {"n": 0, "score": 0.5, "reason": "ok", "best": 0.5, "remaining": 99}
curl -s "$JUDGE_URL/status"
```

You have a **limited number of submissions** (see `remaining`) — evaluate
candidates yourself however you like, and spend submissions on the ones
worth scoring. Being stopped at the deadline is normal; your best
submission so far is what counts.
