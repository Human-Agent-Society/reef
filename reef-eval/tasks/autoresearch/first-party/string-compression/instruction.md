# String compression

Compress `/app/corpus.txt` (about 9 KB of text) as tightly as you can, by
shipping a **decompressor program plus a payload**.

**Reward = corpus bytes / (decompressor source bytes + payload string bytes).**
zlib gets ≈ 3.5; the corpus has structure worth exploiting for more.

## Solution format

```json
{"decompressor": "<python source>", "payload": "<any string, e.g. base64>"}
```

When graded, your decompressor runs as a Python program with a global
variable `PAYLOAD` holding your payload string; it must print the exact
corpus to stdout (byte-exact, 15 s limit, decompressor ≤ 100 KB).

## Scoring integrity

The judge runs your decompressor in a clean environment where **every copy
of the corpus is deleted first** — a "decompressor" that reads the corpus
from disk finds nothing and scores 0. It must reconstruct the text from
the payload alone.

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
