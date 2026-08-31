"""Random search: the smallest possible method under the judge protocol.

This is a method, not a harness. It runs inside the task container, where
``minimal_harness.py`` puts it, and it also runs on its own under
``reef-eval run ... --local``.

Pure stdlib, no LLM, no keys. It shows the whole contract from the
method's side: generate candidates, POST them to ``$JUDGE_URL/submit``,
read the score that comes back (the judge's response is the only
feedback), and stop when the time or submission budget runs out.

Written for the circle-packing task; the protocol is identical for every
task, only the solution format changes.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

JUDGE_URL = os.environ["JUDGE_URL"]
BUDGET_SEC = float(os.environ.get("BUDGET_SEC", "30"))
random.seed(os.environ.get("SEED"))  # unset = system entropy, the default


def submit(solution: dict) -> dict | None:
    request = urllib.request.Request(
        JUDGE_URL + "/submit", data=json.dumps(solution).encode()
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError:
        return None  # 429: out of submissions, stop searching


t0 = time.monotonic()
best = 0.0
while time.monotonic() - t0 < BUDGET_SEC:
    circles = [
        [random.random(), random.random(), random.uniform(0.02, 0.35)] for _ in range(3)
    ]
    result = submit({"circles": circles})
    if result is None:
        break
    best = result["best"]

print(f"random search done: best={best}")
