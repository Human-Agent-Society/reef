#!/bin/bash
# Oracle: the origin — a deliberately mediocre valid point (reward exactly
# 1/3). Proves the pipeline; beating it is the agent's job.
set -euo pipefail
printf '{"x": 0.0, "y": 0.0}' > /tmp/solution.json
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
